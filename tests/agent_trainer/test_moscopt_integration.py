"""MOSCOPT end-to-end integration test — full training run with real data.

Uses:
- Provider/Model from ``~/.summerclaw/config.json`` (safe, no direct read)
- Data from ``resources/trainer-agent/split_jsonl/``
- Config from ``resources/trainer-agent/moscopt.yaml``
- Custom scorer from ``resources/trainer-agent/custom-scorer.py``

Verifies the complete 6-stage pipeline + epoch hooks + test evaluation.

Expected scale:
- data.jsonl (53 items) → train:val = 2:1 → ~35 train, ~18 val
- test.jsonl (124 items) → test split
- batch_size=40 → 1 step/epoch, 4 epochs → 4 total steps
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

# ── Paths ────────────────────────────────────────────────────────────────

RESOURCES = Path(__file__).resolve().parents[2] / "resources" / "trainer-agent"
CONFIG_PATH = Path.home() / ".summerclaw" / "config.json"


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def provider_and_model():
    """Create an LLM provider and model name from the user's config.

    Uses ``SummerClaw.from_config()`` which safely loads
    ``~/.summerclaw/config.json`` without exposing sensitive fields.
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config not found: {CONFIG_PATH}")

    from summerclaw.summerclaw import SummerClaw

    bot = SummerClaw.from_config(CONFIG_PATH)
    provider = bot._loop.provider
    model = bot._loop.model
    return provider, model


@pytest.fixture
def prepared_data(tmp_path: Path):
    """Parse JSONL resources and write train/val/test splits to *tmp_path*.

    Returns a dict with paths:
    - ``data_dir``: directory containing ``train/``, ``val/``, ``test/``
    - ``scorer_path``: path to ``custom-scorer.py``
    - ``train_count`` / ``val_count`` / ``test_count``: item counts
    """
    from summerclaw.agent_trainer.datasets.splitter import (
        parse_file,
        split_with_test,
        write_splits,
    )

    data_items = parse_file(RESOURCES / "split_jsonl" / "data.jsonl")
    test_items = parse_file(RESOURCES / "split_jsonl" / "test.jsonl")

    # Mark all items with custom scorer
    for item in data_items + test_items:
        if not item.get("scorer"):
            item["scorer"] = "custom"

    # data.jsonl → train:val (2:1), test.jsonl → test (separate)
    splits = split_with_test(
        data_items, test_items,
        train_ratio=2, val_ratio=1, seed=42,
    )

    data_dir = tmp_path / "uploaded_data"
    write_splits(splits, data_dir)

    return {
        "data_dir": data_dir,
        "scorer_path": RESOURCES / "custom-scorer.py",
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
    }


@pytest.fixture
def flat_config():
    """Load and flatten the MOSCOPT YAML config."""
    from summerclaw.agent_trainer.config import (
        load_config as load_yaml,
        flatten_config,
    )

    yaml_cfg = load_yaml(str(RESOURCES / "moscopt.yaml"))
    return flatten_config(yaml_cfg)


# ── Main Integration Test ────────────────────────────────────────────────


async def test_moscopt_full_training_run(
    provider_and_model,
    prepared_data,
    flat_config,
    tmp_path,
):
    """Run a complete MOSCOPT training and verify pipeline integrity.

    This test exercises:
    1. Provider initialization from config.json
    2. Data splitting and loading
    3. MOSCOPT algorithm construction with YAML parameters
    4. Environment adapter creation
    5. TrainerEngine orchestration (6-stage pipeline × 4 epochs)
    6. Initial skill generation (LLM-driven)
    7. Baseline evaluation
    8. Per-step rollout/reflect/aggregate/select/update/evaluate
    9. Epoch-end hooks (slow update, meta skill, collective evolution)
    10. Test set evaluation
    11. State persistence (summary, history, algorithm state)
    """
    provider, model = provider_and_model
    flat = flat_config

    # ── 1. Build Algorithm ─────────────────────────────────────────────
    from summerclaw.agent_trainer.algorithms.moscopt.algorithm import (
        MOSCOPTAlgorithm,
    )

    algo = MOSCOPTAlgorithm(
        provider=provider,
        model=model,
        minibatch_size=flat.get("minibatch_size", 8),
        edit_budget=flat.get("edit_budget", 4),
        workers=2,
        optimizer_model=flat.get("optimizer_model"),
        update_mode=flat.get("skill_update_mode", "patch"),
        lr_mode=flat.get("lr_scheduler", "cosine"),
        min_lr=flat.get("min_edit_budget", 2),
        reasoning_effort=flat.get("reasoning_effort", "medium"),
        merge_batch_size=flat.get("merge_batch_size", 8),
        max_analyst_rounds=flat.get("max_analyst_rounds", 3),
        use_slow_update=flat.get("use_slow_update", True),
        use_meta_skill=flat.get("use_meta_skill", True),
        longitudinal_pair_policy=flat.get("longitudinal_pair_policy", "mixed"),
        use_rejected_buffer=flat.get("use_rejected_buffer", True),
        pool_size=flat.get("pool_size", 5),
        activate_count=flat.get("activate_count", 2),
        evolution_interval=flat.get("evolution_interval", 5),
        evolution_count=flat.get("evolution_count", 1),
        gating_granularity=flat.get("gating_granularity", "task"),
        ema_beta=flat.get("ema_beta", 0.3),
        min_activations=flat.get("min_activations", 5),
        diversity_threshold=flat.get("diversity_threshold", 0.85),
        val_sample_ratio=flat.get("val_sample_ratio", 1.0),
    )

    # ── 2. Build Environment ───────────────────────────────────────────
    from summerclaw.agent_trainer.env.summerclaw_env import (
        SummerClawEnvAdapter,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "train-output"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = SummerClawEnvAdapter(
        provider=provider,
        model=model,
        workspace=workspace,
        train_out_dir=out_dir,
        memory_algorithm_name=None,
        context_window_tokens=65536,
        max_tool_iterations=20,
        max_tool_result_chars=16000,
        temperature=0.1,
        max_tokens=8192,
        workers=2,
    )

    # ── 3. Build DataLoader + Engine ──────────────────────────────────
    from summerclaw.agent_trainer.datasets.loader import DataLoader
    from summerclaw.agent_trainer.engine.trainer import TrainerEngine

    loader = DataLoader(str(prepared_data["data_dir"]))
    assert loader.split_names, "No data splits loaded"

    num_epochs = flat.get("num_epochs", 4)
    batch_size = flat.get("batch_size", 40)

    engine = TrainerEngine(
        algorithm=algo,
        env=env,
        data_loader=loader,
        out_dir=out_dir,
        skill_init="",  # Will be auto-generated by LLM
        num_epochs=num_epochs,
        batch_size=batch_size,
        edit_budget=flat.get("edit_budget", 4),
        seed=42,
        eval_test=True,
        max_test_items=20,  # Cap test eval to 20 items (avoid 2×124 full-set rollout)
    )

    # Create task directory and place custom scorer
    engine._ensure_out_dir()
    if prepared_data["scorer_path"].exists():
        shutil.copy2(
            prepared_data["scorer_path"],
            engine.out_dir / "custom-scorer.py",
        )

    # ── 4. Run Training ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MOSCOPT Integration Test — Starting Training")
    print(f"  Provider: {type(provider).__name__}")
    print(f"  Model: {model}")
    print(f"  Train: {prepared_data['train_count']} items")
    print(f"  Val:   {prepared_data['val_count']} items")
    print(f"  Test:  {prepared_data['test_count']} items")
    print(f"  Epochs: {num_epochs}, Batch: {batch_size}")
    print(f"  Pool: N={algo.pool_size}, K={algo.activate_count}")
    print(f"  Out: {engine.out_dir}")
    print(f"{'='*60}\n")

    history = await engine.train()

    # ── 5. Verification Checklist ─────────────────────────────────────

    task_dir = engine.out_dir
    print(f"\n{'='*60}")
    print(f"MOSCOPT Integration Test — Verifying Results")
    print(f"  Task dir: {task_dir}")
    print(f"  Total steps: {history.total_steps}")
    print(f"  Total epochs: {history.total_epochs}")
    print(f"{'='*60}\n")

    # 5.1 summary.json
    summary_path = task_dir / "summary.json"
    assert summary_path.exists(), "summary.json not created"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"  summary.json: OK")

    # 5.2 baseline_score computed
    assert summary["baseline_score"] >= 0, (
        f"baseline_score not computed: {summary['baseline_score']}"
    )
    print(f"  baseline_score: {summary['baseline_score']:.4f}")

    # 5.3 best_score reasonable
    assert summary["best_score"] >= 0, (
        f"best_score invalid: {summary['best_score']}"
    )
    print(f"  best_score: {summary['best_score']:.4f}")

    # 5.4 total_steps correct
    train_count = prepared_data["train_count"]
    steps_per_epoch = max(1, train_count // batch_size)
    expected_total = num_epochs * steps_per_epoch
    assert summary["total_steps"] == expected_total, (
        f"total_steps mismatch: {summary['total_steps']} != {expected_total}"
    )
    print(f"  total_steps: {summary['total_steps']} (expected {expected_total})")

    # 5.5 epoch_stats complete
    # May be fewer if convergence detected early
    assert len(summary["epoch_stats"]) >= 1, "No epoch_stats recorded"
    print(f"  epoch_stats: {len(summary['epoch_stats'])} epochs")

    # 5.6 skill_v0000.md exists
    skills_dir = task_dir / "skills"
    assert skills_dir.exists(), "skills/ directory not created"
    v0 = skills_dir / "skill_v0000.md"
    assert v0.exists(), "Initial skill (skill_v0000.md) not saved"
    v0_content = v0.read_text(encoding="utf-8")
    assert len(v0_content) > 0, "Initial skill is empty"
    print(f"  skill_v0000.md: {len(v0_content)} chars")

    # 5.7 At least 1 skill version
    all_skills = list(skills_dir.glob("skill_v*.md"))
    assert len(all_skills) >= 1, "No skill versions found"
    print(f"  skill versions: {len(all_skills)}")

    # 5.8 config.json exists
    config_path = task_dir / "config.json"
    assert config_path.exists(), "config.json not created"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config.get("algorithm") == "moscopt", (
        f"algorithm mismatch: {config.get('algorithm')}"
    )
    print(f"  config.json: algorithm={config.get('algorithm')}")

    # 5.9 runtime_state completed
    rs_path = task_dir / "runtime_state.json"
    assert rs_path.exists(), "runtime_state.json not created"
    rs = json.loads(rs_path.read_text(encoding="utf-8"))
    assert rs.get("status") in ("completed", "stopped"), (
        f"Training did not complete: status={rs.get('status')}"
    )
    print(f"  runtime_state: status={rs.get('status')}")

    # 5.10 history has steps
    assert history.total_steps > 0, "No training steps recorded"
    print(f"  history.total_steps: {history.total_steps}")

    # 5.11 algorithm_state.json exists
    algo_state_path = task_dir / "algorithm_state.json"
    assert algo_state_path.exists(), "algorithm_state.json not saved"
    algo_state = json.loads(algo_state_path.read_text(encoding="utf-8"))
    # MOSCOPT state should contain pool data
    assert "pool" in algo_state or "skills" in algo_state or len(algo_state) > 0, (
        "algorithm_state appears empty"
    )
    print(f"  algorithm_state.json: {len(algo_state)} top-level keys")

    # 5.12 test_evaluation directory
    test_eval_dir = task_dir / "test_evaluation"
    if flat.get("eval_test", True):
        assert test_eval_dir.exists(), "test_evaluation/ not created"
        print(f"  test_evaluation/: OK")

    # 5.13 training_log.jsonl exists
    log_path = task_dir / "training_log.jsonl"
    assert log_path.exists(), "training_log.jsonl not created"
    log_lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    print(f"  training_log.jsonl: {len(log_lines)} events")

    # 5.14 epoch output directories
    epochs_dir = task_dir / "epochs"
    assert epochs_dir.exists(), "epochs/ directory not created"
    epoch_dirs = list(epochs_dir.iterdir())
    print(f"  epochs/: {len(epoch_dirs)} epoch dirs")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ALL CHECKS PASSED")
    print(f"  Baseline: {summary['baseline_score']:.4f}")
    print(f"  Best:     {summary['best_score']:.4f} (step {summary['best_step']})")
    print(f"  Steps:    {summary['total_steps']}")
    print(f"  Accepts:  {summary.get('total_accepts', '?')}")
    print(f"  Rejects:  {summary.get('total_rejects', '?')}")
    print(f"  Skips:    {summary.get('total_skips', '?')}")
    print(f"  Time:     {summary.get('total_wall_time_s', '?')}s")
    print(f"{'='*60}\n")
