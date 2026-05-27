"""Generic training engine — orchestrates the 6-stage algorithm pipeline.

The :class:`TrainerEngine` is algorithm-agnostic; it delegates all
algorithm-specific logic to a :class:`BaseAlgorithm` instance obtained
from the registry.

Pipeline per step:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject

Dashboard integration:
  The trainer publishes progress via callbacks so the Gradio dashboard
  can display real-time status.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from summerclaw.agent_trainer.base import BaseAlgorithm
from summerclaw.agent_trainer.datasets.loader import DataLoader
from summerclaw.agent_trainer.evaluation.gate import evaluate_gate
from summerclaw.agent_trainer.types import (
    GateResult,
    Patch,
    RawPatch,
    RolloutResult,
    TrainingHistory,
    TrainingStep,
)


# ── Initial skill resolution ──────────────────────────────────────────────

async def _resolve_skill_init(
    skill_init: str,
    skill_init_path: str,
    env: Any,
    data_loader: Any,
    out_dir: "Path",
) -> str:
    """Resolve the initial skill content via file path or LLM generation.

    Called by :meth:`TrainerEngine.train` when ``skill_init`` is empty.

    1. If *skill_init_path* points to a readable file, load it.
    2. Otherwise, if the environment has an LLM provider and the data
       loader has training items, ask the LLM to synthesize an initial
       skill from up to 5 sample items and persist it.
    3. Return empty string as a last resort.
    """
    # Scheme 1: file path
    if skill_init_path:
        p = Path(skill_init_path).expanduser()
        if p.is_file():
            content = p.read_text(encoding="utf-8").strip()
            if content:
                logger.info("[SKILL-INIT] loaded from file: {} ({} chars)", p, len(content))
                return content

    # Scheme 2: LLM generation
    provider = getattr(env, "provider", None)
    model = getattr(env, "model", "")
    if not provider or not model:
        logger.warning("[SKILL-INIT] no LLM provider on env; cannot auto-generate")
        return ""

    if not data_loader:
        logger.warning("[SKILL-INIT] no data loader; cannot auto-generate")
        return ""

    try:
        train_items = data_loader.train.items
    except (KeyError, AttributeError):
        logger.warning("[SKILL-INIT] no train split available")
        return ""

    if not train_items:
        logger.warning("[SKILL-INIT] train split is empty")
        return ""

    try:
        from summerclaw.agent_trainer.algorithms.skillopt.initial_skill import (
            generate_initial_skill_from_data,
        )
        return await generate_initial_skill_from_data(
            provider=provider,
            model=model,
            items=train_items,
            out_dir=str(out_dir),
        )
    except Exception as exc:
        logger.error("[SKILL-INIT] LLM generation failed: {}", exc)
        return ""


# ── Persistence helpers ───────────────────────────────────────────────────

def _skill_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_skill(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _load_skill(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_events_from_jsonl(path: str | Path, max_entries: int = 500) -> list[dict]:
    """Load recent events from a JSONL log file."""
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except (OSError, IOError):
        pass
    return events[-max_entries:] if len(events) > max_entries else events


def _write_jsonl(path: str, records: list[dict]) -> None:
    """Write records as JSON Lines file (append=False, overwrite)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _extract_failure_patterns(results: list[RolloutResult]) -> list[dict]:
    """Aggregate failure patterns from rollout results.

    Returns a list of dicts: ``[{pattern, count, task_ids}, ...]``.
    """
    buckets: dict[str, dict] = {}
    for r in results:
        if r.hard:
            continue
        reason = r.fail_reason or "unknown_failure"
        if reason not in buckets:
            buckets[reason] = {"pattern": reason, "count": 0, "task_ids": []}
        buckets[reason]["count"] += 1
        if r.id:
            buckets[reason]["task_ids"].append(r.id)
    return sorted(buckets.values(), key=lambda x: -x["count"])


def _ensure_selection_eval_artifacts(
    sel_eval_dir: str,
    rollout_results: list[RolloutResult],
    candidate_skill: str,
    val_items: list[dict],
) -> None:
    """Ensure selection_eval directory has predictions/ and results.jsonl.

    If ``algorithm.evaluate()`` already wrote these files we leave them
    untouched.  Otherwise we create them from the rollout results we have
    in memory so the output structure mirrors the official SkillOpt layout.
    """
    pred_dir = os.path.join(sel_eval_dir, "predictions")
    results_jsonl = os.path.join(sel_eval_dir, "results.jsonl")
    # Only write if not already present
    if not os.path.exists(results_jsonl):
        os.makedirs(sel_eval_dir, exist_ok=True)
        _write_jsonl(results_jsonl, [r.to_dict() for r in rollout_results])
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir, exist_ok=True)
        for r in rollout_results:
            _save_json(os.path.join(pred_dir, f"{r.id}.json"), r.to_dict())


# ── Progress callback ────────────────────────────────────────────────────

ProgressCallback = Callable[[str, dict], None]
"""Callback signature: (event_type, payload_dict).

Event types:
  "step_start", "phase_done", "step_done", "epoch_done",
  "training_done", "error"
"""


# ── Trainer Engine ────────────────────────────────────────────────────────

class TrainerEngine:
    """Algorithm-agnostic training engine.

    Parameters
    ----------
    algorithm : BaseAlgorithm
        The pluggable algorithm instance.
    env : SummerClawEnvAdapter
        The environment adapter.
    data_loader : DataLoader
        Training data loader.
    out_dir : str | Path
        Root output directory for this training run.
    skill_init : str
        Initial skill document content.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Items per training batch.
    edit_budget : int
        Maximum edits per step (learning rate L).
    seed : int
        Random seed.
    workers : int
        Max concurrent operations.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Any,
        data_loader: DataLoader | None = None,
        out_dir: str | Path = ".",
        skill_init: str = "",
        skill_init_path: str = "",
        num_epochs: int = 3,
        batch_size: int = 5,
        edit_budget: int = 4,
        seed: int = 42,
        workers: int = 4,
    ):
        self.algorithm = algorithm
        self.env = env
        self.data_loader = data_loader
        self.out_dir = Path(out_dir)
        self._task_dir_created = False
        self.skill_init = skill_init
        self.skill_init_path = skill_init_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.edit_budget = edit_budget
        self.seed = seed
        self.workers = workers

        # Runtime state
        self._current_skill = skill_init
        self._best_skill = skill_init
        self._current_score = -1.0
        self._best_score = -1.0
        self._best_step = 0
        self._history = TrainingHistory()
        self._progress_cb: ProgressCallback | None = None
        self._running = False
        self._cancel_requested = False

        # Event log (shared with dashboard for log streaming)
        self._events: list[dict] = []
        self._events_lock = threading.Lock()
        self._loguru_sink_id: int | None = None

        # Dashboard-initiated training support
        self._training_task: asyncio.Task | None = None
        self._channel_ctx: Any = None
        self._trainer_cfg: dict = {}

        # Checkpoint / resume support
        self._events_flushed: int = 0
        self._last_completed_epoch: int = 0
        self._baseline_score: float = -1.0
        self._skill_init_resolved: bool = False

        # Origin tracking (aligns with official SkillOpt runtime_state)
        self._current_origin: str = "initial_skill"
        self._best_origin: str = "initial_skill"

    # ── Properties ────────────────────────────────────────────────────

    @property
    def history(self) -> TrainingHistory:
        return self._history

    @property
    def current_skill(self) -> str:
        return self._current_skill

    @property
    def best_skill(self) -> str:
        return self._best_skill

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Callbacks ─────────────────────────────────────────────────────

    def set_progress_callback(self, cb: ProgressCallback) -> None:
        """Set a progress callback for dashboard integration."""
        self._progress_cb = cb

    def _emit(self, event: str, payload: dict | None = None) -> None:
        payload = payload or {}
        # Always append to engine-level event log (for /api/logs & Gradio log window)
        with self._events_lock:
            self._events.append({
                "time": time.strftime("%H:%M:%S"),
                "event": event,
                **payload,
            })
            # Keep last 500 events to bound memory
            if len(self._events) > 500:
                self._events = self._events[-500:]
        # Also call external progress callback if set
        if self._progress_cb:
            try:
                self._progress_cb(event, payload)
            except Exception:
                pass
        # Periodic flush to disk (every 20 events)
        if len(self._events) - self._events_flushed >= 20:
            self._flush_events_to_disk()

    def _install_log_sink(self) -> None:
        """Install a loguru sink that captures agent_trainer logs into _events.

        This bridges all ``logger.info/warning/error`` calls from algorithm
        modules (reflect, aggregate, select, slow_update, meta_skill, etc.)
        into the dashboard event stream so the Gradio log window shows the
        same level of detail as the terminal.
        """
        if self._loguru_sink_id is not None:
            return

        def _sink(message):
            if not self._running:
                return
            rec = message.record
            # loguru record uses "name" for the logger's __name__
            name = rec.get("name", "") or ""
            if "agent_trainer" not in name:
                return
            level_name = rec["level"].name
            if level_name not in ("INFO", "WARNING", "ERROR", "CRITICAL"):
                return
            text = str(message).rstrip()
            if not text:
                return
            # Extract just the raw message (not the full loguru formatted line)
            raw_msg = rec.get("message", "")
            if raw_msg:
                text = raw_msg
            # Derive a short module tag: e.g. "algorithms.skillopt.reflect"
            parts = name.split(".")
            if "algorithms" in parts:
                idx = parts.index("algorithms")
                tag = ".".join(parts[idx:])
            elif "engine" in parts:
                tag = "engine"
            elif "env" in parts:
                tag = "env"
            else:
                tag = parts[-1] if parts else name
            with self._events_lock:
                self._events.append({
                    "time": time.strftime("%H:%M:%S"),
                    "event": "log",
                    "level": level_name,
                    "module": tag,
                    "message": text,
                })
                if len(self._events) > 500:
                    self._events = self._events[-500:]

        self._loguru_sink_id = logger.add(_sink, level="INFO")

    def _uninstall_log_sink(self) -> None:
        """Remove the loguru sink installed by :meth:`_install_log_sink`."""
        if self._loguru_sink_id is not None:
            try:
                logger.remove(self._loguru_sink_id)
            except Exception:
                pass
            self._loguru_sink_id = None

    # ── Event log persistence ────────────────────────────────────────

    def _flush_events_to_disk(self) -> None:
        """Append new events to training_log.jsonl."""
        if not self._task_dir_created:
            return
        log_path = self.out_dir / "training_log.jsonl"
        new_events = self._events[self._events_flushed:]
        if not new_events:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for evt in new_events:
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")
            self._events_flushed = len(self._events)
        except OSError:
            pass

    # ── Step-level checkpoint helpers ────────────────────────────────

    @staticmethod
    def _save_step_progress(step_dir: str, progress: dict) -> None:
        """Atomically write step_progress.json."""
        _save_json(os.path.join(step_dir, "step_progress.json"), progress)

    @staticmethod
    def _load_step_progress(step_dir: str) -> dict:
        """Load step_progress.json, return empty dict if absent."""
        data = _load_json(os.path.join(step_dir, "step_progress.json"))
        return data if isinstance(data, dict) else {"completed_phases": []}

    @staticmethod
    def _save_step_artifact(step_dir: str, name: str, data: Any) -> None:
        """Save a JSON artifact into a step directory."""
        _save_json(os.path.join(step_dir, name), data)

    # ── Data loader management ────────────────────────────────────────

    def _ensure_out_dir(self) -> None:
        """Lazily create the timestamped task directory.

        Called before any operation that writes to ``out_dir``.
        On first call, creates a ``<algo>-<YYYYMMDD-HHMMSS>`` subdirectory
        under the initial ``out_dir`` (which is ``train_root``).
        """
        if self._task_dir_created:
            return
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        alg_name = getattr(self.algorithm, "name", "train")
        task_id = f"{alg_name}-{ts}"
        self.out_dir = self.out_dir / task_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._task_dir_created = True
        # Sync env adapter workspace if present
        if hasattr(self.env, "train_workspace"):
            self.env.train_workspace = self.out_dir
            self.env._workspace_ready = False
        logger.info("Task directory created: {}", self.out_dir)

    def set_data_loader(self, loader: DataLoader) -> None:
        """Set or replace the data loader (for dashboard-initiated setup)."""
        self.data_loader = loader

    def _set_task_dir(self, task_dir: str | Path) -> None:
        """Point the engine at an existing task directory.

        Used by the dashboard to resume/continue work on a previously
        created task (e.g. after a process restart).  Marks the task
        directory as already created so :meth:`_ensure_out_dir` becomes
        a no-op.
        """
        task_dir = Path(task_dir)
        if not task_dir.is_dir():
            return
        self.out_dir = task_dir
        self._task_dir_created = True
        # Sync env adapter workspace if present
        if hasattr(self.env, "train_workspace"):
            self.env.train_workspace = task_dir
            self.env._workspace_ready = False
        # Try restoring data loader from uploaded_data
        uploaded = task_dir / "uploaded_data"
        if uploaded.is_dir():
            from summerclaw.agent_trainer.datasets.loader import DataLoader
            loader = DataLoader(str(uploaded))
            if loader.split_names:
                self.data_loader = loader
                logger.info(
                    "Restored data from {}: {}", uploaded, loader.summary(),
                )
        # Restore event log
        log_path = task_dir / "training_log.jsonl"
        if log_path.exists():
            self._events = _load_events_from_jsonl(str(log_path), max_entries=500)
            self._events_flushed = len(self._events)
            logger.info("Restored {} events from log", len(self._events))
        # Restore algorithm state
        algo_state = _load_json(str(task_dir / "algorithm_state.json"))
        if algo_state and hasattr(self.algorithm, 'load_state_dict'):
            try:
                self.algorithm.load_state_dict(algo_state)
                logger.info("Restored algorithm state")
            except Exception as exc:
                logger.warning("Failed to restore algorithm state: {}", exc)
        # Restore runtime state (scores, history)
        state = _load_json(str(task_dir / "runtime_state.json"))
        if state:
            self._current_score = float(state.get("current_score", -1.0))
            self._best_score = float(state.get("best_score", -1.0))
            self._best_step = int(state.get("best_step", 0))
            self._last_completed_epoch = int(state.get("last_completed_epoch", 0))
            self._baseline_score = float(state.get("baseline_score", -1.0))
            self._skill_init_resolved = bool(state.get("skill_init_resolved", False))
            self._current_origin = str(state.get("current_origin", "initial_skill"))
            self._best_origin = str(state.get("best_origin", "initial_skill"))
            # Restore skills from paths
            curr_path = state.get("current_skill_path", "")
            if curr_path and os.path.exists(curr_path):
                self._current_skill = _load_skill(curr_path) or self._current_skill
            best_path = state.get("best_skill_path", "")
            if best_path and os.path.exists(best_path):
                self._best_skill = _load_skill(best_path) or self._best_skill
            # Restore history
            hist_data = _load_json(str(task_dir / "history.json"))
            if hist_data:
                self._history = TrainingHistory.from_dict(hist_data)
        # Restore initial skill
        skill_v0 = task_dir / "skills" / "skill_v0000.md"
        if skill_v0.exists() and (not self.skill_init or not self.skill_init.strip()):
            self.skill_init = _load_skill(str(skill_v0)) or ""
        logger.info("Engine pointed to existing task: {}", task_dir)

    def has_data(self) -> bool:
        """Check if data loader is set and has training data."""
        return self.data_loader is not None and len(self.data_loader.split_names) > 0

    # ── Control ───────────────────────────────────────────────────────

    def request_cancel(self) -> None:
        """Request cancellation of the current training run."""
        self._cancel_requested = True

    # ── Persistence ───────────────────────────────────────────────────

    def _save_state(self, step: int) -> None:
        """Persist runtime state for resume support."""
        skills_dir = self.out_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        _save_skill(str(skills_dir / f"skill_v{step:04d}.md"), self._current_skill)
        _save_skill(str(self.out_dir / "best_skill.md"), self._best_skill)

        _save_json(str(self.out_dir / "history.json"), self._history.to_dict())
        _save_json(str(self.out_dir / "runtime_state.json"), {
            "last_completed_step": step,
            "last_completed_epoch": self._last_completed_epoch,
            "current_score": self._current_score,
            "best_score": self._best_score,
            "best_step": self._best_step,
            "baseline_score": self._baseline_score,
            "skill_init_resolved": self._skill_init_resolved,
            "current_origin": self._current_origin,
            "best_origin": self._best_origin,
            "current_skill_path": str(skills_dir / f"skill_v{step:04d}.md"),
            "best_skill_path": str(self.out_dir / "best_skill.md"),
        })

        # Save algorithm state (scheduler, meta_skill, etc.)
        if hasattr(self.algorithm, 'state_dict'):
            try:
                _save_json(
                    str(self.out_dir / "algorithm_state.json"),
                    self.algorithm.state_dict(),
                )
            except Exception as exc:
                logger.warning("Failed to save algorithm state: {}", exc)

    def _save_algorithm_state(self) -> None:
        """Save algorithm state separately (called at epoch boundaries)."""
        if hasattr(self.algorithm, 'state_dict'):
            try:
                _save_json(
                    str(self.out_dir / "algorithm_state.json"),
                    self.algorithm.state_dict(),
                )
            except Exception as exc:
                logger.warning("Failed to save algorithm state: {}", exc)

    def _compute_epoch_stats(self) -> list[dict]:
        """Build per-epoch statistics from training history.

        Output aligns with the official SkillOpt ``summary.json`` schema.
        """
        stats: list[dict] = []
        for e in range(1, self._history.total_epochs + 1):
            epoch_steps = [s for s in self._history.steps if s.epoch == e]
            if epoch_steps:
                stats.append({
                    "epoch": e,
                    "steps": [s.step for s in epoch_steps],
                    "accepts": sum(1 for s in epoch_steps if "accept" in s.action),
                    "rejects": sum(1 for s in epoch_steps if s.action == "reject"),
                    "skips": sum(1 for s in epoch_steps if "skip" in s.action),
                    "best_score_at_epoch_end": epoch_steps[-1].best_score,
                    "current_score_at_epoch_end": epoch_steps[-1].current_score,
                })
        return stats

    def _load_state(self) -> int:
        """Load runtime state if available. Returns resume step."""
        state = _load_json(str(self.out_dir / "runtime_state.json"))
        if not state:
            return 0

        last_step = int(state.get("last_completed_step", 0))
        self._current_score = float(state.get("current_score", -1.0))
        self._best_score = float(state.get("best_score", -1.0))
        self._best_step = int(state.get("best_step", 0))
        self._last_completed_epoch = int(state.get("last_completed_epoch", 0))
        self._baseline_score = float(state.get("baseline_score", -1.0))
        self._skill_init_resolved = bool(state.get("skill_init_resolved", False))
        self._current_origin = str(state.get("current_origin", "initial_skill"))
        self._best_origin = str(state.get("best_origin", "initial_skill"))

        # Load skills
        curr_path = state.get("current_skill_path", "")
        if curr_path and os.path.exists(curr_path):
            self._current_skill = _load_skill(curr_path) or self.skill_init
        best_path = state.get("best_skill_path", "")
        if best_path and os.path.exists(best_path):
            self._best_skill = _load_skill(best_path) or self.skill_init

        # Load history
        hist_data = _load_json(str(self.out_dir / "history.json"))
        if hist_data:
            self._history = TrainingHistory.from_dict(hist_data)

        # Load algorithm state (scheduler, meta_skill, etc.)
        algo_state = _load_json(str(self.out_dir / "algorithm_state.json"))
        if algo_state and hasattr(self.algorithm, 'load_state_dict'):
            try:
                self.algorithm.load_state_dict(algo_state)
                logger.info("Restored algorithm state from checkpoint")
            except Exception as exc:
                logger.warning("Failed to restore algorithm state: {}", exc)

        # Restore prev_epoch_skill for SkillOpt slow update (from last epoch dir)
        if hasattr(self.algorithm, '_prev_epoch_skill') and self._last_completed_epoch > 0:
            prev_skill_path = (
                self.out_dir / "epochs" / f"epoch_{self._last_completed_epoch:02d}" / "prev_skill.md"
            )
            if prev_skill_path.exists():
                self.algorithm._prev_epoch_skill = _load_skill(str(prev_skill_path)) or ""

        # Restore initial skill from saved file if not already set
        if not self.skill_init or not self.skill_init.strip():
            v0_path = self.out_dir / "skills" / "skill_v0000.md"
            if v0_path.exists():
                self.skill_init = _load_skill(str(v0_path)) or ""
                self._current_skill = self._current_skill or self.skill_init
                self._best_skill = self._best_skill or self.skill_init

        logger.info(
            "Resumed from step {} epoch {} (current={:.4f}, best={:.4f})",
            last_step, self._last_completed_epoch,
            self._current_score, self._best_score,
        )
        return last_step

    # ── Start training from dashboard ────────────────────────────────

    def start_training_async(self) -> str:
        """Start training in background (called from dashboard).

        Returns status message.
        """
        if self._running:
            return "Training already in progress."
        if not self.has_data():
            return "No training data loaded. Upload data first."
        self._ensure_out_dir()

        self._cancel_requested = False

        async def _run():
            try:
                await self.train()
            except Exception as exc:
                logger.error("Training failed: {}", exc, exc_info=True)
                self._emit("error", {"error": str(exc), "detail": repr(exc)})

        loop = asyncio.get_event_loop()
        self._training_task = loop.create_task(_run())
        return "Training started."

    # ── Main training loop ────────────────────────────────────────────

    async def train(self) -> TrainingHistory:
        """Execute the full training loop.

        Returns the training history.
        """
        self._ensure_out_dir()
        self._running = True
        self._cancel_requested = False
        t_start = time.time()

        # Bridge algorithm logs → dashboard event stream
        self._install_log_sink()

        if not self.data_loader:
            self._uninstall_log_sink()
            raise RuntimeError("No data loader set. Upload data before starting training.")

        try:
            # Emit init phase
            self._emit("init", {
                "algorithm": self.algorithm.name,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "edit_budget": self.edit_budget,
                "seed": self.seed,
                "out_dir": str(self.out_dir),
            })

            # Resume check
            resume_from = self._load_state()
            if resume_from > 0:
                logger.info("Resuming from step {}", resume_from)
                self._emit("resume", {"from_step": resume_from})

            # Resolve initial skill (file path or LLM generation)
            if not self._skill_init_resolved and (
                not self.skill_init or not self.skill_init.strip()
            ):
                logger.info("[SKILL-INIT] skill_init is empty; resolving via file or LLM...")
                self.skill_init = await _resolve_skill_init(
                    skill_init=self.skill_init,
                    skill_init_path=self.skill_init_path,
                    env=self.env,
                    data_loader=self.data_loader,
                    out_dir=self.out_dir,
                )
                if not self.skill_init:
                    logger.warning("[SKILL-INIT] no initial skill resolved; training may fail")
                else:
                    # Sync runtime state
                    self._current_skill = self.skill_init
                    self._best_skill = self.skill_init
                self._skill_init_resolved = True

            # Save initial skill
            _save_skill(str(self.out_dir / "skills" / "skill_v0000.md"), self.skill_init)

            # Training parameters (needed for config)
            train_split = self.data_loader.train
            steps_per_epoch = max(1, len(train_split) // self.batch_size)
            total_steps = self.num_epochs * steps_per_epoch

            # Save config (expanded — aligned with official SkillOpt schema)
            _val_items = 0
            try:
                _val_items = len(self.data_loader.val.items)
            except (KeyError, AttributeError):
                pass
            config_dict: dict[str, Any] = {
                # Base parameters
                "algorithm": self.algorithm.name,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "edit_budget": self.edit_budget,
                "seed": self.seed,
                "workers": self.workers,
                # Algorithm configuration
                "update_mode": getattr(self.algorithm, "update_mode", "patch"),
                "lr_scheduler": getattr(self.algorithm, "lr_scheduler_type", "constant"),
                "lr_control_mode": getattr(self.algorithm, "lr_mode", "fixed"),
                "use_slow_update": getattr(self.algorithm, "use_slow_update", False),
                "use_meta_skill": getattr(self.algorithm, "use_meta_skill", False),
                "use_gate": True,
                # Skill sizes
                "skill_init_len": len(self.skill_init),
                # Runtime environment
                "out_root": str(self.out_dir),
                "skill_init_path": self.skill_init_path,
                # Data
                "train_items": len(train_split.items),
                "val_items": _val_items,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
            }
            _save_json(str(self.out_dir / "config.json"), config_dict)
            self._emit("config_saved", config_dict)

            # Baseline evaluation
            if self._current_score < 0:
                logger.info("Computing baseline score on validation set...")
                self._emit("baseline_start", {"message": "Computing baseline score..."})
                try:
                    val_items = self.data_loader.val.items
                except KeyError:
                    val_items = self.data_loader.train.items[:self.batch_size]
                    logger.warning("No val split; using first {} train items", self.batch_size)

                try:
                    baseline_score = await self.algorithm.evaluate(
                        self.env, self.skill_init, val_items,
                        str(self.out_dir / "baseline"),
                    )
                    self._current_score = baseline_score
                    self._best_score = baseline_score
                    self._best_step = 0
                    self._baseline_score = baseline_score
                    self._emit("baseline", {"score": baseline_score})
                    # Checkpoint baseline so it's not re-computed on restart
                    self._save_state(0)
                except Exception as exc:
                    logger.error("Baseline evaluation failed: {}", exc, exc_info=True)
                    self._emit("baseline_error", {"error": str(exc)})

            # Training parameters — already computed above for config.json

            self._emit("training_start", {
                "total_steps": total_steps,
                "num_epochs": self.num_epochs,
                "steps_per_epoch": steps_per_epoch,
                "lr_mode": getattr(self.algorithm, 'lr_mode', 'constant'),
                "update_mode": getattr(self.algorithm, 'update_mode', 'patch'),
            })

            # Notify algorithm of total_steps (for LR scheduler init)
            if hasattr(self.algorithm, "init_training_run"):
                self.algorithm.init_training_run(total_steps)

            global_step = 0
            # Track last-step results per epoch for slow update comparison
            prev_epoch_results: list = []
            prev_epoch_items: list[dict] = []

            for epoch in range(1, self.num_epochs + 1):
                if self._cancel_requested:
                    break

                logger.info("=== EPOCH {}/{} ===", epoch, self.num_epochs)
                self._emit("epoch_start", {"epoch": epoch})

                # Generate batches for this epoch
                batches = list(train_split.iter_batches(
                    self.batch_size, seed=self.seed + epoch * 1000,
                ))

                curr_epoch_results: list = []
                curr_epoch_items: list[dict] = []

                for step_in_epoch, batch_items in enumerate(batches):
                    global_step += 1
                    if global_step <= resume_from:
                        continue
                    if self._cancel_requested:
                        break

                    step_results, step_items = await self._run_step(
                        global_step=global_step,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        items=batch_items,
                        total_steps=total_steps,
                    )
                    # Track last step results for epoch-end comparison
                    curr_epoch_results = step_results
                    curr_epoch_items = step_items

                # Epoch end hook with slow update / meta skill support
                epoch_out_dir = str(self.out_dir / "epochs" / f"epoch_{epoch:02d}")
                os.makedirs(epoch_out_dir, exist_ok=True)
                self._emit("epoch_end_start", {
                    "epoch": epoch,
                    "slow_update": getattr(self.algorithm, 'use_slow_update', False),
                    "meta_skill": getattr(self.algorithm, 'use_meta_skill', False),
                })
                self._current_skill = await self.algorithm.on_epoch_end(
                    epoch, self._history, self._current_skill,
                    prev_results=prev_epoch_results or None,
                    curr_results=curr_epoch_results or None,
                    items=curr_epoch_items or None,
                    out_dir=epoch_out_dir,
                )

                # Save prev_epoch_skill for slow update resume
                if hasattr(self.algorithm, '_prev_epoch_skill'):
                    _save_skill(
                        os.path.join(epoch_out_dir, "prev_skill.md"),
                        self._current_skill,
                    )

                # Save algorithm state at epoch boundary
                self._save_algorithm_state()

                self._last_completed_epoch = epoch
                self._emit("epoch_end_done", {"epoch": epoch})

                # Update epoch tracking for next epoch's slow update
                prev_epoch_results = curr_epoch_results
                prev_epoch_items = curr_epoch_items

                self._history.total_epochs = epoch
                self._emit("epoch_done", {
                    "epoch": epoch,
                    "current_score": self._current_score,
                    "best_score": self._best_score,
                })

            # Training complete
            elapsed = time.time() - t_start

            # Write summary.json (aligned with official SkillOpt)
            summary: dict[str, Any] = {
                "version": "summerclaw-0.1.0",
                "config": config_dict,
                "baseline_score": self._baseline_score,
                "best_score": self._best_score,
                "best_step": self._best_step,
                "current_origin": self._current_origin,
                "best_origin": self._best_origin,
                "total_steps": self._history.total_steps,
                "total_accepts": sum(
                    1 for s in self._history.steps if "accept" in s.action
                ),
                "total_rejects": sum(
                    1 for s in self._history.steps if s.action == "reject"
                ),
                "total_skips": sum(
                    1 for s in self._history.steps if "skip" in s.action
                ),
                "epoch_stats": self._compute_epoch_stats(),
                "total_wall_time_s": round(elapsed, 1),
            }
            _save_json(str(self.out_dir / "summary.json"), summary)

            self._emit("training_done", {
                "elapsed_s": round(elapsed, 1),
                "best_score": self._best_score,
                "best_step": self._best_step,
                "total_steps": self._history.total_steps,
            })

        except Exception as exc:
            logger.error("Training error: {}", exc, exc_info=True)
            self._emit("error", {"error": str(exc), "detail": repr(exc)})
            raise
        finally:
            self._running = False
            self._flush_events_to_disk()
            self._uninstall_log_sink()

        return self._history

    # ── Single step ───────────────────────────────────────────────────

    async def _run_step(
        self,
        global_step: int,
        epoch: int,
        step_in_epoch: int,
        items: list[dict],
        total_steps: int,
    ) -> tuple[list, list[dict]]:
        """Execute one 6-stage training step with phase-level checkpoints.

        If a previous run crashed mid-step, completed phases are restored
        from disk instead of being re-executed (saves LLM calls).

        Returns
        -------
        tuple[list, list[dict]]
            (rollout_results, batch_items) for epoch-end tracking.
        """
        step_t0 = time.time()
        step_dir = str(self.out_dir / "steps" / f"step_{global_step:04d}")
        os.makedirs(step_dir, exist_ok=True)

        self._emit("step_start", {
            "step": global_step,
            "epoch": epoch,
            "total": total_steps,
        })

        step_rec = TrainingStep(step=global_step, epoch=epoch, score=0.0, action="",
                                step_in_epoch=step_in_epoch)
        timing: dict[str, float] = {}

        # Load or create phase-level checkpoint
        progress = self._load_step_progress(step_dir)
        completed = set(progress.get("completed_phases", []))
        if not progress.get("batch_item_ids"):
            progress["batch_item_ids"] = [
                str(item.get("id", i)) for i, item in enumerate(items)
            ]

        # ① ROLLOUT
        if "rollout" not in completed:
            t0 = time.time()
            results = await self.algorithm.rollout(
                self.env, self._current_skill, items, step_dir,
            )
            timing["rollout_s"] = round(time.time() - t0, 1)
            # Save rollout artifacts (summary JSON + per-prediction files + JSONL)
            self._save_step_artifact(step_dir, "rollout_results.json",
                                     [r.to_dict() for r in results])
            self._save_step_artifact(step_dir, "batch_items.json", items)
            rollout_dir = os.path.join(step_dir, "rollout")
            pred_dir = os.path.join(rollout_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            for r in results:
                _save_json(os.path.join(pred_dir, f"{r.id}.json"), r.to_dict())
            _write_jsonl(os.path.join(rollout_dir, "results.jsonl"),
                         [r.to_dict() for r in results])
            completed.add("rollout")
            progress["completed_phases"] = list(completed)
            self._save_step_progress(step_dir, progress)
            self._emit("phase_done", {
                "phase": "rollout", "step": global_step,
                "duration_s": timing["rollout_s"],
                "n_results": len(results),
            })
        else:
            # Restore from checkpoint
            raw = _load_json(os.path.join(step_dir, "rollout_results.json")) or []
            results = [RolloutResult.from_dict(r) for r in raw]
            logger.info("[RESUME] restored {} rollout results for step {}",
                        len(results), global_step)
            self._emit("phase_done", {
                "phase": "rollout", "step": global_step,
                "n_results": len(results), "resumed": True,
            })

        # Collect rollout stats
        rollout_n = len(results) or 1
        rollout_hard = sum(r.hard for r in results) / rollout_n
        rollout_soft = sum(r.soft for r in results) / rollout_n
        step_rec.rollout_hard = round(rollout_hard, 6)
        step_rec.rollout_soft = round(rollout_soft, 6)
        step_rec.rollout_n = len(results)

        # ② REFLECT
        if "reflect" not in completed:
            t0 = time.time()
            raw_patches = await self.algorithm.reflect(
                results, self._current_skill, step_dir,
            )
            timing["reflect_s"] = round(time.time() - t0, 1)
            # Save reflect artifacts (summary + per-patch files)
            self._save_step_artifact(step_dir, "raw_patches.json",
                                     [p.to_dict() for p in raw_patches])
            patches_dir = os.path.join(step_dir, "patches")
            os.makedirs(patches_dir, exist_ok=True)
            fail_idx = succ_idx = 0
            for p in raw_patches:
                d = p.to_dict()
                kind = d.get("kind", "failure")
                if kind == "success":
                    fname = f"minibatch_succ_{succ_idx:03d}.json"
                    succ_idx += 1
                else:
                    fname = f"minibatch_fail_{fail_idx:03d}.json"
                    fail_idx += 1
                _save_json(os.path.join(patches_dir, fname), d)
            completed.add("reflect")
            progress["completed_phases"] = list(completed)
            self._save_step_progress(step_dir, progress)
            self._emit("phase_done", {
                "phase": "reflect", "step": global_step,
                "duration_s": timing["reflect_s"],
                "n_patches": len(raw_patches),
            })
        else:
            raw = _load_json(os.path.join(step_dir, "raw_patches.json")) or []
            raw_patches = [RawPatch.from_dict(p) for p in raw if p]
            logger.info("[RESUME] restored {} patches for step {}",
                        len(raw_patches), global_step)
            self._emit("phase_done", {
                "phase": "reflect", "step": global_step,
                "n_patches": len(raw_patches), "resumed": True,
            })

        # Collect patch stats
        n_fail = sum(1 for p in raw_patches if getattr(p, 'kind', 'failure') != 'success')
        n_succ = len(raw_patches) - n_fail
        step_rec.n_patches = len(raw_patches)
        step_rec.n_failure_patches = n_fail
        step_rec.n_success_patches = n_succ

        if not raw_patches:
            step_rec.action = "skip_no_patches"
            step_rec.score = self._current_score
            step_rec.timing = timing
            step_rec.current_score = self._current_score
            step_rec.best_score = self._best_score
            step_rec.best_step = self._best_step
            step_rec.skill_len = len(self._current_skill)
            step_rec.wall_time_s = round(time.time() - step_t0, 1)
            self._history.add_step(step_rec)
            self._save_state(global_step)
            # Write step_record.json (aligned with official SkillOpt)
            _save_json(os.path.join(step_dir, "step_record.json"), step_rec.to_dict())
            self._emit("step_done", {
                "step": global_step, "action": "skip_no_patches",
            })
            return results, items

        # ③ AGGREGATE
        if "aggregate" not in completed:
            t0 = time.time()
            merged_patch = await self.algorithm.aggregate(
                raw_patches, self._current_skill,
            )
            timing["aggregate_s"] = round(time.time() - t0, 1)
            _save_json(
                os.path.join(step_dir, "merged_patch.json"),
                merged_patch.to_dict(),
            )
            completed.add("aggregate")
            progress["completed_phases"] = list(completed)
            self._save_step_progress(step_dir, progress)
            self._emit("phase_done", {
                "phase": "aggregate", "step": global_step,
                "duration_s": timing["aggregate_s"],
                "n_edits": len(merged_patch.edits),
            })
        else:
            d = _load_json(os.path.join(step_dir, "merged_patch.json")) or {}
            merged_patch = Patch.from_dict(d)
            logger.info("[RESUME] restored merged patch ({} edits) for step {}",
                        len(merged_patch.edits), global_step)
            self._emit("phase_done", {
                "phase": "aggregate", "step": global_step,
                "n_edits": len(merged_patch.edits), "resumed": True,
            })

        step_rec.n_edits_merged = len(merged_patch.edits)

        # ④ SELECT
        if "select" not in completed:
            t0 = time.time()
            step_budget = self.algorithm.get_edit_budget(global_step, total_steps)
            selected_patch = await self.algorithm.select(
                merged_patch, step_budget, self._current_skill,
            )
            timing["select_s"] = round(time.time() - t0, 1)
            _save_json(
                os.path.join(step_dir, "selected_edits.json"),
                selected_patch.to_dict(),
            )
            # Also save as ranked_edits.json (official SkillOpt naming)
            _save_json(
                os.path.join(step_dir, "ranked_edits.json"),
                selected_patch.to_dict(),
            )
            completed.add("select")
            progress["completed_phases"] = list(completed)
            self._save_step_progress(step_dir, progress)
            step_rec.n_edits_applied = len(selected_patch.edits)
            self._emit("phase_done", {
                "phase": "select", "step": global_step,
                "duration_s": timing["select_s"],
                "n_selected": len(selected_patch.edits),
            })
        else:
            d = _load_json(os.path.join(step_dir, "selected_edits.json")) or {}
            selected_patch = Patch.from_dict(d)
            step_rec.n_edits_applied = len(selected_patch.edits)
            logger.info("[RESUME] restored selected patch ({} edits) for step {}",
                        len(selected_patch.edits), global_step)
            self._emit("phase_done", {
                "phase": "select", "step": global_step,
                "n_selected": len(selected_patch.edits), "resumed": True,
            })

        step_rec.edit_budget = self.algorithm.get_edit_budget(global_step, total_steps)
        step_rec.lr_control_mode = getattr(self.algorithm, "lr_mode", "fixed")

        # lr_history.jsonl (autonomous LR mode)
        lr_mode = getattr(self.algorithm, "lr_mode", "fixed")
        if lr_mode == "autonomous" and hasattr(self.algorithm, "_last_lr_decision"):
            decision = self.algorithm._last_lr_decision
            if decision:
                try:
                    with open(os.path.join(str(self.out_dir), "lr_history.jsonl"), "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "epoch": epoch, **decision,
                        }, ensure_ascii=False) + "\n")
                except OSError:
                    pass

        # ⑤ UPDATE
        if "update" not in completed:
            t0 = time.time()
            candidate_skill, apply_report = await self.algorithm.update(
                self._current_skill, selected_patch,
            )
            timing["update_s"] = round(time.time() - t0, 1)
            _save_json(
                os.path.join(step_dir, "edit_apply_report.json"),
                apply_report,
            )
            _save_skill(os.path.join(step_dir, "candidate_skill.md"), candidate_skill)
            completed.add("update")
            progress["completed_phases"] = list(completed)
            self._save_step_progress(step_dir, progress)
            self._emit("phase_done", {
                "phase": "update", "step": global_step,
                "duration_s": timing["update_s"],
                "skill_len": len(candidate_skill),
            })
        else:
            candidate_skill = _load_skill(
                os.path.join(step_dir, "candidate_skill.md")
            ) or self._current_skill
            apply_report = _load_json(
                os.path.join(step_dir, "edit_apply_report.json")
            ) or []
            logger.info("[RESUME] restored candidate skill ({} chars) for step {}",
                        len(candidate_skill), global_step)
            self._emit("phase_done", {
                "phase": "update", "step": global_step,
                "skill_len": len(candidate_skill), "resumed": True,
            })

        step_rec.candidate_skill_len = len(candidate_skill)
        # Build edit_apply_summary (aligned with official)
        if apply_report:
            step_rec.edit_apply_summary = {
                "total": len(apply_report),
                "applied": sum(1 for r in apply_report
                               if str(r.get("status", "")).startswith("applied")),
                "skipped": sum(1 for r in apply_report
                               if str(r.get("status", "")).startswith("skipped")),
                "errors": sum(1 for r in apply_report
                              if r.get("status") == "error"),
            }

        # ⑥ EVALUATE
        if "evaluate" not in completed:
            t0 = time.time()
            try:
                val_items = self.data_loader.val.items
            except KeyError:
                val_items = self.data_loader.train.items[:self.batch_size]

            sel_eval_dir = os.path.join(step_dir, "selection_eval")
            cand_score = await self.algorithm.evaluate(
                self.env, candidate_skill, val_items, sel_eval_dir,
            )
            timing["evaluate_s"] = round(time.time() - t0, 1)

            # Gate decision
            gate = evaluate_gate(
                candidate_skill=candidate_skill,
                cand_hard=cand_score,
                current_skill=self._current_skill,
                current_score=self._current_score,
                best_skill=self._best_skill,
                best_score=self._best_score,
                best_step=self._best_step,
                global_step=global_step,
            )

            # Apply gate decision
            self._current_skill = gate.current_skill
            self._current_score = gate.current_score
            self._best_skill = gate.best_skill
            self._best_score = gate.best_score
            self._best_step = gate.best_step

            # Update origin tracking (aligned with official SkillOpt)
            if gate.action in {"accept", "accept_new_best"}:
                self._current_origin = f"step_{global_step:04d}"
            if gate.action == "accept_new_best":
                self._best_origin = self._current_origin

            step_rec.score = cand_score
            step_rec.action = gate.action
            step_rec.skill_hash = _skill_hash(candidate_skill)
            step_rec.n_edits_rejected = (
                len(selected_patch.edits) if gate.action == "reject" else 0
            )
            step_rec.selection_hard = cand_score
            step_rec.selection_soft = cand_score

            # Save gate decision artifact
            self._save_step_artifact(step_dir, "gate_decision.json", {
                "action": gate.action,
                "cand_score": cand_score,
                "current_score": self._current_score,
                "best_score": self._best_score,
                "best_step": self._best_step,
            })

            # Write selection_eval predictions/results (if evaluate didn't already)
            _ensure_selection_eval_artifacts(
                sel_eval_dir, results, candidate_skill, val_items,
            )

            completed.add("evaluate")
            progress["completed_phases"] = list(completed)
            self._save_step_progress(step_dir, progress)

            # Fill remaining step_rec fields
            step_rec.current_score = self._current_score
            step_rec.best_score = self._best_score
            step_rec.best_step = self._best_step
            step_rec.current_origin = self._current_origin
            step_rec.best_origin = self._best_origin
            step_rec.skill_len = len(self._current_skill)
            step_rec.timing = timing
            step_rec.wall_time_s = round(time.time() - step_t0, 1)

            # Write step_record.json (aligned with official SkillOpt)
            _save_json(os.path.join(step_dir, "step_record.json"), step_rec.to_dict())

            # Write trajectory_digest.json
            failure_patterns = _extract_failure_patterns(results)
            digest: dict[str, Any] = {
                "step": global_step,
                "action": gate.action,
                "n_total": len(results),
                "n_fail": sum(1 for r in results if not r.hard),
                "failure_patterns": failure_patterns,
            }
            if "reject" in gate.action:
                digest["rejected_edits"] = [
                    {"target": e.target, "source": e.content[:80]}
                    for e in selected_patch.edits
                ]
                digest["score_before"] = self._current_score
                digest["score_after"] = cand_score
            _save_json(os.path.join(step_dir, "trajectory_digest.json"), digest)

            # Save state (includes algorithm_state.json)
            self._history.add_step(step_rec)
            self._save_state(global_step)
            self._flush_events_to_disk()

            elapsed = time.time() - step_t0
            self._emit("phase_done", {
                "phase": "evaluate", "step": global_step,
                "duration_s": timing["evaluate_s"],
                "score": cand_score,
                "action": gate.action,
            })
            self._emit("step_done", {
                "step": global_step,
                "epoch": epoch,
                "action": gate.action,
                "score": cand_score,
                "current_score": self._current_score,
                "best_score": self._best_score,
                "elapsed_s": round(elapsed, 1),
            })

            logger.info(
                "Step {} done: action={} score={:.4f} current={:.4f} best={:.4f} ({:.1f}s)",
                global_step, gate.action, cand_score,
                self._current_score, self._best_score, elapsed,
            )
        else:
            # Restore gate decision from disk
            gate_data = _load_json(os.path.join(step_dir, "gate_decision.json")) or {}
            cand_score = float(gate_data.get("cand_score", 0.0))
            step_rec.score = cand_score
            step_rec.action = gate_data.get("action", "")
            step_rec.skill_hash = _skill_hash(candidate_skill)
            step_rec.n_edits_rejected = (
                len(selected_patch.edits) if gate_data.get("action") == "reject" else 0
            )
            step_rec.selection_hard = cand_score
            step_rec.selection_soft = cand_score
            step_rec.current_score = self._current_score
            step_rec.best_score = self._best_score
            step_rec.best_step = self._best_step
            step_rec.current_origin = self._current_origin
            step_rec.best_origin = self._best_origin
            step_rec.skill_len = len(self._current_skill)
            step_rec.timing = timing
            # Note: gate decision was already applied in the previous run,
            # so we do NOT re-apply it. The state was restored via _load_state.
            logger.info("[RESUME] step {} evaluate already done (action={}, score={:.4f})",
                        global_step, step_rec.action, cand_score)
            self._emit("step_done", {
                "step": global_step,
                "epoch": epoch,
                "action": step_rec.action,
                "score": cand_score,
                "current_score": self._current_score,
                "best_score": self._best_score,
                "resumed": True,
            })

        return results, items

    # ── Eval only ─────────────────────────────────────────────────────

    async def eval_only(self, skill_content: str | None = None) -> float:
        """Run evaluation only (no training).

        Parameters
        ----------
        skill_content : str | None
            Skill to evaluate. Uses current skill if None.

        Returns
        -------
        float
            Validation score.
        """
        self._ensure_out_dir()
        skill = skill_content or self._current_skill
        try:
            val_items = self.data_loader.val.items
        except KeyError:
            val_items = self.data_loader.train.items[:self.batch_size]

        eval_dir = str(self.out_dir / "eval_only")
        score = await self.algorithm.evaluate(
            self.env, skill, val_items, eval_dir,
        )
        self._emit("eval_done", {"score": score})
        return score

    # ── Deploy skill ──────────────────────────────────────────────────

    async def deploy_skill(self, target_path: str | Path) -> str:
        """Copy the best skill to the target path.

        Parameters
        ----------
        target_path : str | Path
            Destination file path.

        Returns
        -------
        str
            The deployed skill content.
        """
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._best_skill, encoding="utf-8")
        logger.info("Deployed best skill to {} ({} chars)", target, len(self._best_skill))
        self._emit("deployed", {"path": str(target), "chars": len(self._best_skill)})
        return self._best_skill
