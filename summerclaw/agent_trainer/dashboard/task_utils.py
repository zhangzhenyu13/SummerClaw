"""Task scanning, caching, and history loading utilities.

Provides helpers for discovering and parsing training task directories
stored under ``~/.summerclaw/train-algs`` (or a custom root).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml

from summerclaw.agent_trainer.engine.trainer import _load_json


# ── Defaults ────────────────────────────────────────────────────────────

def _default_train_root() -> Path:
    return Path.home() / ".summerclaw" / "train-algs"


# ── Parsing helpers ─────────────────────────────────────────────────────

def _extract_task_id(dirname: str) -> tuple[str, str]:
    """Split a task directory name into (algorithm_name, timestamp_str)."""
    parts = dirname.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 15 and "-" not in parts[1]:
        pass
    return dirname, ""


def _parse_task_created(dirname: str) -> str:
    """Best-effort parse of creation datetime from task directory name."""
    m = re.search(r"(\d{8})-(\d{6})$", dirname)
    if m:
        date_str, time_str = m.group(1), m.group(2)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return "unknown"


# Heartbeat timeout: if no checkpoint within this many seconds,
# treat the task as crashed even if the gateway process is still alive.
# NOTE: Training runs in a background thread (not subprocess), so PID-based
# checks are unreliable — the PID is always the gateway process.
# Heartbeat freshness is the authoritative signal for liveness.
_HEARTBEAT_TIMEOUT = 120.0  # seconds


def _is_task_actually_running(
    task_dir: Path,
    active_sessions: dict | None = None,
) -> bool:
    """Determine if a task is actively running (shared helper).

    Training runs in a background thread of the gateway process (not as a
    subprocess), so the old PID-based crash detection was unreliable — the
    PID in runtime_state.json was always the gateway PID.

    Checks in order:
      1. active_sessions has running_task_dir pointing here + engine.is_running
      2. run_id match: file's run_id matches a live engine's run_id
      3. Heartbeat freshness (fallback — survives gateway restart)
    """
    task_dir_str = str(task_dir)

    # 1) Check active_sessions via engine.is_running flag
    if active_sessions:
        for info in active_sessions.values():
            eng = info.get("engine")
            if not eng or not getattr(eng, "is_running", False):
                continue
            rtd = info.get("running_task_dir")
            rtd_str = str(rtd) if rtd else str(eng.out_dir)
            if rtd_str == task_dir_str:
                return True

    # Load runtime_state.json for fallback checks (2 & 3)
    rt_path = task_dir / "runtime_state.json"
    try:
        state = _load_json(str(rt_path)) or {}
    except Exception:
        return False
    if state.get("status") != "running":
        return False

    # 2) run_id match: file's run_id belongs to a live engine in this process
    file_run_id = state.get("run_id", "")
    if file_run_id and active_sessions:
        for info in active_sessions.values():
            eng = info.get("engine")
            if eng and getattr(eng, "run_id", None) == file_run_id:
                return True
        # run_id exists in file but no live engine owns it → training is dead
        return False

    # 3) Heartbeat freshness — the only signal that survives gateway restart.
    #    Since training runs in a thread (not subprocess), heartbeat stops
    #    updating when the thread dies, even though the gateway stays alive.
    _hb = state.get("heartbeat_ts", 0)
    _hb_age = time.time() - _hb if _hb else float("inf")
    return _hb_age < _HEARTBEAT_TIMEOUT


# ── Unified status resolution ──────────────────────────────────────────

def _resolve_task_status(
    task_dir: Path,
    persisted_status: str,
    is_archived: bool,
    has_history: bool,
    active_sessions: dict | None = None,
    stop_requested_task_dirs: set[str] | None = None,
) -> str:
    """Unified status resolution used by both list and detail views.

    Priority order:
      1. stopping (running + stop_requested for this task)
      2. running (actively executing)
      3. archived (summary.json exists)
      4. stopped (process died without updating runtime_state.json, or explicitly stopped)
      5. failed
      6. queued
      7. completed
      8. idle (default)
    """
    _is_running = _is_task_actually_running(task_dir, active_sessions)
    _dir_key = str(task_dir)
    _is_stopping = _is_running and (stop_requested_task_dirs and _dir_key in stop_requested_task_dirs)

    if _is_stopping:
        return "stopping"
    if _is_running:
        return "running"
    if is_archived:
        return "archived"
    # Process died without updating runtime_state.json (heartbeat stale, thread dead).
    # This is a crash, not a clean stop — the except block would have written "failed"
    # if it ran. Only show "stopped" if explicitly marked.
    if persisted_status == "running":
        return "failed"
    if persisted_status == "failed":
        return "failed"
    if persisted_status == "stopped":
        return "stopped"
    if persisted_status == "queued":
        return "queued"
    if persisted_status == "completed" or has_history:
        return "completed"
    return "idle"


# ── Task scanning ───────────────────────────────────────────────────────

def _scan_all_tasks(
    train_root: Path,
    active_sessions: dict | None = None,
    max_concurrency: int = 0,
) -> list[dict]:
    """Scan train_root for all task directories and return metadata list.

    Each entry contains:
      task_id, path, algorithm, created, status, best_score, total_steps,
      epochs, batch_size, workers, effective_workers, notes, archived,
      started_at, finished_at, duration_s
    """
    tasks: list[dict] = []
    if not train_root.exists():
        return tasks

    stopping_dirs: set[str] = set()  # dirs where stop has been requested
    if active_sessions:
        for info in active_sessions.values():
            eng = info.get("engine")
            if not eng:
                continue
            running_dir = info.get("running_task_dir")
            dir_key = str(running_dir) if running_dir else str(eng.out_dir)
            if info.get("stop_requested"):
                stopping_dirs.add(dir_key)

    for entry in sorted(train_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue

        task_id = entry.name
        task_dir = entry
        created = _parse_task_created(task_id)

        config = _load_json(str(task_dir / "config.json")) or {}
        state = _load_json(str(task_dir / "runtime_state.json")) or {}
        has_history = (task_dir / "history.json").exists()
        summary = _load_json(str(task_dir / "summary.json")) or {}
        is_archived = bool(summary)

        # Fallback created time from config if dir name doesn't contain timestamp
        if created == "unknown":
            created = config.get("created_at", "unknown")
            # Parse ISO format back to display format if needed
            if created and "T" in str(created):
                try:
                    from datetime import datetime as _dt
                    dt_obj = _dt.fromisoformat(str(created))
                    created = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

        algorithm = config.get("algorithm", task_id.rsplit("-", 2)[0] if "-" in task_id else task_id)
        epochs = config.get("num_epochs", "?")
        batch_size = config.get("batch_size", "?")
        workers = config.get("workers", "?")

        # Compute effective workers: 0 = auto-derive 80% of max_concurrency
        _w_int = int(workers) if isinstance(workers, (int, float)) else 0
        if _w_int > 0:
            effective_workers = _w_int
        elif max_concurrency > 0:
            effective_workers = max(1, int(max_concurrency * 0.8))
        else:
            effective_workers = 4  # fallback

        best_score = state.get("best_score", -1)
        best_step = state.get("best_step", 0)
        baseline_score = state.get("baseline_score", -1)
        total_steps = state.get("last_completed_step", 0)
        total_epochs = state.get("total_epochs", 0)

        # --- Unified status determination ---
        status = _resolve_task_status(
            task_dir,
            state.get("status", ""),
            is_archived,
            has_history,
            active_sessions,
            stopping_dirs,
        )

        tasks.append({
            "task_id": task_id,
            "path": str(task_dir),
            "algorithm": algorithm,
            "created": created,
            "status": status,
            "archived": is_archived,
            "best_score": best_score,
            "baseline_score": baseline_score,
            "best_step": best_step,
            "total_steps": total_steps,
            "total_epochs": total_epochs,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "effective_workers": effective_workers,
            "name": config.get("name", ""),
            "description": config.get("description", ""),
            "notes": "",
            # Timing fields
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
            "duration_s": state.get("total_wall_time_s") or summary.get("total_wall_time_s"),
        })

    # Newest first
    tasks.sort(key=lambda x: x["created"], reverse=True)
    return tasks


# ── Scan cache (TTL) ──────────────────────────────────────────────────

_scan_cache: dict = {"result": None, "ts": 0.0, "mt": 0.0, "session_key": None}
_SCAN_TTL = 5.0  # seconds


def _scan_all_tasks_cached(
    train_root: Path,
    active_sessions: dict | None = None,
    max_concurrency: int = 0,
) -> list[dict]:
    """Cached wrapper around _scan_all_tasks (5s TTL + mtime + session-state check).

    The cache is invalidated when:
    - TTL expires (5s)
    - train_root mtime changes (new/removed task dirs)
    - Any session's running_task_dir or stop_requested changes (status transition)
    """
    now = time.monotonic()
    try:
        mt = train_root.stat().st_mtime
    except OSError:
        mt = 0.0
    # Build a lightweight session-state key so that running_task_dir and
    # stop_requested changes immediately invalidate the cache.
    _session_key: frozenset = frozenset()
    if active_sessions:
        _session_key = frozenset(
            (id(info.get("engine")),
             info.get("running_task_dir", ""),
             bool(info.get("stop_requested")))
            for info in active_sessions.values()
        )
    if (
        _scan_cache["result"] is not None
        and (now - _scan_cache["ts"]) < _SCAN_TTL
        and mt == _scan_cache["mt"]
        and _session_key == _scan_cache["session_key"]
    ):
        return _scan_cache["result"]
    result = _scan_all_tasks(train_root, active_sessions, max_concurrency)
    _scan_cache["result"] = result
    _scan_cache["ts"] = now
    _scan_cache["mt"] = mt
    _scan_cache["session_key"] = _session_key
    return result


# ── History loading ─────────────────────────────────────────────────────

def _load_task_history(task_dir: Path) -> dict:
    """Load saved history.json for a historical (read-only) task."""
    hist_path = task_dir / "history.json"
    if hist_path.exists():
        return _load_json(str(hist_path)) or {}
    return {}


# ── YAML template helpers ───────────────────────────────────────────────

def _find_default_yaml_template() -> Path | None:
    """Locate the default skillopt.yaml template.

    Search order:
      1. ``<project_root>/resources/trainer-skillopt/skillopt.yaml``
      2. ``<project_root>/skillopt.yaml``
    """
    # ui_state.py is at summerclaw/agent_trainer/dashboard/ → 3 levels up
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "resources" / "trainer-skillopt" / "skillopt.yaml",
        project_root / "skillopt.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _generate_default_yaml() -> str:
    """Generate a complete default skillopt.yaml content string."""
    return _DEFAULT_SKILLOPT_YAML


_DEFAULT_SKILLOPT_YAML = """\
# SummerClaw SkillOpt Training Configuration
# Auto-generated — edit as needed

train:
  num_epochs: 3
  batch_size: 5
  workers: 0             # 0 = auto-derive 80% of maxConcurrency
  accumulation: 1
  seed: 42

optimizer:
  learning_rate: 4
  min_learning_rate: 2
  lr_scheduler: constant
  skill_update_mode: patch
  use_slow_update: true
  slow_update_samples: 20
  use_meta_skill: true
  longitudinal_pair_policy: mixed

gradient:
  minibatch_size: 8
  merge_batch_size: 8
  max_analyst_rounds: 3
  failure_only: false

model:
  reasoning_effort: medium
  rewrite_reasoning_effort: ""
  rewrite_max_completion_tokens: 64000

evaluation:
  use_gate: true
  sel_env_num: 0
  test_env_num: 0
  eval_test: true

env:
  name: ""
  skill_init: ""
  split_mode: ratio
  split_ratio: "2:1:7"
  split_seed: 42
  split_dir: ""
  data_path: ""
  exec_timeout: 120
  memory_algorithm: null
  enabled_tools: []
"""


def _apply_params_to_yaml(yaml_path: Path, params: dict[str, Any]) -> None:
    """Update specific parameters in a YAML file and write back.

    *params* uses the **flat** key naming (same as ``_FLATTEN_MAP`` output).
    Only keys present in *params* are changed; everything else is preserved.
    """
    from summerclaw.agent_trainer.config import load_config

    cfg = load_config(str(yaml_path))

    # Map flat keys → structured section paths
    _FLAT_TO_STRUCTURED: dict[str, tuple[str, str]] = {
        "num_epochs":                ("train", "num_epochs"),
        "batch_size":                ("train", "batch_size"),
        "workers":                   ("train", "workers"),
        "accumulation":              ("train", "accumulation"),
        "seed":                      ("train", "seed"),
        "edit_budget":               ("optimizer", "learning_rate"),
        "min_edit_budget":           ("optimizer", "min_learning_rate"),
        "lr_scheduler":              ("optimizer", "lr_scheduler"),
        "skill_update_mode":         ("optimizer", "skill_update_mode"),
        "use_slow_update":           ("optimizer", "use_slow_update"),
        "slow_update_samples":       ("optimizer", "slow_update_samples"),
        "use_meta_skill":            ("optimizer", "use_meta_skill"),
        "longitudinal_pair_policy":  ("optimizer", "longitudinal_pair_policy"),
        "minibatch_size":            ("gradient", "minibatch_size"),
        "merge_batch_size":          ("gradient", "merge_batch_size"),
        "max_analyst_rounds":        ("gradient", "max_analyst_rounds"),
        "analyst_workers":           ("gradient", "analyst_workers"),
        "failure_only":              ("gradient", "failure_only"),
        "reasoning_effort":          ("model", "reasoning_effort"),
        "rewrite_reasoning_effort":  ("model", "rewrite_reasoning_effort"),
        "rewrite_max_completion_tokens": ("model", "rewrite_max_completion_tokens"),
        "use_gate":                  ("evaluation", "use_gate"),
        "sel_env_num":               ("evaluation", "sel_env_num"),
        "test_env_num":              ("evaluation", "test_env_num"),
        "eval_test":                 ("evaluation", "eval_test"),
        "exec_timeout":              ("env", "exec_timeout"),
        "memory_algorithm":          ("env", "memory_algorithm"),
        "enabled_tools":             ("env", "enabled_tools"),
    }

    for flat_key, value in params.items():
        if flat_key in _FLAT_TO_STRUCTURED:
            section, key = _FLAT_TO_STRUCTURED[flat_key]
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section][key] = value
            else:
                cfg.setdefault(section, {})[key] = value
        else:
            # Top-level or unknown key — set at root
            cfg[flat_key] = value

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
