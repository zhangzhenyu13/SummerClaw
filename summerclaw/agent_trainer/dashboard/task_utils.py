"""Task scanning, caching, and history loading utilities.

Provides helpers for discovering and parsing training task directories
stored under ``~/.summerclaw/train-algs`` (or a custom root).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

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


# ── Task scanning ───────────────────────────────────────────────────────

def _scan_all_tasks(
    train_root: Path,
    active_sessions: dict | None = None,
) -> list[dict]:
    """Scan train_root for all task directories and return metadata list.

    Each entry contains:
      task_id, path, algorithm, created, status, best_score, total_steps,
      epochs, batch_size, notes
    """
    tasks: list[dict] = []
    if not train_root.exists():
        return tasks

    active_dirs: set[str] = set()
    if active_sessions:
        for info in active_sessions.values():
            eng = info.get("engine")
            if eng:
                active_dirs.add(str(eng.out_dir))

    for entry in sorted(train_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue

        task_id = entry.name
        task_dir = entry
        created = _parse_task_created(task_id)

        config = _load_json(str(task_dir / "config.json")) or {}
        state = _load_json(str(task_dir / "runtime_state.json")) or {}
        has_history = (task_dir / "history.json").exists()

        algorithm = config.get("algorithm", task_id.rsplit("-", 2)[0] if "-" in task_id else task_id)
        epochs = config.get("num_epochs", "?")
        batch_size = config.get("batch_size", "?")

        best_score = state.get("best_score", -1)
        best_step = state.get("best_step", 0)
        total_steps = state.get("last_completed_step", 0)
        total_epochs = state.get("total_epochs", 0)

        is_active = str(task_dir) in active_dirs

        tasks.append({
            "task_id": task_id,
            "path": str(task_dir),
            "algorithm": algorithm,
            "created": created,
            "status": "running" if is_active else ("completed" if has_history else "idle"),
            "best_score": best_score,
            "best_step": best_step,
            "total_steps": total_steps,
            "total_epochs": total_epochs,
            "epochs": epochs,
            "batch_size": batch_size,
            "notes": "",
        })

    # Newest first
    tasks.sort(key=lambda x: x["created"], reverse=True)
    return tasks


# ── Scan cache (TTL) ──────────────────────────────────────────────────

_scan_cache: dict = {"result": None, "ts": 0.0, "mt": 0.0}
_SCAN_TTL = 10.0  # seconds


def _scan_all_tasks_cached(
    train_root: Path,
    active_sessions: dict | None = None,
) -> list[dict]:
    """Cached wrapper around _scan_all_tasks (10s TTL + mtime check)."""
    now = time.monotonic()
    try:
        mt = train_root.stat().st_mtime
    except OSError:
        mt = 0.0
    if (
        _scan_cache["result"] is not None
        and (now - _scan_cache["ts"]) < _SCAN_TTL
        and mt == _scan_cache["mt"]
    ):
        return _scan_cache["result"]
    result = _scan_all_tasks(train_root, active_sessions)
    _scan_cache["result"] = result
    _scan_cache["ts"] = now
    _scan_cache["mt"] = mt
    return result


# ── History loading ─────────────────────────────────────────────────────

def _load_task_history(task_dir: Path) -> dict:
    """Load saved history.json for a historical (read-only) task."""
    hist_path = task_dir / "history.json"
    if hist_path.exists():
        return _load_json(str(hist_path)) or {}
    return {}
