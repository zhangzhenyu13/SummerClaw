"""Dashboard shared mutable state.

Holds all state that is accessed across API route handlers, replacing
the Gradio-bound ``UIState`` with a plain Python object.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine, _load_json
from summerclaw.agent_trainer.dashboard.task_utils import (
    _load_task_history,
    _parse_task_created,
    _is_task_actually_running,
    _resolve_task_status,
)
from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler


class _DashboardState:
    """Holds mutable state shared across API handlers.

    Replaces the Gradio-bound UIState with a plain Python object that can
    be accessed from any async handler.
    """

    def __init__(
        self,
        engine: TrainerEngine,
        train_root: Path,
        active_sessions: dict,
    ):
        self.engine = engine
        self.train_root = train_root
        self.active_sessions = active_sessions

        # Training control
        self._stop_requested: bool = False
        self._was_running: bool = False

        # Pending create-task settings (applied on next start)
        self._pending_split: dict = {}
        self._pending_scorer_mode: str = "exact_match"

        # Task scheduler (concurrency-aware)
        self.scheduler: _TaskScheduler | None = None

        # Wire progress callback
        engine.set_progress_callback(self._on_progress)

    def _on_progress(self, event_type: str, payload: dict) -> None:
        pass  # Events stored centrally on engine._events

    # -- Notification helper (best-effort, never blocks) -----------------

    def _fire_notify(self, message: str) -> None:
        try:
            _nf = None
            for _sess in self.active_sessions.values():
                _nf = _sess.get("notify_fn")
                if _nf:
                    break
            if _nf:
                _loop = _sess.get("main_loop")
                if _loop and _loop.is_running():
                    asyncio.run_coroutine_threadsafe(_nf(message), _loop)
                else:
                    _loop2 = asyncio.get_event_loop()
                    if _loop2.is_running():
                        _loop2.create_task(_nf(message))
        except Exception:
            pass

    # -- Engine helpers ---------------------------------------------------

    def _maybe_restore_task(self, task_id: str) -> None:
        """Point the engine at the selected task if it differs from current."""
        if not task_id:
            return
        selected_dir = self.train_root / task_id
        if selected_dir.is_dir() and str(selected_dir) != str(self.engine.out_dir):
            self.engine._set_task_dir(selected_dir)

    def get_engine_for_task(self, task_id: str) -> TrainerEngine:
        """Return the engine instance for a given task.

        If the scheduler has a per-task engine for *task_id*, return it.
        Otherwise fall back to the shared template engine.
        """
        if self.scheduler is not None:
            task_engine = self.scheduler.get_task_engine(task_id)
            if task_engine is not None:
                return task_engine
        return self.engine

    def is_task_engine_running(self, task_id: str) -> bool:
        """Return True if a scheduler-managed engine is running for *task_id*."""
        if self.scheduler is not None:
            task_engine = self.scheduler.get_task_engine(task_id)
            if task_engine is not None:
                return getattr(task_engine, "is_running", False)
        return False

    def _apply_yaml_to_engine(self, task_dir: Path) -> None:
        """Load skillopt.yaml from *task_dir* and apply to engine params."""
        yaml_path = task_dir / "skillopt.yaml"
        if not yaml_path.is_file():
            return
        try:
            from summerclaw.agent_trainer.config import load_config, flatten_config
            cfg = load_config(str(yaml_path))
            flat = flatten_config(cfg)
            eng = self.engine

            _CORE = {
                "num_epochs": ("num_epochs", int),
                "batch_size": ("batch_size", int),
                "edit_budget": ("edit_budget", int),
                "seed": ("seed", int),
                "workers": ("workers", int),
                "eval_test": ("eval_test", bool),
            }
            for flat_key, (attr, cast) in _CORE.items():
                if flat_key in flat:
                    setattr(eng, attr, cast(flat[flat_key]))

            # Sync workers to env adapter (controls rollout_batch concurrency)
            env = getattr(eng, "env", None)
            if env is not None and hasattr(env, "workers"):
                yaml_workers = flat.get("workers", 0)
                if yaml_workers > 0:
                    env.workers = int(yaml_workers)
                else:
                    # Auto-derive 80% of provider.max_concurrency
                    provider_max = getattr(env.provider, "max_concurrency", 0) or 0
                    env.workers = max(1, int(provider_max * 0.8)) if provider_max > 0 else 4

            # Reconfigure memory algorithm and tools from task YAML
            if env is not None and hasattr(env, "reconfigure_for_task"):
                _mem_algo = flat.get("memory_algorithm")
                # Normalize: YAML null → Python None; string "null"/"none" → None
                if _mem_algo in (None, "", "null", "none", "Null", "None"):
                    _mem_algo = None
                _enabled_tools = flat.get("enabled_tools")
                if isinstance(_enabled_tools, list) and len(_enabled_tools) > 0:
                    pass  # keep as-is
                else:
                    _enabled_tools = None  # None = all tools
                env.reconfigure_for_task(_mem_algo, _enabled_tools)

                logger.info(
                    "Task config applied — algorithm: skillopt | memory: {} | tools: {}",
                    _mem_algo or "disabled",
                    ", ".join(_enabled_tools) if _enabled_tools else "all",
                )

            algo = eng.algorithm
            _ALGO = {
                "lr_scheduler": ("lr_scheduler_type", str),
                "lr_mode": ("lr_mode", str),
                "edit_budget": ("edit_budget", int),
                "min_edit_budget": ("min_lr", int),
                "skill_update_mode": ("update_mode", str),
                "update_mode": ("update_mode", str),
                "use_slow_update": ("use_slow_update", bool),
                "slow_update_samples": ("slow_update_samples", int),
                "use_meta_skill": ("use_meta_skill", bool),
                "longitudinal_pair_policy": ("longitudinal_pair_policy", str),
                "minibatch_size": ("minibatch_size", int),
                "merge_batch_size": ("merge_batch_size", int),
                "max_analyst_rounds": ("max_analyst_rounds", int),
                "analyst_workers": ("analyst_workers", int),
                "aggregate_workers": ("aggregate_workers", int),
                "evaluate_workers": ("evaluate_workers", int),
                "reasoning_effort": ("reasoning_effort", str),
                "rewrite_reasoning_effort": ("rewrite_reasoning_effort", str),
                "rewrite_max_completion_tokens": ("rewrite_max_completion_tokens", int),
            }
            for flat_key, (attr, cast) in _ALGO.items():
                if flat_key in flat and hasattr(algo, attr):
                    setattr(algo, attr, cast(flat[flat_key]))

            eng._trainer_cfg.update(flat)
            logger.info("Applied skillopt.yaml from {} to engine", task_dir)
        except Exception as exc:
            logger.error("Failed to apply skillopt.yaml from {}: {}", task_dir, exc, exc_info=True)

    # -- Data display helpers ---------------------------------------------

    def get_status_dict(self) -> dict:
        hist = self.engine.history
        if self._stop_requested and self.engine.is_running:
            status = "stopping"
        else:
            status = "running" if self.engine.is_running else "idle"
        return {
            "status": status,
            "best_score": self.engine.best_score,
            "baseline_score": self.engine.baseline_score,
            "best_step": self.engine._best_step,
            "total_steps": hist.total_steps,
            "total_epochs": hist.total_epochs,
        }

    def get_history_rows(self) -> list[dict]:
        steps = self.engine.history.steps
        return [
            {
                "step": s.step, "epoch": s.epoch, "score": round(s.score, 4),
                "action": s.action, "skill_hash": s.skill_hash,
                "edits_applied": s.n_edits_applied, "edits_rejected": s.n_edits_rejected,
            }
            for s in steps
        ]

    def get_log_lines(self) -> list[str]:
        return self.get_log_lines_for_engine(self.engine)

    def get_log_lines_for_engine(self, engine: TrainerEngine) -> list[str]:
        """Return formatted log lines from a specific engine's events."""
        with engine._events_lock:
            recent = list(engine._events[-200:])
        lines = []
        for e in recent:
            ts = e.get("time", "")
            ev = e.get("event", "")
            if ev == "log":
                level = e.get("level", "INFO")
                tag = e.get("module", "")
                msg = e.get("message", "")
                lines.append(f"[{ts}] [{level}] {tag}: {msg}")
            else:
                extra = {k: v for k, v in e.items() if k not in ("time", "event")}
                extra_str = json.dumps(extra, ensure_ascii=False) if extra else ""
                lines.append(f"[{ts}] {ev} {extra_str}")
        return lines

    def get_log_events_for_engine(self, engine: TrainerEngine) -> list[dict]:
        """Return raw event dicts from a specific engine."""
        with engine._events_lock:
            return list(engine._events[-100:])

    def get_score_chart(self) -> list[dict]:
        steps = self.engine.history.steps
        return [{"step": s.step, "score": s.score} for s in steps]

    def get_data_status(self) -> dict:
        if self.engine.has_data():
            summary = self.engine.data_loader.summary()
            data_path = getattr(self.engine.data_loader, "_root", None) or getattr(
                self.engine.data_loader, "root", ""
            )
            return {"loaded": True, "splits": summary, "path": str(data_path)}
        return {"loaded": False}

    def get_data_status_for_task(self, task_id: str) -> dict:
        """Check data status from filesystem for a specific task (engine-independent)."""
        if not task_id:
            return {"loaded": False}
        task_dir = self.train_root / task_id
        data_root = task_dir / "uploaded_data"
        if not data_root.is_dir():
            return {"loaded": False}
        splits: dict[str, int] = {}
        for split_dir in sorted(data_root.iterdir()):
            if not split_dir.is_dir():
                continue
            items_path = split_dir / "items.json"
            if items_path.is_file():
                try:
                    data = _load_json(str(items_path)) or []
                    splits[split_dir.name] = len(data)
                except Exception:
                    splits[split_dir.name] = -1
        if splits:
            return {"loaded": True, "splits": splits, "path": str(data_root)}
        return {"loaded": False}

    def get_task_detail(self, task_id: str) -> dict:
        if not task_id:
            return {}
        task_dir = self.train_root / task_id
        config = _load_json(str(task_dir / "config.json")) or {}
        state = _load_json(str(task_dir / "runtime_state.json")) or {}
        has_history = (task_dir / "history.json").exists()
        summary = _load_json(str(task_dir / "summary.json")) or {}
        is_archived = bool(summary)  # summary.json means training completed
        created = _parse_task_created(task_id) or config.get("created_at", "unknown")
        bs = state.get("best_score", -1)
        baseline = state.get("baseline_score", -1)

        # Timing information
        started_at = state.get("started_at") or config.get("started_at")
        finished_at = state.get("finished_at")
        duration_s = state.get("total_wall_time_s") or summary.get("total_wall_time_s")

        # Build stopping_dirs from active_sessions
        _stopping_dirs: set[str] = set()
        if self.active_sessions:
            for info in self.active_sessions.values():
                if info.get("stop_requested"):
                    eng = info.get("engine")
                    if eng:
                        _d = info.get("running_task_dir") or getattr(eng, "out_dir", None)
                        if _d:
                            _stopping_dirs.add(str(_d))

        # Unified status determination
        status = _resolve_task_status(
            task_dir,
            state.get("status", ""),
            is_archived,
            has_history,
            self.active_sessions,
            _stopping_dirs,
        )

        return {
            "task_id": task_id,
            "name": config.get("name", ""),
            "description": config.get("description", ""),
            "algorithm": config.get("algorithm", "?"),
            "created": created,
            "status": status,
            "archived": is_archived,
            "best_score": bs,
            "baseline_score": baseline,
            "best_step": state.get("best_step", 0),
            "total_steps": state.get("last_completed_step", 0),
            "total_epochs": config.get("num_epochs", 0) if has_history else 0,
            "path": str(task_dir),
            # Timing fields
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": duration_s,
        }

    def get_default_deploy_name(self, task_id: str) -> str:
        if not task_id:
            return "train-skill"
        task_dir = self.train_root / task_id
        config = _load_json(str(task_dir / "config.json")) or {}
        alg = config.get("algorithm", "skill")
        tname = task_id
        if task_id.startswith(f"{alg}-"):
            tname = task_id[len(alg) + 1:]
        return f"train-{alg}-{tname}"

    def get_readonly_history(self, task_id: str) -> list[dict]:
        if not task_id:
            return []
        hist = _load_task_history(self.train_root / task_id)
        steps = hist.get("steps", [])
        return [
            {
                "step": s.get("step"), "epoch": s.get("epoch"),
                "score": round(s.get("score", 0), 4),
                "action": s.get("action"), "skill_hash": s.get("skill_hash", ""),
                "edits_applied": s.get("n_edits_applied", 0),
                "edits_rejected": s.get("n_edits_rejected", 0),
            }
            for s in steps
        ]

    def get_readonly_score_chart(self, task_id: str) -> list[dict]:
        if not task_id:
            return []
        hist = _load_task_history(self.train_root / task_id)
        steps = hist.get("steps", [])
        return [{"step": s.get("step"), "score": s.get("score", 0)} for s in steps]
