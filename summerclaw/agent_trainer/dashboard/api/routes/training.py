"""Training control routes — start / cancel training for a task."""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def register(router: APIRouter, state: _DashboardState) -> None:
    """Register training-control routes on *router*."""
    from summerclaw.agent_trainer.engine.trainer import _load_json

    # ------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------

    @router.post("/api/tasks/{task_id}/start")
    async def start_training(task_id: str, body: dict = None):
        body = body or {}
        try:
            if state.engine.is_running:
                return {"error": "Training already in progress."}

            # Guard: only idle (never started) or stopped (resume) tasks can be started.
            task_dir = state.train_root / task_id
            if not task_dir.is_dir():
                return {"error": f"Task directory not found: {task_id}"}

            summary = _load_json(str(task_dir / "summary.json")) or {}
            rt_state = _load_json(str(task_dir / "runtime_state.json")) or {}
            has_history = (task_dir / "history.json").exists()
            persisted_status = rt_state.get("status", "")

            if summary:
                # Archived / completed with summary — immutable
                return {
                    "error": (
                        "This task has already been completed and is archived. "
                        "Use 'Copy to Create' to start a new run with the same config."
                    ),
                }

            if persisted_status == "completed" or (persisted_status == "" and has_history):
                return {
                    "error": (
                        "This task has already completed. "
                        "Use 'Copy to Create' to start a new run."
                    ),
                }

            if persisted_status == "failed":
                return {
                    "error": (
                        "This task has failed. "
                        "Use 'Copy to Create' to start a new run with the same config."
                    ),
                }

            # Allowed statuses: idle (no history, no status), stopped (resume checkpoint),
            # queued (scheduler-paused, can be force-started manually)

            state._maybe_restore_task(task_id)
            state._apply_yaml_to_engine(state.engine.out_dir)
            if not state.engine.has_data():
                return {"error": "No training data loaded. Upload data first."}
            state.engine._ensure_out_dir()
            skill_path = body.get("skill_init_path", "")
            if skill_path:
                state.engine.skill_init_path = skill_path
                p = Path(skill_path).expanduser()
                if p.is_file():
                    content = p.read_text(encoding="utf-8").strip()
                    if content:
                        state.engine.skill_init = content
                        state.engine._current_skill = content
                        state.engine._best_skill = content
            state.engine._cancel_requested = False
            state._stop_requested = False
            # Clear stop_requested flag on all sessions for this engine
            for _sess in state.active_sessions.values():
                if _sess.get("engine") is state.engine:
                    _sess.pop("stop_requested", None)

            # Unregister from scheduler (task is being manually started)
            if state.scheduler:
                state.scheduler.unregister(task_id)

            # Record the running task dir in the active session so that
            # _scan_all_tasks can reliably identify the running task even
            # when eng.out_dir is changed by _maybe_restore_task().
            _task_dir_str = str(state.engine.out_dir)
            for _sess in state.active_sessions.values():
                if _sess.get("engine") is state.engine:
                    _sess["running_task_dir"] = _task_dir_str

            def _run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(state.engine.train())
                except Exception as exc:
                    logger.error("Training failed: {}", exc, exc_info=True)
                finally:
                    # Clear running_task_dir and stop_requested when training ends
                    for _sess in state.active_sessions.values():
                        if _sess.get("engine") is state.engine:
                            _sess.pop("running_task_dir", None)
                            _sess.pop("stop_requested", None)
                    # Notify scheduler so queued tasks can be promoted
                    if state.scheduler:
                        state.scheduler.on_task_finished(task_id)
                    loop.close()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            msg = f"Training started — task: {task_id}"
            print(msg)
            state._fire_notify(msg)
            return {"status": "started", "task_id": task_id}
        except Exception as exc:
            return {"error": f"Error starting training: {exc}"}

    # ------------------------------------------------------------------
    # Cancel training
    # ------------------------------------------------------------------

    @router.post("/api/tasks/{task_id}/cancel")
    async def cancel_training(task_id: str):
        state.engine.request_cancel()
        state._stop_requested = True
        # Mark stop_requested on sessions so _scan_all_tasks can report "stopping"
        for _sess in state.active_sessions.values():
            if _sess.get("engine") is state.engine:
                _sess["stop_requested"] = True
        msg = f"Training stop requested — task: {task_id}"
        print(msg)
        state._fire_notify(msg)
        return {"status": "cancel_requested"}
