"""Real-time snapshot endpoint — combined data for UI refresh."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def register(router: APIRouter, state: _DashboardState) -> None:
    """Register the real-time snapshot route on *router*."""

    @router.get("/api/realtime")
    async def realtime_snapshot(task_id: str = ""):
        """Combined snapshot for real-time UI refresh (replaces gr.Timer)."""
        from summerclaw.agent_trainer.engine.trainer import _load_json
        from summerclaw.agent_trainer.dashboard.task_utils import _resolve_task_status

        is_active = not task_id or str(state.train_root / task_id) == str(state.engine.out_dir)

        if is_active:
            # Use proper status resolution (includes heartbeat-based crash detection)
            # instead of only checking engine.is_running.
            _task_dir = state.train_root / task_id if task_id else state.engine.out_dir
            _task_rt = _load_json(str(_task_dir / "runtime_state.json")) or {}
            _summary = _load_json(str(_task_dir / "summary.json")) or {}
            _has_history = (_task_dir / "history.json").exists()
            _is_archived = bool(_summary)

            _stopping_dirs: set[str] = set()
            if state.active_sessions:
                for info in state.active_sessions.values():
                    if info.get("stop_requested"):
                        eng = info.get("engine")
                        if eng:
                            _d = info.get("running_task_dir") or getattr(eng, "out_dir", None)
                            if _d:
                                _stopping_dirs.add(str(_d))

            resolved_status = _resolve_task_status(
                _task_dir,
                _task_rt.get("status", ""),
                _is_archived,
                _has_history,
                state.active_sessions,
                _stopping_dirs,
            )
            status = state.get_status_dict()
            status["status"] = resolved_status
            history = state.get_history_rows()
            chart = state.get_score_chart()
            logs = state.get_log_lines()
            data_status = state.get_data_status()
        else:
            # Non-active task: read status from its own runtime_state.json
            # to avoid leaking the active engine's scores/state.
            from summerclaw.agent_trainer.engine.trainer import _load_json
            from summerclaw.agent_trainer.dashboard.task_utils import _resolve_task_status

            _task_dir = state.train_root / task_id
            _task_rt = _load_json(str(_task_dir / "runtime_state.json")) or {}
            _summary = _load_json(str(_task_dir / "summary.json")) or {}
            _has_history = (_task_dir / "history.json").exists()
            _is_archived = bool(_summary)

            # Build stopping_dirs from active_sessions
            _stopping_dirs: set[str] = set()
            if state.active_sessions:
                for info in state.active_sessions.values():
                    if info.get("stop_requested"):
                        eng = info.get("engine")
                        if eng:
                            _d = info.get("running_task_dir") or getattr(eng, "out_dir", None)
                            if _d:
                                _stopping_dirs.add(str(_d))

            status = {
                "status": _resolve_task_status(
                    _task_dir,
                    _task_rt.get("status", ""),
                    _is_archived,
                    _has_history,
                    state.active_sessions,
                    _stopping_dirs,
                ),
                "best_score": _task_rt.get("best_score", -1),
                "baseline_score": _task_rt.get("baseline_score", -1),
                "best_step": _task_rt.get("best_step", 0),
                "total_steps": _task_rt.get("last_completed_step", 0),
                "total_epochs": _task_rt.get("total_epochs", 0),
            }
            history = state.get_readonly_history(task_id)
            chart = state.get_readonly_score_chart(task_id)
            logs = []
            # Check filesystem for data status even for non-active tasks
            data_status = state.get_data_status_for_task(task_id)

        # Detect training completion
        running = state.engine.is_running
        notification = None
        if state._was_running and not running:
            tid = Path(state.engine.out_dir).name
            if state._stop_requested:
                notification = (
                    f"Training stopped — task: {tid} "
                    f"(best={state.engine.best_score:.4f}, "
                    f"steps={state.engine.history.total_steps})"
                )
            else:
                notification = (
                    f"Training completed — task: {tid} "
                    f"(best={state.engine.best_score:.4f}, "
                    f"steps={state.engine.history.total_steps})"
                )
            state._fire_notify(notification)
        state._was_running = running

        if state._stop_requested and not running:
            state._stop_requested = False

        return {
            "status": status,
            "history": history,
            "chart": chart,
            "logs": logs,
            "data_status": data_status,
            "is_running": running,
            "stop_requested": state._stop_requested,
            "notification": notification,
            "deploy_name": state.get_default_deploy_name(task_id) if task_id else "",
        }
