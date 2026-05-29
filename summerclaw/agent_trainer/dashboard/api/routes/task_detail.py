"""Task detail routes — task info, history, skill content."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def register(router: APIRouter, state: _DashboardState) -> None:
    """Register task-detail routes on *router*."""

    @router.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        return state.get_task_detail(task_id)

    @router.get("/api/tasks/{task_id}/history")
    async def get_task_history(task_id: str):
        # Check for a scheduler-managed per-task engine first
        if state.scheduler is not None:
            _te = state.scheduler.get_task_engine(task_id)
            if _te is not None:
                _hist = _te.history
                history = [
                    {
                        "step": s.step, "epoch": s.epoch, "score": round(s.score, 4),
                        "action": s.action, "skill_hash": s.skill_hash,
                        "edits_applied": s.n_edits_applied, "edits_rejected": s.n_edits_rejected,
                    }
                    for s in _hist.steps
                ]
                chart = [{"step": s.step, "score": s.score} for s in _hist.steps]
                return {"history": history, "chart": chart}

        active_dir = str(state.engine.out_dir)
        selected_dir = str(state.train_root / task_id)
        # Only return active engine's history when the queried task IS the active task.
        if selected_dir == active_dir:
            return {"history": state.get_history_rows(), "chart": state.get_score_chart()}
        return {
            "history": state.get_readonly_history(task_id),
            "chart": state.get_readonly_score_chart(task_id),
        }

    @router.get("/api/tasks/{task_id}/skill")
    async def get_task_skill(task_id: str, which: str = "best"):
        # Check for a scheduler-managed per-task engine first
        if state.scheduler is not None:
            _te = state.scheduler.get_task_engine(task_id)
            if _te is not None:
                content = _te.best_skill if which == "best" else _te.current_skill
                return {"content": content, "chars": len(content)}

        active_dir = str(state.engine.out_dir)
        selected_dir = str(state.train_root / task_id)
        if selected_dir == active_dir:
            content = state.engine.best_skill if which == "best" else state.engine.current_skill
            return {"content": content, "chars": len(content)}
        # For historical tasks, read from disk
        task_dir = state.train_root / task_id
        skill_dir = task_dir / "skills"
        if skill_dir.is_dir():
            files = sorted(skill_dir.glob("*.md"))
            if files:
                target = files[-1] if which == "best" else files[0]
                content = target.read_text(encoding="utf-8")
                return {"content": content, "chars": len(content)}
        return {"content": "", "chars": 0}
