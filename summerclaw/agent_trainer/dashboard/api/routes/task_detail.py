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
        active_dir = str(state.engine.out_dir)
        selected_dir = str(state.train_root / task_id)
        # Only return active engine's history when the queried task IS the active task.
        # The old `or state.engine.has_data()` condition leaked the active task's
        # history to any other task that happened to be queried while data was loaded.
        if selected_dir == active_dir:
            return {"history": state.get_history_rows(), "chart": state.get_score_chart()}
        return {
            "history": state.get_readonly_history(task_id),
            "chart": state.get_readonly_score_chart(task_id),
        }

    @router.get("/api/tasks/{task_id}/skill")
    async def get_task_skill(task_id: str, which: str = "best"):
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
