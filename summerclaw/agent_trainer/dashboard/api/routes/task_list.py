"""Task list route — GET /api/tasks (with search, filter, sort, pagination)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def register(router: APIRouter, state: _DashboardState) -> None:
    """Register task-list routes on *router*."""
    from fastapi import APIRouter as _AR  # noqa: F811 (re-import for runtime)
    from summerclaw.agent_trainer.dashboard.task_utils import _scan_all_tasks_cached

    @router.get("/api/tasks")
    async def list_tasks(
        search: str = "",
        status: str = "all",
        sort: str = "created",
        asc: bool = False,
        page: int = 1,
        per_page: int = 10,
    ):
        tasks = _scan_all_tasks_cached(
            state.train_root, state.active_sessions,
            max_concurrency=state.scheduler.max_concurrency if state.scheduler else 0,
        )
        # Filter
        if search:
            q = search.lower()
            tasks = [
                t for t in tasks
                if q in t["task_id"].lower()
                or q in t["algorithm"].lower()
                or q in t.get("name", "").lower()
                or q in t.get("description", "").lower()
            ]
        if status and status != "all":
            tasks = [t for t in tasks if t["status"] == status]
        # Sort
        if sort in ("created", "best_score", "total_steps", "algorithm"):
            tasks.sort(key=lambda x: x.get(sort, ""), reverse=not asc)
        # Paginate
        total = len(tasks)
        total_pages = max(1, -(-total // per_page))
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        return {
            "tasks": tasks[start:end],
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "start": start,
            "end": end,
        }
