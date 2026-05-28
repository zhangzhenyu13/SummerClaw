"""API factory — assembles all routes and state into a FastAPI router.

This is the single entry point used by ``app.py`` to wire up the
dashboard REST API.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine
from summerclaw.agent_trainer.dashboard.task_utils import _default_train_root
from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler
from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def _create_api(
    engine: TrainerEngine,
    train_root: Path | None = None,
    active_sessions: dict | None = None,
):
    """Create FastAPI APIRouter with all dashboard endpoints."""
    try:
        from fastapi import APIRouter
    except ImportError:
        logger.warning("FastAPI not installed; dashboard API disabled")
        return None, None

    if train_root is None:
        train_root = _default_train_root()
    if active_sessions is None:
        active_sessions = {}

    state = _DashboardState(engine, train_root, active_sessions)
    router = APIRouter()

    # -- Health (trivial, kept here for visibility) -----------------------

    @router.get("/api/health")
    async def health():
        return {"status": "ok"}

    # -- Register all route groups ----------------------------------------

    from summerclaw.agent_trainer.dashboard.api.routes import register_all
    register_all(router, state)

    # -- Initialize scheduler ---------------------------------------------

    scheduler = _TaskScheduler(engine, train_root, active_sessions)
    state.scheduler = scheduler
    # NOTE: scheduler.start() is deferred to the FastAPI lifespan so that
    # it runs inside the uvicorn event loop, not the background thread.

    return router, state
