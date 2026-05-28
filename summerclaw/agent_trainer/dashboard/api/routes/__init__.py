"""Route registration helpers.

Each sub-module exposes a ``register(router, state)`` function that
attaches its endpoints to the shared ``APIRouter``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter
    from summerclaw.agent_trainer.dashboard.api.state import _DashboardState


def register_all(router: APIRouter, state: _DashboardState) -> None:
    """Register every route group on *router*."""
    from summerclaw.agent_trainer.dashboard.api.routes import (
        task_list,
        task_detail,
        training,
        task_crud,
        data,
        config,
        realtime,
    )

    task_list.register(router, state)
    task_detail.register(router, state)
    training.register(router, state)
    task_crud.register(router, state)
    data.register(router, state)
    config.register(router, state)
    realtime.register(router, state)
