"""Dashboard REST API package.

Public API (backward-compatible with the old ``api.py`` module):
  - ``_create_api(engine, train_root, active_sessions)`` → ``(router, state)``

Internal sub-modules:
  - ``scheduler``  — concurrency-aware task scheduler
  - ``state``      — shared mutable dashboard state
  - ``factory``    — route assembly and API construction
  - ``routes/``    — one file per route group
"""
from summerclaw.agent_trainer.dashboard.api.factory import _create_api

__all__ = ["_create_api"]
