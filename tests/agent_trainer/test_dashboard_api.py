"""Functional tests for the refactored dashboard API package.

Tests cover:
- Import structure and backward compatibility
- _TaskScheduler: budget calculation, idle/queue/promote logic
- _DashboardState: status dict, history, data status, task detail
- API routes via FastAPI TestClient
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@dataclass
class _MockStep:
    step: int = 0
    epoch: int = 0
    score: float = 0.0
    action: str = ""
    skill_hash: str = ""
    n_edits_applied: int = 0
    n_edits_rejected: int = 0


@dataclass
class _MockHistory:
    steps: list = field(default_factory=list)
    best_score: float = 0.0
    best_step: int = 0
    total_epochs: int = 0
    total_steps: int = 0

    def to_dict(self) -> dict:
        return {
            "steps": [],
            "best_score": self.best_score,
            "best_step": self.best_step,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
        }


class _MockProvider:
    max_concurrency = 10


class _MockEnv:
    provider = _MockProvider()
    workers = 4

    def reconfigure_for_task(self, mem_algo, enabled_tools):
        pass


class _MockAlgorithm:
    lr_scheduler_type = "constant"
    lr_mode = "constant"
    update_mode = "patch"


def _make_mock_engine(
    out_dir: Path | None = None,
    is_running: bool = False,
    has_data: bool = False,
) -> MagicMock:
    """Create a mock TrainerEngine with the attributes used by the API."""
    engine = MagicMock()
    engine.out_dir = out_dir or Path(tempfile.mkdtemp())
    engine.is_running = is_running
    engine.best_score = 0.85
    engine.baseline_score = 0.5
    engine._best_step = 3
    engine.current_score = 0.7
    engine._current_score = 0.7
    engine._current_skill = "skill content"
    engine._best_skill = "best skill content"
    engine._trainer_cfg = {"num_epochs": 3, "batch_size": 5}
    engine._cancel_requested = False
    engine._events = []
    engine._events_lock = threading.Lock()
    engine.skill_init = ""
    engine.skill_init_path = ""
    engine.env = _MockEnv()
    engine.algorithm = _MockAlgorithm()
    engine.data_loader = None
    engine.history = _MockHistory()

    engine.has_data.return_value = has_data
    engine.set_progress_callback = MagicMock()
    engine._ensure_out_dir = MagicMock()
    engine._set_task_dir = MagicMock()
    engine.request_cancel = MagicMock()
    engine.set_data_loader = MagicMock()
    engine.start_training_async = MagicMock(return_value="started")
    engine.best_skill = "best skill content"
    engine.current_skill = "skill content"

    return engine


# ---------------------------------------------------------------------------
# Tests: Import structure & backward compatibility
# ---------------------------------------------------------------------------


class TestImportStructure:
    """Verify the package structure and backward-compatible exports."""

    def test_create_api_importable_from_package(self):
        """The old ``from dashboard.api import _create_api`` must still work."""
        from summerclaw.agent_trainer.dashboard.api import _create_api
        assert callable(_create_api)

    def test_scheduler_importable(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler
        assert _TaskScheduler is not None

    def test_state_importable(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState
        assert _DashboardState is not None

    def test_factory_importable(self):
        from summerclaw.agent_trainer.dashboard.api.factory import _create_api
        assert callable(_create_api)

    def test_routes_register_all_importable(self):
        from summerclaw.agent_trainer.dashboard.api.routes import register_all
        assert callable(register_all)

    def test_all_route_modules_importable(self):
        from summerclaw.agent_trainer.dashboard.api.routes import (
            task_list,
            task_detail,
            training,
            task_crud,
            data,
            config,
            realtime,
        )
        for mod in (task_list, task_detail, training, task_crud, data, config, realtime):
            assert hasattr(mod, "register")
            assert callable(mod.register)

    def test_app_py_import_still_works(self):
        """app.py imports _create_api from dashboard.api — must not break."""
        from summerclaw.agent_trainer.dashboard.app import DashboardServer
        assert DashboardServer is not None


# ---------------------------------------------------------------------------
# Tests: _TaskScheduler
# ---------------------------------------------------------------------------


class TestTaskScheduler:
    """Unit tests for the concurrency-aware task scheduler."""

    def test_max_concurrency_from_provider(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            # Patch config loader to raise so it falls through to provider
            with patch("summerclaw.config.loader.load_config",
                        side_effect=Exception("no config")):
                # Should read from env.provider.max_concurrency (10)
                assert scheduler.max_concurrency == 10

    def test_max_concurrency_fallback(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        engine.env = None  # no env → fallback to 20
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            with patch("summerclaw.config.loader.load_config",
                        side_effect=Exception("no config")):
                assert scheduler.max_concurrency == 20

    def test_used_workers_empty(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            assert scheduler.used_workers() == 0

    def test_left_budget_equals_max_when_idle(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            assert scheduler.left_budget() == scheduler.max_concurrency

    def test_register_and_unregister_idle(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            scheduler.register_idle("task-001")
            assert "task-001" in scheduler._idle_pending
            scheduler.unregister("task-001")
            assert "task-001" not in scheduler._idle_pending

    def test_on_task_finished_clears_state(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            scheduler.register_idle("task-002")
            scheduler.on_task_finished("task-002")
            assert "task-002" not in scheduler._idle_pending

    def test_status_info_structure(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            info = scheduler.status_info()
            assert "max_concurrency" in info
            assert "used_workers" in info
            assert "left_budget" in info
            assert "idle_pending" in info
            assert "queued" in info
            assert "running_tasks" in info

    def test_write_status_creates_file(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler
        from summerclaw.agent_trainer.engine.trainer import _load_json

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir)
            task_dir = train_root / "task-003"
            task_dir.mkdir()
            scheduler = _TaskScheduler(engine, train_root, {})
            scheduler._write_status("task-003", "queued")
            rt = _load_json(str(task_dir / "runtime_state.json"))
            assert rt["status"] == "queued"


# ---------------------------------------------------------------------------
# Tests: _DashboardState
# ---------------------------------------------------------------------------


class TestDashboardState:
    """Unit tests for the shared mutable state object."""

    def test_init_wires_progress_callback(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            engine.set_progress_callback.assert_called_once()

    def test_get_status_dict_idle(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(is_running=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            d = state.get_status_dict()
            assert d["status"] == "idle"
            assert d["best_score"] == 0.85

    def test_get_status_dict_running(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(is_running=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            d = state.get_status_dict()
            assert d["status"] == "running"

    def test_get_status_dict_stopping(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(is_running=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            state._stop_requested = True
            d = state.get_status_dict()
            assert d["status"] == "stopping"

    def test_get_history_rows_empty(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            rows = state.get_history_rows()
            assert rows == []

    def test_get_history_rows_with_steps(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        engine.history.steps = [
            _MockStep(step=1, epoch=1, score=0.5, action="edit"),
            _MockStep(step=2, epoch=1, score=0.7, action="rewrite"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            rows = state.get_history_rows()
            assert len(rows) == 2
            assert rows[0]["step"] == 1
            assert rows[1]["score"] == 0.7

    def test_get_score_chart(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        engine.history.steps = [_MockStep(step=1, score=0.5), _MockStep(step=2, score=0.8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            chart = state.get_score_chart()
            assert chart == [{"step": 1, "score": 0.5}, {"step": 2, "score": 0.8}]

    def test_get_data_status_no_data(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(has_data=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            d = state.get_data_status()
            assert d == {"loaded": False}

    def test_get_task_detail_empty_task_id(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            d = state.get_task_detail("")
            assert d == {}

    def test_get_task_detail_nonexistent(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            d = state.get_task_detail("nonexistent-task")
            # Should not crash, returns empty-ish dict
            assert d["task_id"] == "nonexistent-task"
            assert d["status"] == "idle"

    def test_get_default_deploy_name(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState
        from summerclaw.agent_trainer.engine.trainer import _save_json

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir)
            task_dir = train_root / "skillopt-20260101-120000"
            task_dir.mkdir()
            _save_json(str(task_dir / "config.json"), {"algorithm": "skillopt"})
            state = _DashboardState(engine, train_root, {})
            name = state.get_default_deploy_name("skillopt-20260101-120000")
            assert name == "train-skillopt-20260101-120000"

    def test_get_default_deploy_name_empty(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            assert state.get_default_deploy_name("") == "train-skill"

    def test_maybe_restore_task_noop_when_empty(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            state._maybe_restore_task("")
            engine._set_task_dir.assert_not_called()

    def test_maybe_restore_task_calls_set_task_dir(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir)
            task_dir = train_root / "task-restore"
            task_dir.mkdir()
            state = _DashboardState(engine, train_root, {})
            state._maybe_restore_task("task-restore")
            engine._set_task_dir.assert_called_once_with(task_dir)

    def test_get_data_status_for_task_no_data(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            d = state.get_data_status_for_task("no-such-task")
            assert d == {"loaded": False}

    def test_get_data_status_for_task_with_splits(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState
        from summerclaw.agent_trainer.engine.trainer import _save_json

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir)
            data_root = train_root / "task-data" / "uploaded_data"
            for split in ("train", "val"):
                split_dir = data_root / split
                split_dir.mkdir(parents=True)
                _save_json(str(split_dir / "items.json"), [{"id": "1"}, {"id": "2"}])
            state = _DashboardState(engine, train_root, {})
            d = state.get_data_status_for_task("task-data")
            assert d["loaded"] is True
            assert d["splits"]["train"] == 2
            assert d["splits"]["val"] == 2


# ---------------------------------------------------------------------------
# Tests: _create_api factory
# ---------------------------------------------------------------------------


class TestCreateApi:
    """Test the API factory function."""

    def test_returns_router_and_state(self):
        from summerclaw.agent_trainer.dashboard.api import _create_api

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            router, state = _create_api(engine, train_root=Path(tmpdir))
            assert router is not None
            assert state is not None
            assert state.engine is engine

    def test_state_has_scheduler(self):
        from summerclaw.agent_trainer.dashboard.api import _create_api

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            router, state = _create_api(engine, train_root=Path(tmpdir))
            assert state.scheduler is not None

    def test_active_sessions_default_empty(self):
        from summerclaw.agent_trainer.dashboard.api import _create_api

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            _, state = _create_api(engine, train_root=Path(tmpdir))
            assert state.active_sessions == {}


# ---------------------------------------------------------------------------
# Tests: API routes via TestClient
# ---------------------------------------------------------------------------


class TestApiRoutes:
    """Integration tests using FastAPI TestClient."""

    @pytest.fixture
    def client_and_state(self):
        """Set up a FastAPI app with the dashboard routes and return (client, state)."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from summerclaw.agent_trainer.dashboard.api import _create_api

        engine = _make_mock_engine(has_data=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir)
            router, state = _create_api(engine, train_root=train_root)

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)
            yield client, state, train_root

    # -- Health -----------------------------------------------------------

    def test_health(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    # -- Task list --------------------------------------------------------

    def test_list_tasks_empty(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    # -- Algorithms & tools -----------------------------------------------

    def test_list_algorithms(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/algorithms")
        assert resp.status_code == 200
        algos = resp.json()["algorithms"]
        assert "skillopt" in algos

    def test_list_memory_algorithms(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/memory-algorithms")
        assert resp.status_code == 200
        algos = resp.json()["algorithms"]
        assert "naive_memory" in algos
        assert "mastra_om_memory" in algos

    def test_list_tools(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/tools")
        assert resp.status_code == 200
        cats = resp.json()["categories"]
        keys = {c["key"] for c in cats}
        assert "filesystem" in keys
        assert "browser" in keys

    # -- Config & status --------------------------------------------------

    def test_get_config(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "epochs" in data
        assert "batch_size" in data

    def test_get_status(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_running"] is False
        assert data["best_score"] == 0.85

    def test_get_logs(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert "logs" in resp.json()

    def test_get_engine_history(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/history")
        assert resp.status_code == 200

    def test_get_best_skill(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/best_skill")
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_get_current_skill(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/current_skill")
        assert resp.status_code == 200
        assert "content" in resp.json()

    # -- Scheduler info ---------------------------------------------------

    def test_scheduler_info(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/scheduler")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "max_concurrency" in data

    # -- Realtime snapshot ------------------------------------------------

    def test_realtime_snapshot(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/realtime")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "history" in data
        assert "chart" in data
        assert "logs" in data
        assert "is_running" in data

    # -- Task CRUD --------------------------------------------------------

    def test_create_task(self, client_and_state):
        client, state, train_root = client_and_state
        resp = client.post("/api/tasks", json={
            "name": "Test Task",
            "algorithm": "skillopt",
            "epochs": 5,
            "batch_size": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert "task_id" in data
        task_id = data["task_id"]
        assert (train_root / task_id).is_dir()
        assert (train_root / task_id / "config.json").is_file()
        assert (train_root / task_id / "skillopt.yaml").is_file()

    def test_create_task_no_name(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.post("/api/tasks", json={"algorithm": "skillopt"})
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_delete_task(self, client_and_state):
        client, state, train_root = client_and_state
        # Create a task first
        task_dir = train_root / "task-to-delete"
        task_dir.mkdir()
        resp = client.delete("/api/tasks/task-to-delete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert not task_dir.exists()

    def test_delete_nonexistent_task(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.delete("/api/tasks/nonexistent")
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_get_task_config(self, client_and_state):
        client, state, train_root = client_and_state
        from summerclaw.agent_trainer.engine.trainer import _save_json

        task_dir = train_root / "cfg-task"
        task_dir.mkdir()
        _save_json(str(task_dir / "config.json"), {"algorithm": "skillopt", "name": "cfg"})
        (task_dir / "skills").mkdir()
        resp = client.get("/api/tasks/cfg-task/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["algorithm"] == "skillopt"

    # -- YAML management --------------------------------------------------

    def test_yaml_template(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.get("/api/yaml/template")
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_upload_task_yaml(self, client_and_state):
        client, state, train_root = client_and_state
        task_dir = train_root / "yaml-task"
        task_dir.mkdir()
        yaml_content = "num_epochs: 10\nbatch_size: 20\n"
        resp = client.post("/api/tasks/yaml-task/yaml", json={"content": yaml_content})
        assert resp.status_code == 200
        assert resp.json()["status"] == "uploaded"
        assert (task_dir / "skillopt.yaml").is_file()

    def test_upload_task_yaml_invalid(self, client_and_state):
        client, state, train_root = client_and_state
        task_dir = train_root / "yaml-bad"
        task_dir.mkdir()
        resp = client.post("/api/tasks/yaml-bad/yaml", json={"content": "invalid: [yaml"})
        assert resp.status_code == 200
        assert "error" in resp.json()

    # -- Engine-level control ---------------------------------------------

    def test_cancel_engine(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.post("/api/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancel_requested"
        state.engine.request_cancel.assert_called_once()

    def test_start_engine(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.post("/api/start", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    # -- Scheduler control ------------------------------------------------

    def test_scheduler_stop(self, client_and_state):
        client, state, _ = client_and_state
        resp = client.post("/api/scheduler/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_scheduler_start(self, client_and_state):
        client, state, _ = client_and_state
        # Stop first, then start
        client.post("/api/scheduler/stop")
        resp = client.post("/api/scheduler/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
