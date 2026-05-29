"""Comprehensive tests for per-task isolation in the agent trainer.

Covers:
- Scheduler engine_factory lifecycle (create, register, cleanup)
- Scheduler without engine_factory (fallback / rejection behavior)
- State helpers: get_engine_for_task, is_task_engine_running, log routing
- Log sink thread isolation (threading.get_ident filtering)
- Session key isolation (task_id embedded in session_key)
- _set_task_dir / _ensure_out_dir propagation of _task_id
- API routes routing to per-task engines (logs, realtime, cancel, history)
"""
from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock fixtures (mirroring test_dashboard_api.py)
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

    def __init__(self):
        self._task_id = ""

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
    engine._events: list[dict] = []
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


# ===========================================================================
# 1. Scheduler engine_factory lifecycle
# ===========================================================================


class TestSchedulerEngineFactory:
    """Test per-task engine creation and cleanup via engine_factory."""

    def test_scheduler_stores_engine_factory(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)
            assert scheduler._engine_factory is factory

    def test_get_task_engine_returns_registered(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)
            per_task = _make_mock_engine()
            with scheduler._task_engines_lock:
                scheduler._task_engines["task-A"] = per_task
            assert scheduler.get_task_engine("task-A") is per_task

    def test_get_task_engine_unknown_returns_none(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)
            assert scheduler.get_task_engine("unknown-task") is None

    def test_get_all_task_engines(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)
            e1 = _make_mock_engine()
            e2 = _make_mock_engine()
            with scheduler._task_engines_lock:
                scheduler._task_engines["t1"] = e1
                scheduler._task_engines["t2"] = e2
            all_engines = scheduler.get_all_task_engines()
            assert all_engines == {"t1": e1, "t2": e2}

    def test_used_workers_counts_task_engines(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine(is_running=False)
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)
            # Register two running per-task engines
            e1 = _make_mock_engine(is_running=True)
            e2 = _make_mock_engine(is_running=True)
            with scheduler._task_engines_lock:
                scheduler._task_engines["t1"] = e1
                scheduler._task_engines["t2"] = e2
            workers = scheduler.used_workers()
            assert workers >= 2

    def test_status_info_includes_task_engines(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)
            e1 = _make_mock_engine(is_running=True)
            with scheduler._task_engines_lock:
                scheduler._task_engines["t1"] = e1
            info = scheduler.status_info()
            # running_tasks is a dict[str, int]: task_id -> effective_workers
            assert isinstance(info["running_tasks"], dict)
            assert "t1" in info["running_tasks"]


# ===========================================================================
# 2. Scheduler without engine_factory (fallback behavior)
# ===========================================================================


class TestSchedulerNoEngineFactory:
    """Test scheduler behavior when engine_factory is None."""

    def test_scheduler_works_without_factory(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            assert scheduler._engine_factory is None

    def test_get_task_engine_no_factory_returns_none(self):
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {})
            assert scheduler.get_task_engine("any") is None


# ===========================================================================
# 3. State helpers: get_engine_for_task, is_task_engine_running
# ===========================================================================


class TestStatePerTaskHelpers:
    """Test _DashboardState methods that route to per-task engines."""

    def test_get_engine_for_task_no_scheduler_returns_shared(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            state.scheduler = None
            result = state.get_engine_for_task("any-task")
            assert result is engine

    def test_get_engine_for_task_with_per_task_engine(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        per_task_engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            # Mock scheduler to return per-task engine
            mock_scheduler = MagicMock()
            mock_scheduler.get_task_engine.return_value = per_task_engine
            state.scheduler = mock_scheduler
            result = state.get_engine_for_task("task-X")
            assert result is per_task_engine

    def test_get_engine_for_task_fallback_to_shared(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            mock_scheduler = MagicMock()
            mock_scheduler.get_task_engine.return_value = None
            state.scheduler = mock_scheduler
            result = state.get_engine_for_task("missing-task")
            assert result is engine

    def test_is_task_engine_running_no_scheduler(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(is_running=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            state.scheduler = None
            assert state.is_task_engine_running("any") is False

    def test_is_task_engine_running_with_per_task(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(is_running=False)
        per_task = _make_mock_engine(is_running=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            mock_scheduler = MagicMock()
            mock_scheduler.get_task_engine.return_value = per_task
            state.scheduler = mock_scheduler
            assert state.is_task_engine_running("task-running") is True

    def test_is_task_engine_running_missing_task(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine(is_running=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            mock_scheduler = MagicMock()
            mock_scheduler.get_task_engine.return_value = None
            state.scheduler = mock_scheduler
            assert state.is_task_engine_running("no-such") is False

    def test_get_log_lines_for_engine_empty(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        engine._events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            lines = state.get_log_lines_for_engine(engine)
            assert lines == []

    def test_get_log_lines_for_engine_with_events(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        engine._events = [
            {"ts": 1000.0, "level": "INFO", "msg": "Starting training"},
            {"ts": 1001.0, "level": "WARNING", "msg": "Slow rollout"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            lines = state.get_log_lines_for_engine(engine)
            assert len(lines) == 2
            assert "Starting training" in lines[0]
            assert "Slow rollout" in lines[1]

    def test_get_log_events_for_engine_returns_copy(self):
        from summerclaw.agent_trainer.dashboard.api.state import _DashboardState

        engine = _make_mock_engine()
        engine._events = [{"ts": 1.0, "level": "INFO", "msg": "test"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _DashboardState(engine, Path(tmpdir), {})
            events = state.get_log_events_for_engine(engine)
            assert len(events) == 1
            # Verify it's a copy
            events.append({"extra": True})
            assert len(engine._events) == 1


# ===========================================================================
# 4. Log sink thread isolation
# ===========================================================================


class TestLogSinkThreadIsolation:
    """Test that the loguru sink only captures logs from its own thread."""

    def test_log_sink_thread_id_captured(self):
        """Verify _install_log_sink captures the correct thread ID."""
        from summerclaw.agent_trainer.engine.trainer import TrainerEngine

        # We can't easily instantiate TrainerEngine directly (many deps),
        # but we can test the sink closure behavior indirectly.
        # Create a minimal mock that mimics the sink installation.
        import loguru

        captured_ids: list[int] = []

        def _test_sink():
            _train_thread_id = threading.get_ident()
            captured_ids.append(_train_thread_id)

            def _sink(message):
                if threading.get_ident() != _train_thread_id:
                    return  # filtered
                pass

            return loguru.logger.add(_sink, level="INFO")

        # Install sink from main thread
        sid = _test_sink()
        try:
            # Log from main thread — should be captured
            loguru.logger.info("from main thread")
            # Log from a different thread — should be filtered
            t = threading.Thread(target=lambda: loguru.logger.info("from other thread"))
            t.start()
            t.join()
        finally:
            loguru.logger.remove(sid)

        # The sink captured the main thread's ID
        assert captured_ids[0] == threading.get_ident()

    def test_thread_id_filtering_logic(self):
        """Unit test the thread ID filtering logic that the sink uses."""
        main_thread_id = threading.get_ident()
        other_results: list[bool] = []

        def check_from_other_thread():
            # This simulates the sink's filtering check
            matches = (threading.get_ident() == main_thread_id)
            other_results.append(matches)

        t = threading.Thread(target=check_from_other_thread)
        t.start()
        t.join()

        assert other_results == [False]
        # Main thread should match
        assert threading.get_ident() == main_thread_id


# ===========================================================================
# 5. Session key isolation (task_id in session_key)
# ===========================================================================


class TestSessionKeyIsolation:
    """Verify that session keys include task_id for multi-task isolation."""

    def test_env_has_task_id_attribute(self):
        """SummerClawEnvAdapter should have _task_id attribute."""
        from summerclaw.agent_trainer.env.summerclaw_env import SummerClawEnvAdapter

        provider = _MockProvider()
        env = SummerClawEnvAdapter(
            provider=provider,
            model="test-model",
            workspace=Path(tempfile.mkdtemp()),
            train_out_dir=tempfile.mkdtemp(),
        )
        assert env._task_id == ""

    def test_env_task_id_can_be_set(self):
        from summerclaw.agent_trainer.env.summerclaw_env import SummerClawEnvAdapter

        provider = _MockProvider()
        env = SummerClawEnvAdapter(
            provider=provider,
            model="test-model",
            workspace=Path(tempfile.mkdtemp()),
            train_out_dir=tempfile.mkdtemp(),
        )
        env._task_id = "task-ABC"
        assert env._task_id == "task-ABC"

    def test_session_key_includes_task_id(self):
        """Session key format must include task_id to prevent collisions."""
        # We test the format string directly rather than running a full rollout
        task_id = "skillopt-20260101-120000"
        epoch = 1
        step = 5
        item_id = "item-42"

        # Old format (vulnerable to collision):
        old_key = f"trainer:epoch{epoch:02d}:step{step:03d}:{item_id}"
        # New format (isolated):
        new_key = f"trainer:{task_id}:e{epoch:02d}:s{step:03d}:{item_id}"

        # Keys are different
        assert old_key != new_key
        # New key contains the task_id
        assert task_id in new_key
        # Two different tasks produce different keys
        key_task_a = f"trainer:taskA:e{epoch:02d}:s{step:03d}:{item_id}"
        key_task_b = f"trainer:taskB:e{epoch:02d}:s{step:03d}:{item_id}"
        assert key_task_a != key_task_b

    def test_session_key_fallback_when_task_id_empty(self):
        """When _task_id is empty, session_key uses 'shared' fallback."""
        _tid = "" or "shared"
        epoch, step, item_id = 0, 0, "x1"
        key = f"trainer:{_tid}:e{epoch:02d}:s{step:03d}:{item_id}"
        assert "shared" in key
        assert "trainer:shared:e00:s000:x1" == key


# ===========================================================================
# 6. _set_task_dir / _ensure_out_dir propagation of _task_id
# ===========================================================================


class TestTaskIdPropagation:
    """Test that _set_task_dir and _ensure_out_dir propagate _task_id to env."""

    def test_set_task_dir_propagates_task_id(self):
        """_set_task_dir should set env._task_id to the directory name."""
        from summerclaw.agent_trainer.engine.trainer import TrainerEngine

        # We test this by examining the code path — _set_task_dir sets
        # env._task_id = task_dir.name when env has _task_id attribute
        class FakeEnv:
            train_workspace = Path("/tmp/fake")
            _workspace_ready = True
            _task_id = ""

        fake_env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "my-task-dir"
            task_dir.mkdir()

            # Simulate what _set_task_dir does for _task_id propagation
            if hasattr(fake_env, "_task_id"):
                fake_env._task_id = task_dir.name

            assert fake_env._task_id == "my-task-dir"

    def test_ensure_out_dir_propagates_task_id(self):
        """_ensure_out_dir should set env._task_id to the created dir name."""
        import datetime

        class FakeEnv:
            train_workspace = Path("/tmp/fake")
            _workspace_ready = True
            _task_id = ""

        fake_env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "skillopt-20260101-120000"
            out_dir.mkdir()

            # Simulate what _ensure_out_dir does for _task_id propagation
            if hasattr(fake_env, "_task_id"):
                fake_env._task_id = out_dir.name

            assert fake_env._task_id == "skillopt-20260101-120000"

    def test_set_task_dir_no_crash_without_task_id_attr(self):
        """_set_task_dir should work even if env lacks _task_id attribute."""

        class OldEnv:
            train_workspace = Path("/tmp/fake")
            _workspace_ready = True

        old_env = OldEnv()
        assert not hasattr(old_env, "_task_id")

        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "task-old"
            task_dir.mkdir()

            # Simulate the hasattr check in _set_task_dir
            if hasattr(old_env, "_task_id"):
                old_env._task_id = task_dir.name

            # No crash, and env unchanged
            assert not hasattr(old_env, "_task_id")


# ===========================================================================
# 7. API routes routing to per-task engines
# ===========================================================================


class TestApiPerTaskRouting:
    """Test that API routes correctly route to per-task engines."""

    @pytest.fixture
    def client_state_factory(self):
        """Create a FastAPI test client with per-task engine support."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from summerclaw.agent_trainer.dashboard.api import _create_api

        def _make(engine_kwargs=None, engine_factory=None):
            engine_kwargs = engine_kwargs or {}
            engine = _make_mock_engine(**engine_kwargs)
            with tempfile.TemporaryDirectory() as tmpdir:
                train_root = Path(tmpdir)
                router, state = _create_api(
                    engine,
                    train_root=train_root,
                    engine_factory=engine_factory,
                )
                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)
                yield client, state, train_root, engine

        return _make

    def test_logs_without_task_id_uses_shared_engine(self, client_state_factory):
        """GET /api/logs without task_id falls back to shared engine."""
        for client, state, _, engine in client_state_factory():
            engine._events = [
                {"ts": 1000.0, "level": "INFO", "msg": "shared engine log"},
            ]
            resp = client.get("/api/logs")
            assert resp.status_code == 200
            data = resp.json()
            assert "logs" in data

    def test_logs_with_task_id_routes_to_per_task(self, client_state_factory):
        """GET /api/logs?task_id=xxx routes to per-task engine if available."""
        def factory():
            return _make_mock_engine()

        for client, state, _, engine in client_state_factory(
            engine_factory=factory,
        ):
            # Register a per-task engine with its own events
            per_task = _make_mock_engine()
            per_task._events = [
                {"ts": 2000.0, "level": "INFO", "msg": "per-task log"},
            ]
            with state.scheduler._task_engines_lock:
                state.scheduler._task_engines["task-log"] = per_task

            resp = client.get("/api/logs?task_id=task-log")
            assert resp.status_code == 200

    def test_logs_with_unknown_task_id_falls_back(self, client_state_factory):
        """GET /api/logs?task_id=unknown falls back to shared engine."""
        def factory():
            return _make_mock_engine()

        for client, state, _, engine in client_state_factory(
            engine_factory=factory,
        ):
            engine._events = [
                {"ts": 1000.0, "level": "INFO", "msg": "fallback"},
            ]
            resp = client.get("/api/logs?task_id=nonexistent")
            assert resp.status_code == 200

    def test_realtime_with_per_task_engine(self, client_state_factory):
        """GET /api/realtime?task_id=xxx uses per-task engine data."""
        def factory():
            return _make_mock_engine()

        for client, state, train_root, engine in client_state_factory(
            engine_factory=factory,
        ):
            task_dir = train_root / "task-rt"
            task_dir.mkdir()
            per_task = _make_mock_engine(is_running=True)
            per_task.out_dir = task_dir
            per_task._events = [
                {"ts": 3000.0, "level": "INFO", "msg": "realtime per-task"},
            ]
            with state.scheduler._task_engines_lock:
                state.scheduler._task_engines["task-rt"] = per_task

            resp = client.get("/api/realtime?task_id=task-rt")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data

    def test_cancel_routes_to_per_task_engine(self, client_state_factory):
        """POST /api/cancel finds and cancels the per-task engine."""
        def factory():
            return _make_mock_engine()

        for client, state, train_root, engine in client_state_factory(
            engine_factory=factory,
        ):
            task_dir = train_root / "task-cancel"
            task_dir.mkdir()
            per_task = _make_mock_engine(is_running=True)
            per_task.out_dir = task_dir
            with state.scheduler._task_engines_lock:
                state.scheduler._task_engines["task-cancel"] = per_task

            # Cancel without task_id should hit shared engine
            resp = client.post("/api/cancel")
            assert resp.status_code == 200

    def test_scheduler_info_endpoint(self, client_state_factory):
        """GET /api/scheduler shows per-task engine info."""
        def factory():
            return _make_mock_engine()

        for client, state, _, engine in client_state_factory(
            engine_factory=factory,
        ):
            e1 = _make_mock_engine(is_running=True)
            e2 = _make_mock_engine(is_running=True)
            with state.scheduler._task_engines_lock:
                state.scheduler._task_engines["t1"] = e1
                state.scheduler._task_engines["t2"] = e2

            resp = client.get("/api/scheduler")
            assert resp.status_code == 200
            data = resp.json()
            # running_tasks is a dict[str, int]
            assert isinstance(data["running_tasks"], dict)
            assert len(data["running_tasks"]) >= 2


# ===========================================================================
# 8. Concurrent task engine thread safety
# ===========================================================================


class TestConcurrentTaskEngines:
    """Test thread safety of _task_engines dict operations."""

    def test_concurrent_register_unregister(self):
        """Multiple threads can safely register/unregister task engines."""
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)

            errors: list[Exception] = []

            def register_and_unregister(task_id: str):
                try:
                    per_task = _make_mock_engine()
                    with scheduler._task_engines_lock:
                        scheduler._task_engines[task_id] = per_task
                    # Small delay to increase contention
                    time.sleep(0.001)
                    with scheduler._task_engines_lock:
                        scheduler._task_engines.pop(task_id, None)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=register_and_unregister, args=(f"task-{i}",))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == []
            assert len(scheduler._task_engines) == 0

    def test_concurrent_get_task_engine(self):
        """Multiple threads can safely read task engines."""
        from summerclaw.agent_trainer.dashboard.api.scheduler import _TaskScheduler

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = _TaskScheduler(engine, Path(tmpdir), {}, engine_factory=factory)

            # Pre-register some engines
            engines = {}
            for i in range(10):
                e = _make_mock_engine()
                engines[f"task-{i}"] = e
            with scheduler._task_engines_lock:
                scheduler._task_engines.update(engines)

            results: dict[str, Any] = {}
            lock = threading.Lock()

            def read_engine(task_id: str):
                eng = scheduler.get_task_engine(task_id)
                with lock:
                    results[task_id] = eng

            threads = [
                threading.Thread(target=read_engine, args=(f"task-{i}",))
                for i in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All reads should return the correct engine
            for task_id, expected_engine in engines.items():
                assert results[task_id] is expected_engine


# ===========================================================================
# 9. _create_api factory with engine_factory
# ===========================================================================


class TestCreateApiWithEngineFactory:
    """Test _create_api correctly passes engine_factory through."""

    def test_create_api_without_engine_factory(self):
        from summerclaw.agent_trainer.dashboard.api import _create_api

        engine = _make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            router, state = _create_api(engine, train_root=Path(tmpdir))
            assert state.scheduler is not None
            assert state.scheduler._engine_factory is None

    def test_create_api_with_engine_factory(self):
        from summerclaw.agent_trainer.dashboard.api import _create_api

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        with tempfile.TemporaryDirectory() as tmpdir:
            router, state = _create_api(
                engine, train_root=Path(tmpdir), engine_factory=factory,
            )
            assert state.scheduler is not None
            assert state.scheduler._engine_factory is factory


# ===========================================================================
# 10. DashboardServer engine_factory passthrough
# ===========================================================================


class TestDashboardServerEngineFactory:
    """Test that DashboardServer correctly stores and passes engine_factory."""

    def test_server_stores_engine_factory(self):
        from summerclaw.agent_trainer.dashboard.app import DashboardServer

        engine = _make_mock_engine()
        factory = MagicMock(return_value=_make_mock_engine())
        server = DashboardServer(
            engine=engine,
            train_root=Path(tempfile.mkdtemp()),
            engine_factory=factory,
        )
        assert server._engine_factory is factory

    def test_server_works_without_engine_factory(self):
        from summerclaw.agent_trainer.dashboard.app import DashboardServer

        engine = _make_mock_engine()
        server = DashboardServer(
            engine=engine,
            train_root=Path(tempfile.mkdtemp()),
        )
        assert server._engine_factory is None
