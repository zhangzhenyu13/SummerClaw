"""Tests for TaskRegistry (task_persistence module)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.agent.task_persistence import PersistedTask, TaskRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path: Path) -> TaskRegistry:
    return TaskRegistry(tmp_path)


def _make_task(task_id: str = "abc12345", task_type: str = "subagent") -> PersistedTask:
    return PersistedTask(
        task_id=task_id,
        task_type=task_type,
        label="Test task",
        description="Do something useful",
        origin_channel="telegram",
        origin_chat_id="99999",
        session_key="telegram:99999",
        depth=0,
        started_at=TaskRegistry.now_iso(),
    )


# ---------------------------------------------------------------------------
# TaskRegistry: save / remove / list_all / get
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_save_creates_file(self, registry: TaskRegistry, tmp_path: Path) -> None:
        task = _make_task("task001")
        registry.save(task)
        path = tmp_path / "task_registry" / "task001.json"
        assert path.exists(), "Task file should exist after save()"

    def test_save_content_is_valid_json(self, registry: TaskRegistry, tmp_path: Path) -> None:
        task = _make_task("task002")
        registry.save(task)
        path = tmp_path / "task_registry" / "task002.json"
        data = json.loads(path.read_text())
        assert data["task_id"] == "task002"
        assert data["label"] == "Test task"
        assert data["origin_channel"] == "telegram"

    def test_get_returns_task(self, registry: TaskRegistry) -> None:
        task = _make_task("task003")
        registry.save(task)
        loaded = registry.get("task003")
        assert loaded is not None
        assert loaded.task_id == "task003"
        assert loaded.description == "Do something useful"

    def test_get_missing_returns_none(self, registry: TaskRegistry) -> None:
        assert registry.get("nonexistent") is None

    def test_remove_deletes_file(self, registry: TaskRegistry, tmp_path: Path) -> None:
        task = _make_task("task004")
        registry.save(task)
        registry.remove("task004")
        path = tmp_path / "task_registry" / "task004.json"
        assert not path.exists(), "File should be deleted after remove()"

    def test_remove_nonexistent_is_silent(self, registry: TaskRegistry) -> None:
        # Must not raise
        registry.remove("does_not_exist")

    def test_list_all_empty(self, registry: TaskRegistry) -> None:
        assert registry.list_all() == []

    def test_list_all_returns_all_tasks(self, registry: TaskRegistry) -> None:
        for i in range(3):
            registry.save(_make_task(f"t{i:03d}"))
        tasks = registry.list_all()
        assert len(tasks) == 3
        ids = {t.task_id for t in tasks}
        assert ids == {"t000", "t001", "t002"}

    def test_list_all_excludes_removed(self, registry: TaskRegistry) -> None:
        registry.save(_make_task("keep"))
        registry.save(_make_task("gone"))
        registry.remove("gone")
        tasks = registry.list_all()
        assert len(tasks) == 1
        assert tasks[0].task_id == "keep"

    def test_list_all_skips_corrupt_files(self, registry: TaskRegistry, tmp_path: Path) -> None:
        # Write a bad JSON file alongside a valid one
        bad_path = tmp_path / "task_registry" / "corrupt.json"
        bad_path.write_text("not-json")
        registry.save(_make_task("valid01"))
        tasks = registry.list_all()
        assert len(tasks) == 1
        assert tasks[0].task_id == "valid01"

    def test_roundtrip_main_session_type(self, registry: TaskRegistry) -> None:
        task = _make_task("sess001", task_type="main_session")
        registry.save(task)
        loaded = registry.get("sess001")
        assert loaded is not None
        assert loaded.task_type == "main_session"

    def test_task_id_from_dict_defaults(self) -> None:
        """from_dict should tolerate missing optional fields."""
        data = {
            "task_id": "x1",
            "label": "L",
            "description": "D",
            "origin_channel": "slack",
            "origin_chat_id": "chan",
            "started_at": TaskRegistry.now_iso(),
        }
        task = PersistedTask.from_dict(data)
        assert task.task_type == "subagent"
        assert task.session_key is None
        assert task.depth == 0


# ---------------------------------------------------------------------------
# PersistedTask.elapsed_description
# ---------------------------------------------------------------------------


class TestElapsedDescription:
    def test_seconds(self) -> None:
        from datetime import datetime, timezone, timedelta
        t = (datetime.now(tz=timezone.utc) - timedelta(seconds=30)).isoformat()
        task = _make_task()
        task = PersistedTask(**{**task.__dict__, "started_at": t})
        desc = task.elapsed_description()
        assert "s ago" in desc

    def test_minutes(self) -> None:
        from datetime import datetime, timezone, timedelta
        t = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).isoformat()
        task = PersistedTask(**{**_make_task().__dict__, "started_at": t})
        desc = task.elapsed_description()
        assert "m ago" in desc

    def test_invalid_date_returns_raw(self) -> None:
        task = PersistedTask(**{**_make_task().__dict__, "started_at": "not-a-date"})
        desc = task.elapsed_description()
        assert desc == "not-a-date"


# ---------------------------------------------------------------------------
# SessionManager.list_all_sessions
# ---------------------------------------------------------------------------


class TestListAllSessions:
    def test_returns_sessions_with_metadata(self, tmp_path: Path) -> None:
        from summerclaw.session.manager import Session, SessionManager

        mgr = SessionManager(tmp_path)
        s1 = Session(key="telegram:111")
        s1.metadata["pending_user_turn"] = True
        s1.add_message("user", "hello")
        mgr.save(s1)

        s2 = Session(key="telegram:222")
        s2.add_message("user", "world")
        mgr.save(s2)

        all_sessions = mgr.list_all_sessions()
        keys = {s.key for s in all_sessions}
        assert "telegram:111" in keys
        assert "telegram:222" in keys

    def test_filters_pending_user_turn(self, tmp_path: Path) -> None:
        from summerclaw.session.manager import Session, SessionManager

        mgr = SessionManager(tmp_path)
        s = Session(key="slack:abc")
        s.metadata["pending_user_turn"] = True
        s.add_message("user", "pending request")
        mgr.save(s)

        pending = [
            sess for sess in mgr.list_all_sessions()
            if sess.metadata.get("pending_user_turn")
        ]
        assert len(pending) == 1
        assert pending[0].key == "slack:abc"

    def test_empty_directory(self, tmp_path: Path) -> None:
        from summerclaw.session.manager import SessionManager

        mgr = SessionManager(tmp_path)
        assert mgr.list_all_sessions() == []


# ---------------------------------------------------------------------------
# cmd_tasks (integration smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cmd_tasks_no_tasks(tmp_path: Path) -> None:
    """cmd_tasks should return gracefully when nothing is running."""
    from summerclaw.agent.task_persistence import TaskRegistry
    from summerclaw.bus.events import InboundMessage
    from summerclaw.command.builtin import cmd_tasks
    from summerclaw.command.router import CommandContext
    from summerclaw.session.manager import SessionManager

    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="chat1", content="/tasks",
    )
    # Minimal loop mock
    loop = MagicMock()
    loop.subagents.get_all_statuses.return_value = []
    loop._task_registry = TaskRegistry(tmp_path)
    loop.sessions = SessionManager(tmp_path)
    loop._INTERRUPTED_TASK_ID_KEY = "interrupted_task_id"
    loop._PENDING_USER_TURN_KEY = "pending_user_turn"

    ctx = CommandContext(msg=msg, session=None, key="telegram:chat1", raw="/tasks", loop=loop)
    result = await cmd_tasks(ctx)
    assert "No subagents currently running" in result.content


@pytest.mark.asyncio
async def test_cmd_tasks_resume_missing_id(tmp_path: Path) -> None:
    """Resuming an unknown ID should return a 'not found' message."""
    from summerclaw.agent.task_persistence import TaskRegistry
    from summerclaw.bus.events import InboundMessage
    from summerclaw.command.builtin import cmd_tasks_resume
    from summerclaw.command.router import CommandContext
    from summerclaw.session.manager import SessionManager

    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="chat1", content="/tasks resume xyz",
    )
    loop = MagicMock()
    loop._task_registry = TaskRegistry(tmp_path)
    loop._resume_session_task = AsyncMock(return_value="Task not found or already completed.")

    ctx = CommandContext(
        msg=msg, session=None, key="telegram:chat1",
        raw="/tasks resume xyz", loop=loop, args="xyz",
    )
    result = await cmd_tasks_resume(ctx)
    assert "not found" in result.content.lower() or "Task not found" in result.content


@pytest.mark.asyncio
async def test_cmd_tasks_discard_subagent(tmp_path: Path) -> None:
    """Discarding a known subagent task should remove it from the registry."""
    from summerclaw.agent.task_persistence import TaskRegistry
    from summerclaw.bus.events import InboundMessage
    from summerclaw.command.builtin import cmd_tasks_discard
    from summerclaw.command.router import CommandContext

    registry = TaskRegistry(tmp_path)
    task = _make_task("zzz99999")
    registry.save(task)
    assert registry.get("zzz99999") is not None

    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="chat1",
        content="/tasks discard zzz99999",
    )
    loop = MagicMock()
    loop._task_registry = registry

    ctx = CommandContext(
        msg=msg, session=None, key="telegram:chat1",
        raw="/tasks discard zzz99999", loop=loop, args="zzz99999",
    )
    result = await cmd_tasks_discard(ctx)
    assert "discarded" in result.content.lower()
    assert registry.get("zzz99999") is None
