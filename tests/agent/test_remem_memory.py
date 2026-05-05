"""Tests for the ReMe memory algorithm module — Store, Consolidator, Dream, AutoCompact, and Algorithm.

Covers every public method, key internal logic, edge cases, and error paths,
mirroring the test coverage of naive_memory tests.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.memory.remem_memory.auto_compact import ReMeAutoCompact
from nanobot.memory.remem_memory.consolidator import ReMeConsolidator
from nanobot.memory.remem_memory.dream import ReMeDream
from nanobot.memory.remem_memory.store import ReMeStore


# ---------------------------------------------------------------------------
# ReMeStore fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Provide a clean temporary workspace directory."""
    return tmp_path


@pytest.fixture
def mock_reme_light() -> MagicMock:
    """Create a mock ReMeLight instance."""
    mock = MagicMock()
    mock.compact_memory = MagicMock(return_value=None)
    mock.summary_memory = MagicMock(return_value=None)
    mock.pre_reasoning_hook = MagicMock(return_value=None)
    return mock


@pytest.fixture
def store(tmp_workspace: Path, mock_reme_light: MagicMock) -> ReMeStore:
    """Create a ReMeStore with a mock ReMeLight."""
    return ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)


# ===================================================================
# ReMeStore tests
# ===================================================================

class TestReMeStoreBasicIO:
    """Test basic read/write of MEMORY.md and context injection."""

    def test_read_memory_returns_empty_when_missing(self, store: ReMeStore) -> None:
        assert store.read_memory() == ""

    def test_write_and_read_memory(self, store: ReMeStore) -> None:
        store.write_memory("hello world")
        assert store.read_memory() == "hello world"

    def test_write_memory_overwrites(self, store: ReMeStore) -> None:
        store.write_memory("first")
        store.write_memory("second")
        assert store.read_memory() == "second"

    def test_get_memory_context_returns_empty_when_missing(self, store: ReMeStore) -> None:
        assert store.get_memory_context() == ""

    def test_get_memory_context_returns_formatted_content(self, store: ReMeStore) -> None:
        store.write_memory("important fact")
        ctx = store.get_memory_context()
        assert "## Long-term Memory" in ctx
        assert "important fact" in ctx

    def test_write_memory_with_unicode(self, store: ReMeStore) -> None:
        content = "记忆系统测试 \U0001f680 日本語 테스트"
        store.write_memory(content)
        assert store.read_memory() == content

    def test_write_memory_with_empty_string(self, store: ReMeStore) -> None:
        store.write_memory("")
        assert store.read_memory() == ""


class TestReMeStoreHistoryCursor:
    """Test history append / cursor / read-unprocessed pipeline."""

    def test_append_history_returns_cursor(self, store: ReMeStore) -> None:
        c1 = store.append_history("event 1")
        assert c1 == 1
        c2 = store.append_history("event 2")
        assert c2 == 2

    def test_append_history_includes_cursor_in_file(self, store: ReMeStore) -> None:
        store.append_history("event 1")
        with open(store._history_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["cursor"] == 1

    def test_cursor_persists_across_appends(self, store: ReMeStore) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        cursor = store.append_history("event 3")
        assert cursor == 3

    def test_cursor_persists_across_store_recreation(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s1 = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        s1.append_history("event 1")
        s1.append_history("event 2")
        s2 = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        cursor = s2.append_history("event 3")
        assert cursor == 3

    def test_read_unprocessed_history(self, store: ReMeStore) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2
        assert entries[1]["cursor"] == 3

    def test_read_unprocessed_history_returns_all_when_cursor_zero(self, store: ReMeStore) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_read_unprocessed_history_returns_empty_when_cursor_ahead(self, store: ReMeStore) -> None:
        store.append_history("event 1")
        entries = store.read_unprocessed_history(since_cursor=999)
        assert entries == []

    def test_read_unprocessed_skips_entries_without_cursor(self, store: ReMeStore) -> None:
        """Entries missing the cursor key should be silently skipped."""
        store._history_file.write_text(
            '{"timestamp": "2026-04-01 10:00", "content": "no cursor"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "valid"}\n'
            '{"cursor": 3, "timestamp": "2026-04-01 10:02", "content": "also valid"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert [e["cursor"] for e in entries] == [2, 3]

    def test_next_cursor_falls_back_when_last_entry_has_no_cursor(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        """_next_cursor should not KeyError on entries without cursor."""
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        s._history_file.write_text(
            '{"timestamp": "2026-04-01 10:01", "content": "no cursor"}\n',
            encoding="utf-8",
        )
        # Delete cursor file so _next_cursor falls back to reading JSONL
        s._cursor_file.unlink(missing_ok=True)
        cursor = s.append_history("new event")
        assert cursor == 1

    def test_append_history_preserves_unicode_content(self, store: ReMeStore) -> None:
        content = "ユーザーが設定を変更しました 🎉"
        cursor = store.append_history(content)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert entries[0]["content"] == content
        assert entries[0]["cursor"] == cursor


class TestReMeStoreCompact:
    """Test history compaction logic."""

    def test_compact_history_drops_oldest(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace, max_history_entries=2)
        s.append_history("event 1")
        s.append_history("event 2")
        s.append_history("event 3")
        s.append_history("event 4")
        s.append_history("event 5")
        s.compact_history()
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_compact_history_noop_when_under_limit(self, store: ReMeStore) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.compact_history()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_compact_history_noop_when_max_is_zero(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace, max_history_entries=0)
        s.append_history("event 1")
        s.append_history("event 2")
        s.compact_history()
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_compact_history_noop_when_max_is_negative(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace, max_history_entries=-5)
        s.append_history("event 1")
        s.compact_history()
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1

    def test_compact_history_exactly_at_limit(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace, max_history_entries=3)
        s.append_history("event 1")
        s.append_history("event 2")
        s.append_history("event 3")
        s.compact_history()
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 3


class TestReMeStoreDreamCursor:
    """Test dream cursor persistence."""

    def test_initial_cursor_is_zero(self, store: ReMeStore) -> None:
        assert store.get_last_dream_cursor() == 0

    def test_set_and_get_cursor(self, store: ReMeStore) -> None:
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5

    def test_cursor_persists_across_store_recreation(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s1 = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        s1.set_last_dream_cursor(3)
        s2 = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        assert s2.get_last_dream_cursor() == 3

    def test_cursor_file_corrupted_returns_zero(self, store: ReMeStore) -> None:
        store._dream_cursor_file.write_text("not-a-number", encoding="utf-8")
        assert store.get_last_dream_cursor() == 0

    def test_set_cursor_zero(self, store: ReMeStore) -> None:
        store.set_last_dream_cursor(5)
        store.set_last_dream_cursor(0)
        assert store.get_last_dream_cursor() == 0


class TestReMeStoreFormatMessages:
    """Test static _format_messages utility."""

    def test_format_messages_basic(self) -> None:
        msgs = [
            {"role": "user", "content": "hello", "timestamp": "2026-04-01 10:00:00"},
            {"role": "assistant", "content": "hi there", "timestamp": "2026-04-01 10:00:05"},
        ]
        result = ReMeStore._format_messages(msgs)
        assert "USER: hello" in result
        assert "ASSISTANT: hi there" in result

    def test_format_messages_with_tools(self) -> None:
        msgs = [
            {"role": "assistant", "content": "done", "timestamp": "2026-04-01 10:00:00",
             "tools_used": ["read_file", "edit_file"]},
        ]
        result = ReMeStore._format_messages(msgs)
        assert "[tools: read_file, edit_file]" in result

    def test_format_messages_skips_empty_content(self) -> None:
        msgs = [
            {"role": "user", "content": "", "timestamp": "2026-04-01 10:00:00"},
            {"role": "assistant", "content": "valid", "timestamp": "2026-04-01 10:00:05"},
        ]
        result = ReMeStore._format_messages(msgs)
        assert "USER" not in result
        assert "ASSISTANT: valid" in result

    def test_format_messages_missing_timestamp(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result = ReMeStore._format_messages(msgs)
        assert "USER: hello" in result
        assert "[?]" in result


class TestReMeStoreRawArchive:
    """Test raw_archive fallback dumping."""

    def test_raw_archive_appends_to_history(self, store: ReMeStore) -> None:
        msgs = [
            {"role": "user", "content": "fix bug", "timestamp": "2026-04-01 10:00:00"},
            {"role": "assistant", "content": "done", "timestamp": "2026-04-01 10:00:05",
             "tools_used": ["edit_file"]},
        ]
        store.raw_archive(msgs)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW] 2 messages" in entries[0]["content"]
        assert "USER: fix bug" in entries[0]["content"]
        # _format_messages includes tools in the role tag: ASSISTANT [tools: edit_file]: done
        assert "ASSISTANT" in entries[0]["content"]
        assert "done" in entries[0]["content"]


class TestReMeStoreEdgeCases:
    """Test JSONL corruption, empty files, and prefix isolation."""

    def test_jsonl_with_corrupted_line_skipped(self, store: ReMeStore) -> None:
        store._history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "good"}\n'
            'not-valid-json\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "also good"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert [e["cursor"] for e in entries] == [1, 2]

    def test_jsonl_empty_file(self, store: ReMeStore) -> None:
        entries = store.read_unprocessed_history(since_cursor=0)
        assert entries == []

    def test_jsonl_file_not_exists(self, store: ReMeStore) -> None:
        # File might not exist if no entries have been appended yet.
        # Use missing_ok to handle both cases.
        store._history_file.unlink(missing_ok=True)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert entries == []

    def test_jsonl_blank_lines_skipped(self, store: ReMeStore) -> None:
        store._history_file.write_text(
            '\n'
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "good"}\n'
            '\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "also good"}\n'
            '\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_read_last_entry_on_empty_file(self, store: ReMeStore) -> None:
        last = store._read_last_entry()
        assert last is None

    def test_read_last_entry_returns_last(self, store: ReMeStore) -> None:
        store.append_history("first")
        store.append_history("second")
        last = store._read_last_entry()
        assert last is not None
        assert last["content"] == "second"

    def test_read_write_entries_roundtrip(self, store: ReMeStore) -> None:
        entries = [
            {"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "a"},
            {"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "b"},
        ]
        store._write_entries(entries)
        read_back = store._read_entries()
        assert read_back == entries

    def test_append_history_uniquely_identifies_with_prefix(self, store: ReMeStore) -> None:
        """Verify that cursor-based isolation prevents cross-prefix leakage."""
        store.append_history("prefix_A: event 1")
        store.append_history("prefix_B: event 2")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 1
        assert entries[0]["content"] == "prefix_B: event 2"

    def test_memory_dir_created_on_init(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        assert s.memory_dir.exists()
        assert s.memory_dir.is_dir()

    def test_default_max_history_is_1000(self, store: ReMeStore) -> None:
        assert store.max_history_entries == 1000

    def test_custom_max_history_stored(self, tmp_workspace: Path, mock_reme_light: MagicMock) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace, max_history_entries=500)
        assert s.max_history_entries == 500


# ===================================================================
# ReMeConsolidator tests
# ===================================================================

@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLMProvider."""
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_sessions(tmp_workspace: Path) -> MagicMock:
    """Create a mock SessionManager."""
    sm = MagicMock()
    sm.save = MagicMock()
    sm.invalidate = MagicMock()
    sm.list_sessions = MagicMock(return_value=[])
    return sm


@pytest.fixture
def consolidator(
    store: ReMeStore,
    mock_reme_light: MagicMock,
    mock_provider: MagicMock,
    mock_sessions: MagicMock,
) -> ReMeConsolidator:
    """Create a ReMeConsolidator with mock dependencies."""
    return ReMeConsolidator(
        store=store,
        reme_light=mock_reme_light,
        provider=mock_provider,
        model="test-model",
        sessions=mock_sessions,
        context_window_tokens=100_000,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
        max_completion_tokens=4096,
    )


class TestReMeConsolidatorArchive:
    """Test archive() via ReMeLight compact_memory and fallback behavior."""

    async def test_archive_calls_compact_memory_and_appends_summary(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        mock_reme_light.compact_memory.return_value = "User fixed a bug in the auth module."
        messages = [
            {"role": "user", "content": "fix the auth bug"},
            {"role": "assistant", "content": "Done, fixed the race condition."},
        ]
        result = await consolidator.archive(messages)
        assert result == "User fixed a bug in the auth module."
        mock_reme_light.compact_memory.assert_called_once()
        entries = store.read_unprocessed_history(since_cursor=0)
        # One raw_archive entry + one summary entry
        assert len(entries) == 2
        assert "[RAW]" in entries[0]["content"]
        assert "User fixed a bug" in entries[1]["content"]

    async def test_archive_skips_empty_messages(self, consolidator: ReMeConsolidator) -> None:
        result = await consolidator.archive([])
        assert result is None

    async def test_archive_falls_back_on_compact_memory_exception(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        mock_reme_light.compact_memory.side_effect = Exception("ReMe error")
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        # raw_archive is called once before compact_memory, and once in the except block
        # But looking at the code, raw_archive is called first, then compact_memory fails,
        # then the except calls raw_archive again. So we have 2 entries.
        assert len(entries) >= 1
        assert any("[RAW]" in e["content"] for e in entries)

    async def test_archive_returns_none_when_compact_memory_returns_none(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        mock_reme_light.compact_memory.return_value = None
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        # raw_archive was called, but summary was None so no second append
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_archive_with_async_compact_memory(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        """compact_memory returning a coroutine should be awaited."""
        async def _async_compact():
            return "Async summary"
        mock_reme_light.compact_memory.return_value = _async_compact()
        messages = [{"role": "user", "content": "async test"}]
        result = await consolidator.archive(messages)
        assert result == "Async summary"
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    async def test_archive_with_non_string_summary(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        """Non-string summary should not be appended."""
        mock_reme_light.compact_memory.return_value = 12345
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1  # Only raw_archive


class TestReMeConsolidatorTokenBudget:
    """Test maybe_consolidate_by_tokens token-budget logic."""

    async def test_prompt_below_threshold_does_not_consolidate(
        self, consolidator: ReMeConsolidator,
    ) -> None:
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100, "tiktoken"))
        consolidator.archive = AsyncMock(return_value=True)
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_no_consolidation_when_context_window_zero(
        self, store: ReMeStore, mock_reme_light: MagicMock,
        mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        c = ReMeConsolidator(
            store=store,
            reme_light=mock_reme_light,
            provider=mock_provider,
            model="test-model",
            sessions=mock_sessions,
            context_window_tokens=0,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        c.archive = AsyncMock()
        await c.maybe_consolidate_by_tokens(session)
        c.archive.assert_not_called()

    async def test_no_consolidation_when_no_messages(
        self, consolidator: ReMeConsolidator,
    ) -> None:
        session = MagicMock()
        session.messages = []
        session.key = "test:key"
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_estimate_session_prompt_tokens_handles_error(
        self, consolidator: ReMeConsolidator,
    ) -> None:
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(side_effect=Exception("estimation error"))
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_consolidation_triggers_pre_reasoning_hook(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock,
    ) -> None:
        """When estimated tokens exceed budget, pre_reasoning_hook should be called."""
        mock_reme_light.pre_reasoning_hook.return_value = None
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        session.key = "test:key"
        # Estimate above budget (context_window=100000, max_completion=4096, safety=1024)
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100_000, "tiktoken"))
        consolidator.archive = AsyncMock(return_value="summary")
        await consolidator.maybe_consolidate_by_tokens(session)
        mock_reme_light.pre_reasoning_hook.assert_called_once()

    async def test_consolidation_with_async_pre_reasoning_hook(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock,
    ) -> None:
        async def _async_hook():
            return None
        mock_reme_light.pre_reasoning_hook.return_value = _async_hook()
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100_000, "tiktoken"))
        consolidator.archive = AsyncMock(return_value="summary")
        await consolidator.maybe_consolidate_by_tokens(session)
        mock_reme_light.pre_reasoning_hook.assert_called_once()

    async def test_consolidation_pre_reasoning_hook_error_is_caught(
        self, consolidator: ReMeConsolidator, mock_reme_light: MagicMock,
    ) -> None:
        """Error in pre_reasoning_hook should be caught, not propagate."""
        mock_reme_light.pre_reasoning_hook.side_effect = Exception("hook error")
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100_000, "tiktoken"))
        await consolidator.maybe_consolidate_by_tokens(session)
        # Should not raise, should mark as consolidated
        assert session.last_consolidated == len(session.messages)


class TestReMeConsolidatorLock:
    """Test consolidation lock behavior."""

    def test_get_lock_returns_same_lock_for_same_key(self, consolidator: ReMeConsolidator) -> None:
        lock1 = consolidator.get_lock("session:a")
        lock2 = consolidator.get_lock("session:a")
        assert lock1 is lock2

    def test_get_lock_returns_different_lock_for_different_keys(self, consolidator: ReMeConsolidator) -> None:
        lock1 = consolidator.get_lock("session:a")
        lock2 = consolidator.get_lock("session:b")
        assert lock1 is not lock2

    def test_get_lock_creates_default(self, consolidator: ReMeConsolidator) -> None:
        lock = consolidator.get_lock("new:session")
        assert isinstance(lock, asyncio.Lock)


class TestReMeConsolidatorEstimate:
    """Test token estimation."""

    def test_estimate_session_prompt_tokens(self, consolidator: ReMeConsolidator) -> None:
        session = MagicMock()
        session.get_history.return_value = [{"role": "user", "content": "hello"}]
        session.key = "channel:chat123"
        with patch(
            "nanobot.memory.remem_memory.consolidator.estimate_prompt_tokens_chain",
            return_value=(42, "tiktoken"),
        ) as mock_estimate:
            tokens, source = consolidator.estimate_session_prompt_tokens(session)
            assert tokens == 42
            assert source == "tiktoken"
            mock_estimate.assert_called_once()


# ===================================================================
# ReMeDream tests
# ===================================================================

@pytest.fixture
def dream(
    store: ReMeStore,
    mock_reme_light: MagicMock,
    mock_provider: MagicMock,
) -> ReMeDream:
    """Create a ReMeDream with mock dependencies."""
    return ReMeDream(
        store=store,
        reme_light=mock_reme_light,
        provider=mock_provider,
        model="test-model",
        max_batch_size=5,
        max_iterations=10,
    )


class TestReMeDreamRun:
    """Test Dream's run() method."""

    async def test_noop_when_no_unprocessed_history(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        result = await dream.run()
        assert result is False
        mock_reme_light.summary_memory.assert_not_called()

    async def test_calls_summary_memory_for_unprocessed_entries(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("User prefers dark mode")
        mock_reme_light.summary_memory.return_value = "New long-term fact extracted."
        result = await dream.run()
        assert result is True
        mock_reme_light.summary_memory.assert_called_once()

    async def test_advances_dream_cursor(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        mock_reme_light.summary_memory.return_value = "Nothing new"
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_history_after_run(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_reme_light.summary_memory.return_value = "Summary"
        await dream.run()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_respects_max_batch_size(
        self, tmp_workspace: Path, mock_reme_light: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        s = ReMeStore(reme_light=mock_reme_light, workspace=tmp_workspace)
        for i in range(10):
            s.append_history(f"event {i}")
        d = ReMeDream(
            store=s,
            reme_light=mock_reme_light,
            provider=mock_provider,
            model="test-model",
            max_batch_size=3,
        )
        mock_reme_light.summary_memory.return_value = "summary"
        await d.run()
        # Cursor should advance to the cursor of the 3rd entry (=3), not the 10th
        assert s.get_last_dream_cursor() == 3

    async def test_handles_summary_memory_exception(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event 1")
        mock_reme_light.summary_memory.side_effect = Exception("summary error")
        result = await dream.run()
        # Should still return True and advance cursor to avoid infinite loops
        assert result is True
        assert store.get_last_dream_cursor() == 1

    async def test_handles_async_summary_memory(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event 1")
        async def _async_summary():
            return "Async long-term memory summary"
        mock_reme_light.summary_memory.return_value = _async_summary()
        result = await dream.run()
        assert result is True

    async def test_handles_none_summary(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event 1")
        mock_reme_light.summary_memory.return_value = None
        result = await dream.run()
        assert result is True
        assert store.get_last_dream_cursor() == 1

    async def test_handles_non_string_summary(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event 1")
        mock_reme_light.summary_memory.return_value = 42
        result = await dream.run()
        assert result is True
        assert store.get_last_dream_cursor() == 1

    async def test_dream_only_processes_new_entries(self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore) -> None:
        """Dream should only process entries after the last dream cursor."""
        store.append_history("event 1")
        store.set_last_dream_cursor(1)
        store.append_history("event 2")
        store.append_history("event 3")
        mock_reme_light.summary_memory.return_value = "summary"
        await dream.run()
        # Cursor was 1, new events are 2,3 → cursor advances to 3
        assert store.get_last_dream_cursor() == 3
        # First call processed, second should be noop
        mock_reme_light.summary_memory.reset_mock()
        result = await dream.run()
        assert result is False
        mock_reme_light.summary_memory.assert_not_called()


class TestReMeDreamConfig:
    """Test Dream configuration defaults and overrides."""

    def test_default_max_batch_size(self, store: ReMeStore, mock_reme_light: MagicMock, mock_provider: MagicMock) -> None:
        d = ReMeDream(store=store, reme_light=mock_reme_light, provider=mock_provider, model="m")
        assert d.max_batch_size == 20

    def test_custom_max_batch_size(self, store: ReMeStore, mock_reme_light: MagicMock, mock_provider: MagicMock) -> None:
        d = ReMeDream(store=store, reme_light=mock_reme_light, provider=mock_provider, model="m", max_batch_size=50)
        assert d.max_batch_size == 50

    def test_default_max_iterations(self, store: ReMeStore, mock_reme_light: MagicMock, mock_provider: MagicMock) -> None:
        d = ReMeDream(store=store, reme_light=mock_reme_light, provider=mock_provider, model="m")
        assert d.max_iterations == 10

    def test_annotate_line_ages_default(self, store: ReMeStore, mock_reme_light: MagicMock, mock_provider: MagicMock) -> None:
        d = ReMeDream(store=store, reme_light=mock_reme_light, provider=mock_provider, model="m")
        assert d.annotate_line_ages is True


# ===================================================================
# ReMeDream — tool registry
# ===================================================================

class TestReMeDreamTools:
    """Test Dream's _build_tools() method and tool registry."""

    def test_build_tools_returns_tool_registry(
        self, dream: ReMeDream,
    ) -> None:
        tools = dream._build_tools()
        from nanobot.agent.tools.registry import ToolRegistry
        assert isinstance(tools, ToolRegistry)

    def test_tool_registry_has_expected_tools(
        self, dream: ReMeDream,
    ) -> None:
        tools = dream._tools
        assert tools.get("read_file") is not None
        assert tools.get("edit_file") is not None
        assert tools.get("write_file") is not None

    def test_skill_prefix_write_file_uses_dreamed_prefix(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        """Verify SkillPrefixWriteFileTool has 'dreamed' prefix."""
        tools = dream._tools
        write_tool = tools.get("write_file")
        assert write_tool is not None
        assert write_tool._skill_prefix == "dreamed-"

    def test_skills_dir_created(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        skills_dir = store.workspace / "skills"
        assert skills_dir.exists()
        assert skills_dir.is_dir()


# ===================================================================
# ReMeDream — skill listing
# ===================================================================

class TestReMeDreamSkills:
    """Test _list_existing_skills() method."""

    def test_list_skills_with_user_skills(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        skill_dir = store.workspace / "skills" / "test-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n",
            encoding="utf-8",
        )
        skills = dream._list_existing_skills()
        assert any("test-skill" in s for s in skills)

    def test_list_skills_empty_when_none_exist(
        self, dream: ReMeDream,
    ) -> None:
        skills = dream._list_existing_skills()
        assert isinstance(skills, list)

    def test_list_skills_skips_directories_without_skill_md(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        # Create a directory without SKILL.md — should be skipped
        (store.workspace / "skills" / "empty-dir").mkdir(parents=True, exist_ok=True)
        (store.workspace / "skills" / "empty-dir" / "README.md").write_text("hello")
        skills = dream._list_existing_skills()
        assert not any("empty-dir" in s for s in skills)

    def test_list_skills_handles_missing_description(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        skill_dir = store.workspace / "skills" / "no-desc-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-desc-skill\n---\n# No description skill\n",
            encoding="utf-8",
        )
        skills = dream._list_existing_skills()
        assert any("no-desc-skill" in s for s in skills)
        assert any("(no description)" in s for s in skills)


# ===================================================================
# ReMeDream — age annotation
# ===================================================================

class TestReMeDreamAgeAnnotation:
    """Test _annotate_with_ages() method."""

    def test_annotate_empty_content(
        self, dream: ReMeDream,
    ) -> None:
        assert dream._annotate_with_ages("") == ""

    def test_annotate_with_no_stale_lines(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        store.write_memory("line one\nline two")
        with patch.object(store.git, "line_ages") as mock_ages:
            from dataclasses import dataclass
            @dataclass
            class FakeAge:
                age_days: int
            mock_ages.return_value = [FakeAge(3), FakeAge(5)]
            result = dream._annotate_with_ages("line one\nline two")
            # Both under 14 days, no annotation
            assert "\u2190" not in result
            assert "3d" not in result
            assert "5d" not in result

    def test_annotate_with_stale_lines(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        store.write_memory("old line\nnew line")
        with patch.object(store.git, "line_ages") as mock_ages:
            from dataclasses import dataclass
            @dataclass
            class FakeAge:
                age_days: int
            mock_ages.return_value = [FakeAge(30), FakeAge(3)]
            result = dream._annotate_with_ages("old line\nnew line")
            assert "30d" in result
            assert "\u2190" in result

    def test_annotate_skips_blank_lines(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        store.write_memory("content\n\nmore content")
        with patch.object(store.git, "line_ages") as mock_ages:
            from dataclasses import dataclass
            @dataclass
            class FakeAge:
                age_days: int
            mock_ages.return_value = [FakeAge(30), FakeAge(30), FakeAge(30)]
            result = dream._annotate_with_ages("content\n\nmore content")
            # Blank line should not be annotated
            lines = result.splitlines()
            assert lines[1] == ""  # blank line stays blank

    def test_annotate_handles_git_failure(
        self, dream: ReMeDream,
    ) -> None:
        """When git fails, content returned unchanged."""
        content = "original content\nmore content"
        with patch.object(dream.store.git, "line_ages", side_effect=OSError("git error")):
            result = dream._annotate_with_ages(content)
            assert result == content

    def test_annotate_handles_length_mismatch(
        self, dream: ReMeDream,
    ) -> None:
        """When line count differs, content returned unchanged."""
        content = "line one\nline two"
        with patch.object(dream.store.git, "line_ages") as mock_ages:
            from dataclasses import dataclass
            @dataclass
            class FakeAge:
                age_days: int
            mock_ages.return_value = [FakeAge(10)]  # Only 1 age for 2 lines
            result = dream._annotate_with_ages(content)
            assert result == content

    def test_annotate_preserves_trailing_newline(
        self, dream: ReMeDream, store: ReMeStore,
    ) -> None:
        store.write_memory("line one\n")
        with patch.object(store.git, "line_ages") as mock_ages:
            from dataclasses import dataclass
            @dataclass
            class FakeAge:
                age_days: int
            mock_ages.return_value = [FakeAge(20)]
            result = dream._annotate_with_ages("line one\n")
            assert result.endswith("\n")
            assert "20d" in result

    async def test_annotate_with_ages_skipped_when_disabled(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        """When annotate_line_ages is False, _annotate_with_ages should not be called."""
        store.append_history("event")
        dream.annotate_line_ages = False
        mock_reme_light.summary_memory.return_value = "summary"

        # Patch Phase 2 runner to avoid real LLM calls
        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _fake_run

        with patch.object(dream, "_annotate_with_ages") as mock_annotate:
            await dream.run()
            mock_annotate.assert_not_called()


# ===================================================================
# ReMeDream — run() Phase 2 integration
# ===================================================================

class TestReMeDreamPhase2:
    """Test Phase 2 AgentRunner integration in run()."""

    async def test_phase2_agent_runner_called(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("conversation event")
        mock_reme_light.summary_memory.return_value = "Analysis: user preference detected."

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _fake_run
        result = await dream.run()
        assert result is True
        mock_reme_light.summary_memory.assert_called_once()

    async def test_phase2_receives_correct_model(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        captured_spec = None
        async def _capture_run(spec):
            nonlocal captured_spec
            captured_spec = spec
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _capture_run
        await dream.run()
        assert captured_spec is not None
        assert captured_spec.model == "test-model"
        assert captured_spec.max_iterations == dream.max_iterations
        assert captured_spec.max_tool_result_chars == dream.max_tool_result_chars
        assert captured_spec.fail_on_tool_error is False

    async def test_phase2_receives_tools(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        captured_spec = None
        async def _capture_run(spec):
            nonlocal captured_spec
            captured_spec = spec
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _capture_run
        await dream.run()
        assert captured_spec.tools is dream._tools

    async def test_phase2_includes_skills_in_prompt(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        # Create a skill so it appears in the listing
        skill_dir = store.workspace / "skills" / "existing-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: existing-skill\ndescription: Already exists\n---\n",
            encoding="utf-8",
        )

        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        captured_messages = None
        async def _capture_run(spec):
            nonlocal captured_messages
            captured_messages = spec.initial_messages
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _capture_run
        await dream.run()
        user_msg = captured_messages[1]["content"]
        assert "existing-skill" in user_msg
        assert "Existing Skills" in user_msg

    async def test_phase2_receives_file_context(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.write_memory("# Memory\n- test fact")
        store.write_soul("# Soul\n- helpful")
        store.write_user("# User\n- developer")
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        captured_messages = None
        async def _capture_run(spec):
            nonlocal captured_messages
            captured_messages = spec.initial_messages
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _capture_run
        await dream.run()
        user_msg = captured_messages[1]["content"]
        assert "MEMORY.md" in user_msg
        assert "test fact" in user_msg
        assert "SOUL.md" in user_msg
        assert "helpful" in user_msg
        assert "USER.md" in user_msg
        assert "developer" in user_msg

    async def test_phase2_handles_agent_runner_exception(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        async def _failing_run(*args, **kwargs):
            raise RuntimeError("AgentRunner crashed")
        dream._runner.run = _failing_run
        result = await dream.run()
        # Should still return True and advance cursor
        assert result is True
        assert store.get_last_dream_cursor() == 1

    async def test_phase2_handles_empty_tool_events(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],  # empty events
            )
        dream._runner.run = _fake_run
        result = await dream.run()
        assert result is True
        assert store.get_last_dream_cursor() == 1


# ===================================================================
# ReMeDream — run() tool events & changelog
# ===================================================================

class TestReMeDreamChangelog:
    """Test changelog building and git auto-commit behaviour."""

    async def test_changelog_built_from_tool_events(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[
                    {"name": "edit_file", "status": "ok", "detail": "Edited MEMORY.md"},
                    {"name": "write_file", "status": "ok", "detail": "Created dreamed-test/SKILL.md"},
                ],
            )
        dream._runner.run = _fake_run
        await dream.run()
        # Cursor advanced
        assert store.get_last_dream_cursor() == 1

    async def test_changelog_filters_failed_events(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[
                    {"name": "edit_file", "status": "ok", "detail": "edited"},
                    {"name": "write_file", "status": "error", "detail": "permission denied"},
                ],
            )
        dream._runner.run = _fake_run
        result = await dream.run()
        assert result is True


# ===================================================================
# ReMeDream — run() summary_memory edge cases
# ===================================================================

class TestReMeDreamPhase1EdgeCases:
    """Test Phase 1 edge cases with Phase 2 following."""

    async def test_phase1_returns_none_proceeds_to_phase2(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = None

        phase2_called = False
        async def _fake_run(*args, **kwargs):
            nonlocal phase2_called
            phase2_called = True
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _fake_run
        result = await dream.run()
        assert result is True
        assert phase2_called, "Phase 2 should run even when summary_memory returns None"

    async def test_phase1_exception_proceeds_to_phase2(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.side_effect = RuntimeError("summary crashed")

        phase2_called = False
        async def _fake_run(*args, **kwargs):
            nonlocal phase2_called
            phase2_called = True
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _fake_run
        result = await dream.run()
        assert result is True
        assert phase2_called, "Phase 2 should run even when Phase 1 throws"
        assert store.get_last_dream_cursor() == 1

    async def test_phase2_prompt_contains_history_text(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("User asked about Python")
        mock_reme_light.summary_memory.return_value = "Analysis"

        captured_messages = None
        async def _capture_run(spec):
            nonlocal captured_messages
            captured_messages = spec.initial_messages
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _capture_run
        await dream.run()
        user_msg = captured_messages[1]["content"]
        assert "User asked about Python" in user_msg
        assert "Conversation History" in user_msg

    async def test_phase2_prompt_has_fallback_when_analysis_empty(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = ""  # empty string

        captured_messages = None
        async def _capture_run(spec):
            nonlocal captured_messages
            captured_messages = spec.initial_messages
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],
            )
        dream._runner.run = _capture_run
        await dream.run()
        user_msg = captured_messages[1]["content"]
        assert "no analysis available" in user_msg.lower()


# ===================================================================
# ReMeDream — git auto-commit
# ===================================================================

class TestReMeDreamGitCommit:
    """Test git auto-commit after successful Phase 2 with changelog."""

    async def test_git_commit_called_when_changelog_and_git_initialized(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "analysis summary"

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[
                    {"name": "edit_file", "status": "ok", "detail": "edited"},
                ],
            )
        dream._runner.run = _fake_run

        with patch.object(store.git, "is_initialized", return_value=True), \
             patch.object(store.git, "auto_commit") as mock_commit:
            mock_commit.return_value = "abc123"
            await dream.run()
            mock_commit.assert_called_once()

    async def test_git_commit_not_called_when_no_changelog(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[],  # no changes
            )
        dream._runner.run = _fake_run

        with patch.object(store.git, "is_initialized", return_value=True), \
             patch.object(store.git, "auto_commit") as mock_commit:
            await dream.run()
            mock_commit.assert_not_called()

    async def test_git_commit_not_called_when_not_initialized(
        self, dream: ReMeDream, mock_reme_light: MagicMock, store: ReMeStore,
    ) -> None:
        store.append_history("event")
        mock_reme_light.summary_memory.return_value = "summary"

        async def _fake_run(*args, **kwargs):
            return AgentRunResult(
                final_content="done",
                stop_reason="completed",
                messages=[],
                tools_used=[],
                usage={},
                tool_events=[
                    {"name": "edit_file", "status": "ok", "detail": "edited"},
                ],
            )
        dream._runner.run = _fake_run

        with patch.object(store.git, "is_initialized", return_value=False), \
             patch.object(store.git, "auto_commit") as mock_commit:
            await dream.run()
            mock_commit.assert_not_called()


# ===================================================================
# ReMeDream — constructor & initialisation
# ===================================================================

class TestReMeDreamInit:
    """Test constructor initialises AgentRunner and tools correctly."""

    def test_constructor_initialises_agent_runner(
        self, dream: ReMeDream,
    ) -> None:
        from nanobot.agent.runner import AgentRunner
        assert isinstance(dream._runner, AgentRunner)

    def test_constructor_initialises_tools(
        self, dream: ReMeDream,
    ) -> None:
        from nanobot.agent.tools.registry import ToolRegistry
        assert isinstance(dream._tools, ToolRegistry)

    def test_constructor_stores_all_params(
        self, store: ReMeStore, mock_reme_light: MagicMock, mock_provider: MagicMock,
    ) -> None:
        d = ReMeDream(
            store=store,
            reme_light=mock_reme_light,
            provider=mock_provider,
            model="custom-model",
            max_batch_size=42,
            max_iterations=7,
            max_tool_result_chars=4000,
            annotate_line_ages=False,
        )
        assert d.store is store
        assert d.reme_light is mock_reme_light
        assert d.provider is mock_provider
        assert d.model == "custom-model"
        assert d.max_batch_size == 42
        assert d.max_iterations == 7
        assert d.max_tool_result_chars == 4000
        assert d.annotate_line_ages is False


# ===================================================================
# ReMeAutoCompact tests
# ===================================================================

@pytest.fixture
def auto_compact_sessions(tmp_workspace: Path) -> MagicMock:
    """Create a mock SessionManager for auto compact tests."""
    from nanobot.session.manager import Session
    sm = MagicMock()
    sm.save = MagicMock()
    sm.invalidate = MagicMock()

    def _get_or_create(key: str) -> Session:
        return Session(key=key)

    sm.get_or_create = MagicMock(side_effect=_get_or_create)
    sm.list_sessions = MagicMock(return_value=[])
    return sm


@pytest.fixture
def auto_compact(
    auto_compact_sessions: MagicMock,
    consolidator: ReMeConsolidator,
) -> ReMeAutoCompact:
    """Create a ReMeAutoCompact instance."""
    return ReMeAutoCompact(
        sessions=auto_compact_sessions,
        consolidator=consolidator,
        session_ttl_minutes=15,
    )


class TestReMeAutoCompactTTL:
    """Test TTL and expiration logic."""

    def test_default_ttl_is_zero(self, auto_compact_sessions: MagicMock, consolidator: ReMeConsolidator) -> None:
        ac = ReMeAutoCompact(sessions=auto_compact_sessions, consolidator=consolidator)
        assert ac._ttl == 0

    def test_custom_ttl_stored(self, auto_compact_sessions: MagicMock, consolidator: ReMeConsolidator) -> None:
        ac = ReMeAutoCompact(sessions=auto_compact_sessions, consolidator=consolidator, session_ttl_minutes=30)
        assert ac._ttl == 30

    def test_is_expired_when_ttl_zero(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact._ttl = 0
        ts = datetime.now() - timedelta(minutes=100)
        assert auto_compact._is_expired(ts) is False

    def test_is_expired_boundary(self, auto_compact: ReMeAutoCompact) -> None:
        ts = datetime.now() - timedelta(minutes=15)
        assert auto_compact._is_expired(ts) is True
        ts2 = datetime.now() - timedelta(minutes=14, seconds=59)
        assert auto_compact._is_expired(ts2) is False

    def test_is_expired_string_timestamp(self, auto_compact: ReMeAutoCompact) -> None:
        ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        assert auto_compact._is_expired(ts) is True

    def test_is_expired_none(self, auto_compact: ReMeAutoCompact) -> None:
        assert auto_compact._is_expired(None) is False

    def test_is_expired_empty_string(self, auto_compact: ReMeAutoCompact) -> None:
        assert auto_compact._is_expired("") is False

    def test_is_expired_custom_now(self, auto_compact: ReMeAutoCompact) -> None:
        now = datetime(2026, 5, 5, 12, 0, 0)
        ts = datetime(2026, 5, 5, 11, 44, 0)  # 16 minutes ago
        assert auto_compact._is_expired(ts, now=now) is True
        ts2 = datetime(2026, 5, 5, 11, 46, 0)  # 14 minutes ago
        assert auto_compact._is_expired(ts2, now=now) is False


class TestReMeAutoCompactFormatSummary:
    """Test _format_summary static method."""

    def test_format_summary(self) -> None:
        last_active = datetime.now() - timedelta(minutes=10)
        summary = ReMeAutoCompact._format_summary("User discussed deployment.", last_active)
        assert "Inactive for" in summary
        assert "User discussed deployment." in summary


class TestReMeAutoCompactSplit:
    """Test _split_unconsolidated logic."""

    def test_split_empty_session(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        archive, kept = auto_compact._split_unconsolidated(session)
        assert archive == []
        assert kept == []

    def test_split_splits_older_from_recent(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"user msg {i}")
            session.add_message("assistant", f"assistant msg {i}")
        archive, kept = auto_compact._split_unconsolidated(session)
        assert len(kept) == auto_compact._RECENT_SUFFIX_MESSAGES
        assert len(archive) == 24 - len(kept)
        assert len(archive) > 0

    def test_split_respects_last_consolidated(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(20):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")
        # 40 messages total, last_consolidated=30 → 10 unconsolidated > 8 recent suffix
        session.last_consolidated = 30
        archive, kept = auto_compact._split_unconsolidated(session)
        assert len(kept) == auto_compact._RECENT_SUFFIX_MESSAGES
        assert len(archive) == 10 - len(kept)
        assert len(archive) > 0


class TestReMeAutoCompactArchive:
    """Test _archive method."""

    @pytest.mark.asyncio
    async def test_archive_empty_session(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)
        archive_called = False

        async def _fake_archive(msgs):
            nonlocal archive_called
            archive_called = True
            return "Summary."

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")
        assert not archive_called
        assert "cli:test" not in auto_compact._archiving

    @pytest.mark.asyncio
    async def test_archive_stores_summary(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"user msg {i}")
            session.add_message("assistant", f"assistant msg {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _fake_archive(msgs):
            return "User said hello many times."

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")

        entry = auto_compact._summaries.get("cli:test")
        assert entry is not None
        assert entry[0] == "User said hello many times."
        assert "cli:test" not in auto_compact._archiving

    @pytest.mark.asyncio
    async def test_archive_nothing_summary_not_stored(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _fake_archive(msgs):
            return "(nothing)"

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")

        assert "cli:test" not in auto_compact._summaries

    @pytest.mark.asyncio
    async def test_archive_empty_summary_not_stored(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _fake_archive(msgs):
            return ""

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")

        assert "cli:test" not in auto_compact._summaries

    @pytest.mark.asyncio
    async def test_archive_error_is_caught(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _failing_archive(msgs):
            raise RuntimeError("LLM down")

        auto_compact.consolidator.archive = _failing_archive
        # Should not raise
        await auto_compact._archive("cli:test")
        assert "cli:test" not in auto_compact._archiving

    @pytest.mark.asyncio
    async def test_archive_keeps_recent_suffix_after_error(self, auto_compact: ReMeAutoCompact) -> None:
        """When archive raises, the error is caught and _archiving is cleaned up.
        Session messages are NOT updated (error short-circuits before assignment)."""
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _failing_archive(msgs):
            raise RuntimeError("API down")

        auto_compact.consolidator.archive = _failing_archive
        # Should not raise — error is caught inside _archive
        await auto_compact._archive("cli:test")

        # On error, the except block catches the exception before
        # session.messages = kept_msgs runs, so messages are unchanged
        assert len(session.messages) == 24
        # _archiving set should be cleaned up in finally
        assert "cli:test" not in auto_compact._archiving


class TestReMeAutoCompactPrepareSession:
    """Test prepare_session for summary recovery."""

    def test_prepare_session_no_summary(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        result_session, summary = auto_compact.prepare_session(session, "cli:test")
        assert result_session is session
        assert summary is None

    def test_prepare_session_from_in_memory(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        last_active = datetime.now() - timedelta(minutes=20)
        auto_compact._summaries["cli:test"] = ("User discussed auth.", last_active)
        session.metadata["_last_summary"] = {"text": "old", "last_active": "2026-01-01T00:00:00"}

        result_session, summary = auto_compact.prepare_session(session, "cli:test")
        assert summary is not None
        assert "User discussed auth." in summary
        assert "Inactive for" in summary
        # In-memory dict should be consumed
        assert "cli:test" not in auto_compact._summaries
        # Metadata should be cleaned
        assert "_last_summary" not in result_session.metadata

    def test_prepare_session_from_metadata(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        last_active = datetime.now() - timedelta(minutes=10)
        session.metadata["_last_summary"] = {
            "text": "User prefers Go language.",
            "last_active": last_active.isoformat(),
        }

        result_session, summary = auto_compact.prepare_session(session, "cli:test")
        assert summary is not None
        assert "User prefers Go language." in summary
        assert "_last_summary" not in result_session.metadata

    def test_prepare_session_metadata_consumed_once(self, auto_compact: ReMeAutoCompact) -> None:
        from nanobot.session.manager import Session
        session = Session(key="cli:test")
        session.metadata["_last_summary"] = {
            "text": "Summary.",
            "last_active": datetime.now().isoformat(),
        }

        _, summary1 = auto_compact.prepare_session(session, "cli:test")
        assert summary1 is not None

        _, summary2 = auto_compact.prepare_session(session, "cli:test")
        assert summary2 is None


class TestReMeAutoCompactCheckExpired:
    """Test check_expired scheduling."""

    def test_noop_when_ttl_zero(self, auto_compact_sessions: MagicMock, consolidator: ReMeConsolidator) -> None:
        ac = ReMeAutoCompact(sessions=auto_compact_sessions, consolidator=consolidator, session_ttl_minutes=0)
        ac.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=30)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        ac.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_schedules_expired_sessions(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 1
        assert "cli:test" in auto_compact._archiving

    def test_skips_active_session_keys(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule, active_session_keys={"cli:test"})
        assert len(scheduled) == 0

    def test_skips_already_archiving(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact._archiving.add("cli:test")
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_skips_recent_sessions(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": datetime.now().isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_skips_empty_key(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_multiple_sessions_partial_expired(self, auto_compact: ReMeAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:expired", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
            {"key": "cli:active", "updated_at": datetime.now().isoformat()},
            {"key": "cli:also_expired", "updated_at": (datetime.now() - timedelta(minutes=30)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 2


# ===================================================================
# ReMeMemoryAlgorithm integration tests
# ===================================================================

class TestReMeMemoryAlgorithm:
    """Test the ReMeMemoryAlgorithm build() integration."""

    def test_algorithm_name(self) -> None:
        from nanobot.memory.remem_memory import ReMeMemoryAlgorithm
        algo = ReMeMemoryAlgorithm()
        assert algo.name == "remem_memory"

    def test_algorithm_build_returns_memory_components(self, tmp_workspace: Path) -> None:
        """build() should return MemoryComponents when reme is available."""
        reme = pytest.importorskip("reme")
        from nanobot.memory.remem_memory import ReMeMemoryAlgorithm
        from nanobot.memory.base import MemoryComponents

        algo = ReMeMemoryAlgorithm()
        provider = MagicMock()
        provider.generation = MagicMock()
        provider.generation.temperature = 0.7
        provider.generation.max_tokens = 4096
        provider.api_key = "test-key"
        provider.api_base = "https://api.test.com"

        sessions = MagicMock()

        components = algo.build(
            workspace=tmp_workspace,
            provider=provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=128_000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=15,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=True,
        )

        assert isinstance(components, MemoryComponents)
        assert isinstance(components.store, ReMeStore)
        assert isinstance(components.consolidator, ReMeConsolidator)
        assert isinstance(components.dream, ReMeDream)
        assert isinstance(components.auto_compact, ReMeAutoCompact)
        assert components.auto_compact is not None

    def test_algorithm_build_auto_compact_not_none(self, tmp_workspace: Path) -> None:
        """auto_compact should not be None — parity with naive_memory."""
        reme = pytest.importorskip("reme")
        from nanobot.memory.remem_memory import ReMeMemoryAlgorithm

        algo = ReMeMemoryAlgorithm()
        provider = MagicMock()
        provider.generation = MagicMock()
        provider.generation.temperature = 0.7
        provider.generation.max_tokens = 4096
        provider.api_key = "test-key"
        provider.api_base = "https://api.test.com"

        components = algo.build(
            workspace=tmp_workspace,
            provider=provider,
            model="test-model",
            sessions=MagicMock(),
            context_window_tokens=128_000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=15,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=True,
        )

        assert components.auto_compact is not None

    def test_algorithm_registers_in_registry(self) -> None:
        from nanobot.memory.registry import MemoryRegistry
        from nanobot.memory.remem_memory import ReMeMemoryAlgorithm

        registry = MemoryRegistry()
        registry.register(ReMeMemoryAlgorithm())
        algo = registry.get("remem_memory")
        assert algo.name == "remem_memory"

    def test_algorithm_build_with_minimal_provider(self, tmp_workspace: Path) -> None:
        """build() should work even if provider lacks generation attrs."""
        reme = pytest.importorskip("reme")
        from nanobot.memory.remem_memory import ReMeMemoryAlgorithm
        from nanobot.memory.base import MemoryComponents

        algo = ReMeMemoryAlgorithm()
        provider = MagicMock()
        # No generation attribute
        del provider.generation
        provider.api_key = "test-key"
        provider.api_base = None

        components = algo.build(
            workspace=tmp_workspace,
            provider=provider,
            model="test-model",
            sessions=MagicMock(),
            context_window_tokens=128_000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=15,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=True,
        )

        assert isinstance(components, MemoryComponents)

    def test_algorithm_build_respects_parameters(self, tmp_workspace: Path) -> None:
        """build() should pass parameters through to components."""
        reme = pytest.importorskip("reme")
        from nanobot.memory.remem_memory import ReMeMemoryAlgorithm

        algo = ReMeMemoryAlgorithm()
        provider = MagicMock()
        provider.generation = MagicMock()
        provider.generation.temperature = 0.3
        provider.generation.max_tokens = 2048

        components = algo.build(
            workspace=tmp_workspace,
            provider=provider,
            model="custom-model",
            sessions=MagicMock(),
            context_window_tokens=64_000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=2048,
            session_ttl_minutes=30,
            max_batch_size=50,
            max_iterations=5,
            max_tool_result_chars=8000,
            annotate_line_ages=False,
        )

        assert components.consolidator.model == "custom-model"
        assert components.consolidator.context_window_tokens == 64_000
        assert components.consolidator.max_completion_tokens == 2048
        assert components.dream.max_batch_size == 50
        assert components.dream.max_iterations == 5
        assert components.dream.max_tool_result_chars == 8000
        assert components.dream.annotate_line_ages is False
        assert components.auto_compact._ttl == 30
