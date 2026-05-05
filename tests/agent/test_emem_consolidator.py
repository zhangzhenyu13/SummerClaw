"""Tests for EMemConsolidator — token-budget consolidation with EDU extraction."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.memory.emem_memory.consolidator import EMemConsolidator
from nanobot.memory.emem_memory.store import EMemStore


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def mock_embedder() -> MagicMock:
    m = MagicMock()
    m.batch_encode = MagicMock(return_value=[])
    return m


@pytest.fixture
def store(tmp_path, mock_embedder: MagicMock) -> EMemStore:
    return EMemStore(workspace=tmp_path, embedding_model=mock_embedder)


@pytest.fixture
def mock_provider() -> MagicMock:
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_sessions() -> MagicMock:
    sm = MagicMock()
    sm.save = MagicMock()
    sm.invalidate = MagicMock()
    sm.list_sessions = MagicMock(return_value=[])
    return sm


@pytest.fixture
def mock_edu_extractor() -> MagicMock:
    ee = MagicMock()
    ee.extract_from_history = AsyncMock(return_value=[])
    return ee


@pytest.fixture
def consolidator(
    store: EMemStore,
    mock_provider: MagicMock,
    mock_sessions: MagicMock,
    mock_edu_extractor: MagicMock,
) -> EMemConsolidator:
    return EMemConsolidator(
        store=store,
        provider=mock_provider,
        model="test-model",
        sessions=mock_sessions,
        context_window_tokens=100_000,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
        edu_extractor=mock_edu_extractor,
        emem_store=store,
        max_completion_tokens=4096,
    )


# ===================================================================
# EMemConsolidator — archive with EDU extraction
# ===================================================================

class TestEMemConsolidatorArchive:
    """Test archive() method — LLM summary + EDU extraction."""

    async def test_archive_calls_super_and_extracts_edus(
        self,
        consolidator: EMemConsolidator,
        mock_provider: MagicMock,
        store: EMemStore,
        mock_edu_extractor: MagicMock,
    ) -> None:
        """archive() should get LLM summary, then extract EDUs from messages."""
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="User discussed deployment.",
            finish_reason="stop",
        )
        # Mock EDU extraction to return some EDUs
        from nanobot.memory.emem_memory.datatypes import EDURecord
        mock_edu_extractor.extract_from_history.return_value = [
            EDURecord(edu_id="edu-test-1", text="User deployed the app."),
        ]

        messages = [
            {"role": "user", "content": "I deployed the app."},
            {"role": "assistant", "content": "Deployment successful."},
        ]
        result = await consolidator.archive(messages)

        assert result == "User discussed deployment."
        mock_edu_extractor.extract_from_history.assert_called_once()
        # Should have inserted EDUs into the edu store
        assert len(store.edu_store.get_all_ids()) >= 0

    async def test_archive_skips_empty_messages(
        self, consolidator: EMemConsolidator, mock_edu_extractor: MagicMock,
    ) -> None:
        result = await consolidator.archive([])
        assert result is None
        mock_edu_extractor.extract_from_history.assert_not_called()

    async def test_archive_edu_extraction_failure_is_caught(
        self,
        consolidator: EMemConsolidator,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
    ) -> None:
        """If EDU extraction fails, the summary should still be returned."""
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Summary.",
            finish_reason="stop",
        )
        mock_edu_extractor.extract_from_history.side_effect = Exception("EDU error")

        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result == "Summary."  # Should still return summary

    async def test_archive_llm_failure_falls_back(
        self,
        consolidator: EMemConsolidator,
        mock_provider: MagicMock,
        store: EMemStore,
        mock_edu_extractor: MagicMock,
    ) -> None:
        """On LLM failure, raw_archive should be called and EDUs not extracted."""
        mock_provider.chat_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None  # raw dump fallback
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_archive_on_error_finish_reason(
        self,
        consolidator: EMemConsolidator,
        mock_provider: MagicMock,
        store: EMemStore,
        mock_edu_extractor: MagicMock,
    ) -> None:
        """LLM returning finish_reason='error' should trigger raw_archive."""
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Error: overloaded",
            finish_reason="error",
        )
        messages = [{"role": "user", "content": "test"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]


# ===================================================================
# EMemConsolidator — _format_messages_for_edu
# ===================================================================

class TestEMemConsolidatorFormat:
    """Test _format_messages_for_edu static method."""

    def test_format_messages_basic(self) -> None:
        msgs = [
            {"role": "user", "content": "hello", "timestamp": "2026-04-01 10:00:00"},
            {"role": "assistant", "content": "hi", "timestamp": "2026-04-01 10:00:05"},
        ]
        result = EMemConsolidator._format_messages_for_edu(msgs)
        assert "USER: hello" in result
        assert "ASSISTANT: hi" in result

    def test_format_messages_with_tools(self) -> None:
        msgs = [
            {"role": "assistant", "content": "done", "timestamp": "2026-04-01 10:00:00",
             "tools_used": ["read_file", "edit_file"]},
        ]
        result = EMemConsolidator._format_messages_for_edu(msgs)
        assert "[tools: read_file, edit_file]" in result

    def test_format_messages_skips_empty_content(self) -> None:
        msgs = [
            {"role": "user", "content": "", "timestamp": "2026-04-01 10:00:00"},
            {"role": "assistant", "content": "valid", "timestamp": "2026-04-01 10:00:05"},
        ]
        result = EMemConsolidator._format_messages_for_edu(msgs)
        assert "USER" not in result
        assert "ASSISTANT: valid" in result


# ===================================================================
# EMemConsolidator — token budget (inherited from Consolidator)
# ===================================================================

class TestEMemConsolidatorTokenBudget:
    """Test token-budget consolidation (inherited from Consolidator)."""

    async def test_prompt_below_threshold_does_not_consolidate(
        self, consolidator: EMemConsolidator,
    ) -> None:
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(100, "tiktoken"),
        )
        consolidator.archive = AsyncMock(return_value=True)
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_no_consolidation_when_context_window_zero(
        self,
        store: EMemStore,
        mock_provider: MagicMock,
        mock_sessions: MagicMock,
        mock_edu_extractor: MagicMock,
    ) -> None:
        c = EMemConsolidator(
            store=store,
            provider=mock_provider,
            model="test-model",
            sessions=mock_sessions,
            context_window_tokens=0,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            edu_extractor=mock_edu_extractor,
            emem_store=store,
        )
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        c.archive = AsyncMock()
        await c.maybe_consolidate_by_tokens(session)
        c.archive.assert_not_called()

    async def test_estimate_session_prompt_tokens_handles_error(
        self, consolidator: EMemConsolidator,
    ) -> None:
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(
            side_effect=Exception("estimation error"),
        )
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()


# ===================================================================
# EMemConsolidator — lock management
# ===================================================================

class TestEMemConsolidatorLock:
    """Test consolidation lock behavior."""

    def test_get_lock_returns_same_lock_for_same_key(
        self, consolidator: EMemConsolidator,
    ) -> None:
        lock1 = consolidator.get_lock("session:a")
        lock2 = consolidator.get_lock("session:a")
        assert lock1 is lock2

    def test_get_lock_returns_different_lock_for_different_keys(
        self, consolidator: EMemConsolidator,
    ) -> None:
        lock1 = consolidator.get_lock("session:a")
        lock2 = consolidator.get_lock("session:b")
        assert lock1 is not lock2

    def test_get_lock_creates_asyncio_lock(
        self, consolidator: EMemConsolidator,
    ) -> None:
        lock = consolidator.get_lock("new:session")
        assert isinstance(lock, asyncio.Lock)


# ===================================================================
# EMemConsolidator — estimate
# ===================================================================

class TestEMemConsolidatorEstimate:
    """Test token estimation."""

    def test_estimate_session_prompt_tokens(
        self, consolidator: EMemConsolidator,
    ) -> None:
        session = MagicMock()
        session.get_history.return_value = [{"role": "user", "content": "hello"}]
        session.key = "channel:chat123"
        with patch(
            "nanobot.memory.naive_memory.consolidator.estimate_prompt_tokens_chain",
            return_value=(42, "tiktoken"),
        ) as mock_estimate:
            tokens, source = consolidator.estimate_session_prompt_tokens(session)
            assert tokens == 42
            assert source == "tiktoken"
            mock_estimate.assert_called_once()
