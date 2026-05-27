"""Tests for MastraOM Consolidator — Observer/Reflector pipeline, token budget, Hermes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from summerclaw.memory.mastra_om_memory.consolidator import MastraOMConsolidator
from summerclaw.memory.mastra_om_memory.store import MastraOMStore


@pytest.fixture
def store(tmp_path):
    return MastraOMStore(tmp_path)


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_sessions():
    s = MagicMock()
    s.save = MagicMock()
    return s


@pytest.fixture
def consolidator(store, mock_provider, mock_sessions):
    return MastraOMConsolidator(
        store=store,
        provider=mock_provider,
        model="test-model",
        sessions=mock_sessions,
        context_window_tokens=1000,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
        max_completion_tokens=100,
        message_tokens_threshold=30_000,
        observation_tokens_threshold=40_000,
    )


# ------------------------------------------------------------------
# Boundary picking
# ------------------------------------------------------------------


class TestBoundaryPicking:
    def test_no_boundary_when_nothing_to_remove(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 5
        session.messages = [{"role": "user", "content": "hi"} for _ in range(5)]
        result = consolidator.pick_consolidation_boundary(session, tokens_to_remove=0)
        assert result is None

    def test_no_boundary_when_at_end(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 3
        session.messages = [{"role": "user", "content": "hi"} for _ in range(3)]
        result = consolidator.pick_consolidation_boundary(session, tokens_to_remove=1000)
        assert result is None

    def test_finds_user_turn_boundary(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "help me"},
            {"role": "assistant", "content": "sure"},
        ]
        result = consolidator.pick_consolidation_boundary(session, tokens_to_remove=1)
        assert result is not None
        end_idx, _ = result
        assert end_idx == 2

    def test_returns_last_boundary_if_tokens_insufficient(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
        ]
        # Request impossibly large token removal
        result = consolidator.pick_consolidation_boundary(session, tokens_to_remove=999_999)
        assert result is not None
        end_idx, _ = result
        assert end_idx == 2

    def test_cap_preserves_user_turn_boundary(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user" if i in {0, 50, 61} else "assistant", "content": f"m{i}"}
            for i in range(70)
        ]
        capped = consolidator._cap_consolidation_boundary(session, end_idx=61)
        assert capped == 50  # rewind to last user turn within MAX_CHUNK_MESSAGES (60)

    def test_cap_returns_none_when_no_user_boundary(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user" if i == 0 else "assistant", "content": f"m{i}"}
            for i in range(70)
        ]
        capped = consolidator._cap_consolidation_boundary(session, end_idx=65)
        assert capped is None


# ------------------------------------------------------------------
# observe_and_store
# ------------------------------------------------------------------


class TestObserveAndStore:

    async def test_skips_empty_messages(self, consolidator, mock_provider):
        result = await consolidator.observe_and_store([])
        assert result is None
        mock_provider.chat_with_retry.assert_not_called()

    async def test_raw_dumps_on_llm_failure(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.observe_and_store(messages)
        assert result is None
        entries = store._read_entries()
        assert len(entries) >= 1
        assert any("[RAW]" in e["content"] for e in entries)

    async def test_raw_dumps_on_degenerate_output(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="x" * 50000,  # long enough to trigger degenerate detection
            finish_reason="stop",
        )
        messages = [{"role": "user", "content": "test"}]
        result = await consolidator.observe_and_store(messages)
        assert result is None
        entries = store._read_entries()
        assert any("[RAW]" in e["content"] for e in entries)

    async def test_appends_valid_observations(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="""<observations>
Date: May 9
* 🔴 User prefers dark mode
</observations>""",
            finish_reason="stop",
        )
        messages = [
            {"role": "user", "content": "I prefer dark mode", "timestamp": "2025-05-09 10:00"},
        ]
        result = await consolidator.observe_and_store(messages)
        assert result is not None
        assert "dark mode" in result
        obs = store.read_observations()
        assert "dark mode" in obs
        # Check raw messages stored in history (use _read_entries since read_unprocessed_history returns [])
        entries = store._read_entries()
        assert any("dark mode" in e["content"] for e in entries)
        # Check OM summary stored in om-ops
        om_ops = store.read_om_ops()
        assert any("[OM-OBSERVED]" in e["content"] for e in om_ops)
        # Check history_cursor was embedded
        assert 'history_cursor="' in obs


# ------------------------------------------------------------------
# reflect_and_condense
# ------------------------------------------------------------------


class TestReflectAndCondense:

    async def test_noop_when_empty_observations(self, consolidator, mock_provider):
        result = await consolidator.reflect_and_condense()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()

    async def test_noop_when_below_threshold(self, consolidator, mock_provider, store):
        store.write_observations("Short observation")
        result = await consolidator.reflect_and_condense()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()

    async def test_condenses_when_above_threshold(self, consolidator, mock_provider, store):
        # Create >40k chars of observations (~10k tokens)
        long_obs = "Long observation text. " * 11000
        store.write_observations(long_obs)
        consolidator.observation_tokens_threshold = 10000  # lower threshold for test

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="""<observations>
Date: May 9
* 🔴 Condensed summary
</observations>""",
            finish_reason="stop",
        )
        result = await consolidator.reflect_and_condense()
        assert result is True
        obs = store.read_observations()
        assert "Condensed summary" in obs

    async def test_progressive_compression_retry(self, consolidator, mock_provider, store):
        """When first reflection still too large, retry with higher compression level."""
        long_obs = "Long observation text. " * 11000
        store.write_observations(long_obs)
        consolidator.observation_tokens_threshold = 500  # very low

        # First call: still too large
        # Second call: small enough
        call_count = [0]

        async def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(
                    content="""<observations>
""" + ("Still too long. " * 500) + """</observations>""",
                    finish_reason="stop",
                )
            else:
                return MagicMock(
                    content="""<observations>
* 🔴 Short summary
</observations>""",
                    finish_reason="stop",
                )

        mock_provider.chat_with_retry.side_effect = side_effect
        result = await consolidator.reflect_and_condense()
        assert result is True
        assert call_count[0] == 2
        obs = store.read_observations()
        assert "Short summary" in obs


# ------------------------------------------------------------------
# extract_and_store (Hermes integration)
# ------------------------------------------------------------------


class TestExtractAndStore:

    async def test_skips_empty_messages(self, consolidator):
        result = await consolidator.extract_and_store([])
        assert result == []

    async def test_returns_facts_on_success(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="""<observations>
Date: May 9
* 🔴 User prefers Python over Java
* 🟡 User is working on a web app
</observations>""",
            finish_reason="stop",
        )
        messages = [
            {"role": "user", "content": "I prefer Python", "timestamp": "2025-05-09 10:00"},
            {"role": "user", "content": "building a web app", "timestamp": "2025-05-09 10:01"},
        ]
        facts = await consolidator.extract_and_store(messages)
        assert len(facts) == 2
        assert any("Python" in f for f in facts)
        assert any("web app" in f for f in facts)

    async def test_returns_empty_on_degenerate(self, consolidator, mock_provider):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="x" * 50000,
            finish_reason="stop",
        )
        messages = [{"role": "user", "content": "test"}]
        result = await consolidator.extract_and_store(messages)
        assert result == []

    async def test_returns_empty_on_api_error(self, consolidator, mock_provider):
        mock_provider.chat_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "test"}]
        result = await consolidator.extract_and_store(messages)
        assert result == []


# ------------------------------------------------------------------
# Context injection
# ------------------------------------------------------------------


class TestContextInjection:

    def test_empty_observations_returns_continuation_hint(self, consolidator, store):
        msgs = consolidator.build_context_system_messages()
        assert len(msgs) == 1
        assert "system-reminder" in msgs[0]
        assert "conversation" in msgs[0].lower()

    def test_with_observations_includes_context(self, consolidator, store):
        store.write_observations("Date: May 9\n* 🔴 User prefers dark mode")
        msgs = consolidator.build_context_system_messages()
        assert len(msgs) == 1
        assert "dark mode" in msgs[0]
        assert "observations" in msgs[0]
        assert "IMPORTANT" in msgs[0]


# ------------------------------------------------------------------
# archive (AutoCompact compat)
# ------------------------------------------------------------------


class TestArchive:

    async def test_archive_delegates_to_observe_and_store(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="""<observations>
* 🔴 Test observation
</observations>""",
            finish_reason="stop",
        )
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is not None
        assert "Test observation" in result


# ------------------------------------------------------------------
# Token budget consolidation
# ------------------------------------------------------------------


class TestTokenBudgetConsolidation:

    async def test_prompt_below_budget_does_not_consolidate(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100, "tiktoken"))
        consolidator.observe_and_store = AsyncMock(return_value=True)

        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.observe_and_store.assert_not_called()

    async def test_prompt_above_budget_triggers_observation(self, consolidator, mock_provider):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
        ]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(
            side_effect=[(1200, "tiktoken"), (400, "tiktoken")]
        )
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="""<observations>
* 🔴 Test
</observations>""",
            finish_reason="stop",
        )

        await consolidator.maybe_consolidate_by_tokens(session)
        assert session.last_consolidated > 0
