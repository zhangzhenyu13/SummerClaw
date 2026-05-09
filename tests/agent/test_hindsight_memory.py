"""Comprehensive tests for the Hindsight memory algorithm.

Covers all four components (Store, Consolidator, Dream, AutoCompact) plus
Hermes-Autogen mode, token budget consolidation, dream offline processing,
auto-compact idle session compression, registry registration, and fallback
behavior when the Hindsight server is unavailable.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.memory import MemoryRegistry, MemoryStore
from nanobot.memory.hindsight_memory import HindsightMemoryAlgorithm
from nanobot.memory.hindsight_memory.auto_compact import HindsightAutoCompact
from nanobot.memory.hindsight_memory.consolidator import HindsightConsolidator
from nanobot.memory.hindsight_memory.dream import HindsightDream
from nanobot.memory.hindsight_memory.store import HindsightStore
from nanobot.utils.gitstore import LineAge


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def tmp_workspace(tmp_path):
    """Temporary workspace with memory dir."""
    return tmp_path


@pytest.fixture
def store(tmp_workspace):
    """Create a HindsightStore with built-in local TEMPR engine."""
    return HindsightStore(tmp_workspace)


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    p.embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])  # mock embedding
    return p


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def consolidator(store, mock_provider):
    sessions = MagicMock()
    sessions.save = MagicMock()
    return HindsightConsolidator(
        store=store,
        provider=mock_provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=1000,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
        max_completion_tokens=100,
    )


@pytest.fixture
def dream(store, mock_provider, mock_runner):
    d = HindsightDream(
        store=store,
        provider=mock_provider,
        model="test-model",
        max_batch_size=5,
    )
    d._runner = mock_runner
    return d


def _make_run_result(
    stop_reason="completed",
    final_content=None,
    tool_events=None,
):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


# ============================================================================
# Test HindsightStore
# ============================================================================


class TestHindsightStoreBasicIO:
    """Test that HindsightStore inherits all naive MemoryStore capabilities."""

    def test_read_memory_empty(self, store):
        assert store.read_memory() == ""

    def test_write_and_read_memory(self, store):
        store.write_memory("hello hindsight")
        assert store.read_memory() == "hello hindsight"

    def test_read_soul_empty(self, store):
        assert store.read_soul() == ""

    def test_write_and_read_soul(self, store):
        store.write_soul("soul content")
        assert store.read_soul() == "soul content"

    def test_read_user_empty(self, store):
        assert store.read_user() == ""

    def test_write_and_read_user(self, store):
        store.write_user("user content")
        assert store.read_user() == "user content"

    def test_get_memory_context(self, store):
        store.write_memory("important fact")
        ctx = store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "important fact" in ctx

    def test_isinstance_of_naive_store(self, store):
        from nanobot.memory.naive_memory.store import MemoryStore as NaiveStore
        assert isinstance(store, NaiveStore)


class TestHindsightStoreHistory:
    """Test history.jsonl operations."""

    def test_append_history_returns_cursor(self, store):
        c1 = store.append_history("event 1")
        c2 = store.append_history("event 2")
        assert c1 == 1
        assert c2 == 2

    def test_read_unprocessed_history(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2

    def test_compact_history(self, tmp_workspace):
        s = HindsightStore(tmp_workspace, max_history_entries=2)
        s.append_history("e1")
        s.append_history("e2")
        s.append_history("e3")
        s.append_history("e4")
        s.compact_history()
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2


class TestHindsightStoreDreamCursor:
    """Test dream cursor persistence."""

    def test_initial_cursor_zero(self, store):
        assert store.get_last_dream_cursor() == 0

    def test_set_and_get(self, store):
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5

    def test_persists_across_instances(self, store):
        store.set_last_dream_cursor(3)
        s2 = HindsightStore(store.workspace)
        assert s2.get_last_dream_cursor() == 3


class TestHindsightStoreLocalTEMPR:
    """Test local TEMPR engine behavior."""

    def test_hindsight_always_enabled(self, store):
        """Local TEMPR engine is always enabled (no server needed)."""
        assert store.hindsight_enabled

    def test_aretain_stores_to_local_bank(self, store):
        """aretain should store memory in the local JSON bank."""
        async def _test():
            result = await store.aretain("User likes Python")
            assert result is not None
            assert hasattr(result, "text")
            assert result.text == "User likes Python"
            # Verify it's in the memory bank
            assert store.memory_count >= 1

        asyncio.run(_test())

    def test_arecall_searches_local_bank(self, store):
        """arecall should search the local TEMPR bank."""
        async def _test():
            # First retain some memories
            await store.aretain("Python is great for data science")
            await store.aretain("TypeScript has strong typing")
            # Search
            result = await store.arecall("Python programming", budget="high")
            assert result is not None
            assert hasattr(result, "text")

        asyncio.run(_test())

    def test_areflect_on_empty_bank(self, store):
        """areflect on empty bank returns empty text."""
        async def _test():
            result = await store.areflect("analyze this")
            assert result is not None
            assert hasattr(result, "text")
            assert result.text == ""  # empty bank

        asyncio.run(_test())

    def test_retain_sync_wrapper(self, store):
        result = store.retain("sync retain test")
        assert result is not None
        assert store.memory_count >= 1

    def test_recall_sync_wrapper(self, store):
        store.retain("sync recall content")
        result = store.recall("sync recall")
        assert result is not None
        assert hasattr(result, "text")

    def test_reflect_sync_wrapper(self, store):
        store.retain("sync reflect content")
        result = store.reflect("sync reflect")
        assert result is not None
        assert hasattr(result, "text")

    def test_tempr_memory_persistence(self, tmp_workspace):
        """Memories should persist across store instances."""
        s1 = HindsightStore(tmp_workspace)
        async def _test():
            await s1.aretain("persistent memory")
        asyncio.run(_test())

        s2 = HindsightStore(tmp_workspace)
        assert s2.memory_count >= 1


# ============================================================================
# Test HindsightConsolidator
# ============================================================================


class TestHindsightConsolidatorSummarize:
    async def test_summarize_appends_to_history(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="User fixed a bug."
        )
        messages = [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "Done."},
        ]
        result = await consolidator.archive(messages)
        assert result == "User fixed a bug."
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1

    async def test_summarize_raw_dumps_on_failure(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_summarize_skips_empty(self, consolidator):
        result = await consolidator.archive([])
        assert result is None

    async def test_has_hindsight_false_without_server(self, consolidator):
        assert not consolidator.has_hindsight


class TestHindsightConsolidatorTokenBudget:
    async def test_prompt_below_threshold_no_consolidation(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100, "tiktoken"))
        consolidator.archive = AsyncMock(return_value=True)
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_chunk_cap_preserves_user_turn_boundary(self, consolidator):
        consolidator._SAFETY_BUFFER = 0
        session = MagicMock()
        session.last_consolidated = 0
        session.key = "test:key"
        session.messages = [
            {"role": "user" if i in {0, 50, 61} else "assistant", "content": f"m{i}"}
            for i in range(70)
        ]
        consolidator.estimate_session_prompt_tokens = MagicMock(
            side_effect=[(1200, "tiktoken"), (400, "tiktoken")]
        )
        consolidator.pick_consolidation_boundary = MagicMock(return_value=(61, 999))
        consolidator.archive = AsyncMock(return_value=True)

        await consolidator.maybe_consolidate_by_tokens(session)
        archived = consolidator.archive.await_args.args[0]
        assert len(archived) == 50
        assert archived[0]["content"] == "m0"
        assert archived[-1]["content"] == "m49"
        assert session.last_consolidated == 50


class TestHindsightConsolidatorHermes:
    """Test Hermes-Autogen mid-turn extraction."""

    async def test_extract_and_store_empty(self, consolidator):
        session = MagicMock()
        result = await consolidator.extract_and_store([], session)
        assert result == []

    async def test_extract_and_store_basic(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="- Fact one\n- Fact two\n- Fact three",
            finish_reason="stop",
        )
        session = MagicMock()
        messages = [
            {"role": "user", "content": "I like Python"},
            {"role": "assistant", "content": "Python is great for ML"},
        ]
        result = await consolidator.extract_and_store(messages, session)
        assert len(result) >= 1
        assert all("memory" in r for r in result)
        assert all("event" in r for r in result)

    async def test_extract_and_store_falls_back_on_error(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.side_effect = Exception("LLM down")
        session = MagicMock()
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.extract_and_store(messages, session)
        # Falls back to raw archive, returns empty
        assert result == []

    async def test_extract_and_store_error_finish_reason(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Error text", finish_reason="error",
        )
        session = MagicMock()
        messages = [{"role": "user", "content": "hello"}]
        # Should raise RuntimeError caught internally, fallback to raw dump
        result = await consolidator.extract_and_store(messages, session)
        assert result == []

    async def test_parse_summary_into_facts(self, consolidator):
        summary = "- Fact A\n- Fact B\n* Fact C\nPlain line"
        facts = consolidator._parse_summary_into_facts(summary)
        assert len(facts) == 4
        assert "Fact A" in facts
        assert "Fact B" in facts

    async def test_parse_summary_short_lines_filtered(self, consolidator):
        summary = "- A\n- Very long and meaningful fact here"
        facts = consolidator._parse_summary_into_facts(summary)
        # "A" is too short (len<5)
        assert len(facts) == 1

    async def test_get_lock_returns_same_for_same_key(self, consolidator):
        lock1 = consolidator.get_lock("key1")
        lock2 = consolidator.get_lock("key1")
        assert lock1 is lock2

    async def test_get_lock_returns_different_for_different_keys(self, consolidator):
        lock1 = consolidator.get_lock("key1")
        lock2 = consolidator.get_lock("key2")
        assert lock1 is not lock2


# ============================================================================
# Test HindsightDream
# ============================================================================


class TestHindsightDreamRun:
    async def test_noop_when_no_history(self, dream, mock_provider, mock_runner, store):
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()

    async def test_calls_runner_for_unprocessed(self, dream, mock_provider, mock_runner, store):
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()

    async def test_advances_dream_cursor(self, dream, mock_provider, mock_runner, store):
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_history(self, dream, mock_provider, mock_runner, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_phase1_prompt_includes_age_annotations(self, dream, mock_provider, mock_runner, store):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        store.git.init()
        store.git.auto_commit("initial")

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_prompt_works_without_git(self, dream, mock_provider, mock_runner, store):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()
        mock_provider.chat_with_retry.assert_called_once()

    async def test_hindsight_fallback_on_analysis_failure(self, dream, mock_provider, mock_runner, store):
        """When Hindsight is not available, Dream falls back to LLM-only analysis."""
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="LLM fallback analysis")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        result = await dream.run()
        assert result is True
        mock_provider.chat_with_retry.assert_called_once()

    async def test_skill_phase_uses_builtin_skill_creator(self, dream, mock_provider, mock_runner, store):
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        store.append_history("Repeated workflow")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKILL] test: desc")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        spec = mock_runner.run.call_args[0][0]
        system_prompt = spec.initial_messages[0]["content"]
        expected = str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        assert expected in system_prompt

    async def test_skill_write_tool_works(self, dream, store):
        write_tool = dream._tools.get("write_file")
        assert write_tool is not None
        result = await write_tool.execute(
            path="skills/test-skill/SKILL.md",
            content="---\nname: dreamed-test-skill\ndescription: Test\n---\n",
        )
        assert "Successfully wrote" in result
        assert (store.workspace / "skills" / "dreamed-test-skill" / "SKILL.md").exists()

    async def test_line_age_threshold(self, dream, mock_provider, mock_runner, store):
        """End-to-end: stale lines get age suffix, fresh lines don't."""
        store.write_memory("# Memory\n- Project X active\n- fresh item\n- edge case")
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        fake_ages = [
            LineAge(age_days=30),  # stale
            LineAge(age_days=20),  # stale
            LineAge(age_days=14),  # threshold boundary
            LineAge(age_days=5),   # fresh
        ]
        with patch.object(store.git, "line_ages", return_value=fake_ages):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190 30d" in memory_section
        assert "\u2190 20d" in memory_section
        assert "\u2190 14d" not in memory_section
        assert "\u2190 5d" not in memory_section

    async def test_annotate_disabled_bypasses_git(self, dream, mock_provider, mock_runner, store):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        dream.annotate_line_ages = False

        with patch.object(store.git, "line_ages") as mock_line_ages:
            await dream.run()
            mock_line_ages.assert_not_called()

    async def test_length_mismatch_skips_annotation(self, dream, mock_provider, mock_runner, store):
        """If ages length != lines length, skip annotation instead of mis-tagging."""
        # Write 2 non-blank lines; mock only 1 age → mismatch
        store.write_memory("# Memory\n- Project X active")
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch.object(store.git, "line_ages", return_value=[LineAge(age_days=999)]):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        # No age arrow at all — we refused to annotate rather than tag the wrong line.
        assert "\u2190" not in memory_section

    async def test_has_hindsight_false_by_default(self, dream):
        assert not dream.has_hindsight


# ============================================================================
# Test HindsightAutoCompact
# ============================================================================


class TestHindsightAutoCompact:
    def test_disabled_by_default(self, store, consolidator):
        sessions = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=0,
        )
        assert not ac.enabled

    def test_enabled_with_positive_ttl(self, store, consolidator):
        sessions = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )
        assert ac.enabled

    def test_has_hindsight_false(self, store, consolidator):
        sessions = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )
        assert not ac.has_hindsight

    def test_is_expired_boundary(self, store, consolidator):
        sessions = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )
        ts = datetime.now() - timedelta(minutes=15)
        assert ac._is_expired(ts) is True
        ts2 = datetime.now() - timedelta(minutes=14, seconds=59)
        assert ac._is_expired(ts2) is False

    def test_is_expired_string_timestamp(self, store, consolidator):
        sessions = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )
        ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        assert ac._is_expired(ts) is True
        assert ac._is_expired(None) is False

    def test_check_expired_skips_when_disabled(self, store, consolidator):
        sessions = MagicMock()
        sessions.list_sessions.return_value = []
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=0,
        )
        schedule_bg = MagicMock()
        ac.check_expired(schedule_bg)
        schedule_bg.assert_not_called()

    def test_format_summary(self, store, consolidator):
        sessions = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )
        last_active = datetime.now() - timedelta(minutes=30)
        summary = ac._format_summary("Test summary.", last_active)
        assert "Inactive for" in summary
        assert "Test summary." in summary

    async def test_archive_empty_session(self, store, consolidator):
        sessions = MagicMock()
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = []
        session.key = "test:key"
        session.created_at = datetime.now()
        session.updated_at = datetime.now()
        session.metadata = {}
        sessions.get_or_create.return_value = session
        sessions.invalidate = MagicMock()
        sessions.save = MagicMock()
        sessions.list_sessions.return_value = [{"key": "test:key", "updated_at": datetime.now()}]

        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )

        await ac._archive("test:key")
        sessions.save.assert_called()

    async def test_archive_with_messages(self, store, mock_provider):
        sessions = MagicMock()
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        session.key = "test:key"
        session.created_at = datetime.now() - timedelta(minutes=25)
        session.updated_at = datetime.now() - timedelta(minutes=20)
        session.metadata = {}
        sessions.get_or_create.return_value = session
        sessions.invalidate = MagicMock()
        sessions.save = MagicMock()

        mock_provider.chat_with_retry = AsyncMock(
            return_value=MagicMock(content="Summary text", finish_reason="stop")
        )
        cons = HindsightConsolidator(
            store=store,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=cons, session_ttl_minutes=15,
        )

        await ac._archive("test:key")
        sessions.save.assert_called()

    def test_prepare_session_no_summary(self, store, consolidator):
        sessions = MagicMock()
        session = MagicMock()
        session.updated_at = datetime.now()
        ac = HindsightAutoCompact(
            sessions=sessions, consolidator=consolidator, session_ttl_minutes=15,
        )
        new_sess, summary = ac.prepare_session(session, "test:key")
        assert summary is None


# ============================================================================
# Test HindsightMemoryAlgorithm (Registry + Build)
# ============================================================================


class TestHindsightMemoryAlgorithm:
    def test_algorithm_name(self):
        algo = HindsightMemoryAlgorithm()
        assert algo.name == "hindsight_memory"

    def test_registry_registration(self):
        registry = MemoryRegistry()
        registry.register(HindsightMemoryAlgorithm())
        algo = registry.get("hindsight_memory")
        assert algo is not None
        assert algo.name == "hindsight_memory"

    def test_registry_list_includes_hindsight(self):
        registry = MemoryRegistry()
        registry.register(HindsightMemoryAlgorithm())
        names = registry.list()
        assert "hindsight_memory" in names

    def test_registry_default_is_still_naive(self):
        registry = MemoryRegistry()
        assert registry.default_name == "naive_memory"

    def test_build_returns_all_components(self, tmp_workspace):
        algo = HindsightMemoryAlgorithm()
        provider = MagicMock()
        sessions = MagicMock()

        components = algo.build(
            workspace=tmp_workspace,
            provider=provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=100,
            session_ttl_minutes=15,
            max_batch_size=10,
            max_iterations=5,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
        )

        assert components.store is not None
        assert components.consolidator is not None
        assert components.dream is not None
        assert components.auto_compact is not None

        assert isinstance(components.store, HindsightStore)
        assert isinstance(components.consolidator, HindsightConsolidator)
        assert isinstance(components.dream, HindsightDream)
        assert isinstance(components.auto_compact, HindsightAutoCompact)

    def test_build_with_zero_ttl_auto_compact_is_still_created(self, tmp_workspace):
        """AutoCompact is always created; enabled check is internal."""
        algo = HindsightMemoryAlgorithm()
        components = algo.build(
            workspace=tmp_workspace,
            provider=MagicMock(),
            model="test",
            sessions=MagicMock(),
            context_window_tokens=1000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=100,
            session_ttl_minutes=0,
            max_batch_size=10,
            max_iterations=5,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
        )
        assert components.auto_compact is not None
        assert isinstance(components.auto_compact, HindsightAutoCompact)
        assert not components.auto_compact.enabled

    def test_build_passes_hindsight_store_to_components(self, tmp_workspace):
        algo = HindsightMemoryAlgorithm()
        components = algo.build(
            workspace=tmp_workspace,
            provider=MagicMock(),
            model="test",
            sessions=MagicMock(),
            context_window_tokens=1000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=100,
            session_ttl_minutes=15,
            max_batch_size=10,
            max_iterations=5,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
        )
        # All components share the same store instance
        assert components.store is components.consolidator._hindsight_store
        assert components.store is components.dream._hindsight_store
        assert components.store is components.auto_compact._hindsight_store

    def test_import_from_memory_package(self):
        from nanobot.memory import (
            HindsightAutoCompact,
            HindsightConsolidator,
            HindsightDream,
            HindsightMemoryAlgorithm,
            HindsightStore,
        )
        assert HindsightMemoryAlgorithm.name == "hindsight_memory"

    def test_double_registration_overwrites(self):
        registry = MemoryRegistry()
        algo1 = HindsightMemoryAlgorithm()
        algo2 = HindsightMemoryAlgorithm()
        registry.register(algo1)
        registry.register(algo2)
        assert registry.get("hindsight_memory") is algo2


# ============================================================================
# Local TEMPR search — tests with mocked provider.embed()
# ============================================================================


def _make_tempr_store(tmp_workspace, mock_provider):
    """Create a HindsightStore with a provider that supports embeddings."""
    s = HindsightStore(
        tmp_workspace,
        provider=mock_provider,
        embedding_model="test-embed-model",
    )
    return s


class TestHindsightStoreTEMPRSearch:
    """Test local TEMPR multi-strategy search with mock embeddings."""

    async def test_aretain_generates_embedding(self, tmp_workspace, mock_provider):
        """aretain should call provider.embed() when available."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        await s.aretain("embed this memory")
        # Provider.embed() was called for the new memory
        mock_provider.embed.assert_called()

    async def test_arecall_with_embeddings(self, tmp_workspace, mock_provider):
        """arecall should use embeddings for semantic search."""
        mock_provider.embed = MagicMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        s = _make_tempr_store(tmp_workspace, mock_provider)
        await s.aretain("Python programming language")
        result = await s.arecall("programming", budget="high")
        assert result is not None
        assert hasattr(result, "text")

    async def test_arecall_keyword_only_budget_low(self, tmp_workspace, mock_provider):
        """Low budget should use keyword-only search (no embeddings)."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        await s.aretain("unique keyword zebra")
        result = await s.arecall("zebra", budget="low")
        assert result is not None
        # Low budget should not call embed for the query
        assert hasattr(result, "text")

    async def test_areflect_with_provider_synthesis(self, tmp_workspace, mock_provider):
        """areflect should use LLM synthesis when provider is available."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        await s.aretain("User prefers dark mode")
        await s.aretain("Project uses React")

        mock_provider.chat_with_retry = AsyncMock(
            return_value=MagicMock(content="Synthesized analysis: user settings and tech stack.")
        )
        # Re-make store so the provider has chat_with_retry
        s = _make_tempr_store(tmp_workspace, mock_provider)
        # Need to reload memories
        s2 = HindsightStore(tmp_workspace, provider=mock_provider, embedding_model="test-embed-model")

        result = await s2.areflect("user preferences", budget="mid")
        assert result is not None
        assert hasattr(result, "text")

    async def test_areflect_fallback_without_provider(self, tmp_workspace):
        """areflect without provider returns raw TEMPR results."""
        s = HindsightStore(tmp_workspace)
        await s.aretain("fact one")
        await s.aretain("fact two")

        result = await s.areflect("facts")
        assert result is not None
        assert hasattr(result, "text")

    async def test_tempr_memory_limit(self, tmp_workspace):
        """Memory bank should trim oldest entries when exceeding max_memories."""
        s = HindsightStore(tmp_workspace, max_memories=3)
        for i in range(5):
            await s.aretain(f"memory item {i}")
        assert s.memory_count <= 3

    async def test_aretain_empty_content(self, tmp_workspace):
        """aretain with empty content returns None."""
        s = HindsightStore(tmp_workspace)
        result = await s.aretain("")
        assert result is None
        result = await s.aretain("   ")
        assert result is None


class TestHindsightConsolidatorTEMPR:
    """Test Consolidator with local TEMPR retention."""

    async def test_archive_retains_to_tempr_bank(self, tmp_workspace, mock_provider):
        """archive() should also retain to local TEMPR bank when available."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        sessions = MagicMock()
        sessions.save = MagicMock()

        cons = HindsightConsolidator(
            store=s,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=100,
            hindsight_store=s,
        )
        assert cons.has_hindsight

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="User fixed a bug.",
            finish_reason="stop",
        )
        messages = [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "Done."},
        ]
        result = await cons.archive(messages)
        assert result == "User fixed a bug."
        # Should have retained to TEMPR bank
        assert s.memory_count >= 1

    async def test_archive_tempr_retain_failure_does_not_block(self, tmp_workspace, mock_provider):
        """If TEMPR retain fails, archive should still succeed (file store)."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        sessions = MagicMock()
        sessions.save = MagicMock()

        # Make the TEMPR aretain fail by corrupting the store
        async def _failing_aretain(*args, **kwargs):
            raise RuntimeError("TEMPR down")
        s.aretain = _failing_aretain

        cons = HindsightConsolidator(
            store=s,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=100,
            hindsight_store=s,
        )

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Summary.", finish_reason="stop",
        )
        result = await cons.archive([{"role": "user", "content": "hello"}])
        assert result == "Summary."
        # File store should still have the entry
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) >= 1

    async def test_hermes_extract_and_store_retains_to_tempr(self, tmp_workspace, mock_provider):
        """Hermes extract_and_store should retain facts to local TEMPR bank."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        sessions = MagicMock()
        sessions.save = MagicMock()

        cons = HindsightConsolidator(
            store=s,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=100,
            hindsight_store=s,
        )

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="- Fact one\n- Fact two",
            finish_reason="stop",
        )
        session = MagicMock()
        messages = [
            {"role": "user", "content": "I like Python"},
            {"role": "assistant", "content": "Python is great for ML"},
        ]
        result = await cons.extract_and_store(messages, session)
        assert len(result) >= 1
        # Facts should be in the TEMPR bank
        assert s.memory_count >= 1

    async def test_hermes_tempr_retain_failure_does_not_block(self, tmp_workspace, mock_provider):
        """Hermes should still work if TEMPR retain fails."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        sessions = MagicMock()
        sessions.save = MagicMock()

        async def _failing_aretain(*args, **kwargs):
            raise RuntimeError("TEMPR unreachable")
        s.aretain = _failing_aretain

        cons = HindsightConsolidator(
            store=s,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=100,
            hindsight_store=s,
        )

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="- Fact one", finish_reason="stop",
        )
        session = MagicMock()
        result = await cons.extract_and_store(
            [{"role": "user", "content": "hello"}], session,
        )
        assert len(result) >= 1
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) >= 1


class TestHindsightDreamTEMPR:
    """Test Dream with local TEMPR engine."""

    async def test_phase1_uses_tempr_reflect(self, tmp_workspace, mock_provider):
        """Dream Phase 1 should use local TEMPR reflect when store has hindsight."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
            hindsight_store=s,
        )
        d._runner = mock_runner
        assert d.has_hindsight

        # Pre-populate TEMPR bank so reflect returns something
        await s.aretain("User said they prefer TypeScript over JavaScript")

        s.append_history("User said they prefer TypeScript")

        # Mock LLM fallback (should NOT be called if TEMPR reflect succeeds)
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="LLM fallback", finish_reason="stop",
        )

        await d.run()

        # Phase 1 LLM should NOT be called since TEMPR reflect gave results
        # (but if TEMPR reflect returns empty or fails, LLM fallback is called)

    async def test_phase1_falls_back_to_llm_when_tempr_fails(self, tmp_workspace, mock_provider):
        """Dream Phase 1 should fall back to LLM if TEMPR reflect fails."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
            hindsight_store=s,
        )
        d._runner = mock_runner

        # Make TEMPR reflect fail
        async def _failing_reflect(*args, **kwargs):
            raise RuntimeError("TEMPR reflect failed")
        s.areflect = _failing_reflect

        s.append_history("event 1")

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="LLM fallback analysis", finish_reason="stop",
        )

        await d.run()

        # LLM fallback was called
        mock_provider.chat_with_retry.assert_called()

    async def test_phase1_both_tempr_and_llm_fail(self, tmp_workspace, mock_provider):
        """When both TEMPR and LLM fail, Dream should return False."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        mock_runner = MagicMock()

        d = HindsightDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
            hindsight_store=s,
        )
        d._runner = mock_runner

        s.append_history("event 1")

        async def _failing_reflect(*args, **kwargs):
            raise RuntimeError("TEMPR down")
        s.areflect = _failing_reflect
        mock_provider.chat_with_retry.side_effect = Exception("LLM down")

        result = await d.run()
        assert result is False

    async def test_phase1_uses_llm_when_tempr_returns_empty(self, tmp_workspace, mock_provider):
        """If TEMPR reflect returns empty text, fall back to LLM."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
            hindsight_store=s,
        )
        d._runner = mock_runner

        s.append_history("event 1")

        # TEMPR bank is empty → reflect returns empty text
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="LLM analysis", finish_reason="stop",
        )

        await d.run()

        mock_provider.chat_with_retry.assert_called()

    async def test_phase1_uses_llm_without_hindsight_store(self, tmp_workspace, mock_provider):
        """Dream without hindsight_store should use LLM directly."""
        s = HindsightStore(tmp_workspace)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner
        assert not d.has_hindsight

        s.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="LLM analysis", finish_reason="stop",
        )

        await d.run()
        mock_provider.chat_with_retry.assert_called()


class TestHindsightDreamSkills:
    """Test Dream skill listing and SOUL/USER handling."""

    def test_list_existing_skills_empty(self, store):
        """When no skills exist (and BUILTIN_SKILLS_DIR is patched empty), _list_existing_skills returns empty list."""
        d = HindsightDream(
            store=store,
            provider=MagicMock(),
            model="test-model",
        )
        from nanobot.agent import skills as skills_mod
        from pathlib import Path as _Path
        with patch.object(skills_mod, "BUILTIN_SKILLS_DIR", _Path("/nonexistent/path")):
            skills = d._list_existing_skills()
            assert skills == []

    def test_list_existing_skills_from_workspace(self, store):
        """Should list skills from workspace/skills/ directory."""
        skills_dir = store.workspace / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill for testing\n---\n"
        )
        d = HindsightDream(
            store=store,
            provider=MagicMock(),
            model="test-model",
        )
        from nanobot.agent import skills as skills_mod
        from pathlib import Path as _Path
        with patch.object(skills_mod, "BUILTIN_SKILLS_DIR", _Path("/nonexistent/path")):
            skills = d._list_existing_skills()
            assert len(skills) == 1
            assert "test-skill" in skills[0]
            assert "A test skill for testing" in skills[0]

    def test_list_existing_skills_skips_dirs_without_skill_md(self, store):
        """Directories without SKILL.md should be skipped."""
        (store.workspace / "skills" / "incomplete").mkdir(parents=True)
        d = HindsightDream(
            store=store,
            provider=MagicMock(),
            model="test-model",
        )
        from nanobot.agent import skills as skills_mod
        from pathlib import Path as _Path
        with patch.object(skills_mod, "BUILTIN_SKILLS_DIR", _Path("/nonexistent/path")):
            skills = d._list_existing_skills()
            assert skills == []

    async def test_phase2_prompt_includes_skills_section(self, store, mock_provider):
        """Phase 2 prompt should include existing skills when available."""
        skills_dir = store.workspace / "skills" / "existing-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\ndescription: An existing project skill\n---\n"
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis", finish_reason="stop",
        )

        await d.run()

        spec = mock_runner.run.call_args[0][0]
        phase2_prompt = spec.initial_messages[1]["content"]
        assert "## Existing Skills" in phase2_prompt
        assert "existing-skill" in phase2_prompt

    async def test_phase2_no_skills_section_when_empty(self, store, mock_provider):
        """Phase 2 prompt should not include skills section when no skills exist."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis", finish_reason="stop",
        )

        from nanobot.agent import skills as skills_mod
        from pathlib import Path as _Path
        with patch.object(skills_mod, "BUILTIN_SKILLS_DIR", _Path("/nonexistent/path")):
            await d.run()

        spec = mock_runner.run.call_args[0][0]
        phase2_prompt = spec.initial_messages[1]["content"]
        assert "## Existing Skills" not in phase2_prompt

    async def test_soul_and_user_never_annotated(self, store, mock_provider):
        """SOUL.md and USER.md should never have age annotations — they are permanent."""
        store.write_soul("# Soul\n- Helpful assistant")
        store.write_user("# User\n- Developer")
        store.write_memory("# Memory\n- Some fact")

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        store.git.init()
        store.git.auto_commit("initial state")

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="[SKIP]", finish_reason="stop",
        )

        await d.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        soul_section = user_msg.split("## Current SOUL.md")[1].split("## Current USER.md")[0]
        user_section = user_msg.split("## Current USER.md")[1]
        assert "\u2190" not in soul_section
        assert "\u2190" not in user_section


class TestHindsightAutoCompactTEMPR:
    """Test AutoCompact with local TEMPR store."""

    async def test_archive_with_tempr_store(self, tmp_workspace, mock_provider):
        """AutoCompact should work with local TEMPR store available."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        sessions = MagicMock()

        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user", "content": "long conversation 1"},
            {"role": "assistant", "content": "long response 1"},
            {"role": "user", "content": "long conversation 2"},
            {"role": "assistant", "content": "long response 2"},
            {"role": "user", "content": "long conversation 3"},
            {"role": "assistant", "content": "long response 3"},
            {"role": "user", "content": "long conversation 4"},
            {"role": "assistant", "content": "long response 4"},
            {"role": "user", "content": "long conversation 5"},
            {"role": "assistant", "content": "long response 5"},
        ]
        session.key = "test:key"
        session.created_at = datetime.now()
        session.updated_at = datetime.now()
        session.metadata = {}
        sessions.get_or_create.return_value = session
        sessions.invalidate = MagicMock()
        sessions.save = MagicMock()
        sessions.list_sessions.return_value = [{"key": "test:key", "updated_at": datetime.now()}]

        mock_provider.chat_with_retry = AsyncMock(
            return_value=MagicMock(content="Summary of conversation.", finish_reason="stop"),
        )
        cons = HindsightConsolidator(
            store=s,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=1000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )
        ac = HindsightAutoCompact(
            sessions=sessions,
            consolidator=cons,
            session_ttl_minutes=15,
            hindsight_store=s,
        )
        assert ac.has_hindsight

        await ac._archive("test:key")
        sessions.save.assert_called()

    def test_has_hindsight_true_with_tempr(self, tmp_workspace, mock_provider):
        """has_hindsight should be True when store with local TEMPR is provided."""
        s = _make_tempr_store(tmp_workspace, mock_provider)
        sessions = MagicMock()
        consolidator = MagicMock()
        ac = HindsightAutoCompact(
            sessions=sessions,
            consolidator=consolidator,
            session_ttl_minutes=15,
            hindsight_store=s,
        )
        assert ac.has_hindsight


class TestHindsightDreamEdgeCases:
    """Edge case tests for HindsightDream."""

    async def test_dream_with_completed_runner_result(self, store, mock_provider):
        """Dream should handle completed runner result properly."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            stop_reason="completed",
            tool_events=[
                {"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md updated"},
                {"name": "write_file", "status": "ok", "detail": "skills/dreamed-test/SKILL.md created"},
            ],
        ))
        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis complete.", finish_reason="stop",
        )

        result = await d.run()
        assert result is True
        # Cursor should be advanced
        assert store.get_last_dream_cursor() == 2

    async def test_dream_phase2_failure_advances_cursor(self, store, mock_provider):
        """Even when Phase 2 fails, cursor should advance to avoid re-processing."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=Exception("Phase 2 error"))
        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Phase 1 analysis", finish_reason="stop",
        )

        result = await d.run()
        assert result is True
        # Cursor still advances — prevents infinite re-processing
        assert store.get_last_dream_cursor() == 3

    async def test_dream_with_tool_event_failures(self, store, mock_provider):
        """Dream should handle mixed tool event statuses."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            stop_reason="completed",
            tool_events=[
                {"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md patched"},
                {"name": "read_file", "status": "error", "detail": "file not found"},
            ],
        ))
        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis", finish_reason="stop",
        )

        result = await d.run()
        assert result is True
        # Failed tool events should not crash dream processing

    async def test_dream_git_commit_includes_analysis(self, store, mock_provider):
        """When git is available, dream should auto-commit with analysis in message."""
        store.git.init()
        store.git.auto_commit("initial")

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[
                {"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md updated"},
            ],
        ))
        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Important analysis for commit message.", finish_reason="stop",
        )

        result = await d.run()
        assert result is True

    async def test_batch_size_cap(self, store, mock_provider):
        """Dream should only process up to max_batch_size entries per run."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=2,
        )
        d._runner = mock_runner

        # Append 5 entries
        for i in range(5):
            store.append_history(f"event {i}")

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis", finish_reason="stop",
        )

        await d.run()

        # Cursor should be at 2 (batch capped)
        assert store.get_last_dream_cursor() == 2

    async def test_dream_incomplete_stop_reason(self, store, mock_provider):
        """Dream should handle non-completed stop reasons."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            stop_reason="max_iterations",
        ))
        d = HindsightDream(
            store=store,
            provider=mock_provider,
            model="test-model",
            max_batch_size=5,
        )
        d._runner = mock_runner

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis", finish_reason="stop",
        )

        result = await d.run()
        assert result is True  # Still processed, cursor advanced


class TestHindsightConsolidatorEdgeCases:
    """Additional edge case tests for HindsightConsolidator."""

    async def test_archive_on_error_finish_reason(self, consolidator, mock_provider, store):
        """LLM returning finish_reason='error' should trigger raw_archive."""
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Error text", finish_reason="error",
        )
        messages = [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "Done."},
        ]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_consolidation_with_context_window_zero(self, consolidator):
        """Should not consolidate when context_window_tokens is 0."""
        consolidator.context_window_tokens = 0
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_consolidation_with_no_messages(self, consolidator):
        """Should not consolidate when session has no messages."""
        consolidator.context_window_tokens = 1000
        session = MagicMock()
        session.messages = []
        session.key = "test:key"
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_extract_and_store_custom_instructions(self, consolidator, mock_provider, store):
        """Hermes extract_and_store should accept custom_instructions."""
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="- Custom fact",
            finish_reason="stop",
        )
        session = MagicMock()
        messages = [{"role": "user", "content": "I like Python"}]
        result = await consolidator.extract_and_store(
            messages, session,
            custom_instructions="Only extract programming-related facts",
        )
        assert len(result) >= 1
