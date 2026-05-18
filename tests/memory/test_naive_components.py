"""Comprehensive tests for Naive Memory components — Consolidator, Dream, AutoCompact."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.memory import MemoryStore
from nanobot.memory.naive_memory.auto_compact import AutoCompact
from nanobot.memory.naive_memory.consolidator import Consolidator
from nanobot.memory.naive_memory.dream import Dream

from nanobot.utils.gitstore import LineAge


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    p.token_count_tool_definitions = MagicMock(return_value=200)
    p.token_count_messages = MagicMock(return_value=100)
    return p


@pytest.fixture
def consolidator(store, mock_provider):
    sessions = MagicMock()
    sessions.save = MagicMock()
    return Consolidator(
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
def dream(store, mock_provider):
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=_make_run_result())
    d = Dream(
        store=store,
        provider=mock_provider,
        model="test-model",
        max_batch_size=5,
    )
    d._runner = mock_runner
    return d


def _make_run_result(stop_reason="completed", final_content=None, tool_events=None):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


# ============================================================================
# Consolidator tests
# ============================================================================


class TestConsolidator:
    async def test_archive_appends_to_history(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="User fixed a bug.",
            finish_reason="stop",
        )
        messages = [
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "Done."},
        ]
        result = await consolidator.archive(messages)
        assert result == "User fixed a bug."
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1

    async def test_archive_empty_returns_none(self, consolidator):
        result = await consolidator.archive([])
        assert result is None

    async def test_archive_raw_dumps_on_failure(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_archive_error_finish_reason(self, consolidator, mock_provider, store):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Error text", finish_reason="error",
        )
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_maybe_consolidate_noop_with_no_messages(self, consolidator):
        session = MagicMock()
        session.messages = []
        session.key = "test:key"
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_maybe_consolidate_noop_zero_context_window(self, consolidator):
        consolidator.context_window_tokens = 0
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.archive = AsyncMock()
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_maybe_consolidate_below_budget_noop(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(100, "tiktoken"))
        consolidator.archive = AsyncMock(return_value=True)
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_pick_consolidation_boundary(self, consolidator):
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
            {"role": "assistant", "content": "m4"},
        ]
        consolidator.estimate_session_prompt_tokens = MagicMock(return_value=(50, "tiktoken"))
        boundary = consolidator.pick_consolidation_boundary(session, 30)
        assert boundary is not None

    def test_get_lock_same_key_same_object(self, consolidator):
        lock1 = consolidator.get_lock("key1")
        lock2 = consolidator.get_lock("key1")
        assert lock1 is lock2

    def test_get_lock_different_keys_different(self, consolidator):
        lock1 = consolidator.get_lock("key1")
        lock2 = consolidator.get_lock("key2")
        assert lock1 is not lock2


# ============================================================================
# Dream tests
# ============================================================================


class TestDream:
    async def test_noop_when_no_history(self, dream, mock_provider):
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()

    async def test_runs_phase1_for_unprocessed_history(self, dream, mock_provider, store):
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        result = await dream.run()
        assert result is True
        mock_provider.chat_with_retry.assert_called_once()

    async def test_advances_cursor_after_run(self, dream, mock_provider, store):
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_phase1_failure_returns_false(self, dream, mock_provider, store):
        store.append_history("event 1")
        mock_provider.chat_with_retry.side_effect = Exception("LLM down")
        result = await dream.run()
        assert result is False

    async def test_batch_size_cap(self, mock_provider, store):
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d = Dream(store=store, provider=mock_provider, model="test-model", max_batch_size=2)
        d._runner = mock_runner

        for i in range(5):
            store.append_history(f"event {i}")

        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")
        await d.run()
        assert store.get_last_dream_cursor() == 2

    async def test_dream_phase2_failure_advances_cursor(self, mock_provider, store):
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=Exception("Phase 2 error"))
        d = Dream(store=store, provider=mock_provider, model="test-model")
        d._runner = mock_runner

        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Phase 1 ok")
        result = await d.run()
        assert result is True
        assert store.get_last_dream_cursor() == 2

    async def test_annotate_disabled_bypasses_git(self, mock_provider, store):
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d = Dream(store=store, provider=mock_provider, model="test-model")
        d._runner = mock_runner
        d.annotate_line_ages = False

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")

        with patch.object(store.git, "line_ages") as mock_line_ages:
            await d.run()
            mock_line_ages.assert_not_called()

    async def test_phase1_prompt_includes_file_context(self, dream, mock_provider, store):
        store.write_memory("# Memory\n- Fact")
        store.write_soul("# Soul\n- Helpful")
        store.write_user("# User\n- Developer")
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")

        await dream.run()
        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "Current MEMORY.md" in user_msg
        assert "Current SOUL.md" in user_msg
        assert "Current USER.md" in user_msg

    async def test_phase2_prompt_includes_skills(self, mock_provider, store):
        skills_dir = store.workspace / "skills" / "existing-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("---\ndescription: A test skill\n---\n")

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d = Dream(store=store, provider=mock_provider, model="test-model")
        d._runner = mock_runner

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")

        from nanobot.agent import skills as skills_mod
        from pathlib import Path as _Path
        with patch.object(skills_mod, "BUILTIN_SKILLS_DIR", _Path("/nonexistent/path")):
            await d.run()

        spec = mock_runner.run.call_args[0][0]
        phase2_prompt = spec.initial_messages[1]["content"]
        assert "Existing Skills" in phase2_prompt

    async def test_skill_write_tool(self, store, mock_provider):
        d = Dream(store=store, provider=mock_provider, model="test-model")
        write_tool = d._tools.get("write_file")
        assert write_tool is not None

        result = await write_tool.execute(
            path="skills/test-skill/SKILL.md",
            content="---\nname: dreamed--naive_memory-test-skill\ndescription: Test\n---\n",
        )
        assert "Successfully wrote" in result


# ============================================================================
# Dream line age annotation tests
# ============================================================================


class TestDreamLineAgeAnnotation:
    async def test_stale_lines_get_age_suffix(self, mock_provider, store):
        store.write_memory("# Memory\n- stale line\n- fresh line")
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d = Dream(store=store, provider=mock_provider, model="test-model")
        d._runner = mock_runner

        fake_ages = [
            LineAge(age_days=30),  # stale — gets suffix
            LineAge(age_days=20),  # stale
            LineAge(age_days=5),   # fresh — no suffix
        ]
        with patch.object(store.git, "line_ages", return_value=fake_ages):
            await d.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190 30d" in memory_section
        assert "\u2190 20d" in memory_section
        assert "\u2190 5d" not in memory_section

    async def test_length_mismatch_skips_annotation(self, mock_provider, store):
        store.write_memory("# Memory\n- line one")
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d = Dream(store=store, provider=mock_provider, model="test-model")
        d._runner = mock_runner

        # 2 non-blank lines but only 1 age
        with patch.object(store.git, "line_ages", return_value=[LineAge(age_days=999)]):
            await d.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190" not in memory_section

    async def test_soul_and_user_never_annotated(self, mock_provider, store):
        store.write_soul("# Soul\n- Helpful")
        store.write_user("# User\n- Developer")
        store.write_memory("# Memory\n- Fact")
        store.append_history("event")

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d = Dream(store=store, provider=mock_provider, model="test-model")
        d._runner = mock_runner

        store.git.init()
        store.git.auto_commit("initial")

        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")
        await d.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        soul_section = user_msg.split("## Current SOUL.md")[1].split("## Current USER.md")[0]
        user_section = user_msg.split("## Current USER.md")[1]
        assert "\u2190" not in soul_section
        assert "\u2190" not in user_section


# ============================================================================
# AutoCompact tests
# ============================================================================


class TestAutoCompact:
    def test_disabled_by_default(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=0)
        # When ttl=0, _is_expired always returns False (effectively disabled)
        assert ac._ttl == 0
        assert not ac._is_expired(datetime.now() - timedelta(days=365))

    def test_enabled_with_positive_ttl(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=15)
        assert ac._ttl == 15
        assert ac._is_expired(datetime.now() - timedelta(minutes=20))

    def test_is_expired_true(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=15)
        ts = datetime.now() - timedelta(minutes=15)
        assert ac._is_expired(ts)

    def test_is_expired_false(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=15)
        ts = datetime.now() - timedelta(minutes=14)
        assert not ac._is_expired(ts)

    def test_is_expired_string(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=15)
        ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        assert ac._is_expired(ts)

    def test_is_expired_none(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=15)
        assert not ac._is_expired(None)

    def test_format_summary(self, store, consolidator):
        ac = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=15)
        last_active = datetime.now() - timedelta(minutes=30)
        summary = ac._format_summary("Test summary.", last_active)
        assert "Inactive for" in summary
        assert "Test summary." in summary

    def test_check_expired_skips_when_disabled(self, store, consolidator):
        sessions = MagicMock()
        sessions.list_sessions.return_value = []
        ac = AutoCompact(sessions=sessions, consolidator=consolidator, session_ttl_minutes=0)
        schedule_bg = MagicMock()
        ac.check_expired(schedule_bg)
        schedule_bg.assert_not_called()

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
        sessions.list_sessions.return_value = [
            {"key": "test:key", "updated_at": datetime.now()}
        ]

        ac = AutoCompact(sessions=sessions, consolidator=consolidator, session_ttl_minutes=15)
        await ac._archive("test:key")
        sessions.save.assert_called()

    def test_prepare_session_no_summary(self, store, consolidator):
        sessions = MagicMock()
        session = MagicMock()
        session.updated_at = datetime.now()
        session.metadata = {}
        ac = AutoCompact(sessions=sessions, consolidator=consolidator, session_ttl_minutes=15)
        new_sess, summary = ac.prepare_session(session, "test:key")
        assert summary is None

    def test_enabled_property(self, store, consolidator):
        """Verify AutoCompact behavior is controlled by session_ttl_minutes."""
        # ttl=0 means effectively disabled: _is_expired never returns True
        ac1 = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=0)
        assert ac1._ttl == 0
        assert not ac1._is_expired(datetime.now() - timedelta(days=365))
        # ttl>0 means enabled: expiry detection works
        ac2 = AutoCompact(sessions=MagicMock(), consolidator=consolidator, session_ttl_minutes=30)
        assert ac2._ttl == 30
        assert ac2._is_expired(datetime.now() - timedelta(minutes=35))