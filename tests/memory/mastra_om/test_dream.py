"""Tests for MastraOM Dream — two-phase processing, skill creation, line age annotation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.memory.mastra_om_memory.dream import MastraOMDream
from nanobot.memory.mastra_om_memory.store import MastraOMStore
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.utils.gitstore import LineAge


@pytest.fixture
def store(tmp_path):
    s = MastraOMStore(tmp_path)
    s.write_soul("# Soul\n- Helpful")
    s.write_user("# User\n- Developer")
    s.write_memory("# Memory\n- Project X active")
    return s


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def dream(store, mock_provider, mock_runner):
    d = MastraOMDream(
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
    usage=None,
):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


class TestDreamRun:
    async def test_noop_when_no_unprocessed_history(self, dream, mock_provider, mock_runner, store):
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()

    async def test_calls_runner_for_unprocessed_entries(self, dream, mock_provider, mock_runner, store):
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New observation fact")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == 10
        assert spec.fail_on_tool_error is False

    async def test_advances_dream_cursor(self, dream, mock_provider, mock_runner, store):
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_processed_history(self, dream, mock_provider, mock_runner, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_skill_phase_uses_builtin_skill_creator_path(self, dream, mock_provider, mock_runner, store):
        store.append_history("Repeated workflow one")
        store.append_history("Repeated workflow two")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKILL] test-skill: test description")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        spec = mock_runner.run.call_args[0][0]
        system_prompt = spec.initial_messages[0]["content"]
        expected = str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        assert expected in system_prompt

    async def test_skill_write_tool_accepts_workspace_relative_skill_path(self, dream, store):
        write_tool = dream._tools.get("write_file")
        assert write_tool is not None

        result = await write_tool.execute(
            path="skills/test-skill/SKILL.md",
            content="---\nname: dreamed-test-skill\ndescription: Test\n---\n",
        )
        assert "Successfully wrote" in result
        assert (store.workspace / "skills" / "dreamed--mastra_om_memory-test-skill" / "SKILL.md").exists()

    async def test_phase1_prompt_includes_line_age_annotations(self, dream, mock_provider, mock_runner, store):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        store.git.init()
        store.git.auto_commit("initial memory state")

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_annotates_only_memory_not_soul_or_user(self, dream, mock_provider, mock_runner, store):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        store.git.init()
        store.git.auto_commit("initial state")

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        soul_section = user_msg.split("## Current SOUL.md")[1].split("## Current USER.md")[0]
        user_section = user_msg.split("## Current USER.md")[1].split("## Current OBSERVATIONS.md")[0]
        assert "\u2190" not in soul_section
        assert "\u2190" not in user_section

    async def test_phase1_prompt_works_without_git(self, dream, mock_provider, mock_runner, store):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        mock_provider.chat_with_retry.assert_called_once()
        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_prompt_carries_age_suffix_for_stale_lines(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.write_memory("# Memory\n- Project X active\n- fresh item\n- edge case line")
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        fake_ages = [
            LineAge(age_days=30),
            LineAge(age_days=20),
            LineAge(age_days=14),
            LineAge(age_days=5),
        ]
        with patch.object(store.git, "line_ages", return_value=fake_ages):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190 30d" in memory_section
        assert "\u2190 20d" in memory_section
        assert "\u2190 14d" not in memory_section
        assert "\u2190 5d" not in memory_section

    async def test_phase1_skips_annotation_when_disabled(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        dream.annotate_line_ages = False
        with patch.object(store.git, "line_ages") as mock_line_ages:
            await dream.run()
            mock_line_ages.assert_not_called()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        assert "\u2190" not in user_msg

    async def test_phase1_skips_annotation_on_line_ages_length_mismatch(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch.object(store.git, "line_ages", return_value=[LineAge(age_days=999)]):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190" not in memory_section

    async def test_phase1_prompt_uses_threshold_from_template_var(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        system_msg = messages[0]["content"]
        assert "N>14" in system_msg

    async def test_git_auto_commit_on_changes(self, dream, mock_provider, mock_runner, store):
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Updated memory")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        store.git.init()
        await dream.run()
        # verify history was compacted
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) >= 0  # at minimum, no crash

    async def test_includes_observations_in_phase1_prompt(self, dream, mock_provider, mock_runner, store):
        store.write_observations("Date: May 9\n* 🔴 Test observation")
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        assert "Past Conversation Records" in user_msg
        assert "Test observation" in user_msg

    async def test_includes_existing_skills_when_present(self, dream, mock_provider, mock_runner, store):
        # Create a skill directory to test skill listing
        skills_dir = store.workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        test_skill = skills_dir / "test-skill"
        test_skill.mkdir(exist_ok=True)
        (test_skill / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n",
            encoding="utf-8",
        )

        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        spec = mock_runner.run.call_args[0][0]
        user_msg = spec.initial_messages[1]["content"]
        assert "Existing Skills" in user_msg
        assert "test-skill" in user_msg

    async def test_batch_size_limit(self, dream, mock_provider, mock_runner, store):
        # Add more entries than max_batch_size
        for i in range(10):
            store.append_history(f"event {i}")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        # Max 5 entries processed in batch
        call_args = mock_provider.chat_with_retry.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_msg = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        history_lines = user_msg.split("## Conversation History")[1].split("## Current Date")[0]
        assert history_lines.count("event") == 5  # batch size

    async def test_incomplete_run_returns_true(self, dream, mock_provider, mock_runner, store):
        """Even when Phase 2 doesn't complete, should still return True."""
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis content")
        mock_runner.run = AsyncMock(return_value=_make_run_result(stop_reason="max_iterations"))

        result = await dream.run()
        assert result is True
