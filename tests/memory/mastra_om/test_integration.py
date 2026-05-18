"""Integration tests for MastraOM — full pipeline, algorithm registration, Hermes/Dream modes."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from nanobot.memory.mastra_om_memory import MastraOMMemoryAlgorithm
from nanobot.memory.mastra_om_memory.store import MastraOMStore
from nanobot.memory.mastra_om_memory.consolidator import MastraOMConsolidator
from nanobot.memory.mastra_om_memory.dream import MastraOMDream
from nanobot.memory.mastra_om_memory.auto_compact import MastraOMAutoCompact
from nanobot.memory.mastra_om_memory.observer import (
    build_observer_system_prompt,
    format_messages_for_observer,
    OBSERVER_SYSTEM_PROMPT,
    OBSERVATION_CONTINUATION_HINT,
    OBSERVATION_CONTEXT_PROMPT,
    OBSERVATION_CONTEXT_INSTRUCTIONS,
)
from nanobot.memory.mastra_om_memory.reflector import (
    build_reflector_system_prompt,
    REFLECTOR_SYSTEM_PROMPT,
    COMPRESSION_GUIDANCE,
)
from nanobot.memory.base import MemoryAlgorithm, MemoryComponents


# ------------------------------------------------------------------
# Algorithm registration
# ------------------------------------------------------------------


class TestAlgorithmRegistration:
    """Verify the MastraOMMemoryAlgorithm integrates with the registry."""

    def test_algorithm_name(self):
        algo = MastraOMMemoryAlgorithm()
        assert algo.name == "mastra_om_memory"
        assert isinstance(algo, MemoryAlgorithm)

    def test_build_returns_memory_components(self, tmp_path):
        algo = MastraOMMemoryAlgorithm()
        provider = MagicMock()
        sessions = MagicMock()

        components = algo.build(
            workspace=tmp_path,
            provider=provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=128_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=30,
            max_batch_size=10,
            max_iterations=8,
            max_tool_result_chars=16_000,
            annotate_line_ages=True,
        )

        assert isinstance(components, MemoryComponents)
        assert isinstance(components.store, MastraOMStore)
        assert isinstance(components.consolidator, MastraOMConsolidator)
        assert isinstance(components.dream, MastraOMDream)
        assert isinstance(components.auto_compact, MastraOMAutoCompact)

    def test_build_without_auto_compact(self, tmp_path):
        algo = MastraOMMemoryAlgorithm()
        provider = MagicMock()
        sessions = MagicMock()

        components = algo.build(
            workspace=tmp_path,
            provider=provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=128_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=0,  # disabled
            max_batch_size=10,
            max_iterations=8,
            max_tool_result_chars=16_000,
            annotate_line_ages=True,
        )

        assert components.auto_compact is None


# ------------------------------------------------------------------
# Store component in isolation
# ------------------------------------------------------------------


class TestStoreIntegration:
    """Verify the Store works correctly within the MemoryComponents workflow."""

    def test_store_creates_expected_files(self, tmp_path):
        store = MastraOMStore(tmp_path)
        assert store.observations_file.parent == tmp_path / "memory"
        assert store.history_file == tmp_path / "memory" / "history.jsonl"
        assert store.soul_file == tmp_path / "SOUL.md"
        assert store.user_file == tmp_path / "USER.md"

    def test_store_read_write_cycle(self, tmp_path):
        store = MastraOMStore(tmp_path)

        # Write initial observations
        store.write_observations("Initial obs")
        assert store.read_observations() == "Initial obs"

        # Append more
        store.append_observations("New obs")
        assert "Initial obs" in store.read_observations()
        assert "New obs" in store.read_observations()

        # Replace (reflector)
        store.replace_observations("Condensed version")
        assert store.read_observations() == "Condensed version"

    def test_store_memory_context(self, tmp_path):
        store = MastraOMStore(tmp_path)
        store.write_observations("Date: May 9\n* 🔴 User likes Python")
        store.write_memory("# Long-term\n- Project X is important")

        context = store.get_memory_context()
        assert "Python" in context
        assert "Project X" in context
        assert "Past Conversation Records" in context


# ------------------------------------------------------------------
# Consolidator component with real store
# ------------------------------------------------------------------


class TestConsolidatorIntegration:
    """Verify Consolidator works with real Store."""

    async def test_build_context_messages_with_real_store(self, tmp_path):
        store = MastraOMStore(tmp_path)
        mock_provider = MagicMock()
        consolidator = MastraOMConsolidator(
            store=store,
            provider=mock_provider,
            model="test",
            sessions=MagicMock(),
            context_window_tokens=128_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
        )

        # Empty observations → continuation hint only
        msgs = consolidator.build_context_system_messages()
        assert len(msgs) == 1
        assert "system-reminder" in msgs[0]

        # With observations → full context
        store.write_observations("Date: May 9\n* 🔴 User test")
        msgs = consolidator.build_context_system_messages()
        assert len(msgs) == 1
        assert "User test" in msgs[0]
        assert "observations" in msgs[0]


# ------------------------------------------------------------------
# Dream component with real store
# ------------------------------------------------------------------


class TestDreamIntegration:
    """Verify Dream handles real Store data."""

    async def test_dream_reads_current_files(self, tmp_path):
        store = MastraOMStore(tmp_path)
        store.write_soul("Soul content")
        store.write_user("User content")
        store.write_memory("Memory content")
        store.write_observations("Obs content")

        mock_provider = MagicMock()
        mock_provider.chat_with_retry = AsyncMock(return_value=MagicMock(content="[SKIP]"))
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=MagicMock(
            stop_reason="completed", tool_events=[],
        ))

        dream = MastraOMDream(
            store=store,
            provider=mock_provider,
            model="test",
            max_batch_size=5,
        )
        dream._runner = mock_runner

        # No history → noop
        result = await dream.run()
        assert result is False

        # With history → runs
        store.append_history("Test event")
        result = await dream.run()
        assert result is True

    async def test_dream_skips_when_no_history(self, tmp_path):
        store = MastraOMStore(tmp_path)
        mock_provider = MagicMock()

        dream = MastraOMDream(
            store=store,
            provider=mock_provider,
            model="test",
        )

        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()


# ------------------------------------------------------------------
# Full pipeline: algorithm → build → components
# ------------------------------------------------------------------


class TestFullBuild:
    """Verify the complete build pipeline creates all expected components."""

    def test_full_build_with_defaults(self, tmp_path):
        algo = MastraOMMemoryAlgorithm()
        provider = MagicMock()
        sessions = MagicMock()

        components = algo.build(
            workspace=tmp_path,
            provider=provider,
            model="gpt-5-mini",
            sessions=sessions,
            context_window_tokens=200_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=8192,
            session_ttl_minutes=60,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=32_000,
            annotate_line_ages=True,
        )

        # Verify component types
        assert isinstance(components.store, MastraOMStore)
        assert isinstance(components.consolidator, MastraOMConsolidator)
        assert isinstance(components.dream, MastraOMDream)
        assert isinstance(components.auto_compact, MastraOMAutoCompact)

        # Verify store workspace
        assert components.store.workspace == tmp_path

        # Verify consolidator thresholds (OM defaults)
        assert components.consolidator.message_tokens_threshold == 30_000
        assert components.consolidator.observation_tokens_threshold == 40_000

    def test_store_files_exist_after_build(self, tmp_path):
        """Store initialization should ensure memory directory exists."""
        algo = MastraOMMemoryAlgorithm()
        provider = MagicMock()
        sessions = MagicMock()

        components = algo.build(
            workspace=tmp_path,
            provider=provider,
            model="test",
            sessions=sessions,
            context_window_tokens=128_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=0,
            max_batch_size=10,
            max_iterations=8,
            max_tool_result_chars=16_000,
            annotate_line_ages=True,
        )

        assert (tmp_path / "memory").exists()
        assert isinstance(components.store, MastraOMStore)


# ------------------------------------------------------------------
# Observer/Reflector prompt consistency
# ------------------------------------------------------------------


class TestPromptConsistency:
    """Verify Observer and Reflector prompts are consistent with each other."""

    def test_observer_prompt_references_observations(self):
        prompt = build_observer_system_prompt()
        assert "observations" in prompt.lower()
        assert "OUTPUT FORMAT" in prompt

    def test_reflector_prompt_references_observer_instructions(self):
        prompt = build_reflector_system_prompt()
        assert "observational-memory-instruction" in prompt
        assert "CRITICAL: DISTINGUISH" in prompt

    def test_observer_continuation_hints_are_consistent(self):
        assert len(OBSERVATION_CONTINUATION_HINT) > 0
        assert len(OBSERVATION_CONTEXT_PROMPT) > 0
        assert "IMPORTANT" in OBSERVATION_CONTEXT_INSTRUCTIONS

    def test_compression_levels_are_progressive(self):
        """Higher compression levels should have more aggressive guidance."""
        for i in range(1, 5):
            assert COMPRESSION_GUIDANCE[i] != ""
        # Each level should be different
        for i in range(1, 4):
            assert COMPRESSION_GUIDANCE[i] != COMPRESSION_GUIDANCE[i + 1]

    def test_default_system_prompts_are_non_empty(self):
        assert len(OBSERVER_SYSTEM_PROMPT) > 100
        assert len(REFLECTOR_SYSTEM_PROMPT) > 100


# ------------------------------------------------------------------
# Hermes mode integration
# ------------------------------------------------------------------


class TestHermesIntegration:
    """Verify Hermes extract_and_store works within the full pipeline."""

    async def test_extract_and_store_with_real_store(self, tmp_path):
        store = MastraOMStore(tmp_path)
        mock_provider = MagicMock()
        mock_provider.chat_with_retry = AsyncMock(return_value=MagicMock(
            content="""<observations>
Date: May 9
* 🔴 User prefers TypeScript
* 🟡 User is building a REST API
</observations>""",
            finish_reason="stop",
        ))

        consolidator = MastraOMConsolidator(
            store=store,
            provider=mock_provider,
            model="test",
            sessions=MagicMock(),
            context_window_tokens=128_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
        )

        messages = [
            {"role": "user", "content": "I use TypeScript for all my projects", "timestamp": "2025-05-09 10:00"},
            {"role": "user", "content": "Building a REST API now", "timestamp": "2025-05-09 10:01"},
        ]

        facts = await consolidator.extract_and_store(messages)
        assert len(facts) >= 1
        assert any("TypeScript" in f for f in facts)

        # Observations should be stored
        obs = store.read_observations()
        assert "TypeScript" in obs

    async def test_extract_and_store_empty_on_failure(self, tmp_path):
        store = MastraOMStore(tmp_path)
        mock_provider = MagicMock()
        mock_provider.chat_with_retry = AsyncMock(side_effect=Exception("API error"))

        consolidator = MastraOMConsolidator(
            store=store,
            provider=mock_provider,
            model="test",
            sessions=MagicMock(),
            context_window_tokens=128_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
        )

        messages = [{"role": "user", "content": "test"}]
        facts = await consolidator.extract_and_store(messages)
        assert facts == []


# ------------------------------------------------------------------
# Message formatting integration
# ------------------------------------------------------------------


class TestMessageFormattingIntegration:
    """Verify message formatting works with typical message patterns."""

    def test_mixed_role_messages(self):
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2025-05-09 10:00"},
            {"role": "assistant", "content": "Hi!", "timestamp": "2025-05-09 10:01"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [{"function": {"name": "read_file"}}, {"function": {"name": "grep"}}],
                "timestamp": "2025-05-09 10:02",
            },
            {
                "role": "tool",
                "content": "File content",
                "name": "read_file",
                "timestamp": "2025-05-09 10:03",
            },
        ]
        result = format_messages_for_observer(messages)
        assert "User" in result
        assert "Assistant" in result
        assert "Tool Call" in result
        assert "Tool Result" in result
        assert "read_file" in result
        assert "grep" in result

    def test_cross_date_messages(self):
        messages = [
            {"role": "user", "content": "Day 1", "timestamp": "2025-05-09 10:00"},
            {"role": "user", "content": "Day 2", "timestamp": "2025-05-10 10:00"},
        ]
        result = format_messages_for_observer(messages)
        assert "2025-05-09" in result
        assert "2025-05-10" in result
