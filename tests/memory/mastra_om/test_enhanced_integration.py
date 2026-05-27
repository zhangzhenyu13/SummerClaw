"""Enhanced integration tests for MastraOM with all three improvements.

End-to-end tests covering:
- Enhanced Observer → groups wrapping → context with retrieval instructions
- Observer → groups → Reflector → groups reconciliation
- Async buffering → activation → merged observations
- Full pipeline: buffering + groups + enhanced prompts
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.memory.mastra_om_memory.groups import (
    parse_observation_groups,
    strip_observation_groups,
    reconcile_observation_groups_from_reflection,
    OBSERVATION_RETRIEVAL_INSTRUCTIONS,
    wrap_in_observation_group,
    build_message_range,
)
from summerclaw.memory.mastra_om_memory.observer import (
    build_observer_system_prompt,
    build_observer_prompt,
    parse_observer_output,
    OBSERVATION_CONTEXT_INSTRUCTIONS,
)
from summerclaw.memory.mastra_om_memory.reflector import (
    build_reflector_system_prompt,
    parse_reflector_output,
)
from summerclaw.memory.mastra_om_memory.buffering import (
    BufferingCoordinator,
    BufferingStore,
    BufferedChunk,
    async_buffer_observe,
    activate_buffered_observations,
)
from summerclaw.memory.mastra_om_memory.consolidator import MastraOMConsolidator


# ── Mock helpers ───────────────────────────────────────────────────────────

def make_mock_provider():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock()
    return provider


def make_mock_store(observations: str = ""):
    store = MagicMock()
    store.read_observations.return_value = observations
    store.append_observations = MagicMock()
    store.append_history = MagicMock(return_value=1)
    store._next_cursor = MagicMock(return_value=1)
    store.replace_observations = MagicMock()
    store.increment_generation = MagicMock()
    return store


def make_mock_sessions():
    sessions = MagicMock()
    sessions.list_sessions.return_value = []
    sessions.save = MagicMock()
    return sessions


# ── Enhanced Observer → Groups → Context ───────────────────────────────────

class TestObserverGroupsContextPipeline:
    """End-to-end: enhanced Observer output → groups wrapping → context with retrieval."""

    @pytest.mark.asyncio
    async def test_observe_and_store_wraps_with_groups(self):
        """observe_and_store should wrap observations with group tags when messages have IDs."""
        provider = make_mock_provider()
        raw_observer_output = (
            "<observations>\n"
            "Date: May 9, 2026\n"
            "* 🔴 (14:30) User requested dark mode support\n"
            "* 🟡 (14:32) Agent opened ThemeProvider.tsx\n"
            "</observations>\n"
            "<current-task>Implement dark mode toggle</current-task>\n"
            "<suggested-response>Dark mode is ready for testing</suggested-response>"
        )
        provider.chat_with_retry.return_value = MagicMock(
            content=raw_observer_output,
            finish_reason="stop",
        )

        store = make_mock_store()
        sessions = make_mock_sessions()

        consolidator = MastraOMConsolidator(
            store=store,
            provider=provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=128000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )

        messages = [
            {"id": "msg_001", "role": "user", "content": "Can you add dark mode?"},
            {"id": "msg_002", "role": "assistant", "content": "Sure, let me check the theme setup."},
        ]

        result = await consolidator.observe_and_store(messages)

        # Verify observations were wrapped with group tags
        assert result is not None
        assert '<observation-group' in result
        assert 'range="msg_001:msg_002"' in result
        assert 'id="' in result

        # Verify group can be parsed back
        groups = parse_observation_groups(result)
        assert len(groups) == 1
        assert groups[0].range == "msg_001:msg_002"
        assert "dark mode" in groups[0].content

    @pytest.mark.asyncio
    async def test_context_includes_retrieval_instructions_when_groups_present(self):
        """When observations contain groups, context should include retrieval instructions."""
        provider = make_mock_provider()
        raw_output = (
            "<observations>\n"
            "* 🔴 User prefers dark theme\n"
            "</observations>"
        )
        provider.chat_with_retry.return_value = MagicMock(
            content=raw_output, finish_reason="stop",
        )

        store = make_mock_store()
        # Pre-populate observations with groups
        store.read_observations.return_value = (
            '<observation-group id="abc" range="msg_001:msg_010">\n'
            '* 🔴 User prefers dark theme\n'
            '</observation-group>'
        )

        sessions = make_mock_sessions()
        consolidator = MastraOMConsolidator(
            store=store, provider=provider, model="test-model",
            sessions=sessions, context_window_tokens=128000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )

        context_msgs = consolidator.build_context_system_messages()

        assert len(context_msgs) == 1
        # Should contain retrieval instructions (auto-recall description)
        assert 'Observation Memory' in context_msgs[0]
        assert 'automatically recalls' in context_msgs[0]

    def test_context_without_groups_no_retrieval_instructions(self):
        """Without groups, context should NOT include retrieval instructions."""
        store = make_mock_store()
        stores = make_mock_sessions()

        # Only plain observation text, no groups
        store.read_observations.return_value = (
            "Date: May 9, 2026\n"
            "* 🔴 User prefers dark theme\n"
        )

        consolidator = MastraOMConsolidator(
            store=store, provider=make_mock_provider(), model="test-model",
            sessions=stores, context_window_tokens=128000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )

        context_msgs = consolidator.build_context_system_messages()
        assert len(context_msgs) == 1
        assert 'Recall — looking up source messages' not in context_msgs[0]


# ── Observer → Groups → Reflector → Reconciliation ────────────────────

class TestObserverGroupsReflectorReconciliation:
    """End-to-end: groups survive the Reflector condensation pipeline."""

    def test_reconcile_preserves_message_range_provenance(self):
        """After reflection, observation group provenance should be preserved."""
        source = (
            '<observation-group id="g1" range="msg_001:msg_010">\n'
            '* 🔴 User asked about auth system\n'
            '* 🟡 Agent browsed auth module\n'
            '</observation-group>'
        )

        # Simulated reflector output with ## Group headings
        reflected = (
            '## Group `g1`\n'
            '_range: `msg_001:msg_010`_\n\n'
            '* 🔴 User asked about auth system\n'
            '* 🟡 Agent browsed auth module (redundant calls omitted)\n'
        )

        result = reconcile_observation_groups_from_reflection(reflected, source)
        assert result is not None
        assert '<observation-group' in result
        assert 'kind="reflection"' in result

        # Verify groups can be parsed back
        groups = parse_observation_groups(result)
        assert len(groups) >= 1
        assert groups[0].kind == "reflection"

    def test_reflector_output_parsing_handles_groups(self):
        """parse_reflector_output should reconcile groups when source_observations provided."""
        source = (
            '<observation-group id="abc" range="msg_001:msg_005">\n'
            '* 🔴 Hello world\n'
            '</observation-group>'
        )

        # Simulated reflector output (plain text without groups)
        reflector_raw = (
            "<observations>\n"
            "* 🔴 User greeted the assistant\n"
            "</observations>"
        )

        result = parse_reflector_output(reflector_raw, source_observations=source)
        assert result["degenerate"] is False
        assert "observations" in result
        # Should have reconciled groups
        if result["observations"]:
            assert '<observation-group' in result["observations"]

    def test_reflector_system_prompt_includes_enhanced_observer_instructions(self):
        """Reflector should reference the enhanced Observer instructions."""
        prompt = build_reflector_system_prompt()
        assert 'PRESERVE UNUSUAL PHRASING' in prompt
        assert 'USER ASSERTIONS ARE AUTHORITATIVE' in prompt


# ── Enhanced Observer Prompt Quality ───────────────────────────────────

class TestEnhancedObserverPromptQuality:
    """Verify the enhanced Observer prompt contains all official quality improvements."""

    def test_system_prompt_has_detailed_extraction_instructions(self):
        prompt = build_observer_system_prompt()
        # All 6 details preservation categories
        assert '1. RECOMMENDATION LISTS' in prompt
        assert '2. NAMES, HANDLES, AND IDENTIFIERS' in prompt
        assert '3. CREATIVE CONTENT' in prompt
        assert '4. TECHNICAL/NUMERICAL RESULTS' in prompt
        assert '5. QUANTITIES AND COUNTS' in prompt
        assert '6. ROLE/PARTICIPATION STATEMENTS' in prompt

    def test_context_instructions_have_planned_actions(self):
        assert 'PLANNED ACTIONS' in OBSERVATION_CONTEXT_INSTRUCTIONS
        assert 'assume they completed the action' in OBSERVATION_CONTEXT_INSTRUCTIONS

    def test_context_instructions_have_system_reminders(self):
        assert 'SYSTEM REMINDERS' in OBSERVATION_CONTEXT_INSTRUCTIONS
        assert '<system-reminder>' in OBSERVATION_CONTEXT_INSTRUCTIONS


# ── Buffering → Activation → Merged Observations ───────────────────────

class TestBufferingActivationPipeline:
    """End-to-end: async buffering → chunk storage → activation → merged log."""

    @pytest.mark.asyncio
    async def test_buffering_and_activation_flow(self):
        """Full flow: buffer 2 chunks, then activate into observation log."""
        coordinator = BufferingCoordinator(buffer_tokens=0.2, buffer_activation=0.5)
        buffering_store = BufferingStore()
        mock_consolidator = MagicMock()
        mock_consolidator.store = make_mock_store()
        mock_consolidator._observe_messages = AsyncMock()

        messages_1 = [
            {"id": "m1", "role": "user", "content": "Hi there"},
            {"id": "m2", "role": "assistant", "content": "Hello!"},
        ]
        messages_2 = [
            {"id": "m3", "role": "user", "content": "What's the weather?"},
            {"id": "m4", "role": "assistant", "content": "Sunny today!"},
        ]

        lock_key = coordinator.get_lock_key(thread_id="test-thread")

        # Buffer first chunk
        mock_consolidator._observe_messages.return_value = {
            "observations": "* 🔴 User greeted assistant\n", "degenerate": False,
        }
        await async_buffer_observe(
            mock_consolidator, messages_1, coordinator, lock_key, buffering_store,
        )

        # Buffer second chunk
        mock_consolidator._observe_messages.return_value = {
            "observations": "* 🔴 User asked about weather\n", "degenerate": False,
        }
        await async_buffer_observe(
            mock_consolidator, messages_2, coordinator, lock_key, buffering_store,
        )

        # Both chunks buffered
        chunks = buffering_store.get_chunks(lock_key)
        assert len(chunks) == 2

        # Activate
        result = await activate_buffered_observations(
            coordinator, lock_key, buffering_store, mock_consolidator,
        )

        assert result is not None
        assert "greeted assistant" in result
        assert "weather" in result

        # Chunks cleared after activation
        assert buffering_store.get_chunks(lock_key) == []

        # Observations appended to store
        mock_consolidator.store.append_observations.assert_called_once()
        mock_consolidator.store.append_om_ops.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffered_observations_have_group_tags(self):
        """Buffered observations should include <observation-group> tags."""
        coordinator = BufferingCoordinator()
        buffering_store = BufferingStore()
        mock_consolidator = MagicMock()
        mock_consolidator.store = make_mock_store()
        mock_consolidator._observe_messages = AsyncMock()
        mock_consolidator._observe_messages.return_value = {
            "observations": "* 🔴 User asked a question\n", "degenerate": False,
        }

        messages = [
            {"id": "buf_001", "role": "user", "content": "Question?"},
            {"id": "buf_002", "role": "assistant", "content": "Answer."},
        ]
        lock_key = coordinator.get_lock_key(thread_id="test")

        await async_buffer_observe(
            mock_consolidator, messages, coordinator, lock_key, buffering_store,
        )

        chunks = buffering_store.get_chunks(lock_key)
        assert len(chunks) == 1
        assert '<observation-group' in chunks[0].observations
        assert 'range="buf_001:buf_002"' in chunks[0].observations

        # Message IDs should be tracked in chunk metadata
        assert "buf_001" in chunks[0].message_ids
        assert "buf_002" in chunks[0].message_ids


# ── Full Pipeline Integration ──────────────────────────────────────────

class TestFullPipelineIntegration:
    """Complete end-to-end: enhanced prompts + groups + buffering + activation."""

    def test_all_components_coexist(self):
        """Verify all three improvements can be imported and used together."""
        # Enhanced Observer prompts
        prompt = build_observer_system_prompt(
            instruction="Test instruction",
            include_thread_title=True,
        )
        assert 'THREAD ATTRIBUTION' in prompt
        assert '<thread-title>' in prompt
        assert 'Test instruction' in prompt

        # Groups
        gid = str(uuid.uuid4())[:8]
        wrapped = wrap_in_observation_group(
            observations="* 🔴 Integrated test",
            range_spec="msg_001:msg_999",
            group_id=gid,
        )
        groups = parse_observation_groups(wrapped)
        assert len(groups) == 1
        assert groups[0].id == gid

        # Stripped context
        stripped = strip_observation_groups(wrapped)
        assert '* 🔴 Integrated test' in stripped
        assert '<observation-group' not in stripped

        # Buffering
        coordinator = BufferingCoordinator(buffer_tokens=0.2)
        assert coordinator.is_async_observation_enabled()

        # Retrieval instructions exist
        assert 'recall' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_observer_prompt_with_enhanced_instructions(self):
        """The task prompt builder works with enhanced system prompts."""
        from summerclaw.memory.mastra_om_memory.observer import build_observer_task_prompt

        task_prompt = build_observer_task_prompt(
            existing_observations="* 🔴 Previous observation",
            prior_current_task="Working on feature X",
            prior_suggested_response="Continue implementation",
        )
        assert "Previous Observations" in task_prompt
        assert "Prior Thread Metadata" in task_prompt
        assert "Your Task" in task_prompt

    def test_build_message_range_with_realistic_ids(self):
        """build_message_range handles realistic message IDs."""
        messages = [
            {"id": "msg_a1b2c3d4", "role": "user", "content": "Hello"},
            {"id": "msg_e5f6g7h8", "role": "assistant", "content": "Hi"},
        ]
        range_spec = build_message_range(messages)
        assert range_spec == "msg_a1b2c3d4:msg_e5f6g7h8"

    @pytest.mark.asyncio
    async def test_multiple_threads_buffering_isolation(self):
        """Different threads should have independent buffering state."""
        coordinator = BufferingCoordinator()
        buffering_store = BufferingStore()
        mock = MagicMock()
        mock.store = make_mock_store()
        mock._observe_messages = AsyncMock()
        mock._observe_messages.return_value = {
            "observations": "* 🔴 Test\n", "degenerate": False,
        }

        msgs = [{"id": "m1", "role": "user", "content": "Hello"}]

        # Thread A
        lock_a = coordinator.get_lock_key(thread_id="thread-A")
        await async_buffer_observe(mock, msgs, coordinator, lock_a, buffering_store)
        assert len(buffering_store.get_chunks(lock_a)) == 1

        # Thread B
        lock_b = coordinator.get_lock_key(thread_id="thread-B")
        await async_buffer_observe(mock, msgs, coordinator, lock_b, buffering_store)
        assert len(buffering_store.get_chunks(lock_b)) == 1

        # Independent
        assert lock_a != lock_b
        assert len(buffering_store.get_chunks(lock_a)) == 1
        assert len(buffering_store.get_chunks(lock_b)) == 1

    @pytest.mark.asyncio
    async def test_disabled_buffering_no_op(self):
        """When buffering is disabled, no background operations occur."""
        coordinator = BufferingCoordinator(buffer_tokens=None)
        assert coordinator.is_async_observation_enabled() is False

        lock_key = coordinator.get_lock_key(thread_id="test")
        assert coordinator.should_trigger_async_observation(
            current_tokens=6000, lock_key=lock_key,
        ) is False
