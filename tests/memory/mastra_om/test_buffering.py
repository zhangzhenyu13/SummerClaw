"""Tests for MastraOM Async Buffering — background pre-computation.

Comprehensive coverage of the buffering module:
- BufferingCoordinator: enable checks, trigger decisions, boundary tracking
- BufferingStore: chunk add/get/pop/clear
- async_buffer_observe: background Observer call
- activate_buffered_observations: chunk activation
- await_buffering: waiting for in-flight operations
- Integration with Consolidator
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.memory.mastra_om_memory.buffering import (
    BufferingCoordinator,
    BufferingStore,
    BufferedChunk,
    async_buffer_observe,
    activate_buffered_observations,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def coordinator():
    """Default BufferingCoordinator with async observation enabled."""
    return BufferingCoordinator(buffer_tokens=0.2, buffer_activation=0.5)


@pytest.fixture
def buffering_store():
    """Empty BufferingStore."""
    return BufferingStore()


@pytest.fixture
def mock_consolidator():
    """Mock MastraOMConsolidator for buffering tests."""
    mock = MagicMock()
    mock.store = MagicMock()
    mock.store.append_observations = MagicMock()
    mock.store.append_history = MagicMock()
    mock._observe_messages = AsyncMock()
    return mock


# ── BufferingCoordinator: enable checks ────────────────────────────────────

class TestBufferingCoordinatorEnableChecks:
    """Tests for is_async_observation_enabled and is_async_reflection_enabled."""

    def test_observation_enabled_by_default(self, coordinator):
        assert coordinator.is_async_observation_enabled() is True

    def test_observation_disabled_when_none(self):
        c = BufferingCoordinator(buffer_tokens=None)
        assert c.is_async_observation_enabled() is False

    def test_observation_disabled_when_zero(self):
        c = BufferingCoordinator(buffer_tokens=0)
        assert c.is_async_observation_enabled() is False

    def test_reflection_enabled_by_default(self, coordinator):
        assert coordinator.is_async_reflection_enabled() is True

    def test_reflection_disabled_when_none(self):
        c = BufferingCoordinator(buffer_activation=None)
        assert c.is_async_reflection_enabled() is False


# ── BufferingCoordinator: lock keys ────────────────────────────────────────

class TestLockKeys:
    """Tests for lock key generation."""

    def test_thread_scope_key(self, coordinator):
        key = coordinator.get_lock_key(thread_id="thread_123")
        assert key == "thread:thread_123"

    def test_thread_scope_unknown(self, coordinator):
        key = coordinator.get_lock_key(thread_id=None)
        assert key == "thread:unknown"

    def test_resource_scope_key(self):
        c = BufferingCoordinator(scope="resource")
        key = c.get_lock_key(thread_id="t1", resource_id="res_456")
        assert key == "resource:res_456"

    def test_resource_scope_falls_back_to_thread(self):
        c = BufferingCoordinator(scope="resource")
        key = c.get_lock_key(thread_id="t1", resource_id=None)
        assert key == "thread:t1"


# ── BufferingCoordinator: trigger decision ─────────────────────────────────

class TestShouldTriggerAsyncObservation:
    """Tests for should_trigger_async_observation interval detection."""

    def test_triggers_when_crossing_first_boundary(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        # 6000 tokens > 0, so interval 0 → 1 (with buffer_tokens=0.2, effective=0.2*30000=6000)
        # But wait: effective_buffer_tokens = 0.2 * current_tokens? No...
        # buffer_tokens is the fraction per interval relative to message_tokens_threshold
        # Actually looking at the official code more carefully:
        # bufferTokens is a number (e.g., 0.2 = 6000 if messageTokens==30000)
        # Wait, it's a fraction of message_tokens_threshold. 0.2 * 30000 = 6000.
        # But we pass buffer_tokens as 0.2, not as a raw number...
        # Let me re-read the should_trigger_async_observation code.
        # effective_buffer_tokens = buffer_tokens (which is 0.2 in this case, not 6000)
        # Hmm, that seems wrong. buffer_tokens should be the actual token count per interval.
        # In the official code, bufferTokens = this.observationConfig.bufferTokens!
        # And bufferTokens is 0.2 in constants. So it's a raw number, not a fraction.
        # But then 0.2 * 30000 = 6000 as an interval? No, bufferTokens=0.2 means 
        # the interval size is 0.2 (in tokens?) ... 

        # Looking at the official code again:
        # const bufferTokens = this.observationConfig.bufferTokens!;
        # bufferTokens = 0.2 (from constants)
        # effectiveBufferTokens = currentTokens >= rampPoint ? bufferTokens / 2 : bufferTokens;
        # Hmm, with 0.2 as the number, the intervals would be every 0.2 tokens, which makes no sense.
        # 
        # Wait, in Mastra, bufferTokens in constants.ts = 0.2 as number | undefined
        # And the comment says "Buffer every 20% of messageTokens"
        # So it IS a fraction. But then the shouldTrigger check uses it as a raw number...
        # That seems like a design choice in the official code. Let me look again.
        #
        # Actually looking more carefully at official code:
        # const bufferTokens = this.observationConfig.bufferTokens!;
        # So bufferTokens = 0.2 (from constants)
        # Then effectiveBufferTokens = currentTokens >= rampPoint ? bufferTokens / 2 : bufferTokens
        # = 0.1 or 0.2
        # currentInterval = Math.floor(currentTokens / effectiveBufferTokens)
        # = Math.floor(currentTokens / 0.2) = Math.floor(currentTokens * 5)
        # So this triggers very frequently. This seems like the buffering triggers at 
        # every 0.2-token boundary, which would be at every token boundary effectively.
        #
        # Hmm, actually I think in the official code, bufferTokens is NOT a fraction of
        # messageTokens but rather is used as the interval size directly. But 0.2 tokens 
        # as interval is tiny. Maybe in the official code, this is always pre-calculated 
        # to the actual token count?
        #
        # Let me re-read: "Buffer every 20% of messageTokens" = 0.2 * 30000 = 6000 tokens
        # So for the official code, bufferTokens=0.2 actually represents 6000 tokens interval.
        # But the code treats it as a raw number. So maybe 0.2 should be pre-multiplied...
        # 
        # Actually no. Looking at the Mastra code flow:
        # 1. Buffer every 20% of messageTokens means trigger at 6000, 12000, 18000, 24000 tokens
        # 2. bufferTokens = 0.2, messageTokensThreshold = 30000
        # 3. In should_trigger, effectiveBufferTokens = 0.2 (or 0.1 near ramp)
        # 4. currentInterval = floor(currentTokens / 0.2) = floor(6000/0.2) = 30000
        # 5. That gives HUGE interval numbers, effectively triggering on every token change
        #
        # This CAN'T be right. Let me look at this differently.
        #
        # Hmm, actually I think in the official code they compute bufferTokens differently.
        # Let me check: in the code flow, when buffer is launched, maybe bufferTokens is 
        # computed as message_tokens_threshold * bufferTokens?
        # 
        # Looking at the mastra code on line 126 of buffering-coordinator.ts:
        # const bufferTokens = this.observationConfig.bufferTokens!;
        # 
        # And in constants.ts, bufferTokens = 0.2
        # The comment says "Buffer every 20% of messageTokens"
        #
        # I think the issue is that in my Python adaptation, I should be multiplying 
        # buffer_tokens by message_tokens_threshold. Let me fix this.
        #
        # Actually wait. Let me re-read the Mastra code more carefully:
        # bufferTokens: 0.2 as number | undefined, // Buffer every 20% of messageTokens
        # 
        # It says 0.2 as a number. But the comment says it's a fraction. So maybe 
        # in the shouldTrigger code, the 0.2 IS used as a fraction but it's pre-multiplied 
        # before being stored?
        #
        # Or maybe the interval is calculated differently. Let me trace through:
        # - messageTokens = 30000
        # - bufferTokens = 0.2 (from config)
        # Then in shouldTriggerAsyncObservation:
        # - currentTokens = e.g., 6000
        # - effectiveBufferTokens = 0.2 (not near ramp)
        # - currentInterval = floor(6000 / 0.2) = 30000
        # - lastInterval = 0 (initially)
        # - shouldTrigger = True (30000 > 0)
        #
        # Then markBufferedBoundary(token_count=0.2 * estimated_tokens?) - no, it records the 
        # current token count. So lastInterval = floor(6000 / 0.2) = 30000
        #
        # Then at next check with 12000 tokens:
        # currentInterval = floor(12000 / 0.2) = 60000
        # lastInterval = 30000
        # shouldTrigger = True (60000 > 30000)
        #
        # This works! The boundary detection IS correct. The issue is that the buffer_tokens is 
        # treated as a token count, but 0.2 is a tiny number. However, since we're checking 
        # floor(tokens / buffer_tokens), the interval count grows precisely as tokens grow.
        # With buffer_tokens=0.2, the interval number grows by 1 for every 0.2 tokens, which 
        # means it changes very frequently and the trigger condition would fire almost constantly.
        #
        # I think the issue is clearer now. In my implementation, I should multiply buffer_tokens 
        # by message_tokens_threshold first. So buffer_tokens = 0.2 * 30000 = 6000.
        #
        # Then:
        # - currentTokens = 6000
        # - effectiveBufferTokens = 6000
        # - currentInterval = floor(6000 / 6000) = 1
        # - lastInterval = 0
        # - shouldTrigger = True
        #
        # - currentTokens = 12000
        # - effectiveBufferTokens = 6000
        # - currentInterval = floor(12000 / 6000) = 2
        # - lastInterval = 1
        # - shouldTrigger = True
        #
        # This makes much more sense!
        
        # So the issue is my should_trigger_async_observation needs to use the 
        # message_tokens_threshold * buffer_tokens as the effective buffer size.
        # But I pass buffer_tokens=0.2 and message_tokens_threshold=30000
        # I need to compute: effective = message_tokens_threshold * buffer_tokens = 6000
        
        result = coordinator.should_trigger_async_observation(
            current_tokens=6000,
            lock_key=lock_key,
            message_tokens_threshold=30000,
        )
        # With buffer_tokens=0.2, message_threshold=30000
        # effective = 30000 * 0.2 = 6000
        # currentInterval = floor(6000/6000) = 1, lastInterval = 0 → trigger
        assert result is True

    def test_does_not_trigger_same_interval(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        # First trigger at 6000 tokens
        assert coordinator.should_trigger_async_observation(
            current_tokens=6000, lock_key=lock_key, message_tokens_threshold=30000,
        ) is True
        coordinator.mark_buffered_boundary(lock_key, 6000)

        # Same boundary, should NOT trigger again
        assert coordinator.should_trigger_async_observation(
            current_tokens=6000, lock_key=lock_key, message_tokens_threshold=30000,
        ) is False

    def test_triggers_at_next_interval(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        coordinator.mark_buffered_boundary(lock_key, 6000)
        # Next boundary: 12000 tokens → interval 2
        assert coordinator.should_trigger_async_observation(
            current_tokens=12000, lock_key=lock_key, message_tokens_threshold=30000,
        ) is True

    def test_ramp_point_uses_half_interval(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        # buffer_tokens = 0.2 * 30000 = 6000
        # ramp_point = 30000 - 6000 * 1.1 = 23400
        # Below ramp: effective = 6000; Above ramp: effective = 3000
        coordinator.mark_buffered_boundary(lock_key, 12000)

        # At 18000 (below ramp), effective = 6000, interval = floor(18000/6000) = 3
        # Last interval = floor(12000/6000) = 2
        assert coordinator.should_trigger_async_observation(
            current_tokens=18000, lock_key=lock_key, message_tokens_threshold=30000,
        ) is True

        # Now at ramp: 24000 >= 23400, effective = 3000
        coordinator.mark_buffered_boundary(lock_key, 24000)
        # interval = floor(24000/3000) = 8
        # Should trigger at 27000: interval = floor(27000/3000) = 9
        assert coordinator.should_trigger_async_observation(
            current_tokens=27000, lock_key=lock_key, message_tokens_threshold=30000,
        ) is True

    def test_no_trigger_when_disabled(self):
        c = BufferingCoordinator(buffer_tokens=None)
        assert c.should_trigger_async_observation(6000, "key") is False

    def test_no_trigger_when_buffering_in_progress(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        buf_key = coordinator._obs_buf_key(lock_key)
        # Simulate in-flight operation
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        coordinator._async_buffering_ops[buf_key] = future
        try:
            result = coordinator.should_trigger_async_observation(6000, lock_key)
            assert result is False
        finally:
            coordinator._async_buffering_ops.pop(buf_key, None)
            loop.close()


# ── BufferingStore ─────────────────────────────────────────────────────────

class TestBufferingStore:
    """Tests for BufferingStore chunk management."""

    def test_add_and_get_chunks(self, buffering_store):
        chunk = BufferedChunk(cycle_id="c1", observations="* 🔴 Test", token_count=100)
        buffering_store.add_chunk("key1", chunk)
        chunks = buffering_store.get_chunks("key1")
        assert len(chunks) == 1
        assert chunks[0].cycle_id == "c1"

    def test_get_returns_empty_for_unknown_key(self, buffering_store):
        assert buffering_store.get_chunks("nonexistent") == []

    def test_pop_removes_chunks(self, buffering_store):
        chunk = BufferedChunk(cycle_id="c1", observations="test")
        buffering_store.add_chunk("key1", chunk)
        popped = buffering_store.pop_chunks("key1")
        assert len(popped) == 1
        assert buffering_store.get_chunks("key1") == []

    def test_clear_removes_chunks(self, buffering_store):
        chunk = BufferedChunk(cycle_id="c1", observations="test")
        buffering_store.add_chunk("key1", chunk)
        buffering_store.clear("key1")
        assert buffering_store.get_chunks("key1") == []

    def test_multiple_chunks_same_key(self, buffering_store):
        buffering_store.add_chunk("key1", BufferedChunk(cycle_id="c1", observations="a"))
        buffering_store.add_chunk("key1", BufferedChunk(cycle_id="c2", observations="b"))
        assert len(buffering_store.get_chunks("key1")) == 2

    def test_separate_keys(self, buffering_store):
        buffering_store.add_chunk("key1", BufferedChunk(cycle_id="c1", observations="a"))
        buffering_store.add_chunk("key2", BufferedChunk(cycle_id="c2", observations="b"))
        assert len(buffering_store.get_chunks("key1")) == 1
        assert len(buffering_store.get_chunks("key2")) == 1


# ── BufferedChunk ──────────────────────────────────────────────────────────

class TestBufferedChunk:
    """Tests for BufferedChunk dataclass."""

    def test_default_values(self):
        chunk = BufferedChunk(cycle_id="c1", observations="test")
        assert chunk.token_count == 0
        assert chunk.message_ids == []
        assert chunk.message_tokens == 0
        assert chunk.last_observed_at == 0.0
        assert chunk.suggested_continuation is None
        assert chunk.current_task is None

    def test_full_construction(self):
        chunk = BufferedChunk(
            cycle_id="c1",
            observations="* 🔴 Test",
            token_count=50,
            message_ids=["m1", "m2"],
            message_tokens=200,
            last_observed_at=12345.6,
            suggested_continuation="Continue",
            current_task="Testing",
        )
        assert chunk.cycle_id == "c1"
        assert chunk.token_count == 50
        assert chunk.message_ids == ["m1", "m2"]
        assert chunk.suggested_continuation == "Continue"
        assert chunk.current_task == "Testing"


# ── async_buffer_observe ───────────────────────────────────────────────────

class TestAsyncBufferObserve:
    """Tests for async_buffer_observe background Observer call."""

    @pytest.mark.asyncio
    async def test_buffers_observation(self, coordinator, buffering_store, mock_consolidator):
        mock_consolidator._observe_messages.return_value = {
            "observations": "* 🔴 Test observation\n",
            "degenerate": False,
        }
        messages = [
            {"id": "msg_001", "role": "user", "content": "Hello world"},
        ]
        lock_key = coordinator.get_lock_key(thread_id="test")

        await async_buffer_observe(
            consolidator=mock_consolidator,
            messages=messages,
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
        )

        chunks = buffering_store.get_chunks(lock_key)
        assert len(chunks) == 1
        assert "* 🔴 Test observation" in chunks[0].observations
        # Should NOT modify active observations (only buffered)
        mock_consolidator.store.append_observations.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_empty_result(self, coordinator, buffering_store, mock_consolidator):
        mock_consolidator._observe_messages.return_value = {
            "observations": "",
            "degenerate": False,
        }
        messages = [{"id": "msg_001", "role": "user", "content": "Hello"}]
        lock_key = coordinator.get_lock_key(thread_id="test")

        await async_buffer_observe(
            consolidator=mock_consolidator,
            messages=messages,
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
        )

        assert buffering_store.get_chunks(lock_key) == []

    @pytest.mark.asyncio
    async def test_skips_degenerate_result(self, coordinator, buffering_store, mock_consolidator):
        mock_consolidator._observe_messages.return_value = {
            "observations": "",
            "degenerate": True,
        }
        messages = [{"id": "msg_001", "role": "user", "content": "Hello"}]
        lock_key = coordinator.get_lock_key(thread_id="test")

        await async_buffer_observe(
            consolidator=mock_consolidator,
            messages=messages,
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
        )

        assert buffering_store.get_chunks(lock_key) == []

    @pytest.mark.asyncio
    async def test_marks_boundary_after_buffering(self, coordinator, buffering_store, mock_consolidator):
        mock_consolidator._observe_messages.return_value = {
            "observations": "* 🔴 Test\n",
            "degenerate": False,
        }
        messages = [{"id": "msg_001", "role": "user", "content": "x" * 100}]
        lock_key = coordinator.get_lock_key(thread_id="test")

        await async_buffer_observe(
            consolidator=mock_consolidator,
            messages=messages,
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
        )

        buf_key = coordinator._obs_buf_key(lock_key)
        assert buf_key in coordinator._last_buffered_boundary

    @pytest.mark.asyncio
    async def test_different_threads_independent(self, coordinator, buffering_store, mock_consolidator):
        mock_consolidator._observe_messages.return_value = {
            "observations": "* 🔴 Test\n",
            "degenerate": False,
        }
        messages_t1 = [{"id": "a1", "role": "user", "content": "Thread 1"}]
        messages_t2 = [{"id": "b1", "role": "user", "content": "Thread 2"}]
        lock_key_1 = coordinator.get_lock_key(thread_id="t1")
        lock_key_2 = coordinator.get_lock_key(thread_id="t2")

        await async_buffer_observe(mock_consolidator, messages_t1, coordinator, lock_key_1, buffering_store)
        await async_buffer_observe(mock_consolidator, messages_t2, coordinator, lock_key_2, buffering_store)

        assert len(buffering_store.get_chunks(lock_key_1)) == 1
        assert len(buffering_store.get_chunks(lock_key_2)) == 1

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, coordinator, buffering_store, mock_consolidator):
        mock_consolidator._observe_messages.side_effect = RuntimeError("LLM error")
        messages = [{"id": "msg_001", "role": "user", "content": "Hello"}]
        lock_key = coordinator.get_lock_key(thread_id="test")

        # Should not raise
        await async_buffer_observe(
            consolidator=mock_consolidator,
            messages=messages,
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
        )

        # No chunks stored on error
        assert buffering_store.get_chunks(lock_key) == []


# ── activate_buffered_observations ─────────────────────────────────────────

class TestActivateBufferedObservations:
    """Tests for activate_buffered_observations."""

    @pytest.mark.asyncio
    async def test_activates_chunks_into_store(self, coordinator, buffering_store, mock_consolidator):
        chunk = BufferedChunk(
            cycle_id=str(uuid.uuid4()),
            observations="* 🔴 Buffered observation\n",
            message_ids=["m1", "m2"],
        )
        lock_key = coordinator.get_lock_key(thread_id="test")
        buffering_store.add_chunk(lock_key, chunk)

        result = await activate_buffered_observations(
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
            consolidator=mock_consolidator,
        )

        assert result is not None
        assert "* 🔴 Buffered observation" in result
        mock_consolidator.store.append_observations.assert_called_once()
        mock_consolidator.store.append_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_chunks(self, coordinator, buffering_store, mock_consolidator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        result = await activate_buffered_observations(
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
            consolidator=mock_consolidator,
        )
        assert result is None
        mock_consolidator.store.append_observations.assert_not_called()

    @pytest.mark.asyncio
    async def test_combines_multiple_chunks(self, coordinator, buffering_store, mock_consolidator):
        buffering_store.add_chunk("key1", BufferedChunk(
            cycle_id="c1", observations="* 🔴 First", message_ids=["m1"],
        ))
        buffering_store.add_chunk("key1", BufferedChunk(
            cycle_id="c2", observations="* 🔴 Second", message_ids=["m2"],
        ))
        lock_key = coordinator.get_lock_key(thread_id="test")
        # Fix key mismatch
        buffering_store._chunks = {}
        buffering_store.add_chunk(lock_key, BufferedChunk(
            cycle_id="c1", observations="* 🔴 First", message_ids=["m1"],
        ))
        buffering_store.add_chunk(lock_key, BufferedChunk(
            cycle_id="c2", observations="* 🔴 Second", message_ids=["m2"],
        ))

        result = await activate_buffered_observations(
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
            consolidator=mock_consolidator,
        )

        assert result is not None
        assert "* 🔴 First" in result
        assert "* 🔴 Second" in result

    @pytest.mark.asyncio
    async def test_cleans_up_after_activation(self, coordinator, buffering_store, mock_consolidator):
        chunk = BufferedChunk(
            cycle_id=str(uuid.uuid4()),
            observations="* 🔴 Test",
            message_ids=["m1"],
        )
        lock_key = coordinator.get_lock_key(thread_id="test")
        buffering_store.add_chunk(lock_key, chunk)

        await activate_buffered_observations(
            buffer_coordinator=coordinator,
            lock_key=lock_key,
            buffering_store=buffering_store,
            consolidator=mock_consolidator,
        )

        # Chunks should be removed
        assert buffering_store.get_chunks(lock_key) == []


# ── await_buffering ────────────────────────────────────────────────────────

class TestAwaitBuffering:
    """Tests for BufferingCoordinator.await_buffering."""

    @pytest.mark.asyncio
    async def test_returns_immediately_when_no_ops(self):
        # Should not raise or hang
        await BufferingCoordinator.await_buffering(
            thread_id="test", timeout_ms=100,
        )

    @pytest.mark.asyncio
    async def test_awaits_in_flight_ops(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        buf_key = coordinator._obs_buf_key(lock_key)

        async def slow_op():
            await asyncio.sleep(0.1)
            return None

        task = asyncio.create_task(slow_op())
        coordinator._async_buffering_ops[buf_key] = task
        try:
            await BufferingCoordinator.await_buffering(
                thread_id="test", timeout_ms=5000,
            )
        finally:
            coordinator._async_buffering_ops.pop(buf_key, None)

    @pytest.mark.asyncio
    async def test_timeout_does_not_raise(self):
        lock_key = "thread:timeout_test"
        buf_key = f"obs:{lock_key}"

        async def never_completes():
            await asyncio.sleep(10)

        task = asyncio.create_task(never_completes())
        BufferingCoordinator._async_buffering_ops[buf_key] = task
        try:
            # Should not raise despite timeout
            await BufferingCoordinator.await_buffering(
                thread_id="timeout_test", timeout_ms=100,
            )
        finally:
            BufferingCoordinator._async_buffering_ops.pop(buf_key, None)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ── Cleanup ────────────────────────────────────────────────────────────────

class TestCleanup:
    """Tests for cleanup_static_maps."""

    def test_full_cleanup_removes_all_state(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        obs_key = coordinator._obs_buf_key(lock_key)

        # Set up state
        coordinator._last_buffered_boundary[obs_key] = 6000
        coordinator._last_buffered_at_time[obs_key] = 12345.0
        coordinator.cleanup_static_maps(lock_key)

        assert obs_key not in coordinator._last_buffered_boundary
        assert obs_key not in coordinator._last_buffered_at_time

    def test_partial_cleanup_preserves_op_state(self, coordinator):
        lock_key = coordinator.get_lock_key(thread_id="test")
        obs_key = coordinator._obs_buf_key(lock_key)

        coordinator._last_buffered_boundary[obs_key] = 6000
        coordinator.cleanup_static_maps(lock_key, activated_message_ids=["m1"])

        assert obs_key not in coordinator._last_buffered_boundary
