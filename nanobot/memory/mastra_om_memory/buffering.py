"""MastraOM Async Buffering — background pre-computation of observations.

Based on Mastra's buffering-coordinator.ts and async-buffer.ts.
The buffering system runs Observer calls in the background at regular
token intervals, so observations are pre-computed before they're needed.

Key concepts:
- bufferTokens: fraction of message_tokens_threshold at which to trigger
  background observation (default 0.2 = every 20% of threshold)
- bufferActivation: fraction for activating buffered reflections (default 0.5)
- Buffered chunks are stored separately until the sync observation threshold
  triggers activation, at which point they're merged in.
- Static coordinator maps track in-flight operations per thread/resource.

Integration:
    Consolidator.maybe_consolidate_by_tokens() checks BufferingCoordinator
    at each step. If a buffer boundary is crossed, it launches an async
    Observer call. On sync threshold crossing, buffered chunks are activated.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory.mastra_om_memory.consolidator import MastraOMConsolidator


# ---------------------------------------------------------------------------
# Buffered chunk data structure
# ---------------------------------------------------------------------------

@dataclass
class BufferedChunk:
    """A single buffered observation chunk, pre-computed by async Observer."""
    cycle_id: str
    observations: str
    token_count: int = 0
    message_ids: list[str] = field(default_factory=list)
    message_tokens: int = 0
    last_observed_at: float = 0.0
    suggested_continuation: str | None = None
    current_task: str | None = None


# ---------------------------------------------------------------------------
# BufferingCoordinator — static state machine
# ---------------------------------------------------------------------------

class BufferingCoordinator:
    """Static coordinator for async observation/reflection buffering.

    Static maps are shared across all MastraOMConsolidator instances because
    buffering state needs to be visible regardless of which instance checks.
    This mirrors the official Mastra implementation where BufferingCoordinator
    uses static class-level maps.

    Configuration:
        buffer_tokens: Fraction of message_tokens_threshold for buffer intervals.
            Default 0.2 = every 20%. Set None/0 to disable async observation.
        buffer_activation: Fraction of observation_tokens_threshold to activate
            buffered reflection. Default 0.5. Set None/0 to disable.
        scope: "thread" or "resource" — determines lock key granularity.
    """

    # Track in-flight async buffering operations per scope key.
    # Key format: "obs:{lockKey}" or "refl:{lockKey}"
    _async_buffering_ops: dict[str, asyncio.Task | asyncio.Future] = {}

    # Track the last token boundary at which we started buffering.
    # Key format: "obs:{lockKey}" or "refl:{lockKey}"
    _last_buffered_boundary: dict[str, int] = {}

    # Track the timestamp cursor for buffered messages.
    # Key format: "obs:{lockKey}"
    _last_buffered_at_time: dict[str, float] = {}

    # Track cycle IDs for in-flight buffered reflections.
    # Key format: "refl:{lockKey}"
    _reflection_buffer_cycle_ids: dict[str, str] = {}

    def __init__(
        self,
        buffer_tokens: float | None = 0.2,
        buffer_activation: float | None = 0.5,
        scope: str = "thread",
    ):
        self.buffer_tokens = buffer_tokens
        self.buffer_activation = buffer_activation
        self.scope = scope

    # -- key helpers ---------------------------------------------------------

    def get_lock_key(self, thread_id: str | None = None, resource_id: str | None = None) -> str:
        if self.scope == "resource" and resource_id:
            return f"resource:{resource_id}"
        return f"thread:{thread_id or 'unknown'}"

    def _obs_buf_key(self, lock_key: str) -> str:
        return f"obs:{lock_key}"

    def _refl_buf_key(self, lock_key: str) -> str:
        return f"refl:{lock_key}"

    # -- enable checks -------------------------------------------------------

    def is_async_observation_enabled(self) -> bool:
        return self.buffer_tokens is not None and self.buffer_tokens > 0

    def is_async_reflection_enabled(self) -> bool:
        return self.buffer_activation is not None and self.buffer_activation > 0

    # -- in-flight check -----------------------------------------------------

    def is_async_buffering_in_progress(self, buffer_key: str) -> bool:
        return buffer_key in self._async_buffering_ops

    # -- trigger decision ----------------------------------------------------

    def should_trigger_async_observation(
        self,
        current_tokens: int,
        lock_key: str,
        message_tokens_threshold: int | None = None,
        db_boundary: int = 0,
    ) -> bool:
        """Check if we've crossed a new buffer interval boundary.

        Uses interval-based detection: computes which buffer interval the
        current token count falls in, and compares with the last interval
        at which buffering was triggered.

        Buffer size = buffer_tokens * message_tokens_threshold.
        With defaults (0.2 * 30000 = 6000), triggers every 6000 tokens.

        Near the threshold (ramp point), the effective buffer size is halved
        to trigger more frequent pre-computation.

        Args:
            current_tokens: Current unobserved message token count.
            lock_key: Scope key for this thread/resource.
            message_tokens_threshold: The sync observation threshold.
            db_boundary: Last buffered token boundary from storage.

        Returns:
            True if a new async observation should be launched.
        """
        if not self.is_async_observation_enabled():
            return False

        buf_key = self._obs_buf_key(lock_key)
        if self.is_async_buffering_in_progress(buf_key):
            return False

        buffer_tokens_fraction = self.buffer_tokens  # type: ignore[assignment]
        assert buffer_tokens_fraction is not None  # guarded by is_async_observation_enabled

        # Compute effective buffer size: fraction * threshold
        threshold = message_tokens_threshold or 30_000
        buffer_tokens = buffer_tokens_fraction * threshold

        mem_boundary = self._last_buffered_boundary.get(buf_key, 0)
        last_boundary = max(db_boundary, mem_boundary)

        # Near the full threshold, use half-sized intervals for more frequent pre-computation
        ramp_point = threshold - buffer_tokens * 1.1
        effective_buffer_tokens = (
            buffer_tokens / 2 if current_tokens >= ramp_point else buffer_tokens
        )

        current_interval = int(current_tokens / effective_buffer_tokens)
        last_interval = int(last_boundary / effective_buffer_tokens)

        should_trigger = current_interval > last_interval

        if should_trigger:
            logger.info(
                "[OM:buffer] trigger decision: tokens={}, bufferTokens={}, effective={}, "
                "currentInterval={}, lastInterval={}, trigger=True",
                current_tokens, buffer_tokens, effective_buffer_tokens,
                current_interval, last_interval,
            )

        return should_trigger

    # -- mark boundary -------------------------------------------------------

    def mark_buffered_boundary(
        self,
        lock_key: str,
        token_count: int,
        timestamp: float | None = None,
    ) -> None:
        """Record that buffering was triggered at this token boundary."""
        buf_key = self._obs_buf_key(lock_key)
        self._last_buffered_boundary[buf_key] = token_count
        if timestamp is not None:
            self._last_buffered_at_time[buf_key] = timestamp

    # -- await buffering -----------------------------------------------------

    @classmethod
    async def await_buffering(
        cls,
        thread_id: str | None = None,
        resource_id: str | None = None,
        scope: str = "thread",
        timeout_ms: int = 30000,
    ) -> None:
        """Await any in-flight async buffering operations.

        Args:
            thread_id: Thread identifier.
            resource_id: Resource identifier.
            scope: "thread" or "resource".
            timeout_ms: Maximum wait time in milliseconds.
        """
        lock_key = (
            f"resource:{resource_id}"
            if scope == "resource" and resource_id
            else f"thread:{thread_id or 'unknown'}"
        )
        obs_key = f"obs:{lock_key}"
        refl_key = f"refl:{lock_key}"

        promises: list[asyncio.Task | asyncio.Future] = []
        obs_op = cls._async_buffering_ops.get(obs_key)
        if obs_op:
            promises.append(obs_op)
        refl_op = cls._async_buffering_ops.get(refl_key)
        if refl_op:
            promises.append(refl_op)

        if not promises:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*promises, return_exceptions=True),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.debug("[OM:buffer] await_buffering timed out after {}ms", timeout_ms)

    # -- cleanup -------------------------------------------------------------

    def cleanup_static_maps(
        self,
        lock_key: str,
        activated_message_ids: list[str] | None = None,
    ) -> None:
        """Clean up static maps for a lock key to prevent memory leaks.

        Args:
            lock_key: The scope key to clean up.
            activated_message_ids: If provided, only clear boundary/time state
                (partial cleanup after activation).
        """
        obs_key = self._obs_buf_key(lock_key)
        refl_key = self._refl_buf_key(lock_key)

        if activated_message_ids is not None:
            # Partial cleanup: only clear boundary/time state
            self._last_buffered_boundary.pop(obs_key, None)
            self._last_buffered_at_time.pop(obs_key, None)
        else:
            # Full cleanup
            self._last_buffered_at_time.pop(obs_key, None)
            self._last_buffered_boundary.pop(obs_key, None)
            self._last_buffered_boundary.pop(refl_key, None)
            self._async_buffering_ops.pop(obs_key, None)
            self._async_buffering_ops.pop(refl_key, None)
            self._reflection_buffer_cycle_ids.pop(refl_key, None)


# ---------------------------------------------------------------------------
# Buffering store extension — in-memory buffered chunks
# ---------------------------------------------------------------------------

class BufferingStore:
    """In-memory store for buffered observation chunks.

    Buffered chunks are pre-computed by async Observer calls and stored here
    until they are activated (merged into permanent observation storage).

    In the official Mastra implementation, buffered chunks are persisted to DB.
    Here we use in-memory storage for simplicity, compatible with nanobot's
    file-based storage model.
    """

    def __init__(self):
        # lock_key → list of BufferedChunk
        self._chunks: dict[str, list[BufferedChunk]] = {}

    def add_chunk(self, lock_key: str, chunk: BufferedChunk) -> None:
        """Add a buffered chunk for a lock key."""
        self._chunks.setdefault(lock_key, []).append(chunk)

    def get_chunks(self, lock_key: str) -> list[BufferedChunk]:
        """Get all buffered chunks for a lock key."""
        return list(self._chunks.get(lock_key, []))

    def pop_chunks(self, lock_key: str) -> list[BufferedChunk]:
        """Get and remove all buffered chunks for a lock key."""
        return self._chunks.pop(lock_key, [])

    def clear(self, lock_key: str) -> None:
        """Clear all buffered chunks for a lock key."""
        self._chunks.pop(lock_key, None)


# ---------------------------------------------------------------------------
# Async buffer observation function
# ---------------------------------------------------------------------------

async def async_buffer_observe(
    consolidator: "MastraOMConsolidator",
    messages: list[dict],
    buffer_coordinator: BufferingCoordinator,
    lock_key: str,
    buffering_store: BufferingStore,
) -> None:
    """Launch an async buffered observation in the background.

    This is a fire-and-forget operation: it calls the Observer on the given
    messages and stores the result as a buffered chunk. It does NOT modify
    the active observation log until activation.

    Args:
        consolidator: The Consolidator instance to use for Observer calls.
        messages: Messages to observe.
        buffer_coordinator: The coordinator tracking this operation.
        lock_key: Scope key for thread/resource.
        buffering_store: Store for buffered chunks.
    """
    import uuid
    from nanobot.memory.mastra_om_memory.groups import build_message_range, wrap_in_observation_group

    cycle_id = str(uuid.uuid4())
    started_at = time.time()

    buf_key = buffer_coordinator._obs_buf_key(lock_key)

    # Register as in-flight
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    buffer_coordinator._async_buffering_ops[buf_key] = future

    logger.info(
        "[OM:buffer] async observation started: cycle={}, msgs={}, range={}",
        cycle_id[:8], len(messages), build_message_range(messages),
    )

    try:
        result = await consolidator._observe_messages(
            messages=messages,
            existing_observations="",
        )

        if not result or result.get("degenerate") or not result.get("observations", "").strip():
            logger.debug("[OM:buffer] async buffer returned empty/degenerate, skipping")
            future.set_result(None)
            return

        observations_text = result["observations"]

        # Wrap with observation group
        message_range = build_message_range(messages)
        if message_range:
            observations_text = wrap_in_observation_group(
                observations=observations_text,
                range_spec=message_range,
            )

        # Estimate token count
        observation_tokens = len(observations_text) // 4
        message_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)

        msg_ids = [m.get("id", "") for m in messages if m.get("id")]

        chunk = BufferedChunk(
            cycle_id=cycle_id,
            observations=observations_text,
            token_count=observation_tokens,
            message_ids=msg_ids,
            message_tokens=message_tokens,
            last_observed_at=started_at,
            suggested_continuation=result.get("suggested_continuation"),
            current_task=result.get("current_task"),
        )

        buffering_store.add_chunk(lock_key, chunk)

        # Mark the boundary
        buffer_coordinator.mark_buffered_boundary(
            lock_key=lock_key,
            token_count=message_tokens,
            timestamp=started_at,
        )

        logger.info(
            "[OM:buffer] async observation complete: cycle={}, msgs={}, obs_chars={}",
            cycle_id[:8], len(messages), len(observations_text),
        )

        future.set_result(None)

    except Exception as exc:
        logger.exception("[OM:buffer] async observation failed: {}", exc)
        future.set_exception(exc)

    finally:
        # Clean up from in-flight (keep boundary info for interval tracking)
        buffer_coordinator._async_buffering_ops.pop(buf_key, None)


# ---------------------------------------------------------------------------
# Activate buffered observations
# ---------------------------------------------------------------------------

async def activate_buffered_observations(
    buffer_coordinator: BufferingCoordinator,
    lock_key: str,
    buffering_store: BufferingStore,
    consolidator: "MastraOMConsolidator",
) -> str | None:
    """Activate buffered observation chunks — merge into active observation log.

    Called when the sync observation threshold is crossed. Takes all buffered
    chunks, appends them to the observation log, and cleans up the buffer.

    Args:
        buffer_coordinator: The coordinator tracking buffering state.
        lock_key: Scope key.
        buffering_store: Store holding buffered chunks.
        consolidator: Consolidator for accessing the store.

    Returns:
        Combined observations text from activated chunks, or None if empty.
    """
    chunks = buffering_store.pop_chunks(lock_key)
    if not chunks:
        return None

    # Combine all buffered observations
    combined_parts: list[str] = []
    for chunk in chunks:
        combined_parts.append(chunk.observations)

    combined = "\n\n".join(combined_parts)

    if not combined.strip():
        return None

    # Append to observation log
    consolidator.store.append_observations(combined)
    consolidator.store.append_history(
        f"[OM-BUFFER-ACTIVATED] {len(chunks)} buffered chunks → "
        f"{len(combined)} chars of observations"
    )

    # Clean up state
    msg_ids = [mid for chunk in chunks for mid in chunk.message_ids]
    buffer_coordinator.cleanup_static_maps(lock_key, activated_message_ids=msg_ids)

    logger.info(
        "[OM:buffer] activated {} buffered chunks: {} chars",
        len(chunks), len(combined),
    )

    return combined
