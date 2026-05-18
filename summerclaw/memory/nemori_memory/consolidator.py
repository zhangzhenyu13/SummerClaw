"""Nemori Consolidator — orchestrates the full nemori memory pipeline.

This is the central pipeline orchestrator that mirrors nemori's MemorySystem:
  message buffer → segment → generate episodes → extract semantics → merge

Designed for async background processing per the user's preference.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from summerclaw.memory.nemori_memory.models import Message

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.memory.nemori_memory.store import NemoriStore
    from summerclaw.memory.nemori_memory.segmenter import BatchSegmenter
    from summerclaw.memory.nemori_memory.episode_generator import EpisodeGenerator
    from summerclaw.memory.nemori_memory.semantic_generator import SemanticGenerator
    from summerclaw.memory.nemori_memory.merger import EpisodeMerger
    from summerclaw.memory.nemori_memory.search import UnifiedSearch

logger = logging.getLogger("nemori")

_MAX_LOCKS = 10_000


class NemoriConsolidator:
    """Core pipeline orchestrator for the Nemori memory algorithm.

    Usage:
        consolidator = NemoriConsolidator(store, segmenter, episode_gen, semantic_gen, merger)
        await consolidator.ingest(user_id, agent_id, messages)
        await consolidator.flush(user_id, agent_id)
    """

    def __init__(
        self,
        store: "NemoriStore",
        segmenter: "BatchSegmenter",
        episode_generator: "EpisodeGenerator",
        semantic_generator: "SemanticGenerator",
        merger: "EpisodeMerger | None" = None,
        search: "UnifiedSearch | None" = None,
        *,
        buffer_size_min: int = 2,
        batch_threshold: int = 20,
        episode_min_messages: int = 2,
        enable_semantic: bool = True,
        enable_merging: bool = True,
    ) -> None:
        self._store = store
        self._segmenter = segmenter
        self._episode_gen = episode_generator
        self._semantic_gen = semantic_generator
        self._merger = merger
        self._search = search

        self._buffer_size_min = buffer_size_min
        self._batch_threshold = batch_threshold
        self._episode_min_messages = episode_min_messages
        self._enable_semantic = enable_semantic
        self._enable_merging = enable_merging and merger is not None

        self._user_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._user_locks:
            if len(self._user_locks) >= _MAX_LOCKS:
                self._user_locks.popitem(last=False)
            self._user_locks[key] = asyncio.Lock()
        self._user_locks.move_to_end(key)
        return self._user_locks[key]

    # ── Public API ──────────────────────────────────────────────────────

    async def ingest(
        self, user_id: str, agent_id: str, messages: list[Message]
    ) -> None:
        """Add messages to buffer. Triggers background processing if threshold met.

        Args:
            user_id: User identifier.
            agent_id: Agent namespace.
            messages: Messages to buffer.
        """
        self._store.push_messages(messages)
        count = self._store.count_unprocessed()
        if count >= self._buffer_size_min:
            task = asyncio.create_task(self._process_background(user_id, agent_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def flush(self, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        """Force processing of all buffered messages.

        Returns:
            List of generated episode dicts.
        """
        episodes = await self._process(user_id, agent_id)
        return [e.to_dict() for e in episodes]

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait for all background processing tasks to complete."""
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            done, not_done = await asyncio.wait(pending, timeout=timeout)
            for t in not_done:
                t.cancel()

    # ── Internal pipeline ───────────────────────────────────────────────

    async def _process_background(self, user_id: str, agent_id: str) -> None:
        """Background processing wrapper that catches and logs errors."""
        try:
            await self._process(user_id, agent_id)
        except Exception as e:
            logger.error(
                "Background nemori processing failed for %s/%s: %s",
                agent_id, user_id, e,
            )

    async def _process(self, user_id: str, agent_id: str) -> list[Any]:
        """Process buffered messages through the full nemori pipeline."""
        key = f"{agent_id}:{user_id}"
        async with self._get_lock(key):
            messages = self._store.get_unprocessed_messages()
            if not messages:
                return []

            message_ids = [m.message_id for m in messages]

            # Step 1: Segment (if batch is large enough)
            if len(messages) >= self._batch_threshold:
                groups = await self._segmenter.segment(messages)
            else:
                groups = [{"messages": messages, "topic": "conversation"}]

            # Step 2-5: Generate episodes → semantics → merge
            episodes: list[Any] = []
            for group in groups:
                group_msgs = group["messages"]
                if len(group_msgs) < self._episode_min_messages:
                    continue

                # Generate episode
                episode = await self._episode_gen.generate(
                    user_id, agent_id, group_msgs, group.get("topic", "conversation")
                )
                self._store.save_episode(episode)

                # Check for merge
                if self._enable_merging and self._merger:
                    merged, merged_ep, old_id = await self._merger.check_and_merge(
                        episode, agent_id
                    )
                    if merged and merged_ep and old_id:
                        self._store.delete_episode(old_id)
                        self._store.delete_episode(episode.id)
                        self._store.save_episode(merged_ep)
                        episode = merged_ep

                episodes.append(episode)

                # Generate semantic memories
                if self._enable_semantic:
                    await self._extract_semantics(user_id, agent_id, episode)

            # Mark messages as processed
            self._store.mark_messages_processed(message_ids)
            self._store.compact_buffer()

            logger.info(
                "Nemori processing complete for %s/%s: %d messages → %d episodes",
                agent_id, user_id, len(messages), len(episodes),
            )
            return episodes

    async def _extract_semantics(
        self, user_id: str, agent_id: str, episode: Any
    ) -> None:
        """Extract semantic memories for a freshly created episode."""
        try:
            existing = self._store.list_semantics(user_id, agent_id)
            memories = await self._semantic_gen.generate(
                user_id, agent_id, episode, existing
            )
            if memories:
                self._store.save_semantic_batch(memories)
                logger.info(
                    "Generated %d semantic memories for %s/%s",
                    len(memories), agent_id, user_id,
                )
        except Exception as e:
            logger.error(
                "Semantic generation failed for %s/%s: %s", agent_id, user_id, e
            )

    async def maybe_consolidate_by_tokens(self, session: Any) -> None:
        """Token-budget-aware consolidation entry point required by AgentLoop.

        Nemori uses its own buffer-based processing pipeline instead of
        traditional message archiving. This method triggers processing
        if there are unprocessed messages in the buffer.

        Args:
            session: Session object containing messages and metadata.
        """
        # Extract user_id and agent_id from session
        user_id = getattr(session, "user_id", "default")
        agent_id = getattr(session, "agent_id", "default")

        # Get unprocessed message count
        unprocessed_count = self._store.count_unprocessed()

        if unprocessed_count >= self._buffer_size_min:
            # Trigger background processing
            task = asyncio.create_task(self._process_background(user_id, agent_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            logger.debug(
                "Nemori maybe_consolidate_by_tokens triggered for %s/%s: %d unprocessed messages",
                agent_id, user_id, unprocessed_count,
            )
        else:
            logger.debug(
                "Nemori maybe_consolidate_by_tokens idle for %s/%s: %d unprocessed messages (min=%d)",
                agent_id, user_id, unprocessed_count, self._buffer_size_min,
            )
