"""Tests for NemoriConsolidator — full memory pipeline orchestrator.

Uses mocked sub-components to verify pipeline flow without real LLM calls.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from nanobot.memory.nemori_memory.consolidator import NemoriConsolidator
from nanobot.memory.nemori_memory.models import Episode, Message
from nanobot.memory.nemori_memory.store import NemoriStore


@pytest.fixture
def store(tmp_path):
    return NemoriStore(Path(tmp_path), backend="file")


@pytest.fixture
def mock_segmenter():
    m = MagicMock()
    m.segment = AsyncMock()
    return m


@pytest.fixture
def mock_episode_gen():
    m = MagicMock()
    m.generate = AsyncMock()
    return m


@pytest.fixture
def mock_semantic_gen():
    m = MagicMock()
    m.generate = AsyncMock()
    return m


@pytest.fixture
def mock_merger():
    m = MagicMock()
    m.check_and_merge = AsyncMock(return_value=(False, None, None))
    return m


@pytest.fixture
def consolidator(store, mock_segmenter, mock_episode_gen, mock_semantic_gen, mock_merger):
    return NemoriConsolidator(
        store=store,
        segmenter=mock_segmenter,
        episode_generator=mock_episode_gen,
        semantic_generator=mock_semantic_gen,
        merger=mock_merger,
        buffer_size_min=2,
        batch_threshold=20,
        episode_min_messages=2,
        enable_semantic=True,
        enable_merging=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Ingest / flush / drain
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriConsolidatorIngest:
    """ingest() — buffer messages, trigger background processing."""

    @pytest.mark.asyncio
    async def test_ingest_buffers_messages(self, consolidator, store):
        msgs = [Message(role="user", content="hello")]
        await consolidator.ingest("u1", "default", msgs)
        assert store.count_unprocessed() == 1

    @pytest.mark.asyncio
    async def test_ingest_triggers_processing_at_threshold(
        self, consolidator, store, mock_episode_gen,
    ):
        """When buffer reaches buffer_size_min, background processing starts."""
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        msgs = [Message(role="user", content="a"), Message(role="user", content="b")]
        await consolidator.ingest("u1", "default", msgs)
        # Background processing should have been triggered
        # Wait for background task to complete
        await consolidator.drain(timeout=5)
        # Messages should be processed
        assert store.count_unprocessed() == 0

    @pytest.mark.asyncio
    async def test_ingest_below_threshold_no_processing(self, consolidator, store, mock_episode_gen):
        """Single message below threshold does not trigger processing."""
        msgs = [Message(role="user", content="hello")]
        await consolidator.ingest("u1", "default", msgs)
        await asyncio.sleep(0.1)
        # Background task might have started since buffer_size_min=2 but there's only 1 msg
        # Actually count >= buffer_size_min (2) check: 1 >= 2 is False → no processing
        await consolidator.drain(timeout=2)
        assert store.count_unprocessed() == 1

    @pytest.mark.asyncio
    async def test_ingest_background_error_does_not_crash(
        self, consolidator, store, mock_episode_gen,
    ):
        """If background processing fails, it should not propagate."""
        mock_episode_gen.generate.side_effect = RuntimeError("boom")
        msgs = [Message(role="user", content="a"), Message(role="user", content="b")]
        await consolidator.ingest("u1", "default", msgs)
        await consolidator.drain(timeout=5)
        # Should not raise


# ────────────────────────────────────────────────────────────────────────────
# Flush (force processing)
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriConsolidatorFlush:
    """flush() — force immediate processing."""

    @pytest.mark.asyncio
    async def test_flush_processes_messages(
        self, consolidator, store, mock_episode_gen,
    ):
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        msgs = [Message(role="user", content="a"), Message(role="user", content="b")]
        store.push_messages(msgs)
        result = await consolidator.flush("u1", "default")
        assert len(result) == 1
        assert result[0]["title"] == "T"

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, consolidator):
        result = await consolidator.flush("u1", "default")
        assert result == []


# ────────────────────────────────────────────────────────────────────────────
# Pipeline flow (with mocks)
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriConsolidatorPipeline:
    """Full pipeline flow: segment → generate → semantic → merge."""

    @pytest.mark.asyncio
    async def test_pipeline_calls_segmenter_for_large_batches(
        self, consolidator, store, mock_segmenter, mock_episode_gen,
    ):
        """When batch >= batch_threshold, segmenter is called."""
        store.push_messages([Message(role="user", content=f"msg-{i}") for i in range(25)])
        mock_segmenter.segment.return_value = [
            {"messages": store.get_unprocessed_messages()[:10], "topic": "A"},
            {"messages": store.get_unprocessed_messages()[10:], "topic": "B"},
        ]
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        acc = consolidator
        # Flush will call _process which checks len(messages) >= batch_threshold
        # We have 25 msgs, threshold=20, so segmenter should be called
        await acc.flush("u1", "default")
        if acc._batch_threshold <= 25:
            assert mock_segmenter.segment.called

    @pytest.mark.asyncio
    async def test_pipeline_small_batch_skips_segmenter(
        self, consolidator, store, mock_segmenter, mock_episode_gen,
    ):
        """Small batches use single group, no segmenter call."""
        store.push_messages([Message(role="user", content="hi")])
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        # Only 1 msg < batch_threshold=20, segmenter not called
        await consolidator.flush("u1", "default")
        assert not mock_segmenter.segment.called

    @pytest.mark.asyncio
    async def test_pipeline_calls_semantic_generator(
        self, consolidator, store, mock_episode_gen, mock_semantic_gen,
    ):
        """Semantic generator is called after episode creation."""
        store.push_messages([Message(role="user", content="a"), Message(role="user", content="b")])
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        mock_semantic_gen.generate.return_value = []
        await consolidator.flush("u1", "default")
        assert mock_semantic_gen.generate.called

    @pytest.mark.asyncio
    async def test_pipeline_disabled_semantic(
        self, store, mock_episode_gen, mock_semantic_gen, mock_segmenter,
    ):
        """When enable_semantic=False, semantic generator is not called."""
        c = NemoriConsolidator(
            store=store, segmenter=mock_segmenter,
            episode_generator=mock_episode_gen,
            semantic_generator=mock_semantic_gen,
            enable_semantic=False,
        )
        store.push_messages([Message(role="user", content="a"), Message(role="user", content="b")])
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        await c.flush("u1", "default")
        assert not mock_semantic_gen.generate.called

    @pytest.mark.asyncio
    async def test_pipeline_disabled_merging(
        self, store, mock_episode_gen, mock_merger, mock_segmenter, mock_semantic_gen,
    ):
        """When enable_merging=False, merger is not called."""
        c = NemoriConsolidator(
            store=store, segmenter=mock_segmenter,
            episode_generator=mock_episode_gen,
            semantic_generator=mock_semantic_gen,
            merger=mock_merger, enable_merging=False,
        )
        store.push_messages([Message(role="user", content="a"), Message(role="user", content="b")])
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        await c.flush("u1", "default")
        assert not mock_merger.check_and_merge.called

    @pytest.mark.asyncio
    async def test_pipeline_merge_saves_merged_episode(
        self, consolidator, store, mock_episode_gen, mock_merger,
    ):
        """When merge happens, old episodes are deleted and merged one saved."""
        ep1 = Episode(user_id="u1", title="Old", content="old", source_messages=[])
        store.save_episode(ep1)
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="New", content="new", source_messages=[], id="new-ep",
        )
        merged = Episode(
            user_id="u1", title="Merged", content="merged",
            source_messages=[], id="merged-ep",
        )
        mock_merger.check_and_merge.return_value = (True, merged, ep1.id)

        store.push_messages([Message(role="user", content="a"), Message(role="user", content="b")])
        await consolidator.flush("u1", "default")

        # Old episode should be deleted
        assert store.get_episode(ep1.id, "u1") is None
        # New episode should be deleted and replaced by merged
        assert store.get_episode("merged-ep", "u1") is not None


# ────────────────────────────────────────────────────────────────────────────
# Lock management
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriConsolidatorLock:
    """Per-user lock management."""

    def test_same_key_same_lock(self, consolidator):
        lock1 = consolidator._get_lock("agent:user")
        lock2 = consolidator._get_lock("agent:user")
        assert lock1 is lock2

    def test_different_keys_different_locks(self, consolidator):
        lock1 = consolidator._get_lock("agent:user1")
        lock2 = consolidator._get_lock("agent:user2")
        assert lock1 is not lock2

    def test_lock_lru_eviction(self, store, mock_segmenter, mock_episode_gen, mock_semantic_gen):
        """When lock count exceeds _MAX_LOCKS, oldest is evicted."""
        from nanobot.memory.nemori_memory.consolidator import _MAX_LOCKS

        c = NemoriConsolidator(
            store=store, segmenter=mock_segmenter,
            episode_generator=mock_episode_gen,
            semantic_generator=mock_semantic_gen,
        )
        for i in range(_MAX_LOCKS + 10):
            c._get_lock(f"key-{i}")
        assert len(c._user_locks) <= _MAX_LOCKS


# ────────────────────────────────────────────────────────────────────────────
# Drain
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriConsolidatorDrain:
    """drain() — wait for background tasks."""

    @pytest.mark.asyncio
    async def test_drain_no_tasks(self, consolidator):
        await consolidator.drain(timeout=1)

    @pytest.mark.asyncio
    async def test_drain_with_pending_tasks(
        self, consolidator, store, mock_episode_gen,
    ):
        """Drain waits for background tasks to complete."""
        async def slow_gen(*args, **kw):
            await asyncio.sleep(0.3)
            return Episode(user_id="u1", title="T", content="C", source_messages=[])

        mock_episode_gen.generate = slow_gen
        msgs = [Message(role="user", content="a"), Message(role="user", content="b")]
        store.push_messages(msgs)
        # Trigger background processing
        task = asyncio.create_task(consolidator._process_background("u1", "default"))
        consolidator._tasks.add(task)
        await consolidator.drain(timeout=5)
        assert store.count_unprocessed() == 0


# ────────────────────────────────────────────────────────────────────────────
# maybe_consolidate_by_tokens
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriConsolidatorMaybeConsolidateByTokens:
    """maybe_consolidate_by_tokens() — token-budget consolidation entry point."""

    @pytest.mark.asyncio
    async def test_maybe_consolidate_triggers_when_above_threshold(
        self, consolidator, store, mock_episode_gen,
    ):
        """When unprocessed messages >= buffer_size_min, processing is triggered."""
        mock_episode_gen.generate.return_value = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
        )
        # Add messages to store
        msgs = [
            Message(role="user", content="a"),
            Message(role="user", content="b"),
        ]
        store.push_messages(msgs)

        # Create mock session
        session = MagicMock()
        session.user_id = "u1"
        session.agent_id = "default"

        # Call maybe_consolidate_by_tokens
        await consolidator.maybe_consolidate_by_tokens(session)

        # Wait for background processing
        await consolidator.drain(timeout=5)

        # Messages should be processed
        assert store.count_unprocessed() == 0

    @pytest.mark.asyncio
    async def test_maybe_consolidate_idle_when_below_threshold(
        self, consolidator, store,
    ):
        """When unprocessed messages < buffer_size_min, no processing occurs."""
        # Add only 1 message (below buffer_size_min=2)
        msgs = [Message(role="user", content="hello")]
        store.push_messages(msgs)

        # Create mock session
        session = MagicMock()
        session.user_id = "u1"
        session.agent_id = "default"

        # Call maybe_consolidate_by_tokens
        await consolidator.maybe_consolidate_by_tokens(session)

        # Give some time for any potential background tasks
        await asyncio.sleep(0.1)
        await consolidator.drain(timeout=2)

        # Message should still be unprocessed
        assert store.count_unprocessed() == 1

    @pytest.mark.asyncio
    async def test_maybe_consolidate_uses_default_ids(self, consolidator, store, mock_episode_gen):
        """When session lacks user_id/agent_id, defaults are used."""
        mock_episode_gen.generate.return_value = Episode(
            user_id="default", title="T", content="C", source_messages=[],
        )
        msgs = [
            Message(role="user", content="a"),
            Message(role="user", content="b"),
        ]
        store.push_messages(msgs)

        # Session without user_id/agent_id attributes
        session = MagicMock(spec=[])

        await consolidator.maybe_consolidate_by_tokens(session)
        await consolidator.drain(timeout=5)

        assert store.count_unprocessed() == 0

    @pytest.mark.asyncio
    async def test_maybe_consolidate_error_handling(
        self, consolidator, store, mock_episode_gen,
    ):
        """Errors in background processing should not propagate."""
        mock_episode_gen.generate.side_effect = RuntimeError("test error")
        msgs = [
            Message(role="user", content="a"),
            Message(role="user", content="b"),
        ]
        store.push_messages(msgs)

        session = MagicMock()
        session.user_id = "u1"
        session.agent_id = "default"

        # Should not raise
        await consolidator.maybe_consolidate_by_tokens(session)
        await consolidator.drain(timeout=5)
