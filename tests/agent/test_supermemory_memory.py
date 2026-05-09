"""Comprehensive tests for the Supermemory memory algorithm.

Covers all components: SupermemoryStore, SupermemoryConsolidator,
SupermemoryDream, SupermemoryAutoCompact, and full-pipeline integration.

References:
- Supermemory research: https://supermemory.ai/research/
- Supermemory source: /home/bird/mem-algs/supermemory
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.memory.supermemory_memory.store import (
    MemoryEdge,
    MemoryNode,
    MemoryRelation,
    SourceChunk,
    SupermemoryStore,
)
from nanobot.memory.supermemory_memory.consolidator import SupermemoryConsolidator
from nanobot.memory.supermemory_memory.dream import SupermemoryDream
from nanobot.memory.supermemory_memory.auto_compact import SupermemoryAutoCompact
from nanobot.memory.supermemory_memory import SupermemoryMemoryAlgorithm
from nanobot.memory.base import MemoryComponents
from nanobot.agent.runner import AgentRunResult
from nanobot.utils.gitstore import LineAge


# ===================================================================
# Helpers
# ===================================================================

def _make_run_result(
    stop_reason: str = "completed",
    tool_events: list[dict] | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        final_content=stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def store(tmp_path: Path) -> SupermemoryStore:
    return SupermemoryStore(tmp_path)


@pytest.fixture
def mock_provider() -> MagicMock:
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_sessions() -> MagicMock:
    sm = MagicMock()
    sm.save = MagicMock()
    sm.invalidate = MagicMock()
    sm.list_sessions = MagicMock(return_value=[])
    return sm


@pytest.fixture
def consolidator(
    store: SupermemoryStore,
    mock_provider: MagicMock,
    mock_sessions: MagicMock,
) -> SupermemoryConsolidator:
    return SupermemoryConsolidator(
        store=store,
        provider=mock_provider,
        model="test-model",
        sessions=mock_sessions,
        context_window_tokens=100_000,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
        max_completion_tokens=4096,
    )


@pytest.fixture
def mock_runner() -> MagicMock:
    return MagicMock()


@pytest.fixture
def dream(
    store: SupermemoryStore,
    mock_provider: MagicMock,
    mock_runner: MagicMock,
) -> SupermemoryDream:
    d = SupermemoryDream(
        store=store,
        provider=mock_provider,
        model="test-model",
        max_batch_size=5,
    )
    d._runner = mock_runner
    return d


# ===================================================================
# SupermemoryStore — Data model
# ===================================================================

class TestMemoryNode:
    """Test MemoryNode dataclass — the core Supermemory data structure."""

    def test_create_basic_node(self) -> None:
        node = MemoryNode(
            id="n1",
            memory="User prefers dark mode",
            content="Original message about dark mode",
            document_date="2026-05-01",
        )
        assert node.id == "n1"
        assert node.memory == "User prefers dark mode"
        assert node.version == 1
        assert node.is_latest is True
        assert node.is_forgotten is False
        assert node.root_memory_id == "n1"
        assert node.parent_memory_id is None
        assert isinstance(node.created_at, str)

    def test_node_to_dict_and_back(self) -> None:
        node = MemoryNode(
            id="n1",
            memory="User is a Python developer",
            content="Source chunk content",
            document_date="2026-05-01",
            event_date="2025-01-15",
            version=2,
            parent_memory_id="n0",
        )
        d = node.to_dict()
        restored = MemoryNode.from_dict(d)
        assert restored.id == node.id
        assert restored.memory == node.memory
        assert restored.event_date == "2025-01-15"
        assert restored.version == 2
        assert restored.parent_memory_id == "n0"

    def test_node_with_embedding(self) -> None:
        node = MemoryNode(
            id="n1",
            memory="Test memory",
            embedding=[0.1, 0.2, 0.3],
        )
        assert node.embedding == [0.1, 0.2, 0.3]

    def test_forgotten_node(self) -> None:
        node = MemoryNode(
            id="n1",
            memory="Outdated fact",
            is_forgotten=True,
            forget_reason="User corrected this",
        )
        assert node.is_forgotten is True
        assert node.forget_reason == "User corrected this"


class TestMemoryEdge:
    """Test MemoryEdge dataclass — relationships between memory nodes."""

    def test_create_edge(self) -> None:
        edge = MemoryEdge(
            id="e1",
            source_id="n2",
            target_id="n1",
            edge_type=MemoryRelation.UPDATES,
        )
        assert edge.id == "e1"
        assert edge.source_id == "n2"
        assert edge.target_id == "n1"
        assert edge.edge_type == MemoryRelation.UPDATES
        assert isinstance(edge.created_at, str)

    def test_edge_serialization(self) -> None:
        edge = MemoryEdge(
            id="e1",
            source_id="n2",
            target_id="n1",
            edge_type=MemoryRelation.EXTENDS,
        )
        d = edge.to_dict()
        assert d["edge_type"] == "extends"
        restored = MemoryEdge.from_dict(d)
        assert restored.edge_type == MemoryRelation.EXTENDS

    def test_all_relation_types(self) -> None:
        for rt in MemoryRelation:
            edge = MemoryEdge(id="e", source_id="a", target_id="b", edge_type=rt)
            assert edge.edge_type == rt


class TestSourceChunk:
    """Test SourceChunk — source conversation chunks for hybrid search."""

    def test_create_chunk(self) -> None:
        chunk = SourceChunk(
            id="c1",
            content="User: I like dark mode\nAssistant: Noted!",
            document_date="2026-05-01",
            memory_ids=["m1", "m2"],
        )
        assert chunk.id == "c1"
        assert chunk.memory_ids == ["m1", "m2"]

    def test_chunk_serialization(self) -> None:
        chunk = SourceChunk(
            id="c1",
            content="Conversation text",
        )
        d = chunk.to_dict()
        restored = SourceChunk.from_dict(d)
        assert restored.id == "c1"
        assert restored.content == "Conversation text"


# ===================================================================
# SupermemoryStore — Basic I/O
# ===================================================================

class TestSupermemoryStoreBasicIO:
    """Test SupermemoryStore inherits MemoryStore functionality correctly."""

    def test_read_write_memory(self, store: SupermemoryStore) -> None:
        store.write_memory("Supermemory content")
        assert store.read_memory() == "Supermemory content"

    def test_read_write_soul(self, store: SupermemoryStore) -> None:
        store.write_soul("soul")
        assert store.read_soul() == "soul"

    def test_read_write_user(self, store: SupermemoryStore) -> None:
        store.write_user("user")
        assert store.read_user() == "user"

    def test_history_operations(self, store: SupermemoryStore) -> None:
        c1 = store.append_history("event 1")
        c2 = store.append_history("event 2")
        assert c1 == 1
        assert c2 == 2
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_dream_cursor(self, store: SupermemoryStore) -> None:
        assert store.get_last_dream_cursor() == 0
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5


# ===================================================================
# SupermemoryStore — Graph operations
# ===================================================================

class TestSupermemoryStoreGraph:
    """Test memory graph (nodes + edges) operations."""

    def test_add_and_get_node(self, store: SupermemoryStore) -> None:
        node = MemoryNode(id="n1", memory="User prefers Python")
        store.add_node(node)
        retrieved = store.get_node("n1")
        assert retrieved is not None
        assert retrieved.memory == "User prefers Python"

    def test_list_nodes_excludes_forgotten(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="Active fact")
        n2 = MemoryNode(id="n2", memory="Forgotten fact", is_forgotten=True)
        store.add_node(n1)
        store.add_node(n2)
        nodes = store.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].id == "n1"

    def test_list_nodes_include_forgotten(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="Active")
        n2 = MemoryNode(id="n2", memory="Forgotten", is_forgotten=True)
        store.add_node(n1)
        store.add_node(n2)
        nodes = store.list_nodes(include_forgotten=True)
        assert len(nodes) == 2

    def test_get_latest_nodes(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="v1", version=1, is_latest=True,
                        root_memory_id="root-1")
        n2 = MemoryNode(id="n2", memory="v2", version=2, is_latest=False,
                        root_memory_id="root-1", parent_memory_id="n1")
        store.add_node(n1)
        store.add_node(n2)
        latest = store.get_latest_nodes()
        assert len(latest) == 1
        assert latest[0].id == "n1"

    def test_forget_node(self, store: SupermemoryStore) -> None:
        node = MemoryNode(id="n1", memory="Old fact")
        store.add_node(node)
        store.forget_node("n1", reason="User changed their mind")
        n = store.get_node("n1")
        assert n is not None
        assert n.is_forgotten is True
        assert n.forget_reason == "User changed their mind"

    def test_add_edge(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="source")
        n2 = MemoryNode(id="n2", memory="target")
        store.add_node(n1)
        store.add_node(n2)
        edge = MemoryEdge(id="e1", source_id="n1", target_id="n2",
                          edge_type=MemoryRelation.EXTENDS)
        store.add_edge(edge)
        edges = store.get_edges_for_node("n1")
        assert len(edges) == 1
        assert edges[0].edge_type == MemoryRelation.EXTENDS

    def test_add_edge_fails_for_missing_node(self, store: SupermemoryStore) -> None:
        edge = MemoryEdge(id="e1", source_id="nonexistent", target_id="n2",
                          edge_type=MemoryRelation.EXTENDS)
        with pytest.raises(ValueError):
            store.add_edge(edge)

    def test_duplicate_edge_prevention(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="a")
        n2 = MemoryNode(id="n2", memory="b")
        store.add_node(n1)
        store.add_node(n2)
        e1 = MemoryEdge(id="e1", source_id="n1", target_id="n2",
                        edge_type=MemoryRelation.EXTENDS)
        e2 = MemoryEdge(id="e2", source_id="n1", target_id="n2",
                        edge_type=MemoryRelation.EXTENDS)
        store.add_edge(e1)
        result = store.add_edge(e2)  # Should return existing edge ID
        assert result == "e1"
        assert len(store.get_edges_for_node("n1")) == 1

    def test_graph_persistence(self, tmp_path: Path) -> None:
        """Graph should persist across store instances."""
        s1 = SupermemoryStore(tmp_path)
        node = MemoryNode(id="n1", memory="persistent fact")
        s1.add_node(node)

        s2 = SupermemoryStore(tmp_path)
        retrieved = s2.get_node("n1")
        assert retrieved is not None
        assert retrieved.memory == "persistent fact"


# ===================================================================
# SupermemoryStore — Relational versioning
# ===================================================================

class TestSupermemoryStoreVersioning:
    """Test Supermemory-style relational versioning (updates, extends, derives)."""

    def test_create_new_version(self, store: SupermemoryStore) -> None:
        """Creating a new version should preserve the old as history."""
        old = MemoryNode(id="n1", memory="Fav color: blue")
        store.add_node(old)

        new = store.create_new_version(
            old_node_id="n1",
            new_memory="Fav color: green",
        )
        assert new.id != old.id
        assert new.memory == "Fav color: green"
        assert new.version == 2
        assert new.is_latest is True
        assert new.parent_memory_id == "n1"
        assert new.root_memory_id == "n1"

        # Old node should be marked as not latest
        old_after = store.get_node("n1")
        assert old_after is not None
        assert old_after.is_latest is False

        # Version chain
        chain = store.get_version_chain("n1")
        assert len(chain) == 2
        assert chain[0].version == 1
        assert chain[1].version == 2

    def test_extend_memory(self, store: SupermemoryStore) -> None:
        """Extend should add detail without overriding."""
        base = MemoryNode(id="n1", memory="Works at Acme Corp")
        store.add_node(base)

        extended = store.extend_memory(
            source_node_id="n1",
            extension_memory="Job title: Senior Engineer",
        )
        assert extended.id != "n1"
        assert extended.memory == "Job title: Senior Engineer"

        edges = store.get_edges_for_node(extended.id)
        assert len(edges) == 1
        assert edges[0].edge_type == MemoryRelation.EXTENDS

    def test_derive_memory(self, store: SupermemoryStore) -> None:
        """Derive should create inferred memory from multiple sources."""
        n1 = MemoryNode(id="n1", memory="Born in Paris")
        n2 = MemoryNode(id="n2", memory="Speaks French fluently")
        store.add_node(n1)
        store.add_node(n2)

        derived = store.derive_memory(
            source_ids=["n1", "n2"],
            derived_memory="Likely French nationality",
        )
        assert derived.memory == "Likely French nationality"

        edges = store.get_edges_for_node(derived.id)
        assert len(edges) == 2
        assert all(e.edge_type == MemoryRelation.DERIVES for e in edges)


# ===================================================================
# SupermemoryStore — Chunks and Search
# ===================================================================

class TestSupermemoryStoreChunks:
    """Test source chunk storage and retrieval."""

    def test_add_and_get_chunk(self, store: SupermemoryStore) -> None:
        chunk = SourceChunk(
            id="c1",
            content="User: hello\nAssistant: hi",
            document_date="2026-05-01",
        )
        store.add_chunk(chunk)
        retrieved = store.get_chunk("c1")
        assert retrieved is not None
        assert retrieved.content == "User: hello\nAssistant: hi"

    def test_get_chunks_for_memory(self, store: SupermemoryStore) -> None:
        chunk = SourceChunk(
            id="c1",
            content="conversation",
            memory_ids=["m1"],
        )
        store.add_chunk(chunk)
        chunks = store.get_chunks_for_memory("m1")
        assert len(chunks) == 1
        assert chunks[0].id == "c1"

    def test_list_chunks(self, store: SupermemoryStore) -> None:
        c1 = SourceChunk(id="c1", content="chunk 1")
        c2 = SourceChunk(id="c2", content="chunk 2")
        store.add_chunk(c1)
        store.add_chunk(c2)
        assert len(store.list_chunks()) == 2


class TestSupermemoryStoreSearch:
    """Test memory search operations."""

    def test_keyword_search(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="User prefers Python for backend",
                        content="discussion about Python")
        n2 = MemoryNode(id="n2", memory="User likes TypeScript for frontend",
                        content="discussion about TypeScript")
        n3 = MemoryNode(id="n3", memory="Unrelated fact about weather")
        store.add_node(n1)
        store.add_node(n2)
        store.add_node(n3)

        results = store.search_memories_by_keyword("python")
        assert len(results) >= 1
        assert results[0].id == "n1"

        results = store.search_memories_by_keyword("typescript")
        assert len(results) >= 1
        assert results[0].id == "n2"

    def test_keyword_search_no_results(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="User prefers Python")
        store.add_node(n1)
        results = store.search_memories_by_keyword("haskell")
        assert len(results) == 0

    def test_keyword_search_excludes_forgotten(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="Python is great", is_forgotten=True)
        store.add_node(n1)
        results = store.search_memories_by_keyword("python")
        assert len(results) == 0

    def test_keyword_search_excludes_non_latest(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="Python v1", is_latest=True)
        n2 = MemoryNode(id="n2", memory="Python v2", is_latest=False)
        store.add_node(n1)
        store.add_node(n2)
        results = store.search_memories_by_keyword("python")
        assert len(results) == 1
        assert results[0].id == "n1"


# ===================================================================
# SupermemoryStore — Context and Stats
# ===================================================================

class TestSupermemoryStoreContext:
    """Test memory context generation."""

    def test_get_memory_context_empty(self, store: SupermemoryStore) -> None:
        ctx = store.get_memory_context()
        assert ctx == ""

    def test_get_memory_context_with_memory(self, store: SupermemoryStore) -> None:
        store.write_memory("# Facts\n- Important fact")
        ctx = store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "Important fact" in ctx

    def test_get_memory_context_with_graph(self, store: SupermemoryStore) -> None:
        node = MemoryNode(
            id="n1",
            memory="User prefers dark mode",
            event_date="2026-04-01",
        )
        store.add_node(node)
        ctx = store.get_memory_context()
        assert "Memory Graph" in ctx
        assert "User prefers dark mode" in ctx
        assert "event: 2026-04-01" in ctx

    def test_get_memory_context_with_version(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="v1", version=1, is_latest=True)
        n2 = MemoryNode(id="n2", memory="v2", version=3, is_latest=True)
        store.add_node(n1)
        store.add_node(n2)
        ctx = store.get_memory_context()
        assert "(v3)" in ctx


class TestSupermemoryStoreStats:
    """Test statistics reporting."""

    def test_stats_empty(self, store: SupermemoryStore) -> None:
        stats = store.stats()
        assert stats["total_nodes"] == 0
        assert stats["active_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_stats_with_data(self, store: SupermemoryStore) -> None:
        n1 = MemoryNode(id="n1", memory="active")
        n2 = MemoryNode(id="n2", memory="forgotten", is_forgotten=True)
        store.add_node(n1)
        store.add_node(n2)
        edge = MemoryEdge(id="e1", source_id="n1", target_id="n2",
                          edge_type=MemoryRelation.EXTENDS)
        store.add_edge(edge)

        stats = store.stats()
        assert stats["total_nodes"] == 2
        assert stats["active_nodes"] == 1
        assert stats["forgotten_nodes"] == 1
        assert stats["total_edges"] == 1
        assert stats["edges_by_type"]["extends"] == 1


# ===================================================================
# SupermemoryConsolidator — Chunk operations
# ===================================================================

class TestSupermemoryConsolidatorChunking:
    """Test chunk-based message decomposition."""

    def test_chunk_empty_messages(self) -> None:
        chunks = SupermemoryConsolidator._chunk_messages([])
        assert chunks == []

    def test_chunk_single_message(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        chunks = SupermemoryConsolidator._chunk_messages(msgs)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_chunk_splits_at_user_boundaries(self) -> None:
        msgs = [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 1000},
            {"role": "user", "content": "c" * 1000},
            {"role": "assistant", "content": "d" * 1000},
        ]
        # With token estimate ~char/4, each message is ~250 tokens
        chunks = SupermemoryConsolidator._chunk_messages(msgs, max_chunk_tokens=600)
        # Should split into roughly 2 chunks
        assert len(chunks) >= 1

    def test_format_chunk(self) -> None:
        msgs = [
            {"role": "user", "content": "hello",
             "timestamp": "2026-05-01 10:00:00"},
            {"role": "assistant", "content": "hi",
             "timestamp": "2026-05-01 10:00:05"},
        ]
        formatted = SupermemoryConsolidator._format_chunk_for_memory_generation(msgs)
        assert "hello" in formatted
        assert "hi" in formatted
        assert "2026-05-01 10:00" in formatted


# ===================================================================
# SupermemoryConsolidator — Archive
# ===================================================================

class TestSupermemoryConsolidatorArchive:
    """Test chunk-based archive with memory generation."""

    async def test_archive_calls_super_and_processes_chunks(
        self,
        consolidator: SupermemoryConsolidator,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="User discussed deployment.",
            finish_reason="stop",
        )
        messages = [
            {"role": "user", "content": "I deployed the app to production.",
             "timestamp": "2026-05-01 10:00:00"},
            {"role": "assistant", "content": "Deployment successful.",
             "timestamp": "2026-05-01 10:00:05"},
        ]
        result = await consolidator.archive(messages)
        assert result == "User discussed deployment."

        # Should have created memory nodes
        stats = store.stats()
        assert stats["total_nodes"] >= 1

        # Should have stored source chunks
        assert len(store.list_chunks()) >= 1

    async def test_archive_skips_empty_messages(
        self, consolidator: SupermemoryConsolidator,
    ) -> None:
        result = await consolidator.archive([])
        assert result is None

    async def test_archive_llm_failure_falls_back(
        self,
        consolidator: SupermemoryConsolidator,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        mock_provider.chat_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "hello"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]

    async def test_archive_error_finish_reason(
        self,
        consolidator: SupermemoryConsolidator,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Error: overloaded",
            finish_reason="error",
        )
        messages = [{"role": "user", "content": "test"}]
        result = await consolidator.archive(messages)
        assert result is None
        entries = store.read_unprocessed_history(since_cursor=0)
        assert "[RAW]" in entries[0]["content"]


# ===================================================================
# SupermemoryConsolidator — Token Budget
# ===================================================================

class TestSupermemoryConsolidatorTokenBudget:
    """Test token-budget consolidation (inherited from Consolidator)."""

    async def test_prompt_below_threshold_skips(
        self, consolidator: SupermemoryConsolidator,
    ) -> None:
        session = MagicMock()
        session.last_consolidated = 0
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(100, "tiktoken"),
        )
        consolidator.archive = AsyncMock(return_value=True)
        await consolidator.maybe_consolidate_by_tokens(session)
        consolidator.archive.assert_not_called()

    async def test_no_consolidation_when_context_zero(
        self,
        store: SupermemoryStore,
        mock_provider: MagicMock,
        mock_sessions: MagicMock,
    ) -> None:
        c = SupermemoryConsolidator(
            store=store,
            provider=mock_provider,
            model="test-model",
            sessions=mock_sessions,
            context_window_tokens=0,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        session.key = "test:key"
        c.archive = AsyncMock()
        await c.maybe_consolidate_by_tokens(session)
        c.archive.assert_not_called()


# ===================================================================
# SupermemoryConsolidator — Relationship detection
# ===================================================================

class TestSupermemoryConsolidatorRelationships:
    """Test relationship detection between new and existing memories."""

    async def test_detect_relationships_with_high_similarity(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        # Pre-populate with an existing memory
        existing = MemoryNode(
            id="old-1",
            memory="User favorite color is blue",
        )
        store.add_node(existing)

        # New memory that updates old information
        new_node = MemoryNode(
            id="new-1",
            memory="User favorite color is now blue dark navy",
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) == 1
        # With Jaccard fallback, high overlap should detect UPDATES or EXTENDS
        assert edges[0].edge_type in (MemoryRelation.UPDATES, MemoryRelation.EXTENDS)

    async def test_detect_relationships_with_moderate_similarity(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        existing = MemoryNode(
            id="old-1",
            memory="User works at Acme Corporation in the engineering team",
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="User works at Acme Corporation as senior engineer software",
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) == 1
        # Moderate Jaccard overlap may produce EXTENDS or DERIVES
        assert edges[0].edge_type in (MemoryRelation.UPDATES, MemoryRelation.EXTENDS, MemoryRelation.DERIVES)


# ===================================================================
# SupermemoryConsolidator — Lock management
# ===================================================================

class TestSupermemoryConsolidatorLock:
    """Test consolidation lock behavior."""

    def test_get_lock_same_key(self, consolidator: SupermemoryConsolidator) -> None:
        lock1 = consolidator.get_lock("session:a")
        lock2 = consolidator.get_lock("session:a")
        assert lock1 is lock2

    def test_get_lock_different_keys(
        self, consolidator: SupermemoryConsolidator,
    ) -> None:
        lock1 = consolidator.get_lock("session:a")
        lock2 = consolidator.get_lock("session:b")
        assert lock1 is not lock2

    def test_get_lock_is_asyncio_lock(
        self, consolidator: SupermemoryConsolidator,
    ) -> None:
        lock = consolidator.get_lock("new:session")
        assert isinstance(lock, asyncio.Lock)


# ===================================================================
# SupermemoryDream — run() basic behavior
# ===================================================================

class TestSupermemoryDreamRun:
    """Test SupermemoryDream run() method."""

    async def test_noop_when_no_unprocessed_history(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
    ) -> None:
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()

    async def test_calls_runner_for_unprocessed_entries(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{
                "name": "edit_file",
                "status": "ok",
                "detail": "memory/MEMORY.md",
            }],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == 10
        assert spec.fail_on_tool_error is False

    async def test_advances_dream_cursor(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_history_after_run(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_respects_max_batch_size(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        s = SupermemoryStore(tmp_path)
        for i in range(10):
            s.append_history(f"event {i}")
        d = SupermemoryDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            max_batch_size=3,
        )
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_r = MagicMock()
        mock_r.run = AsyncMock(return_value=_make_run_result())
        d._runner = mock_r
        await d.run()
        assert s.get_last_dream_cursor() == 3


# ===================================================================
# SupermemoryDream — Phase 1 analysis
# ===================================================================

class TestSupermemoryDreamPhase1:
    """Test Phase 1: LLM analysis with graph context."""

    async def test_phase1_includes_graph_context(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        # Add some graph nodes so context is non-empty
        node = MemoryNode(
            id="n1",
            memory="User prefers Python",
            event_date="2026-04-01",
        )
        store.add_node(node)

        store.append_history("User discussed Python")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        # Verify Phase 1 prompt includes graph context
        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages",
                       call_args[1].get("messages"))[1]["content"]
        assert "Memory Graph Summary" in user_msg

    async def test_phase1_llm_error_is_caught(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.side_effect = Exception("LLM error")
        result = await dream.run()
        assert result is False

    async def test_phase1_prompt_includes_all_files(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.write_soul("# Soul\n- Helpful")
        store.write_user("# User\n- Developer")
        store.write_memory("# Memory\n- Active project")
        store.append_history("some event")

        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages",
                       call_args[1].get("messages"))[1]["content"]
        assert "## Current MEMORY.md" in user_msg
        assert "## Current SOUL.md" in user_msg
        assert "## Current USER.md" in user_msg
        assert "## Current Date" in user_msg


# ===================================================================
# SupermemoryDream — Phase 2 AgentRunner
# ===================================================================

class TestSupermemoryDreamPhase2:
    """Test Phase 2: AgentRunner for file editing."""

    async def test_phase2_calls_agent_runner(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{
                "name": "edit_file",
                "status": "ok",
                "detail": "memory/MEMORY.md",
            }],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()

    async def test_phase2_spec_parameters(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == dream.max_iterations
        assert spec.fail_on_tool_error is False

    async def test_phase2_exception_is_caught(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run.side_effect = Exception("Runner error")
        result = await dream.run()
        # Should still return True and advance cursor
        assert result is True
        assert store.get_last_dream_cursor() == 1

    async def test_phase2_skill_creation_path(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        store.append_history("Repeated workflow")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="[SKILL] test-skill: description",
        )
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        spec = mock_runner.run.call_args[0][0]
        system_prompt = spec.initial_messages[0]["content"]
        expected = str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        assert expected in system_prompt


# ===================================================================
# SupermemoryDream — Git auto-commit
# ===================================================================

class TestSupermemoryDreamGit:
    """Test git auto-commit after successful dream."""

    async def test_git_commit_on_success(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event")
        store.write_memory("# Updated memory")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Memory analysis complete.",
        )
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{
                "name": "edit_file",
                "status": "ok",
                "detail": "memory/MEMORY.md",
            }],
        ))
        store.git.init()
        await dream.run()
        assert store.git.is_initialized()

    async def test_git_no_commit_without_changelog(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[],
        ))
        store.git.init()
        await dream.run()


# ===================================================================
# SupermemoryDream — Age annotation
# ===================================================================

class TestSupermemoryDreamAgeAnnotation:
    """Test MEMORY.md line age annotation."""

    async def test_phase1_prompt_works_without_git(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        mock_provider.chat_with_retry.assert_called_once()

    async def test_annotate_with_ages_skipped_when_disabled(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.append_history("some event")
        dream.annotate_line_ages = False
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch.object(store.git, "line_ages") as mock_line_ages:
            await dream.run()
            mock_line_ages.assert_not_called()

    async def test_phase1_prompt_carries_age_suffix(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        store.write_memory("# Memory\n- Project X active\n- fresh item\n- edge case")
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
        user_msg = call_args.kwargs.get("messages",
                       call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split(
            "## Current SOUL.md",
        )[0]
        assert "\u2190 30d" in memory_section
        assert "\u2190 20d" in memory_section
        assert "\u2190 14d" not in memory_section
        assert "\u2190 5d" not in memory_section


# ===================================================================
# SupermemoryDream — Skill listing
# ===================================================================

class TestSupermemoryDreamSkills:
    """Test existing skill listing."""

    def test_list_skills_with_user_skills(
        self, dream: SupermemoryDream, store: SupermemoryStore,
    ) -> None:
        skill_dir = store.workspace / "skills" / "test-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n",
            encoding="utf-8",
        )
        skills = dream._list_existing_skills()
        assert any("test-skill" in s for s in skills)

    def test_list_skills_empty_when_none_exist(
        self, dream: SupermemoryDream,
    ) -> None:
        skills = dream._list_existing_skills()
        assert isinstance(skills, list)


# ===================================================================
# SupermemoryDream — Configuration
# ===================================================================

class TestSupermemoryDreamConfig:
    """Test Dream configuration defaults and overrides."""

    def test_default_max_batch_size(
        self,
        store: SupermemoryStore,
        mock_provider: MagicMock,
    ) -> None:
        d = SupermemoryDream(store=store, provider=mock_provider, model="m")
        assert d.max_batch_size == 20

    def test_custom_max_batch_size(
        self,
        store: SupermemoryStore,
        mock_provider: MagicMock,
    ) -> None:
        d = SupermemoryDream(
            store=store, provider=mock_provider, model="m",
            max_batch_size=50,
        )
        assert d.max_batch_size == 50

    def test_default_max_iterations(
        self,
        store: SupermemoryStore,
        mock_provider: MagicMock,
    ) -> None:
        d = SupermemoryDream(store=store, provider=mock_provider, model="m")
        assert d.max_iterations == 10

    def test_annotate_line_ages_default(
        self,
        store: SupermemoryStore,
        mock_provider: MagicMock,
    ) -> None:
        d = SupermemoryDream(store=store, provider=mock_provider, model="m")
        assert d.annotate_line_ages is True


# ===================================================================
# SupermemoryAutoCompact
# ===================================================================

class TestSupermemoryAutoCompact:
    """Test SupermemoryAutoCompact idle session compression."""

    def test_is_expired_within_ttl(
        self,
        store: SupermemoryStore,
        consolidator: SupermemoryConsolidator,
        mock_sessions: MagicMock,
    ) -> None:
        """A recently updated session should not be expired."""
        ac = SupermemoryAutoCompact(
            sessions=mock_sessions,
            consolidator=consolidator,
            session_ttl_minutes=60,
        )
        recent = datetime.now()
        assert ac._is_expired(recent) is False

    def test_is_expired_beyond_ttl(
        self,
        store: SupermemoryStore,
        consolidator: SupermemoryConsolidator,
        mock_sessions: MagicMock,
    ) -> None:
        ac = SupermemoryAutoCompact(
            sessions=mock_sessions,
            consolidator=consolidator,
            session_ttl_minutes=1,
        )
        # 2 minutes ago
        old = datetime(2020, 1, 1)
        assert ac._is_expired(old, now=datetime(2020, 1, 1, 0, 3)) is True

    def test_is_expired_ttl_zero(
        self,
        store: SupermemoryStore,
        consolidator: SupermemoryConsolidator,
        mock_sessions: MagicMock,
    ) -> None:
        ac = SupermemoryAutoCompact(
            sessions=mock_sessions,
            consolidator=consolidator,
            session_ttl_minutes=0,
        )
        assert ac._is_expired(datetime(2020, 1, 1)) is False

    def test_format_summary(
        self,
        store: SupermemoryStore,
        consolidator: SupermemoryConsolidator,
        mock_sessions: MagicMock,
    ) -> None:
        ac = SupermemoryAutoCompact(
            sessions=mock_sessions,
            consolidator=consolidator,
        )
        summary = ac._format_summary(
            "User worked on project X.",
            datetime(2026, 1, 1),
        )
        assert "Inactive for" in summary
        assert "project X" in summary

    def test_check_expired_no_sessions(
        self,
        store: SupermemoryStore,
        consolidator: SupermemoryConsolidator,
        mock_sessions: MagicMock,
    ) -> None:
        mock_sessions.list_sessions.return_value = []
        ac = SupermemoryAutoCompact(
            sessions=mock_sessions,
            consolidator=consolidator,
            session_ttl_minutes=60,
        )
        schedule_calls: list[str] = []

        def schedule(coro: object) -> None:
            schedule_calls.append(str(coro))

        ac.check_expired(schedule)
        assert len(schedule_calls) == 0


# ===================================================================
# SupermemoryMemoryAlgorithm — Full pipeline
# ===================================================================

class TestSupermemoryMemoryAlgorithm:
    """Test full algorithm build and registration."""

    def test_build_returns_memory_components(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        algo = SupermemoryMemoryAlgorithm()
        assert algo.name == "supermemory_memory"

        sessions = MagicMock()
        sessions.save = MagicMock()
        sessions.invalidate = MagicMock()
        sessions.list_sessions = MagicMock(return_value=[])

        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=100_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=60,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
            embedding_config=None,
        )

        assert isinstance(components, MemoryComponents)
        assert isinstance(components.store, SupermemoryStore)
        assert isinstance(components.consolidator, SupermemoryConsolidator)
        assert isinstance(components.dream, SupermemoryDream)
        assert isinstance(components.auto_compact, SupermemoryAutoCompact)

    def test_algorithm_registers_in_registry(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        from nanobot.memory.registry import MemoryRegistry

        registry = MemoryRegistry()
        registry.register(SupermemoryMemoryAlgorithm())

        algo = registry.get("supermemory_memory")
        assert algo.name == "supermemory_memory"

        assert "supermemory_memory" in registry.list()

    def test_full_memory_context_after_adding_nodes(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        """End-to-end: build algorithm, add nodes, verify context includes graph."""
        sessions = MagicMock()
        sessions.save = MagicMock()
        sessions.invalidate = MagicMock()
        sessions.list_sessions = MagicMock(return_value=[])

        algo = SupermemoryMemoryAlgorithm()
        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=100_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=60,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
        )

        sm_store: SupermemoryStore = components.store
        sm_store.write_memory("# Long-term facts\n- Project X is active")

        # Add some memory graph nodes
        n1 = MemoryNode(id="n1", memory="User prefers Python",
                        event_date="2026-05-01")
        n2 = MemoryNode(id="n2", memory="User likes dark mode")
        sm_store.add_node(n1)
        sm_store.add_node(n2)

        # Add relationship
        edge = MemoryEdge(id="e1", source_id="n1", target_id="n2",
                          edge_type=MemoryRelation.EXTENDS)
        sm_store.add_edge(edge)

        # Get context
        ctx = sm_store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "Memory Graph" in ctx
        assert "User prefers Python" in ctx
        assert "User likes dark mode" in ctx


# ===================================================================
# SupermemoryMemoryAlgorithm — Relational versioning workflow
# ===================================================================

class TestSupermemoryVersioningWorkflow:
    """Test end-to-end Supermemory-style relational versioning workflow."""

    def test_full_version_chain_workflow(
        self, store: SupermemoryStore,
    ) -> None:
        """Simulate a typical Supermemory workflow:
        1. User states a fact
        2. Later user updates the fact → creates version chain
        3. Additional detail is added → extends
        4. Derived inference from combining memories
        """
        # Step 1: Initial fact
        initial = MemoryNode(
            id="m1",
            memory="User favorite programming language is Python",
            document_date="2026-04-01",
            event_date="2026-01-01",
        )
        store.add_node(initial)

        # Step 2: User updates (contradiction) → new version
        updated = store.create_new_version(
            old_node_id="m1",
            new_memory="User favorite programming language is now Rust",
            event_date="2026-05-01",
        )
        assert updated.version == 2
        assert updated.is_latest is True

        # Verify old is preserved
        old = store.get_node("m1")
        assert old is not None
        assert old.is_latest is False

        # Step 3: Additional detail → extends
        extended = store.extend_memory(
            source_node_id=updated.id,
            extension_memory="User uses Rust for systems programming",
        )

        # Step 4: Derived inference
        derived = store.derive_memory(
            source_ids=[updated.id, extended.id],
            derived_memory="User is interested in systems-level programming",
        )

        # Verify the full state
        stats = store.stats()
        assert stats["total_nodes"] == 4
        assert stats["active_nodes"] == 3  # m1 is not latest
        assert stats["version_chains"] == 1
        assert stats["edges_by_type"]["updates"] == 1
        assert stats["edges_by_type"]["extends"] == 1
        assert stats["edges_by_type"]["derives"] == 2

        # Version chain verification
        chain = store.get_version_chain("m1")
        assert len(chain) == 2

        # Latest nodes should include updated version but not old
        latest = store.get_latest_nodes()
        latest_ids = {n.id for n in latest}
        assert updated.id in latest_ids
        assert "m1" not in latest_ids


# ===================================================================
# SupermemoryMemoryAlgorithm — Hermes (SkillAutogen) coverage
# ===================================================================

class TestSupermemoryHermesIntegration:
    """Verify Supermemory works with Hermes skill auto-generation.

    Hermes (skill_autogen.py) uses the MemoryStore from any algorithm
    to discover existing skills and write new ones. SupermemoryStore
    extends MemoryStore, so it must be compatible with Hermes.
    """

    def test_store_workspace_is_accessible_for_skills(
        self, store: SupermemoryStore,
    ) -> None:
        """SkillAutogen reads from store.workspace/skills/"""
        skills_dir = store.workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        assert skills_dir.exists()

    def test_store_read_write_memory_for_hermes(
        self, store: SupermemoryStore,
    ) -> None:
        """Hermes reads MEMORY.md and SOUL.md for context."""
        store.write_memory("# Facts\n- User prefers Python")
        store.write_soul("# Soul\n- Helpful assistant")
        assert "Python" in store.read_memory()
        assert "Helpful" in store.read_soul()

    def test_graph_serialization_roundtrip(
        self, tmp_path: Path,
    ) -> None:
        """Verify the full graph serialization round-trip is reliable."""
        s1 = SupermemoryStore(tmp_path)

        n1 = MemoryNode(id="n1", memory="Fact A")
        n2 = MemoryNode(id="n2", memory="Fact B", version=2,
                        parent_memory_id="n1")
        s1.add_node(n1)
        s1.add_node(n2)

        e1 = MemoryEdge(id="e1", source_id="n2", target_id="n1",
                        edge_type=MemoryRelation.UPDATES)
        s1.add_edge(e1)

        c1 = SourceChunk(id="c1", content="Conversation A",
                         memory_ids=["n1", "n2"])
        s1.add_chunk(c1)

        # Re-read from a fresh store instance
        s2 = SupermemoryStore(tmp_path)
        assert s2.get_node("n1") is not None
        assert s2.get_node("n2") is not None
        assert s2.get_node("n2").version == 2  # type: ignore[union-attr]
        assert len(s2.get_edges_for_node("n2")) == 1
        assert s2.get_chunk("c1") is not None

    def test_node_compact_removes_old_forgotten(
        self, tmp_path: Path,
    ) -> None:
        """When max_nodes is exceeded, forgotten nodes should be removed first."""
        store = SupermemoryStore(tmp_path, max_nodes=3)
        for i in range(5):
            node = MemoryNode(
                id=f"n{i}",
                memory=f"fact {i}",
                is_forgotten=(i < 2),  # First 2 are forgotten
            )
            store.add_node(node)
        assert len(store._nodes) <= 3
        # Forgotten nodes should be removed first
        remaining_ids = set(store._nodes.keys())
        assert "n0" not in remaining_ids
        assert "n1" not in remaining_ids


# ===================================================================
# SupermemoryStore — Embedding semantic search (NEW)
# ===================================================================

class TestSupermemoryStoreEmbeddingSearch:
    """Test embedding-based semantic search operations."""

    def test_cosine_similarity_identical(self) -> None:
        """Cosine similarity of identical vectors should be ~1.0."""
        vec = [1.0, 2.0, 3.0]
        sim = SupermemoryStore._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self) -> None:
        """Cosine similarity of orthogonal vectors should be 0.0."""
        sim = SupermemoryStore._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim - 0.0) < 1e-9

    def test_cosine_similarity_empty(self) -> None:
        """Empty vectors should return 0.0."""
        assert SupermemoryStore._cosine_similarity([], [1.0]) == 0.0
        assert SupermemoryStore._cosine_similarity([1.0], []) == 0.0

    def test_cosine_similarity_mismatched_lengths(self) -> None:
        """Vectors of different lengths should return 0.0."""
        assert SupermemoryStore._cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_search_memories_by_embedding_finds_match(
        self, store: SupermemoryStore,
    ) -> None:
        """Semantic search should find nodes with similar embeddings."""
        n1 = MemoryNode(
            id="n1", memory="User prefers Python",
            embedding=[1.0, 0.0, 0.0],
        )
        n2 = MemoryNode(
            id="n2", memory="User likes TypeScript",
            embedding=[0.0, 1.0, 0.0],
        )
        n3 = MemoryNode(
            id="n3", memory="User enjoys programming",
            embedding=[1.0, 0.1, 0.0],  # Similar to n1
        )
        store.add_node(n1)
        store.add_node(n2)
        store.add_node(n3)

        # Search with embedding similar to n1
        results = store.search_memories_by_embedding(
            [1.0, 0.05, 0.0], limit=5, threshold=0.5,
        )
        assert len(results) >= 2
        ids = {r[0].id for r in results}
        assert "n1" in ids
        assert "n3" in ids  # Should match n3 too (similar to n1)

    def test_search_memories_by_embedding_no_embedding_skipped(
        self, store: SupermemoryStore,
    ) -> None:
        """Nodes without embeddings should be skipped."""
        n1 = MemoryNode(id="n1", memory="No embedding")
        store.add_node(n1)
        results = store.search_memories_by_embedding([1.0, 0.0, 0.0])
        assert len(results) == 0

    def test_search_memories_by_embedding_respects_threshold(
        self, store: SupermemoryStore,
    ) -> None:
        """Results below threshold should be excluded."""
        n1 = MemoryNode(
            id="n1", memory="Python",
            embedding=[1.0, 0.0, 0.0],
        )
        store.add_node(n1)
        # Query embedding is orthogonal → sim ≈ 0
        results = store.search_memories_by_embedding(
            [0.0, 1.0, 0.0], threshold=0.5,
        )
        assert len(results) == 0

    def test_search_memories_by_embedding_empty_query(self, store: SupermemoryStore) -> None:
        """Empty query embedding should return nothing."""
        results = store.search_memories_by_embedding([])
        assert results == []

    def test_search_memories_by_embedding_excludes_forgotten(
        self, store: SupermemoryStore,
    ) -> None:
        """Forgotten nodes should not appear in search results."""
        n1 = MemoryNode(
            id="n1", memory="Active fact",
            embedding=[1.0, 0.0, 0.0],
        )
        n2 = MemoryNode(
            id="n2", memory="Forgotten fact",
            embedding=[1.0, 0.0, 0.0], is_forgotten=True,
        )
        store.add_node(n1)
        store.add_node(n2)
        results = store.search_memories_by_embedding([1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0][0].id == "n1"

    def test_search_memories_by_embedding_excludes_non_latest(
        self, store: SupermemoryStore,
    ) -> None:
        """Non-latest versions should not appear."""
        n1 = MemoryNode(
            id="n1", memory="v1", is_latest=True,
            embedding=[1.0, 0.0, 0.0],
        )
        n2 = MemoryNode(
            id="n2", memory="v2", is_latest=False,
            embedding=[1.0, 0.0, 0.0],
        )
        store.add_node(n1)
        store.add_node(n2)
        results = store.search_memories_by_embedding([1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0][0].id == "n1"

    def test_set_node_embedding(self, store: SupermemoryStore) -> None:
        """Should set and persist embedding on an existing node."""
        node = MemoryNode(id="n1", memory="Test")
        store.add_node(node)
        result = store.set_node_embedding("n1", [0.1, 0.2, 0.3])
        assert result is True
        retrieved = store.get_node("n1")
        assert retrieved is not None
        assert retrieved.embedding == [0.1, 0.2, 0.3]

    def test_set_node_embedding_nonexistent(self, store: SupermemoryStore) -> None:
        """Setting embedding on nonexistent node should return False."""
        result = store.set_node_embedding("nonexistent", [1.0])
        assert result is False

    def test_get_nodes_without_embeddings(self, store: SupermemoryStore) -> None:
        """Should identify nodes missing embeddings."""
        n1 = MemoryNode(id="n1", memory="Has embedding", embedding=[1.0])
        n2 = MemoryNode(id="n2", memory="No embedding")
        n3 = MemoryNode(id="n3", memory="Forgotten", is_forgotten=True)
        store.add_node(n1)
        store.add_node(n2)
        store.add_node(n3)
        missing = store.get_nodes_without_embeddings()
        assert len(missing) == 1
        assert missing[0].id == "n2"

    def test_search_memories_by_embedding_scores_ordered(
        self, store: SupermemoryStore,
    ) -> None:
        """Results should be sorted by descending similarity."""
        n1 = MemoryNode(
            id="n1", memory="Very similar",
            embedding=[1.0, 0.0, 0.0],
        )
        n2 = MemoryNode(
            id="n2", memory="Less similar",
            embedding=[0.7, 0.7, 0.0],  # cos ≈ 0.707 with [1,0,0]
        )
        store.add_node(n1)
        store.add_node(n2)
        results = store.search_memories_by_embedding(
            [1.0, 0.0, 0.0], threshold=0.0,
        )
        assert len(results) == 2
        assert results[0][0].id == "n1"  # Higher score first
        assert results[0][1] >= results[1][1]


# ===================================================================
# SupermemoryStore — Static/Dynamic profiling (NEW)
# ===================================================================

class TestSupermemoryStoreStaticDynamic:
    """Test static (long-term) vs dynamic (transient) memory profiling."""

    def test_get_static_memories(self, store: SupermemoryStore) -> None:
        """Should return only static, active memories."""
        n1 = MemoryNode(id="n1", memory="Python preference", is_static=True)
        n2 = MemoryNode(id="n2", memory="Current task", is_static=False)
        n3 = MemoryNode(id="n3", memory="Forgotten static", is_static=True,
                        is_forgotten=True)
        n4 = MemoryNode(id="n4", memory="Old version", is_static=True,
                        is_latest=False)
        store.add_node(n1)
        store.add_node(n2)
        store.add_node(n3)
        store.add_node(n4)

        static = store.get_static_memories()
        assert len(static) == 1
        assert static[0].id == "n1"

    def test_get_dynamic_memories(self, store: SupermemoryStore) -> None:
        """Should return only dynamic, active memories."""
        n1 = MemoryNode(id="n1", memory="Static", is_static=True)
        n2 = MemoryNode(id="n2", memory="Dynamic task", is_static=False)
        n3 = MemoryNode(id="n3", memory="Forgotten dynamic", is_static=False,
                        is_forgotten=True)
        store.add_node(n1)
        store.add_node(n2)
        store.add_node(n3)

        dynamic = store.get_dynamic_memories()
        assert len(dynamic) == 1
        assert dynamic[0].id == "n2"

    def test_mark_static_true(self, store: SupermemoryStore) -> None:
        """Should mark a memory as static."""
        node = MemoryNode(id="n1", memory="Dynamic fact", is_static=False)
        store.add_node(node)
        result = store.mark_static("n1", True)
        assert result is True
        retrieved = store.get_node("n1")
        assert retrieved is not None
        assert retrieved.is_static is True

    def test_mark_static_false(self, store: SupermemoryStore) -> None:
        """Should mark a static memory as dynamic."""
        node = MemoryNode(id="n1", memory="Static fact", is_static=True)
        store.add_node(node)
        result = store.mark_static("n1", False)
        assert result is True
        retrieved = store.get_node("n1")
        assert retrieved.is_static is False

    def test_mark_static_nonexistent(self, store: SupermemoryStore) -> None:
        """Marking nonexistent node should return False."""
        result = store.mark_static("nonexistent")
        assert result is False

    def test_memory_context_includes_static_dynamic_sections(
        self, store: SupermemoryStore,
    ) -> None:
        """Context should have separate Static Knowledge and Dynamic Context sections."""
        n1 = MemoryNode(id="n1", memory="Python preference", is_static=True)
        n2 = MemoryNode(id="n2", memory="Current feature work", is_static=False)
        store.add_node(n1)
        store.add_node(n2)

        ctx = store.get_memory_context()
        assert "Static Knowledge" in ctx
        assert "Dynamic Context" in ctx
        assert "Python preference" in ctx
        assert "Current feature work" in ctx

    def test_memory_context_with_expiry_hint(self, store: SupermemoryStore) -> None:
        """Dynamic memories with forget_after should show expiry hint."""
        node = MemoryNode(
            id="n1", memory="Temp task",
            is_static=False,
            forget_after="2026-06-01T00:00:00",
        )
        store.add_node(node)
        ctx = store.get_memory_context()
        assert "expires: 2026-06-01" in ctx

    def test_stats_includes_static_dynamic_counts(self, store: SupermemoryStore) -> None:
        """Stats should report static and dynamic memory counts."""
        n1 = MemoryNode(id="n1", memory="Static", is_static=True)
        n2 = MemoryNode(id="n2", memory="Dynamic", is_static=False)
        store.add_node(n1)
        store.add_node(n2)

        stats = store.stats()
        assert stats["static_memories"] == 1
        assert stats["dynamic_memories"] == 1

    def test_stats_includes_embedded_nodes(self, store: SupermemoryStore) -> None:
        """Stats should count nodes with embeddings."""
        n1 = MemoryNode(id="n1", memory="With embedding", embedding=[1.0])
        n2 = MemoryNode(id="n2", memory="Without embedding")
        store.add_node(n1)
        store.add_node(n2)

        stats = store.stats()
        assert stats["embedded_nodes"] == 1


# ===================================================================
# SupermemoryStore — Auto-forgetting (NEW)
# ===================================================================

class TestSupermemoryStoreAutoForget:
    """Test automatic forgetting of expired dynamic memories."""

    def test_auto_forget_expired_dynamic(self, store: SupermemoryStore) -> None:
        """Expired dynamic memories should be auto-forgotten."""
        node = MemoryNode(
            id="n1", memory="Old task",
            is_static=False,
            forget_after="2020-01-01T00:00:00",  # Way in the past
        )
        store.add_node(node)
        forgotten = store.auto_forget_expired()
        assert forgotten == 1
        n = store.get_node("n1")
        assert n is not None
        assert n.is_forgotten is True
        assert "Auto-forgotten" in (n.forget_reason or "")

    def test_auto_forget_preserves_static(self, store: SupermemoryStore) -> None:
        """Static memories should never be auto-forgotten, even if expired."""
        node = MemoryNode(
            id="n1", memory="Permanent preference",
            is_static=True,
            forget_after="2020-01-01T00:00:00",  # Expired but static
        )
        store.add_node(node)
        forgotten = store.auto_forget_expired()
        assert forgotten == 0
        n = store.get_node("n1")
        assert n is not None
        assert n.is_forgotten is False

    def test_auto_forget_future_date_not_expired(self, store: SupermemoryStore) -> None:
        """Memories with future forget_after should NOT be forgotten."""
        node = MemoryNode(
            id="n1", memory="Current sprint task",
            is_static=False,
            forget_after="2099-12-31T00:00:00",  # Far future
        )
        store.add_node(node)
        forgotten = store.auto_forget_expired()
        assert forgotten == 0
        n = store.get_node("n1")
        assert n is not None
        assert n.is_forgotten is False

    def test_auto_forget_no_forget_after(self, store: SupermemoryStore) -> None:
        """Memories without forget_after should not be affected."""
        node = MemoryNode(id="n1", memory="No expiry", is_static=False)
        store.add_node(node)
        forgotten = store.auto_forget_expired()
        assert forgotten == 0

    def test_auto_forget_already_forgotten_skipped(self, store: SupermemoryStore) -> None:
        """Already-forgotten nodes should be skipped."""
        node = MemoryNode(
            id="n1", memory="Already gone",
            is_static=False,
            is_forgotten=True,
            forget_after="2020-01-01T00:00:00",
        )
        store.add_node(node)
        forgotten = store.auto_forget_expired()
        assert forgotten == 0

    def test_auto_forget_multiple_expired(self, store: SupermemoryStore) -> None:
        """Multiple expired memories should be forgotten."""
        for i in range(3):
            node = MemoryNode(
                id=f"n{i}", memory=f"Old task {i}",
                is_static=False,
                forget_after="2020-01-01T00:00:00",
            )
            store.add_node(node)
        # Add one non-expired
        n_future = MemoryNode(
            id="n_future", memory="Future task",
            is_static=False,
            forget_after="2099-01-01T00:00:00",
        )
        store.add_node(n_future)
        forgotten = store.auto_forget_expired()
        assert forgotten == 3

    def test_auto_forget_saves_graph(self, tmp_path: Path) -> None:
        """After auto-forgetting, graph should persist correctly."""
        s1 = SupermemoryStore(tmp_path)
        node = MemoryNode(
            id="n1", memory="Expired task",
            is_static=False,
            forget_after="2020-01-01T00:00:00",
        )
        s1.add_node(node)
        s1.auto_forget_expired()

        # Reload from disk
        s2 = SupermemoryStore(tmp_path)
        n = s2.get_node("n1")
        assert n is not None
        assert n.is_forgotten is True


# ===================================================================
# SupermemoryConsolidator — LLM extraction (NEW)
# ===================================================================

class TestSupermemoryConsolidatorLLMExtraction:
    """Test LLM-based memory extraction from conversation chunks."""

    def test_parse_extraction_response_valid_json(self) -> None:
        """Should parse a valid JSON response correctly."""
        response = '{"memories": [{"text": "User prefers Python", "is_static": true}]}'
        result = SupermemoryConsolidator._parse_extraction_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "User prefers Python"
        assert result[0]["is_static"] is True

    def test_parse_extraction_response_with_markdown_fence(self) -> None:
        """Should handle JSON inside markdown code fences."""
        response = '''```json
{"memories": [
  {"text": "User likes dark mode", "is_static": true},
  {"text": "User is deploying today", "is_static": false, "forget_after": "2026-06-01"}
]}
```'''
        result = SupermemoryConsolidator._parse_extraction_response(response)
        assert result is not None
        assert len(result) == 2
        assert result[0]["text"] == "User likes dark mode"
        assert result[1]["forget_after"] == "2026-06-01"

    def test_parse_extraction_response_with_event_date(self) -> None:
        """Should extract event_date from response."""
        response = '{"memories": [{"text": "User started Rust in 2025", "is_static": true, "event_date": "2025-01-01"}]}'
        result = SupermemoryConsolidator._parse_extraction_response(response)
        assert result is not None
        assert result[0]["event_date"] == "2025-01-01"

    def test_parse_extraction_response_empty(self) -> None:
        """Empty response should return None."""
        assert SupermemoryConsolidator._parse_extraction_response("") is None
        assert SupermemoryConsolidator._parse_extraction_response("   ") is None

    def test_parse_extraction_response_invalid_json(self) -> None:
        """Invalid JSON should return None."""
        result = SupermemoryConsolidator._parse_extraction_response("not json at all")
        assert result is None

    def test_parse_extraction_response_empty_memories(self) -> None:
        """Response with empty memories array should return None."""
        result = SupermemoryConsolidator._parse_extraction_response(
            '{"memories": []}',
        )
        assert result is None

    def test_parse_extraction_response_missing_memories_key(self) -> None:
        """Response without 'memories' key should return None."""
        result = SupermemoryConsolidator._parse_extraction_response(
            '{"other_key": "value"}',
        )
        assert result is None

    def test_parse_extraction_response_filters_empty_text(self) -> None:
        """Items with empty text should be filtered out."""
        response = '{"memories": [{"text": "", "is_static": true}, {"text": "Valid", "is_static": false}]}'
        result = SupermemoryConsolidator._parse_extraction_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "Valid"

    async def test_generate_memories_with_llm_success(
        self,
        consolidator: SupermemoryConsolidator,
        mock_provider: MagicMock,
    ) -> None:
        """Should use LLM to generate properly classified memories."""
        mock_provider.chat = AsyncMock(return_value=MagicMock(
            content='{"memories": [{"text": "User prefers Python", "is_static": true}, {"text": "User deploying auth module", "is_static": false, "forget_after": "2026-06-01"}]}',
        ))

        chunk = [
            {"role": "user", "content": "I prefer Python for backend work.",
             "timestamp": "2026-05-01 10:00:00"},
            {"role": "assistant", "content": "Noted.",
             "timestamp": "2026-05-01 10:00:05"},
        ]
        nodes = await consolidator._generate_memories_from_chunk(chunk, "chunk-1")
        assert len(nodes) == 2

        # First memory is static (preference)
        static_nodes = [n for n in nodes if n.is_static]
        dynamic_nodes = [n for n in nodes if not n.is_static]
        assert len(static_nodes) == 1
        assert len(dynamic_nodes) == 1
        assert static_nodes[0].memory == "User prefers Python"
        assert dynamic_nodes[0].forget_after == "2026-06-01"

    async def test_generate_memories_llm_failure_falls_back(
        self,
        consolidator: SupermemoryConsolidator,
        mock_provider: MagicMock,
    ) -> None:
        """When LLM fails, should fall back to heuristic extraction."""
        mock_provider.chat.side_effect = Exception("API error")

        chunk = [
            {"role": "user", "content": "I really enjoy coding in Rust for systems work.",
             "timestamp": "2026-05-01 10:00:00"},
        ]
        nodes = await consolidator._generate_memories_from_chunk(chunk, "chunk-2")
        assert len(nodes) == 1
        assert "User stated:" in nodes[0].memory
        assert nodes[0].is_static is False  # Heuristic doesn't classify as static

    async def test_generate_memories_llm_not_implemented_falls_back(
        self,
        store: SupermemoryStore,
        mock_provider: MagicMock,
        mock_sessions: MagicMock,
    ) -> None:
        """When provider has no chat (NotImplementedError), should use heuristic."""
        mock_provider.chat = AsyncMock(side_effect=NotImplementedError("No chat"))

        c = SupermemoryConsolidator(
            store=store,
            provider=mock_provider,
            model="test-model",
            sessions=mock_sessions,
            context_window_tokens=100_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
        )

        chunk = [
            {"role": "user", "content": "I use VSCode as my editor.",
             "timestamp": "2026-05-01 10:00:00"},
        ]
        nodes = await c._generate_memories_from_chunk(chunk, "chunk-3")
        assert len(nodes) == 1

    async def test_generate_memories_with_event_date(
        self,
        consolidator: SupermemoryConsolidator,
        mock_provider: MagicMock,
    ) -> None:
        """LLM extraction should preserve event_date."""
        mock_provider.chat = AsyncMock(return_value=MagicMock(
            content='{"memories": [{"text": "User started at Acme Corp in March 2020", "is_static": true, "event_date": "2020-03-01"}]}',
        ))

        chunk = [
            {"role": "user", "content": "I joined Acme Corp in March 2020.",
             "timestamp": "2026-05-01 10:00:00"},
        ]
        nodes = await consolidator._generate_memories_from_chunk(chunk, "chunk-4")
        assert len(nodes) == 1
        assert nodes[0].event_date == "2020-03-01"

    def test_jaccard_similarity_identical(self) -> None:
        """Jaccard similarity of identical texts should be 1.0."""
        sim = SupermemoryConsolidator._jaccard_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_similarity_disjoint(self) -> None:
        """Jaccard similarity of completely different texts should be 0.0."""
        sim = SupermemoryConsolidator._jaccard_similarity("hello world", "foo bar baz")
        assert sim == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        """Jaccard with partial overlap should be between 0 and 1."""
        sim = SupermemoryConsolidator._jaccard_similarity(
            "hello world foo", "hello world bar",
        )
        assert 0.0 < sim < 1.0

    def test_cosine_similarity_consolidator(self) -> None:
        """Consolidator cosine similarity should match mathematical expectation."""
        sim = SupermemoryConsolidator._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-9

        sim = SupermemoryConsolidator._cosine_similarity([1.0, 1.0], [1.0, 1.0])
        assert abs(sim - 1.0) < 1e-9


# ===================================================================
# SupermemoryConsolidator — Embedding-based relationship detection (NEW)
# ===================================================================

class TestSupermemoryConsolidatorEmbeddingRelationships:
    """Test embedding-based relationship detection between memories."""

    async def test_embedding_based_updates_detection(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        """With embeddings, high cosine similarity should detect UPDATE."""
        existing = MemoryNode(
            id="old-1",
            memory="User favorite color is blue",
            embedding=[1.0, 0.0, 0.0],
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="User favorite color is now green",
            embedding=[0.95, 0.05, 0.0],  # High cosine similarity ≈ 0.998
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) >= 1
        assert edges[0].edge_type == MemoryRelation.UPDATES

    async def test_embedding_based_extends_detection(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        """Moderate cosine similarity should detect EXTENDS."""
        existing = MemoryNode(
            id="old-1",
            memory="User works at Acme Corp",
            embedding=[1.0, 0.0, 0.0],
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="User is a Senior Engineer at Acme Corp",
            embedding=[0.7, 0.7, 0.0],  # Cosine ≈ 0.707 → EXTENDS
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) >= 1
        assert edges[0].edge_type == MemoryRelation.EXTENDS

    async def test_jaccard_fallback_when_no_embeddings(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        """When nodes have no embeddings, should fall back to Jaccard."""
        existing = MemoryNode(
            id="old-1",
            memory="User favorite programming language is Python for backend development",
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="User favorite programming language is Python for backend",
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) >= 1
        # High word overlap → UPDATE with Jaccard
        assert edges[0].edge_type == MemoryRelation.UPDATES

    async def test_no_relationship_below_threshold(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        """Memories below similarity threshold should not create edges."""
        existing = MemoryNode(
            id="old-1",
            memory="User prefers Python for coding",
            embedding=[1.0, 0.0, 0.0],
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="The weather is nice today",
            embedding=[0.0, 0.0, 1.0],  # Orthogonal → cos = 0
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) == 0

    async def test_mixed_embedding_presence(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        """When one node has embedding and other doesn't, use Jaccard fallback."""
        existing = MemoryNode(
            id="old-1",
            memory="User favourite color blue for the dark theme",
            embedding=[1.0, 0.0, 0.0],  # Has embedding
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="User favourite color blue dark theme looks nice",
            # No embedding → should use Jaccard
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) >= 1

    async def test_relationship_logs_score(
        self,
        consolidator: SupermemoryConsolidator,
        store: SupermemoryStore,
    ) -> None:
        """Relationship detection should log the similarity score."""
        existing = MemoryNode(
            id="old-1",
            memory="Test memory",
            embedding=[1.0, 0.0],
        )
        store.add_node(existing)

        new_node = MemoryNode(
            id="new-1",
            memory="Test memory updated",
            embedding=[0.8, 0.2],
        )
        store.add_node(new_node)

        await consolidator._detect_relationships([new_node])

        edges = store.get_edges_for_node("new-1")
        assert len(edges) >= 1


# ===================================================================
# SupermemoryMemoryAlgorithm — EmbeddingConfig wiring (NEW)
# ===================================================================

class TestSupermemoryEmbeddingConfigWiring:
    """Test embedding_config is properly wired through build()."""

    def test_build_with_embedding_config(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        """When embedding_config is provided, consolidator should use its model."""
        algo = SupermemoryMemoryAlgorithm()
        sessions = MagicMock()
        sessions.save = MagicMock()
        sessions.invalidate = MagicMock()
        sessions.list_sessions = MagicMock(return_value=[])

        # Create a mock embedding_config with a specific model
        embedding_config = MagicMock()
        embedding_config.model = "text-embedding-3-large"

        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="gpt-4",
            sessions=sessions,
            context_window_tokens=100_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=60,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
            embedding_config=embedding_config,
        )

        assert components.consolidator.embedding_model == "text-embedding-3-large"

    def test_build_without_embedding_config(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        """Without embedding_config, consolidator should fall back to chat model."""
        algo = SupermemoryMemoryAlgorithm()
        sessions = MagicMock()
        sessions.save = MagicMock()
        sessions.invalidate = MagicMock()
        sessions.list_sessions = MagicMock(return_value=[])

        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="gpt-4",
            sessions=sessions,
            context_window_tokens=100_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=60,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
            embedding_config=None,
        )

        assert components.consolidator.embedding_model == "gpt-4"

    def test_build_with_embedding_config_no_model_attr(
        self,
        tmp_path: Path,
        mock_provider: MagicMock,
    ) -> None:
        """If embedding_config has no model attribute, fall back to chat model."""
        algo = SupermemoryMemoryAlgorithm()
        sessions = MagicMock()
        sessions.save = MagicMock()
        sessions.invalidate = MagicMock()
        sessions.list_sessions = MagicMock(return_value=[])

        # embedding_config without model attribute
        embedding_config = MagicMock(spec=[])  # No 'model' attribute

        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="claude-3",
            sessions=sessions,
            context_window_tokens=100_000,
            build_messages=MagicMock(return_value=[]),
            get_tool_definitions=MagicMock(return_value=[]),
            max_completion_tokens=4096,
            session_ttl_minutes=60,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=8000,
            annotate_line_ages=True,
            embedding_config=embedding_config,
        )

        assert components.consolidator.embedding_model == "claude-3"


# ===================================================================
# SupermemoryMemoryAlgorithm — Auto-forgetting in Dream (NEW)
# ===================================================================

class TestSupermemoryDreamAutoForget:
    """Test that Dream run() triggers auto-forgetting."""

    async def test_dream_calls_auto_forget(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        """Dream run should call auto_forget_expired after processing."""
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        # Add an expired dynamic memory to trigger auto-forget
        node = MemoryNode(
            id="exp-1", memory="Old task",
            is_static=False,
            forget_after="2020-01-01T00:00:00",
        )
        store.add_node(node)

        result = await dream.run()
        assert result is True

        # Verify the memory was forgotten
        n = store.get_node("exp-1")
        assert n is not None
        assert n.is_forgotten is True

    async def test_dream_auto_forget_preserves_static(
        self,
        dream: SupermemoryDream,
        mock_provider: MagicMock,
        mock_runner: MagicMock,
        store: SupermemoryStore,
    ) -> None:
        """Dream should NOT auto-forget static memories."""
        store.append_history("event 1")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        # Static memory with expired date
        node = MemoryNode(
            id="static-1", memory="Permanent fact",
            is_static=True,
            forget_after="2020-01-01T00:00:00",
        )
        store.add_node(node)

        await dream.run()

        n = store.get_node("static-1")
        assert n is not None
        assert n.is_forgotten is False
