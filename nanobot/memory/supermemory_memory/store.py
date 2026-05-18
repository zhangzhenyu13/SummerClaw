"""Supermemory store — file I/O layer with memory graph, chunks, and relationship tracking.

Extends the naive MemoryStore with Supermemory-specific data structures:
- memory_graph.json : Nodes (atomic memories) + Edges (updates/extends/derives relationships)
- chunks/           : Source conversation chunks for hybrid search
- Temporal grounding: documentDate (conversation time) + eventDate (described event time)
- Relational versioning: version chains when facts are updated
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nanobot.memory.embedding_store import EmbeddingStore, batch_cosine_np
from nanobot.memory.naive_memory.store import MemoryStore
from nanobot.utils.helpers import ensure_dir


class MemoryRelation(str, Enum):
    """Relationship types between memory nodes."""
    UPDATES = "updates"    # Contradicts or updates old information (version chain)
    EXTENDS = "extends"    # Adds detail without contradiction
    DERIVES = "derives"    # Inferred from combining multiple distinct memories


@dataclass
class MemoryNode:
    """An atomic piece of information — a single memory.

    Supermemory's key insight: generate atomic memories from chunks
    to resolve ambiguous references and enable high-precision retrieval.
    """
    id: str
    memory: str                     # The atomic memory text
    content: str = ""               # Original source chunk content
    document_date: str = ""         # When the conversation took place
    event_date: str | None = None   # When the described event actually occurred
    version: int = 1
    is_latest: bool = True
    is_forgotten: bool = False
    forget_after: str | None = None
    forget_reason: str | None = None
    parent_memory_id: str | None = None   # Previous version in chain
    root_memory_id: str | None = None     # First version in chain
    is_static: bool = False         # Static (long-term knowledge) vs dynamic (transient)
    created_at: str = ""
    updated_at: str = ""
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.root_memory_id is None:
            self.root_memory_id = self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory": self.memory,
            "content": self.content,
            "document_date": self.document_date,
            "event_date": self.event_date,
            "version": self.version,
            "is_latest": self.is_latest,
            "is_forgotten": self.is_forgotten,
            "forget_after": self.forget_after,
            "forget_reason": self.forget_reason,
            "parent_memory_id": self.parent_memory_id,
            "root_memory_id": self.root_memory_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_static": self.is_static,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryEdge:
    """A semantic relationship between two memory nodes."""
    id: str
    source_id: str   # The memory that references the target
    target_id: str   # The referenced memory
    edge_type: MemoryRelation
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEdge":
        edge_type = data.get("edge_type", "extends")
        if isinstance(edge_type, str):
            edge_type = MemoryRelation(edge_type)
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=edge_type,
            created_at=data.get("created_at", ""),
        )


@dataclass
class SourceChunk:
    """A source conversation chunk for hybrid search."""
    id: str
    content: str                     # Raw chunk text
    document_date: str = ""          # Conversation timestamp
    memory_ids: list[str] = field(default_factory=list)  # Generated memory node IDs
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "document_date": self.document_date,
            "memory_ids": self.memory_ids,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceChunk":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SupermemoryStore(MemoryStore):
    """Extends MemoryStore with Supermemory graph, chunks, and relationship tracking.

    On-disk layout (inside workspace/memory/):
        MEMORY.md         — formatted memory context (inherited)
        history.jsonl     — raw conversation history (inherited)
        memory_graph.json — nodes + edges for the memory knowledge graph
        chunks/           — source conversation chunks for hybrid search
                             (chunk_<uuid>.json)

    The SOUL.md and USER.md files remain in the workspace root (inherited).
    """

    _DEFAULT_MAX_NODES = 5000

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for empty vectors, mismatched lengths, or zero-norm vectors.
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __init__(
        self,
        workspace: Path,
        max_history_entries: int = MemoryStore._DEFAULT_MAX_HISTORY,
        max_nodes: int = _DEFAULT_MAX_NODES,
        algo_name: str | None = None,
    ) -> None:
        # Ensure supermemory logs are always visible even when the CLI's
        # ``logger.disable("nanobot")`` silences the parent namespace.
        logger.enable("nanobot.memory.supermemory_memory")

        super().__init__(workspace, max_history_entries=max_history_entries, algo_name=algo_name)
        self.max_nodes = max_nodes
        self._graph_file = self.memory_dir / "memory_graph.json"
        self._chunks_dir = ensure_dir(self.memory_dir / "chunks")

        # Migrate legacy supermemory-specific files if needed
        if algo_name:
            self._migrate_supermemory_legacy()

        # Embedding store — numpy binary chunked files, decoupled from JSON content
        self._embeddings = EmbeddingStore(self.memory_dir, prefix="supermemory_embeddings")

        # In-memory caches (lazy-loaded, flushed on write)
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: dict[str, MemoryEdge] = {}
        self._chunks: dict[str, SourceChunk] = {}
        self._load_graph()

    def _migrate_supermemory_legacy(self) -> None:
        """Migrate supermemory-specific files from the legacy location."""
        from nanobot.memory.migrate import maybe_migrate_legacy_files
        old_memory_dir = self.workspace / "memory"
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=self.workspace,
            files=["memory_graph.json"],
            dirs=["chunks"],
        )

    # ------------------------------------------------------------------
    # Graph persistence
    # ------------------------------------------------------------------

    def _load_graph(self) -> None:
        """Load memory graph (nodes + edges) from disk."""
        if not self._graph_file.exists():
            return
        try:
            raw = json.loads(self._graph_file.read_text(encoding="utf-8"))
            self._nodes = {
                nid: MemoryNode.from_dict(nd)
                for nid, nd in raw.get("nodes", {}).items()
            }
            self._edges = {
                eid: MemoryEdge.from_dict(ed)
                for eid, ed in raw.get("edges", {}).items()
            }

            # Migrate old-format embeddings (stored inline in JSON nodes) to the
            # numpy binary EmbeddingStore, then strip them from records.
            _has_old_embeddings = any(
                isinstance(n.embedding, list) and len(n.embedding) > 0
                for n in self._nodes.values()
            )
            if _has_old_embeddings:
                logger.info(
                    "SupermemoryStore: detected old-format embeddings in JSON — migrating to .npy"
                )
                migrated = 0
                for node in self._nodes.values():
                    emb = node.embedding
                    if emb and isinstance(emb, list) and len(emb) > 0:
                        self._embeddings.add(node.id, emb)
                        node.embedding = None
                        migrated += 1
                if migrated > 0:
                    self._save_graph()
                    logger.info(
                        "SupermemoryStore: migrated {} embeddings, JSON cleaned", migrated,
                    )

            # Load chunks
            for chunk_file in sorted(self._chunks_dir.glob("chunk_*.json")):
                try:
                    chunk = SourceChunk.from_dict(
                        json.loads(chunk_file.read_text(encoding="utf-8"))
                    )
                    self._chunks[chunk.id] = chunk
                except Exception:
                    logger.warning("Failed to load chunk file {}", chunk_file)
        except Exception:
            logger.exception("Failed to load memory graph, starting fresh")
            self._nodes = {}
            self._edges = {}
            self._chunks = {}

    def _save_graph(self) -> None:
        """Persist memory graph to disk (embeddings live in EmbeddingStore, not JSON)."""
        try:
            raw = {
                "nodes": {
                    nid: {k: v for k, v in n.to_dict().items() if k != "embedding"}
                    for nid, n in self._nodes.items()
                },
                "edges": {eid: e.to_dict() for eid, e in self._edges.items()},
            }
            self._graph_file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save memory graph")

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: MemoryNode) -> str:
        """Add or update a memory node. Returns the node ID."""
        is_new = node.id not in self._nodes
        self._nodes[node.id] = node
        # Store embedding in decoupled EmbeddingStore if present
        if node.embedding and isinstance(node.embedding, list) and len(node.embedding) > 0:
            self._embeddings.add(node.id, node.embedding)
        self._save_graph()
        self._compact_nodes()
        if is_new:
            logger.debug(
                "Memory node added: id={}, memory='{}', static={}, nodes_total={}",
                node.id[:8], node.memory[:80], node.is_static, len(self._nodes),
            )
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def list_nodes(self, include_forgotten: bool = False) -> list[MemoryNode]:
        nodes = list(self._nodes.values())
        if not include_forgotten:
            nodes = [n for n in nodes if not n.is_forgotten]
        return sorted(nodes, key=lambda n: n.created_at, reverse=True)

    def get_latest_nodes(self) -> list[MemoryNode]:
        """Return only the latest version of each memory chain."""
        return [n for n in self._nodes.values() if n.is_latest and not n.is_forgotten]

    def forget_node(self, node_id: str, reason: str = "") -> None:
        """Mark a node as forgotten."""
        node = self._nodes.get(node_id)
        if node:
            node.is_forgotten = True
            node.forget_reason = reason
            node.forget_after = datetime.now().isoformat()
            node.updated_at = datetime.now().isoformat()
            self._save_graph()
            logger.info(
                "Memory forgotten: id={}, memory='{}', reason='{}'",
                node_id[:8], node.memory[:80], reason or "(unspecified)",
            )

    def _compact_nodes(self) -> None:
        """Drop oldest nodes if exceeding max_nodes, keeping forgotten first."""
        if self.max_nodes <= 0:
            return
        if len(self._nodes) <= self.max_nodes:
            return
        # Sort: forgotten first, then by created_at ascending (oldest first)
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: (not n.is_forgotten, n.created_at),
        )
        to_remove = len(self._nodes) - self.max_nodes
        for node in sorted_nodes[:to_remove]:
            del self._nodes[node.id]
            # Also remove related edges
            edge_ids_to_remove = [
                eid for eid, e in self._edges.items()
                if e.source_id == node.id or e.target_id == node.id
            ]
            for eid in edge_ids_to_remove:
                del self._edges[eid]
        logger.warning(
            "Node compaction: removed {} oldest nodes (max={}, was={})",
            to_remove, self.max_nodes, len(self._nodes) + to_remove,
        )
        self._save_graph()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge) -> str:
        """Add a relationship edge between two memory nodes."""
        # Verify both nodes exist
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} not found")

        # Avoid duplicate edges of same type between same nodes
        for existing in self._edges.values():
            if (existing.source_id == edge.source_id
                    and existing.target_id == edge.target_id
                    and existing.edge_type == edge.edge_type):
                return existing.id

        self._edges[edge.id] = edge
        self._save_graph()
        source_mem = self._nodes[edge.source_id].memory[:60]
        target_mem = self._nodes[edge.target_id].memory[:60]
        logger.debug(
            "Memory edge added: {} {} -> {} ('{}' -> '{}'), edges_total={}",
            edge.id[:8], edge.edge_type.value, edge.target_id[:8],
            source_mem, target_mem, len(self._edges),
        )
        return edge.id

    def get_edges_for_node(self, node_id: str) -> list[MemoryEdge]:
        """Get all edges connected to a node."""
        return [
            e for e in self._edges.values()
            if e.source_id == node_id or e.target_id == node_id
        ]

    def get_version_chain(self, root_memory_id: str) -> list[MemoryNode]:
        """Get all versions in a memory chain, oldest first."""
        chain = [n for n in self._nodes.values()
                 if n.root_memory_id == root_memory_id]
        return sorted(chain, key=lambda n: n.version)

    # ------------------------------------------------------------------
    # Relationship management (Supermemory-style versioning)
    # ------------------------------------------------------------------

    def create_new_version(
        self,
        old_node_id: str,
        new_memory: str,
        new_content: str = "",
        event_date: str | None = None,
    ) -> MemoryNode:
        """Create a new version of an existing memory (updates relationship).

        This implements Supermemory's relational versioning: when a fact changes,
        we create a version chain rather than overwriting the old memory.
        """
        old = self._nodes.get(old_node_id)
        if not old:
            raise ValueError(f"Node {old_node_id} not found")

        # Mark old version as not latest
        old.is_latest = False
        old.updated_at = datetime.now().isoformat()

        # Create new version
        new_node = MemoryNode(
            id=str(uuid.uuid4()),
            memory=new_memory,
            content=new_content or old.content,
            document_date=datetime.now().strftime("%Y-%m-%d"),
            event_date=event_date,
            version=old.version + 1,
            is_latest=True,
            parent_memory_id=old.id,
            root_memory_id=old.root_memory_id or old.id,
        )
        self._nodes[new_node.id] = new_node

        # Create updates edge
        edge = MemoryEdge(
            id=str(uuid.uuid4()),
            source_id=new_node.id,
            target_id=old.id,
            edge_type=MemoryRelation.UPDATES,
        )
        self._edges[edge.id] = edge

        self._save_graph()
        logger.info(
            "Memory version created: '{}' (v{}) updates '{}' (v{})",
            new_node.memory[:60], new_node.version, old.memory[:60], old.version,
        )
        return new_node

    def extend_memory(
        self,
        source_node_id: str,
        extension_memory: str,
        extension_content: str = "",
    ) -> MemoryNode:
        """Add detail to an existing memory (extends relationship)."""
        src = self._nodes.get(source_node_id)
        if not src:
            raise ValueError(f"Node {source_node_id} not found")

        new_node = MemoryNode(
            id=str(uuid.uuid4()),
            memory=extension_memory,
            content=extension_content or src.content,
            document_date=datetime.now().strftime("%Y-%m-%d"),
            event_date=src.event_date,
        )
        self._nodes[new_node.id] = new_node

        edge = MemoryEdge(
            id=str(uuid.uuid4()),
            source_id=new_node.id,
            target_id=source_node_id,
            edge_type=MemoryRelation.EXTENDS,
        )
        self._edges[edge.id] = edge

        self._save_graph()
        logger.debug(
            "Memory extended: '{}' extends '{}'",
            extension_memory[:60], src.memory[:60],
        )
        return new_node

    def derive_memory(
        self,
        source_ids: list[str],
        derived_memory: str,
        derived_content: str = "",
    ) -> MemoryNode:
        """Create inferred memory from combining multiple existing memories (derives relationship)."""
        new_node = MemoryNode(
            id=str(uuid.uuid4()),
            memory=derived_memory,
            content=derived_content,
            document_date=datetime.now().strftime("%Y-%m-%d"),
        )
        self._nodes[new_node.id] = new_node

        for src_id in source_ids:
            if src_id in self._nodes:
                edge = MemoryEdge(
                    id=str(uuid.uuid4()),
                    source_id=new_node.id,
                    target_id=src_id,
                    edge_type=MemoryRelation.DERIVES,
                )
                self._edges[edge.id] = edge

        self._save_graph()
        logger.debug(
            "Memory derived: '{}' from {} source(s)",
            derived_memory[:60], len(source_ids),
        )
        return new_node

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: SourceChunk) -> str:
        """Store a source conversation chunk."""
        self._chunks[chunk.id] = chunk
        chunk_file = self._chunks_dir / f"chunk_{chunk.id}.json"
        chunk_file.write_text(
            json.dumps(chunk.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(
            "Source chunk added: id={}, mem_refs={}, chunks_total={}",
            chunk.id[:8], len(chunk.memory_ids), len(self._chunks),
        )
        return chunk.id

    def get_chunk(self, chunk_id: str) -> SourceChunk | None:
        return self._chunks.get(chunk_id)

    def get_chunks_for_memory(self, memory_id: str) -> list[SourceChunk]:
        """Get source chunks associated with a memory node (hybrid search)."""
        return [c for c in self._chunks.values() if memory_id in c.memory_ids]

    def list_chunks(self) -> list[SourceChunk]:
        return sorted(self._chunks.values(), key=lambda c: c.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def search_memories_by_keyword(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Simple keyword-based search across memory nodes.

        For full semantic search, use the embedding-based hybrid search
        which combines semantic similarity on memories with source chunk retrieval.
        """
        query_lower = query.lower()
        results: list[tuple[MemoryNode, int]] = []

        for node in self._nodes.values():
            if node.is_forgotten:
                continue
            if not node.is_latest:
                continue
            score = 0
            memory_lower = node.memory.lower()
            content_lower = node.content.lower()

            # Exact match
            if query_lower in memory_lower:
                score += 10
            if query_lower in content_lower:
                score += 5

            # Word overlap
            query_words = set(query_lower.split())
            memory_words = set(memory_lower.split())
            content_words = set(content_lower.split())

            memory_overlap = len(query_words & memory_words)
            content_overlap = len(query_words & content_words)
            score += memory_overlap * 3 + content_overlap * 1

            if score > 0:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top = [r[0] for r in results[:limit]]
        if top:
            logger.debug(
                "Keyword search: query='{}' → {} results (top: '{}')",
                query[:60], len(top), top[0].memory[:80],
            )
        else:
            logger.debug("Keyword search: query='{}' → no results", query[:60])
        return top

    def search_memories_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[tuple[MemoryNode, float]]:
        """Semantic search using embedding cosine similarity via EmbeddingStore.

        Uses numpy-accelerated batch cosine similarity on the decoupled
        EmbeddingStore for efficient vector search across all indexed nodes.

        Args:
            query_embedding: The embedding vector of the search query.
            limit: Maximum number of results to return.
            threshold: Minimum cosine similarity score (0.0 to 1.0).

        Returns:
            List of (MemoryNode, score) tuples sorted by descending score.
        """
        if not query_embedding:
            return []

        mem_ids, emb_matrix = self._embeddings.get_all_embeddings()
        if len(mem_ids) == 0 or emb_matrix.shape[1] == 0:
            return []

        q_emb = np.array(query_embedding, dtype=np.float32)
        scores = batch_cosine_np(q_emb, emb_matrix)

        scored: list[tuple[MemoryNode, float]] = []
        for mid, sim in zip(mem_ids, scores):
            sim_f = float(sim)
            if sim_f < threshold:
                continue
            node = self._nodes.get(mid)
            if node is None or node.is_forgotten or not node.is_latest:
                continue
            scored.append((node, sim_f))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]
        if top:
            logger.debug(
                "Embedding search: {} results (top score={:.3f}, '{}')",
                len(top), top[0][1], top[0][0].memory[:80],
            )
        else:
            logger.debug("Embedding search: no results above threshold")
        return top

    def set_node_embedding(self, node_id: str, embedding: list[float]) -> bool:
        """Set the embedding for an existing memory node via EmbeddingStore.

        Embeddings are stored in the decoupled numpy binary store,
        not inline in the JSON graph file.

        Returns True if the node was found and updated, False otherwise.
        """
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("set_node_embedding: node {} not found", node_id[:8])
            return False
        self._embeddings.add(node_id, embedding)
        # Also set on the in-memory node for transient access (consolidator)
        node.embedding = embedding
        node.updated_at = datetime.now().isoformat()
        # Save graph without embedding field
        self._save_graph()
        logger.debug(
            "Embedding set for node {} (dim={})", node_id[:8], len(embedding),
        )
        return True

    def get_nodes_without_embeddings(self) -> list[MemoryNode]:
        """Return latest, active nodes that are missing embeddings in EmbeddingStore."""
        return [
            n for n in self._nodes.values()
            if n.is_latest and not n.is_forgotten
            and self._embeddings.get(n.id) is None
        ]

    # ------------------------------------------------------------------
    # Static / Dynamic memory profiling
    # ------------------------------------------------------------------

    def get_static_memories(self) -> list[MemoryNode]:
        """Return static (long-term knowledge) memories.

        Static memories represent enduring facts about the user:
        preferences, identity, permanent knowledge, skills, etc.
        These should be preserved indefinitely.
        """
        return [
            n for n in self._nodes.values()
            if n.is_latest and not n.is_forgotten and n.is_static
        ]

    def get_dynamic_memories(self) -> list[MemoryNode]:
        """Return dynamic (transient/contextual) memories.

        Dynamic memories represent ephemeral states: current tasks,
        recent conversations, temporary contexts. These may be
        candidates for automatic forgetting after expiration.
        """
        return [
            n for n in self._nodes.values()
            if n.is_latest and not n.is_forgotten and not n.is_static
        ]

    def mark_static(self, node_id: str, static: bool = True) -> bool:
        """Mark a memory node as static (or dynamic).

        Returns True if the node was found and updated.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.is_static = static
        node.updated_at = datetime.now().isoformat()
        self._save_graph()
        logger.debug(
            "Memory marked as {}: id={}, memory='{}'",
            "static" if static else "dynamic", node_id[:8], node.memory[:60],
        )
        return True

    # ------------------------------------------------------------------
    # Automatic forgetting
    # ------------------------------------------------------------------

    def auto_forget_expired(self) -> int:
        """Automatically forget dynamic memories whose forget_after date has passed.

        Only applies to dynamic (non-static) memories with a forget_after
        timestamp set. Static memories are never auto-forgotten.

        Returns the number of memories that were forgotten.
        """
        now = datetime.now().isoformat()
        forgotten_count = 0

        for node in list(self._nodes.values()):
            if node.is_forgotten or not node.is_latest:
                continue
            if node.is_static:
                continue
            if not node.forget_after:
                continue
            if node.forget_after > now:
                continue  # Not yet expired

            # Mark as forgotten
            node.is_forgotten = True
            node.forget_reason = (
                f"Auto-forgotten: forget_after={node.forget_after}, "
                f"now={now[:10]}"
            )
            node.updated_at = now
            forgotten_count += 1
            logger.debug(
                "Auto-forgot dynamic memory '{}': {}",
                node.memory[:60], node.forget_after,
            )

        if forgotten_count > 0:
            self._save_graph()
            logger.info("Auto-forgot {} expired dynamic memories", forgotten_count)

        return forgotten_count

    # ------------------------------------------------------------------
    # Format output for context injection
    # ------------------------------------------------------------------

    def get_memory_context(self) -> str:
        """Build the Supermemory-enhanced context for injection into the LLM prompt.

        Includes: long-term memory (MEMORY.md), plus a summary of the memory graph
        showing latest facts organized by static/dynamic category with relationship hints.
        """
        parts: list[str] = []

        # Standard long-term memory
        long_term = self.read_memory()
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")

        # Memory graph summary organized by static/dynamic
        static_memories = self.get_static_memories()
        dynamic_memories = self.get_dynamic_memories()

        if static_memories or dynamic_memories:
            lines = ["## Memory Graph (Latest Facts)"]

            if static_memories:
                lines.append(f"\n### Static Knowledge ({len(static_memories)} facts)")
                for node in static_memories[:30]:
                    meta = self._format_node_meta(node)
                    lines.append(f"- {node.memory}{meta}")

            if dynamic_memories:
                lines.append(f"\n### Dynamic Context ({len(dynamic_memories)} facts)")
                for node in dynamic_memories[:20]:
                    meta = self._format_node_meta(node)
                    lines.append(f"- {node.memory}{meta}")

            parts.append("\n".join(lines))

        result = "\n\n".join(parts) if parts else ""
        if result:
            logger.debug(
                "Memory context built: {} total nodes (static={}, dynamic={}), "
                "{} edges, {} chars",
                len(static_memories) + len(dynamic_memories),
                len(static_memories), len(dynamic_memories),
                len(self._edges), len(result),
            )
        return result

    def _format_node_meta(self, node: MemoryNode) -> str:
        """Format metadata suffix for a memory node in context display."""
        meta = ""
        if node.event_date:
            meta += f" [event: {node.event_date}]"
        if node.version > 1:
            meta += f" (v{node.version})"
        if node.forget_after:
            meta += f" [expires: {node.forget_after[:10]}]"
        # Look up relationships
        edges = self.get_edges_for_node(node.id)
        rel_hints = []
        for e in edges:
            target = self.get_node(e.target_id if e.source_id == node.id else e.source_id)
            if target and target.is_latest:
                rel_hints.append(f"{e.edge_type.value} → {target.memory[:60]}")
        if rel_hints:
            meta += " [" + "; ".join(rel_hints[:2]) + "]"
        return meta

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return memory graph statistics for monitoring."""
        nodes = list(self._nodes.values())
        # Count version chains: groups of nodes sharing a root_memory_id
        chain_groups: dict[str, list[MemoryNode]] = {}
        for n in nodes:
            root = n.root_memory_id or n.id
            chain_groups.setdefault(root, []).append(n)
        version_chains = len([
            root for root, members in chain_groups.items()
            if len(members) > 1
        ])

        return {
            "total_nodes": len(nodes),
            "active_nodes": len([n for n in nodes if n.is_latest and not n.is_forgotten]),
            "forgotten_nodes": len([n for n in nodes if n.is_forgotten]),
            "static_memories": len([n for n in nodes if n.is_latest and not n.is_forgotten and n.is_static]),
            "dynamic_memories": len([n for n in nodes if n.is_latest and not n.is_forgotten and not n.is_static]),
            "version_chains": version_chains,
            "total_edges": len(self._edges),
            "total_chunks": len(self._chunks),
            "embedded_nodes": self._embeddings.get_embedding_count(),
            "edges_by_type": {
                rt.value: len([e for e in self._edges.values() if e.edge_type == rt])
                for rt in MemoryRelation
            },
        }
