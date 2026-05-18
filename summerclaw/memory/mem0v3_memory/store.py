"""Mem0V3 store — file-based vector store with entity linking and SQLite message log.

Ports the core storage architecture of mem0 v3 to summerclaw's file-based
workspace model.  Embeddings are stored in numpy binary chunks via
EmbeddingStore, decoupled from the JSON records.

Layers:
  - Memory records: JSON file with {id, text, hash, lemmatized, entities, created_at, ...}
  - Entity store:   JSON file with {id, text, type, linked_memory_ids}
  - Message log:    SQLite (mirrors mem0's SQLiteManager)
  - BM25 index:     simple TF-IDF inverted index (no external lib needed)
  - Embedding store: numpy .npy chunks (mem0v3_mem_embeddings_*, mem0v3_ent_embeddings_*)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from summerclaw.memory.embedding_store import EmbeddingStore, batch_cosine_np
from summerclaw.memory.migrate import maybe_migrate_legacy_files

# ---------------------------------------------------------------------------
# Simple BM25-like inverted index
# ---------------------------------------------------------------------------

class BM25Index:
    """Minimal TF-IDF inverted index for keyword search.

    No external dependencies — built entirely on Python dicts and sets.
    Designed for small-to-medium memory collections (up to ~100k records).
    """

    def __init__(self) -> None:
        # inverted_index[term] = {doc_id: term_frequency}
        self._inverted: dict[str, dict[str, int]] = {}
        # doc_lengths[doc_id] = total_terms
        self._doc_lengths: dict[str, int] = {}
        self._doc_count = 0
        self._avg_dl = 0.0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        """Add or update a document in the index."""
        self.remove(doc_id)  # remove old entry first
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            self._inverted.setdefault(term, {})[doc_id] = freq
        self._doc_lengths[doc_id] = len(tokens)
        self._doc_count += 1
        if self._doc_count > 0:
            self._avg_dl = sum(self._doc_lengths.values()) / self._doc_count

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._doc_lengths:
            return
        for term_dict in self._inverted.values():
            term_dict.pop(doc_id, None)
        self._doc_lengths.pop(doc_id, None)
        self._doc_count -= 1
        if self._doc_count > 0:
            self._avg_dl = sum(self._doc_lengths.values()) / self._doc_count

    def search(self, query_tokens: list[str], top_k: int = 60) -> list[tuple[str, float]]:
        """BM25-style keyword search. Returns (doc_id, score) sorted by score desc."""
        if not query_tokens or self._doc_count == 0:
            return []

        k1 = 1.2
        b = 0.75
        scores: dict[str, float] = {}

        for term in set(query_tokens):
            postings = self._inverted.get(term, {})
            if not postings:
                continue
            idf = math.log(1 + (self._doc_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, tf in postings.items():
                dl = self._doc_lengths.get(doc_id, 1)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / max(self._avg_dl, 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def to_dict(self) -> dict:
        return {
            "inverted": {k: dict(v) for k, v in self._inverted.items()},
            "doc_lengths": dict(self._doc_lengths),
            "doc_count": self._doc_count,
            "avg_dl": self._avg_dl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        idx = cls()
        idx._inverted = {k: dict(v) for k, v in data.get("inverted", {}).items()}
        idx._doc_lengths = dict(data.get("doc_lengths", {}))
        idx._doc_count = data.get("doc_count", 0)
        idx._avg_dl = data.get("avg_dl", 0.0)
        return idx


# ---------------------------------------------------------------------------
# Simple SQLite message log (mirrors mem0's SQLiteManager)
# ---------------------------------------------------------------------------

class MessageLog:
    """Lightweight SQLite-based message log for context gathering."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_scope TEXT,
                    role TEXT,
                    content TEXT,
                    name TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()
            conn.close()

    def save_messages(self, messages: list[dict], session_scope: str) -> None:
        if not messages:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                for msg in messages:
                    conn.execute(
                        "INSERT INTO messages (id, session_scope, role, content, name, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), session_scope, msg.get("role"),
                         msg.get("content", ""), msg.get("name"), now),
                    )
                # Keep only the most recent 20 messages per scope
                conn.execute(
                    "DELETE FROM messages WHERE session_scope = ? AND id NOT IN ("
                    "  SELECT id FROM messages WHERE session_scope = ? "
                    "  ORDER BY created_at DESC LIMIT 20"
                    ")",
                    (session_scope, session_scope),
                )
                conn.commit()
            finally:
                conn.close()

    def get_last_messages(self, session_scope: str, limit: int = 10) -> list[dict]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(
                    "SELECT role, content, name, created_at FROM ("
                    "  SELECT role, content, name, created_at FROM messages "
                    "  WHERE session_scope = ? ORDER BY created_at DESC LIMIT ?"
                    ") ORDER BY created_at ASC",
                    (session_scope, limit),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        return [{"role": r[0], "content": r[1], "name": r[2], "created_at": r[3]} for r in rows]


# ---------------------------------------------------------------------------
# Mem0V3Store
# ---------------------------------------------------------------------------

_DEFAULT_ENTITY_SIMILARITY_THRESHOLD = 0.85


class Mem0V3Store:
    """File-based vector store implementing mem0 v3's storage architecture.

    All data lives under ``workspace/memory/``:
      - ``mem0v3_memories.json`` — memory records with embeddings
      - ``mem0v3_entities.json``  — entity index with embeddings
      - ``mem0v3_bm25.json``      — persisted BM25 index
      - ``mem0v3_messages.db``    — SQLite message log
      - ``MEMORY.md``             — human-readable memory file (Dream output)
      - ``history.jsonl``         — conversation history (shared with naive)
    """

    def __init__(
        self,
        workspace: Path,
        *,
        entity_similarity_threshold: float = _DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
        algo_name: str | None = None,
    ):
        self.workspace = Path(workspace) if not isinstance(workspace, Path) else workspace
        self.entity_similarity_threshold = entity_similarity_threshold

        if algo_name:
            self._algo_name = algo_name
            self.memory_dir = self.workspace / "memory" / algo_name
        else:
            self._algo_name = None
            self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._memories_path = self.memory_dir / "mem0v3_memories.json"
        self._entities_path = self.memory_dir / "mem0v3_entities.json"
        self._bm25_path = self.memory_dir / "mem0v3_bm25.json"
        self._db_path = self.memory_dir / "mem0v3_messages.db"
        self._memory_md_path = self.memory_dir / "MEMORY.md"
        self.memory_file = self._memory_md_path
        self.history_file = self.memory_dir / "history.jsonl"

        # Embedding stores — numpy binary chunked files, decoupled from JSON
        self._mem_embeddings = EmbeddingStore(self.memory_dir, prefix="mem0v3_mem_embeddings")
        self._ent_embeddings = EmbeddingStore(self.memory_dir, prefix="mem0v3_ent_embeddings")

        # Migrate legacy shared files if needed
        if algo_name:
            self._migrate_from_legacy()

        # In-memory state
        self._memories: dict[str, dict] = {}   # memory_id -> record
        self._entities: dict[str, dict] = {}   # entity_id -> record
        self._bm25 = BM25Index()
        self._messages = MessageLog(str(self._db_path))

        self._load()

    def _migrate_from_legacy(self) -> None:
        """Migrate data from the legacy shared location to the algorithm-specific dir."""
        old_memory_dir = self.workspace / "memory"
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=self.workspace,
            files=[
                "mem0v3_memories.json",
                "mem0v3_entities.json",
                "mem0v3_bm25.json",
                "mem0v3_messages.db",
                "MEMORY.md",
                "history.jsonl",
            ],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all state from disk, migrating old inline embeddings to EmbeddingStore."""
        # Memories
        if self._memories_path.exists():
            try:
                with open(self._memories_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memories = data.get("memories", {})

                # Migrate old-format embeddings to EmbeddingStore
                _has_old_mem_embs = any(
                    isinstance(rec.get("embedding"), list) and len(rec.get("embedding", [])) > 0
                    for rec in self._memories.values()
                )
                if _has_old_mem_embs:
                    logger.info(
                        "Mem0V3Store: detected old-format memory embeddings in JSON — migrating to .npy"
                    )
                    mem_migrated = 0
                    for mem_id, rec in self._memories.items():
                        emb = rec.get("embedding")
                        if emb and isinstance(emb, list) and len(emb) > 0:
                            self._mem_embeddings.add(mem_id, emb)
                            rec.pop("embedding", None)
                            mem_migrated += 1
                    if mem_migrated > 0:
                        self._save_memories()
                        logger.info(
                            "Mem0V3Store: migrated {} memory embeddings, JSON cleaned", mem_migrated,
                        )

                # Rebuild BM25 from loaded memories
                for mem_id, rec in self._memories.items():
                    tokens = rec.get("lemmatized", "").split()
                    if tokens:
                        for t in tokens:
                            self._bm25._inverted.setdefault(t, {})[mem_id] = \
                                self._bm25._inverted.setdefault(t, {}).get(mem_id, 0) + 1
                    self._bm25._doc_lengths[mem_id] = len(tokens)
                self._bm25._doc_count = len(self._memories)
                if self._bm25._doc_count > 0:
                    self._bm25._avg_dl = sum(self._bm25._doc_lengths.values()) / self._bm25._doc_count
            except Exception:
                logger.warning("Failed to load memories; starting fresh")
                self._memories = {}

        # Entities
        if self._entities_path.exists():
            try:
                with open(self._entities_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entities = data.get("entities", {})

                # Migrate old-format entity embeddings to EmbeddingStore
                _has_old_ent_embs = any(
                    isinstance(rec.get("embedding"), list) and len(rec.get("embedding", [])) > 0
                    for rec in self._entities.values()
                )
                if _has_old_ent_embs:
                    logger.info(
                        "Mem0V3Store: detected old-format entity embeddings in JSON — migrating to .npy"
                    )
                    ent_migrated = 0
                    for eid, erec in self._entities.items():
                        emb = erec.get("embedding")
                        if emb and isinstance(emb, list) and len(emb) > 0:
                            self._ent_embeddings.add(eid, emb)
                            erec.pop("embedding", None)
                            ent_migrated += 1
                    if ent_migrated > 0:
                        self._save_entities()
                        logger.info(
                            "Mem0V3Store: migrated {} entity embeddings, JSON cleaned", ent_migrated,
                        )
            except Exception:
                logger.warning("Failed to load entities; starting fresh")
                self._entities = {}

    def _save_memories(self) -> None:
        """Persist memories to JSON (embeddings live in EmbeddingStore, not JSON)."""
        clean: dict[str, Any] = {}
        for mid, rec in self._memories.items():
            clean[mid] = {k: v for k, v in rec.items() if k != "embedding"}
        data: dict[str, Any] = {"memories": clean, "version": 2}
        # Write atomically
        tmp = self._memories_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._memories_path)

    def _save_entities(self) -> None:
        """Persist entities to JSON (embeddings live in EmbeddingStore, not JSON)."""
        clean: dict[str, Any] = {}
        for eid, erec in self._entities.items():
            clean[eid] = {k: v for k, v in erec.items() if k != "embedding"}
        data: dict[str, Any] = {"entities": clean, "version": 2}
        tmp = self._entities_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._entities_path)

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def insert_memories_batch(
        self,
        records: list[dict],
    ) -> list[str]:
        """Batch-insert memory records.

        Each record is a dict with keys:
          - text (required)
          - embedding (optional, list[float])
          - lemmatized (optional, str)
          - hash (optional, str; auto-generated if absent)
          - created_at (optional, ISO str)
          - metadata (optional, dict)

        Returns list of inserted memory IDs.
        """
        now = datetime.now(timezone.utc).isoformat()
        inserted: list[str] = []

        for rec in records:
            text = rec.get("text", "")
            if not text:
                continue
            mem_id = str(uuid.uuid4())
            mem_hash = rec.get("hash") or hashlib.md5(text.encode()).hexdigest()

            # Check for duplicates
            existing_hash = any(
                m.get("hash") == mem_hash for m in self._memories.values()
            )
            if existing_hash:
                continue

            lemmatized = rec.get("lemmatized", text)
            tokens = lemmatized.split()

            record = {
                "id": mem_id,
                "text": text,
                "hash": mem_hash,
                "lemmatized": lemmatized,
                "created_at": rec.get("created_at", now),
                "updated_at": now,
                "metadata": rec.get("metadata", {}),
            }
            self._memories[mem_id] = record
            self._bm25.add(mem_id, tokens)

            # Store embedding in decoupled EmbeddingStore
            emb = rec.get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                self._mem_embeddings.add(mem_id, emb)
            inserted.append(mem_id)

        if inserted:
            self._save_memories()
        return inserted

    def get_memory(self, memory_id: str) -> dict | None:
        """Retrieve a single memory by ID."""
        return self._memories.get(memory_id)

    def get_all_memories(self) -> list[dict]:
        """Return all memory records."""
        return [
            {**rec, "embedding": None}  # strip embeddings for memory efficiency
            for rec in self._memories.values()
        ]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        if memory_id not in self._memories:
            return False
        del self._memories[memory_id]
        self._bm25.remove(memory_id)
        self._mem_embeddings.remove(memory_id)
        self._save_memories()
        return True

    def update_memory_metadata(self, memory_id: str, metadata: dict) -> bool:
        """Update metadata for a memory (e.g., access_history).

        Args:
            memory_id: The memory ID to update.
            metadata: New metadata dict to merge with existing.

        Returns:
            True if updated, False if memory not found.
        """
        if memory_id not in self._memories:
            return False
        
        # Merge metadata
        existing_metadata = self._memories[memory_id].get("metadata", {})
        existing_metadata.update(metadata)
        self._memories[memory_id]["metadata"] = existing_metadata
        
        # Persist to disk
        self._save_memories()
        return True

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query_embedding: list[float],
        top_k: int = 60,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Cosine similarity search via EmbeddingStore (numpy-accelerated)."""
        if not self._memories:
            return []

        mem_ids, emb_matrix = self._mem_embeddings.get_all_embeddings()
        if len(mem_ids) == 0 or emb_matrix.shape[1] == 0:
            return []

        q_emb = np.array(query_embedding, dtype=np.float32)
        scores = batch_cosine_np(q_emb, emb_matrix)

        ranked = sorted(
            zip(mem_ids, scores), key=lambda x: x[1], reverse=True
        )
        results = []
        for mid, score in ranked:
            score_f = float(score)
            if score_f < threshold:
                continue
            if len(results) >= top_k:
                break
            rec = self._memories.get(mid)
            if rec:
                results.append({"id": mid, "score": score_f, "payload": deepcopy(rec)})
        return results

    # ------------------------------------------------------------------
    # Keyword (BM25) search
    # ------------------------------------------------------------------

    def search_keyword(
        self,
        query_tokens: list[str],
        top_k: int = 60,
    ) -> list[dict]:
        """BM25 keyword search."""
        ranked = self._bm25.search(query_tokens, top_k=top_k)
        results = []
        for doc_id, score in ranked:
            rec = self._memories.get(doc_id)
            if rec:
                results.append({"id": doc_id, "score": score, "payload": deepcopy(rec)})
        return results

    # ------------------------------------------------------------------
    # Entity store
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        entity_text: str,
        entity_type: str,
        memory_id: str,
        embedding: list[float] | None = None,
    ) -> str | None:
        """Upsert an entity, linking it to a memory.

        Uses EmbeddingStore for entity embedding comparison and storage.
        If an entity with similar embedding already exists (cosine > threshold),
        append memory_id to its linked list. Otherwise create a new entity.
        """
        if embedding is None:
            embedding = []

        # Try to match existing entity by semantic similarity via EmbeddingStore
        best_id: str | None = None
        best_score = 0.0
        if embedding:
            ent_ids, ent_emb_matrix = self._ent_embeddings.get_all_embeddings()
            if len(ent_ids) > 0 and ent_emb_matrix.shape[1] > 0:
                q_emb = np.array(embedding, dtype=np.float32)
                ent_scores = batch_cosine_np(q_emb, ent_emb_matrix)
                for eid, sim in zip(ent_ids, ent_scores):
                    sim_f = float(sim)
                    if sim_f > self.entity_similarity_threshold and sim_f > best_score:
                        best_score = sim_f
                        best_id = eid

        if best_id is not None:
            # Update existing entity
            erec = self._entities[best_id]
            linked = set(erec.get("linked_memory_ids", []))
            linked.add(memory_id)
            erec["linked_memory_ids"] = sorted(linked)
            self._save_entities()
            return best_id

        # Create new entity
        eid = str(uuid.uuid4())
        self._entities[eid] = {
            "id": eid,
            "text": entity_text,
            "type": entity_type,
            "linked_memory_ids": [memory_id],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if embedding:
            self._ent_embeddings.add(eid, embedding)
        self._save_entities()
        return eid

    def search_entities(
        self,
        query_embedding: list[float],
        top_k: int = 500,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Search entity store by embedding similarity via EmbeddingStore."""
        if not self._entities:
            return []

        eids, emb_matrix = self._ent_embeddings.get_all_embeddings()
        if len(eids) == 0 or emb_matrix.shape[1] == 0:
            return []

        q_emb = np.array(query_embedding, dtype=np.float32)
        scores = batch_cosine_np(q_emb, emb_matrix)
        ranked = sorted(zip(eids, scores), key=lambda x: x[1], reverse=True)

        results = []
        for eid, score in ranked:
            score_f = float(score)
            if score_f < threshold:
                continue
            if len(results) >= top_k:
                break
            erec = self._entities.get(eid)
            if erec:
                results.append({
                    "id": eid,
                    "score": score_f,
                    "payload": deepcopy(erec),
                    "linked_memory_ids": erec.get("linked_memory_ids", []),
                })
        return results

    # ------------------------------------------------------------------
    # Message log (delegate)
    # ------------------------------------------------------------------

    def save_messages(self, messages: list[dict], session_scope: str) -> None:
        self._messages.save_messages(messages, session_scope)

    def get_last_messages(self, session_scope: str, limit: int = 10) -> list[dict]:
        return self._messages.get_last_messages(session_scope, limit)

    # ------------------------------------------------------------------
    # MEMORY.md (Dream output)
    # ------------------------------------------------------------------

    def read_memory(self) -> str:
        """Read MEMORY.md content (compatibility alias for ContextBuilder)."""
        return self.read_memory_md()

    def get_memory_context(self) -> str:
        """Return long-term memory formatted for context injection."""
        long_term = self.read_memory_md()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def read_unprocessed_history(self, since_cursor: int) -> list[dict]:
        """Return history entries with cursor > *since_cursor*.

        mem0v3 uses vector-based memory retrieval instead of raw history
        injection.  Returns empty list — recent history is not injected
        into the system prompt for this memory algorithm.
        """
        return []

    def get_last_dream_cursor(self) -> int:
        """Return the last dream consolidation cursor.

        mem0v3 dream runs on its own consolidation schedule and does not
        use the cursor-based history injection mechanism.  Always returns 0.
        """
        return 0

    def read_memory_md(self) -> str:
        try:
            return self._memory_md_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def write_memory_md(self, content: str) -> None:
        self._memory_md_path.write_text(content, encoding="utf-8")
        logger.debug("MEMORY.md updated ({} chars)", len(content))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def memory_count(self) -> int:
        return len(self._memories)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    def stats(self) -> dict:
        return {
            "memories": self.memory_count,
            "entities": self.entity_count,
            "bm25_docs": self._bm25._doc_count,
            "memory_embeddings": self._mem_embeddings.get_embedding_count(),
            "entity_embeddings": self._ent_embeddings.get_embedding_count(),
        }
