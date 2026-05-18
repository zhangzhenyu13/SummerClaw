"""Nemori file-based storage layer with optional PG + Qdrant backends.

Default: JSON file storage (zero extra dependencies).
Optional: PostgreSQL + Qdrant (production-grade, requires asyncpg + qdrant_client).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nanobot.memory.embedding_store import EmbeddingStore, batch_cosine_np
from nanobot.memory.nemori_memory.models import Episode, Message, SemanticMemory

logger = logging.getLogger("nemori")


# ── File-based storage ────────────────────────────────────────────────────


class _FileStore:
    """Thread-safe file-based JSON store for episodes / semantic memories."""

    def __init__(self, file_path: Path) -> None:
        self._path = file_path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._write([])

    def _read(self) -> list[dict[str, Any]]:
        try:
            text = self._path.read_text(encoding="utf-8")
            return json.loads(text) if text.strip() else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, data: list[dict[str, Any]]) -> None:
        with self._lock:
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get_all(self) -> list[dict[str, Any]]:
        return self._read()

    def get_by_id(self, item_id: str) -> dict[str, Any] | None:
        for item in self._read():
            if item.get("id") == item_id:
                return item
        return None

    def get_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        id_set = set(ids)
        return [item for item in self._read() if item.get("id") in id_set]

    def save(self, item: dict[str, Any]) -> None:
        items = self._read()
        item_id = item.get("id")
        for i, existing in enumerate(items):
            if existing.get("id") == item_id:
                items[i] = item
                self._write(items)
                return
        items.append(item)
        self._write(items)

    def delete(self, item_id: str) -> None:
        items = [i for i in self._read() if i.get("id") != item_id]
        self._write(items)

    def delete_by_filter(self, **filters: Any) -> None:
        items = self._read()
        kept = []
        for item in items:
            if not all(item.get(k) == v for k, v in filters.items()):
                kept.append(item)
        self._write(kept)


class _FileBufferStore:
    """Thread-safe JSONL message buffer (append-only)."""

    def __init__(self, file_path: Path) -> None:
        self._path = file_path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def push(self, messages: list[Message]) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                for msg in messages:
                    record = {
                        "message_id": msg.message_id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata,
                        "processed": False,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_unprocessed(self) -> list[Message]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []

        result: list[Message] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not record.get("processed", False):
                    result.append(Message.from_dict(record))
            except json.JSONDecodeError:
                continue
        return result

    def count_unprocessed(self) -> int:
        return len(self.get_unprocessed())

    def mark_processed(self, message_ids: list[str]) -> None:
        if not message_ids:
            return
        id_set = set(message_ids)
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return

        with self._lock:
            with open(self._path, "w", encoding="utf-8") as f:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("message_id") in id_set:
                        record["processed"] = True
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def delete_processed(self) -> None:
        """Remove all processed messages from the buffer file."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return

        with self._lock:
            with open(self._path, "w", encoding="utf-8") as f:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not record.get("processed", False):
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── NemoriStore — facade over file-based (and optionally PG+Qdrant) storage ─


class NemoriStore:
    """Unified storage facade for the Nemori memory algorithm.

    Defaults to file-based storage (zero dependencies).
    Set ``backend="postgres"`` and provide ``dsn`` to use the async PG+Qdrant backend.
    """

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
        backend: str = "file",
        config: dict[str, Any] | None = None,
        algo_name: str | None = None,
    ) -> None:
        self.workspace = workspace
        self._backend = backend
        self._config = config or {}

        if algo_name:
            data_dir = workspace / "memory" / algo_name
        else:
            data_dir = workspace / "memory" / "nemori"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = data_dir

        # Public attribute aliases for ContextBuilder (identity template path resolution)
        self.memory_file = data_dir / "MEMORY.md"
        self.history_file = data_dir / "history.jsonl"

        # Migrate legacy data
        if algo_name:
            self._migrate_from_legacy_shared()
            self._migrate_from_legacy_nemori()

        if backend == "file":
            self._episodes = _FileStore(data_dir / "episodes.json")
            self._semantics = _FileStore(data_dir / "semantic_memories.json")
            self._buffer = _FileBufferStore(data_dir / "message_buffer.jsonl")
            self._pg: Any = None
            self._qdrant: Any = None

            # Embedding stores — numpy binary chunked files, decoupled from JSON
            self._ep_embeddings = EmbeddingStore(data_dir, prefix="nemori_ep_embeddings")
            self._sem_embeddings = EmbeddingStore(data_dir, prefix="nemori_sem_embeddings")

            # Migrate old inline embeddings from JSON to EmbeddingStore
            self._migrate_embeddings()
        elif backend == "postgres":
            # Lazy import to avoid hard dependency
            self._episodes: Any = None  # will be set in _init_pg
            self._semantics: Any = None
            self._buffer: Any = None
            self._pg: Any = None
            self._qdrant: Any = None
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    def _migrate_from_legacy_shared(self) -> None:
        """Migrate from the legacy shared ``memory/`` directory to the algo-specific dir."""
        from nanobot.memory.migrate import maybe_migrate_legacy_files
        old_memory_dir = self.workspace / "memory"
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=self.workspace,
            files=[
                "MEMORY.md",
                "SOUL.md",
                "USER.md",
                "history.jsonl",
            ],
        )

    def _migrate_from_legacy_nemori(self) -> None:
        """Migrate from legacy ``memory/nemori/`` to the algorithm-specific directory."""
        old_data_dir = self.workspace / "memory" / "nemori"
        if not old_data_dir.is_dir():
            return
        import shutil
        for fname in ("episodes.json", "semantic_memories.json", "message_buffer.jsonl"):
            src = old_data_dir / fname
            dst = self.memory_dir / fname
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                from loguru import logger
                logger.info("Migrated {} -> {}", src, dst)

    def _migrate_embeddings(self) -> None:
        """Migrate old inline embeddings from JSON files to EmbeddingStore."""
        _migrated = 0

        # Migrate episode embeddings
        for item in self._episodes.get_all():
            emb = item.get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                self._ep_embeddings.add(item["id"], emb)
                item.pop("embedding", None)
                _migrated += 1
        if _migrated > 0:
            # Rewrite episodes without embeddings
            all_data = self._episodes.get_all()
            self._episodes._write(all_data)

        # Migrate semantic embeddings
        sem_migrated = 0
        for item in self._semantics.get_all():
            emb = item.get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                self._sem_embeddings.add(item["id"], emb)
                item.pop("embedding", None)
                sem_migrated += 1
        if sem_migrated > 0:
            all_data = self._semantics.get_all()
            self._semantics._write(all_data)

        if _migrated > 0 or sem_migrated > 0:
            logger = logging.getLogger("nemori")
            logger.info(
                "NemoriStore: migrated %d episode + %d semantic embeddings to .npy",
                _migrated, sem_migrated,
            )

    # -- episodes ------------------------------------------------------------

    def save_episode(self, episode: Episode) -> None:
        if self._backend == "file":
            data = episode.to_dict()
            emb = data.pop("embedding", None)
            self._episodes.save(data)
            if emb and isinstance(emb, list) and len(emb) > 0:
                self._ep_embeddings.add(data["id"], emb)
        else:
            raise NotImplementedError("PG backend not yet connected; call init_async first")

    def get_episode(self, episode_id: str, user_id: str, agent_id: str = "default") -> Episode | None:
        if self._backend == "file":
            data = self._episodes.get_by_id(episode_id)
            if data and data.get("user_id") == user_id and data.get("agent_id", "default") == agent_id:
                return Episode.from_dict(data)
            return None
        raise NotImplementedError("PG backend not yet connected")

    def list_episodes(self, user_id: str, agent_id: str = "default", limit: int = 100) -> list[Episode]:
        if self._backend == "file":
            all_data = self._episodes.get_all()
            filtered = [
                d for d in all_data
                if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id
            ]
            filtered.sort(key=lambda d: d.get("created_at", ""), reverse=True)
            return [Episode.from_dict(d) for d in filtered[:limit]]
        raise NotImplementedError("PG backend not yet connected")

    def get_episodes_batch(self, episode_ids: list[str], user_id: str, agent_id: str = "default") -> list[Episode]:
        if self._backend == "file":
            data_list = self._episodes.get_batch(episode_ids)
            return [
                Episode.from_dict(d) for d in data_list
                if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id
            ]
        raise NotImplementedError("PG backend not yet connected")

    def delete_episode(self, episode_id: str) -> None:
        if self._backend == "file":
            self._episodes.delete(episode_id)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def delete_episodes_by_user(self, user_id: str, agent_id: str = "default") -> None:
        if self._backend == "file":
            self._episodes.delete_by_filter(user_id=user_id, agent_id=agent_id)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def search_episodes_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[Episode]:
        """Simple keyword-based text search (file backend only)."""
        if self._backend != "file":
            raise NotImplementedError("PG backend not yet connected")
        query_lower = query.lower()
        all_data = self._episodes.get_all()
        scored: list[tuple[int, dict]] = []
        for d in all_data:
            if d.get("user_id") != user_id:
                continue
            if d.get("agent_id", "default") != agent_id:
                continue
            text = (d.get("title", "") + " " + d.get("content", "")).lower()
            score = text.count(query_lower)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Episode.from_dict(d) for _, d in scored[:top_k]]

    # -- semantic memories ---------------------------------------------------

    def save_semantic(self, memory: SemanticMemory) -> None:
        if self._backend == "file":
            data = memory.to_dict()
            emb = data.pop("embedding", None)
            self._semantics.save(data)
            if emb and isinstance(emb, list) and len(emb) > 0:
                self._sem_embeddings.add(data["id"], emb)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def save_semantic_batch(self, memories: list[SemanticMemory]) -> None:
        for m in memories:
            self.save_semantic(m)

    def get_semantic(self, memory_id: str, user_id: str, agent_id: str = "default") -> SemanticMemory | None:
        if self._backend == "file":
            data = self._semantics.get_by_id(memory_id)
            if data and data.get("user_id") == user_id and data.get("agent_id", "default") == agent_id:
                return SemanticMemory.from_dict(data)
            return None
        raise NotImplementedError("PG backend not yet connected")

    def list_semantics(
        self, user_id: str, agent_id: str = "default", memory_type: str | None = None
    ) -> list[SemanticMemory]:
        if self._backend == "file":
            all_data = self._semantics.get_all()
            filtered = [
                d for d in all_data
                if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id
            ]
            if memory_type:
                filtered = [d for d in filtered if d.get("memory_type") == memory_type]
            filtered.sort(key=lambda d: d.get("created_at", ""), reverse=True)
            return [SemanticMemory.from_dict(d) for d in filtered]
        raise NotImplementedError("PG backend not yet connected")

    def get_semantics_batch(self, ids: list[str], user_id: str, agent_id: str = "default") -> list[SemanticMemory]:
        if self._backend == "file":
            data_list = self._semantics.get_batch(ids)
            return [
                SemanticMemory.from_dict(d) for d in data_list
                if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id
            ]
        raise NotImplementedError("PG backend not yet connected")

    def delete_semantic(self, memory_id: str) -> None:
        if self._backend == "file":
            self._semantics.delete(memory_id)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def delete_semantics_by_user(self, user_id: str, agent_id: str = "default") -> None:
        if self._backend == "file":
            self._semantics.delete_by_filter(user_id=user_id, agent_id=agent_id)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def search_semantics_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]:
        """Simple keyword-based text search (file backend only)."""
        if self._backend != "file":
            raise NotImplementedError("PG backend not yet connected")
        query_lower = query.lower()
        all_data = self._semantics.get_all()
        scored: list[tuple[int, dict]] = []
        for d in all_data:
            if d.get("user_id") != user_id:
                continue
            if d.get("agent_id", "default") != agent_id:
                continue
            text = d.get("content", "").lower()
            score = text.count(query_lower)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [SemanticMemory.from_dict(d) for _, d in scored[:top_k]]

    # -- message buffer ------------------------------------------------------

    def push_messages(self, messages: list[Message]) -> None:
        if self._backend == "file":
            self._buffer.push(messages)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def get_unprocessed_messages(self) -> list[Message]:
        if self._backend == "file":
            return self._buffer.get_unprocessed()
        raise NotImplementedError("PG backend not yet connected")

    def count_unprocessed(self) -> int:
        if self._backend == "file":
            return self._buffer.count_unprocessed()
        raise NotImplementedError("PG backend not yet connected")

    def mark_messages_processed(self, message_ids: list[str]) -> None:
        if self._backend == "file":
            self._buffer.mark_processed(message_ids)
        else:
            raise NotImplementedError("PG backend not yet connected")

    def compact_buffer(self) -> None:
        """Remove all processed messages from the buffer file."""
        if self._backend == "file":
            self._buffer.delete_processed()
        else:
            raise NotImplementedError("PG backend not yet connected")

    # -- vector search (stub for file backend) -------------------------------

    def search_episodes_by_vector(
        self, embedding: list[float], user_id: str, agent_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Vector search via EmbeddingStore (numpy-accelerated)."""
        if self._backend != "file":
            raise NotImplementedError("PG backend not yet connected")

        ep_ids, emb_matrix = self._ep_embeddings.get_all_embeddings()
        if len(ep_ids) == 0 or emb_matrix.shape[1] == 0:
            return []

        q_emb = np.array(embedding, dtype=np.float32)
        scores = batch_cosine_np(q_emb, emb_matrix)

        # Filter by user_id/agent_id
        scored: list[tuple[str, float]] = []
        all_data = self._episodes.get_all()
        ep_map: dict[str, dict] = {}
        for d in all_data:
            if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id:
                ep_map[d["id"]] = d

        for eid, sim in zip(ep_ids, scores):
            if eid in ep_map:
                scored.append((eid, float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"id": eid, "score": s} for eid, s in scored[:top_k]]

    def search_semantics_by_vector(
        self, embedding: list[float], user_id: str, agent_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Vector search via EmbeddingStore (numpy-accelerated)."""
        if self._backend != "file":
            raise NotImplementedError("PG backend not yet connected")

        sem_ids, emb_matrix = self._sem_embeddings.get_all_embeddings()
        if len(sem_ids) == 0 or emb_matrix.shape[1] == 0:
            return []

        q_emb = np.array(embedding, dtype=np.float32)
        scores = batch_cosine_np(q_emb, emb_matrix)

        # Filter by user_id/agent_id
        all_data = self._semantics.get_all()
        sem_map: dict[str, dict] = {}
        for d in all_data:
            if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id:
                sem_map[d["id"]] = d

        scored: list[tuple[str, float]] = []
        for sid, sim in zip(sem_ids, scores):
            if sid in sem_map:
                scored.append((sid, float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"id": sid, "score": s} for sid, s in scored[:top_k]]

    # -- memory context (required by ContextBuilder) -------------------------

    def get_memory_context(self) -> str:
        """Return Nemori memory context for system prompt injection.

        Combines recent episodes and semantic memories into a formatted string.
        This method is called by ContextBuilder to inject long-term memory
        into the agent's system prompt.

        Note: Uses default user/agent IDs since ContextBuilder calls this
        without parameters. For parameterized access, use get_memory_context_for().

        Returns:
            Formatted memory context string, or empty string if no memories exist.
        """
        return self.get_memory_context_for(user_id="default", agent_id="default")

    def get_memory_context_for(self, user_id: str, agent_id: str = "default") -> str:
        """Return Nemori memory context for specific user/agent.

        Args:
            user_id: User identifier.
            agent_id: Agent namespace (defaults to "default").

        Returns:
            Formatted memory context string, or empty string if no memories exist.
        """
        parts: list[str] = []

        # Get recent episodes
        try:
            episodes = self.list_episodes(user_id, agent_id, limit=10)
            if episodes:
                episode_texts = []
                for ep in episodes:
                    episode_texts.append(
                        f"### {ep.title}\n{ep.content}"
                    )
                parts.append(
                    "## Recent Episodes\n" + "\n\n".join(episode_texts)
                )
        except Exception as e:
            logger.warning(f"Failed to load episodes for memory context: {e}")

        # Get semantic memories
        try:
            semantics = self.list_semantics(user_id, agent_id)
            if semantics:
                # Group by type
                by_type: dict[str, list[str]] = {}
                for mem in semantics[:20]:  # Limit to avoid too much context
                    mem_type = mem.memory_type or "general"
                    if mem_type not in by_type:
                        by_type[mem_type] = []
                    by_type[mem_type].append(f"- {mem.content}")

                semantic_parts = []
                for mem_type, contents in by_type.items():
                    semantic_parts.append(
                        f"### {mem_type.title()} Knowledge\n" + "\n".join(contents)
                    )

                if semantic_parts:
                    parts.append(
                        "## Semantic Knowledge\n" + "\n\n".join(semantic_parts)
                    )
        except Exception as e:
            logger.warning(f"Failed to load semantic memories for memory context: {e}")

        if not parts:
            return ""

        return "\n\n---\n\n".join(parts)

    # -- legacy compatibility methods (required by ContextBuilder) -----------

    def read_memory(self) -> str:
        """Read MEMORY.md content for legacy compatibility."""
        return self._read_file_content(self.memory_file)

    def read_unprocessed_history(self, since_cursor: int = 0) -> list[dict[str, Any]]:
        """Return history entries with cursor > since_cursor.
        
        For Nemori, this reads from history.jsonl if it exists,
        otherwise returns empty list.
        """
        if not self.history_file.exists():
            return []
        
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if entry.get("cursor", 0) > since_cursor:
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def get_last_dream_cursor(self) -> int:
        """Get the last processed cursor position for dream processing."""
        cursor_file = self.memory_dir / ".dream_cursor"
        if cursor_file.exists():
            try:
                return int(cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        """Set the last processed cursor position for dream processing."""
        cursor_file = self.memory_dir / ".dream_cursor"
        cursor_file.write_text(str(cursor), encoding="utf-8")

    def _read_file_content(self, file_path: Path) -> str:
        """Safely read file content, returning empty string if not found."""
        try:
            return file_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return ""
