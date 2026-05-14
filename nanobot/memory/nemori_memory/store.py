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

    def __init__(
        self,
        workspace: Path,
        backend: str = "file",
        config: dict[str, Any] | None = None,
        algo_name: str | None = None,
    ) -> None:
        self._workspace = workspace
        self._backend = backend
        self._config = config or {}

        if algo_name:
            data_dir = workspace / "memory" / algo_name
            # Migrate from legacy "memory/nemori/" to "memory/nemori_memory/"
            self._migrate_from_legacy(workspace, data_dir)
        else:
            data_dir = workspace / "memory" / "nemori"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Public attribute aliases for ContextBuilder (identity template path resolution)
        self.memory_file = data_dir / "MEMORY.md"
        self.history_file = data_dir / "history.jsonl"

        if backend == "file":
            self._episodes = _FileStore(data_dir / "episodes.json")
            self._semantics = _FileStore(data_dir / "semantic_memories.json")
            self._buffer = _FileBufferStore(data_dir / "message_buffer.jsonl")
            self._pg: Any = None
            self._qdrant: Any = None
        elif backend == "postgres":
            # Lazy import to avoid hard dependency
            self._episodes: Any = None  # will be set in _init_pg
            self._semantics: Any = None
            self._buffer: Any = None
            self._pg: Any = None
            self._qdrant: Any = None
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    def _migrate_from_legacy(self, workspace: Path, data_dir: Path) -> None:
        """Migrate from legacy "memory/nemori/" to the algorithm-specific directory."""
        old_data_dir = workspace / "memory" / "nemori"
        if not old_data_dir.is_dir():
            return
        # Migrate individual files from old_data_dir to data_dir
        import shutil
        for fname in ("episodes.json", "semantic_memories.json", "message_buffer.jsonl"):
            src = old_data_dir / fname
            dst = data_dir / fname
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                from loguru import logger
                logger.info("Migrated {} -> {}", src, dst)

    # -- episodes ------------------------------------------------------------

    def save_episode(self, episode: Episode) -> None:
        if self._backend == "file":
            self._episodes.save(episode.to_dict())
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
            self._semantics.save(memory.to_dict())
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
        """Vector search — only meaningful with PG+Qdrant backend."""
        if self._backend != "file":
            raise NotImplementedError("PG backend not yet connected")
        # File backend: fall back to listing all and doing cosine similarity
        all_data = self._episodes.get_all()
        candidates = [
            d for d in all_data
            if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id
            and d.get("embedding") is not None
        ]
        if not candidates:
            return []
        scored = []
        for d in candidates:
            sim = self._cosine_similarity(embedding, d["embedding"])
            scored.append({"id": d["id"], "score": sim})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_semantics_by_vector(
        self, embedding: list[float], user_id: str, agent_id: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Vector search — only meaningful with PG+Qdrant backend."""
        if self._backend != "file":
            raise NotImplementedError("PG backend not yet connected")
        all_data = self._semantics.get_all()
        candidates = [
            d for d in all_data
            if d.get("user_id") == user_id and d.get("agent_id", "default") == agent_id
            and d.get("embedding") is not None
        ]
        if not candidates:
            return []
        scored = []
        for d in candidates:
            sim = self._cosine_similarity(embedding, d["embedding"])
            scored.append({"id": d["id"], "score": sim})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
