"""EMem store — persistent storage for EDUs, arguments, and sessions."""

from __future__ import annotations

import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from loguru import logger

from nanobot.memory.emem_memory.datatypes import (
    ArgumentRecord,
    EDURecord,
    SessionRecord,
    compute_mdhash_id,
)

T = TypeVar("T")


class ContentStore(Generic[T]):
    """Generic content store with optional embedding support.

    Stores content objects identified by hash IDs, with optional
    embedding vectors for similarity search.

    Type Parameters:
        T: The content type stored (e.g. EDURecord, str).
    """

    def __init__(
        self,
        db_dir: Path,
        namespace: str,
        batch_size: int = 32,
        embedding_model: Any | None = None,
        text_extraction_fn: Callable[[T], str] | None = None,
        enable_embeddings: bool = True,
    ):
        self.db_dir = db_dir
        self.namespace = namespace
        self.batch_size = batch_size
        self.enable_embeddings = enable_embeddings

        if self.enable_embeddings:
            if embedding_model is None:
                raise ValueError("embedding_model is required when enable_embeddings=True")
            if text_extraction_fn is None:
                raise ValueError("text_extraction_fn is required when enable_embeddings=True")

        self.embedding_model = embedding_model
        self.text_extraction_fn = text_extraction_fn

        db_dir.mkdir(parents=True, exist_ok=True)

        self.content_filename = db_dir / f"content_{namespace}.pkl"
        self.embedding_filename = db_dir / f"embeddings_{namespace}.parquet"

        self._load_data()

    # ------------------------------------------------------------------ helpers

    def _compute_content_hash(self, content: T) -> str:
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, (EDURecord, ArgumentRecord, SessionRecord)):
            content_str = json.dumps(content.__dict__, sort_keys=True, default=str)
        elif hasattr(content, "__dict__"):
            content_str = json.dumps(content.__dict__, sort_keys=True, default=str)
        else:
            content_str = str(content)
        return compute_mdhash_id(content_str, prefix=self.namespace + "-")

    # ------------------------------------------------------------------ load / save

    def _load_data(self) -> None:
        # Content
        if self.content_filename.exists():
            with open(self.content_filename, "rb") as f:
                data = pickle.load(f)
            self.hash_ids: list[str] = data["hash_ids"]
            self.contents: list[T] = data["contents"]
            logger.debug(f"Loaded {len(self.hash_ids)} records from {self.content_filename}")
        else:
            self.hash_ids, self.contents = [], []

        # Embeddings
        if self.enable_embeddings and self.embedding_filename.exists():
            import pandas as pd

            df = pd.read_parquet(self.embedding_filename)
            emb_hash_ids = df["hash_id"].values.tolist()
            self.embeddings: list = df["embedding"].values.tolist()
            if set(emb_hash_ids) != set(self.hash_ids):
                logger.warning(
                    "Embedding hash IDs don't match content; rebuilding embeddings"
                )
                self.embeddings = []
            else:
                emb_dict = {h: e for h, e in zip(emb_hash_ids, self.embeddings)}
                self.embeddings = [emb_dict[h] for h in self.hash_ids]
                logger.debug(
                    f"Loaded {len(self.embeddings)} embeddings from {self.embedding_filename}"
                )
        else:
            self.embeddings = []

        self._build_indices()

    def _build_indices(self) -> None:
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": c}
            for h, c in zip(self.hash_ids, self.contents)
        }

    def _save_data(self) -> None:
        content_data = {"hash_ids": self.hash_ids, "contents": self.contents}
        with open(self.content_filename, "wb") as f:
            pickle.dump(content_data, f)
        logger.debug(f"Saved {len(self.hash_ids)} records to {self.content_filename}")

        if self.enable_embeddings and self.embeddings:
            import pandas as pd

            emb_df = pd.DataFrame({
                "hash_id": self.hash_ids,
                "embedding": self.embeddings,
            })
            emb_df.to_parquet(self.embedding_filename, index=False)
            logger.debug(f"Saved {len(self.embeddings)} embeddings to {self.embedding_filename}")

        self._build_indices()

    # ------------------------------------------------------------------ CRUD

    def insert_content(
        self, contents: list[T], encoding_instruction: str | None = None,
    ) -> list[str]:
        """Insert content objects, computing embeddings if enabled."""
        content_dict = {}
        for c in contents:
            hid = self._compute_content_hash(c)
            content_dict[hid] = {"content": c}

        all_ids = list(content_dict.keys())
        if not all_ids:
            return []

        existing = set(self.hash_id_to_row.keys())
        missing_ids = [hid for hid in all_ids if hid not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new, "
            f"{len(all_ids) - len(missing_ids)} existing records"
        )
        if not missing_ids:
            return []

        contents_to_store = [content_dict[hid]["content"] for hid in missing_ids]

        missing_embeddings = None
        if self.enable_embeddings:
            texts = [self.text_extraction_fn(c) for c in contents_to_store]  # type: ignore[misc]
            if encoding_instruction is None:
                missing_embeddings = self.embedding_model.batch_encode(texts)
            else:
                missing_embeddings = self.embedding_model.batch_encode(
                    texts, instruction=encoding_instruction,
                )

        self._upsert(missing_ids, contents_to_store, missing_embeddings)
        return missing_ids

    def insert_strings(
        self, texts: list[str], encoding_instruction: str | None = None,
    ) -> list[str]:
        """Insert plain strings (backward compatibility)."""
        return self.insert_content(texts, encoding_instruction)  # type: ignore[arg-type]

    def _upsert(
        self,
        hash_ids: list[str],
        contents: list[T],
        embeddings: list | None = None,
    ) -> None:
        self.hash_ids.extend(hash_ids)
        self.contents.extend(contents)
        if self.enable_embeddings and embeddings is not None:
            self.embeddings.extend(embeddings)
        self._save_data()

    def delete(self, hash_ids: list[str]) -> None:
        indices = sorted(
            (self.hash_id_to_idx[h] for h in hash_ids if h in self.hash_id_to_idx),
            reverse=True,
        )
        for idx in indices:
            self.hash_ids.pop(idx)
            self.contents.pop(idx)
            if self.enable_embeddings and self.embeddings:
                self.embeddings.pop(idx)
        self._save_data()

    # ------------------------------------------------------------------ accessors

    def get_row(self, hash_id: str) -> dict[str, Any]:
        return self.hash_id_to_row[hash_id]

    def get_content(self, hash_id: str) -> T:
        idx = self.hash_id_to_idx[hash_id]
        return self.contents[idx]

    def get_rows(self, hash_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {h: self.hash_id_to_row[h] for h in hash_ids if h in self.hash_id_to_row}

    def get_all_ids(self) -> list[str]:
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.hash_id_to_row)

    def get_all_contents(self) -> list[T]:
        return deepcopy(self.contents)

    def get_embedding(self, hash_id: str, dtype: type = np.float32) -> np.ndarray | None:
        if not self.enable_embeddings or not self.embeddings:
            return None
        if hash_id not in self.hash_id_to_idx:
            return None
        idx = self.hash_id_to_idx[hash_id]
        return np.array(self.embeddings[idx], dtype=dtype)

    def get_embeddings(
        self, hash_ids: list[str], dtype: type = np.float32,
    ) -> list[np.ndarray]:
        if not self.enable_embeddings or not self.embeddings or not hash_ids:
            return []
        indices = np.array(
            [self.hash_id_to_idx[h] for h in hash_ids if h in self.hash_id_to_idx],
            dtype=np.intp,
        )
        if len(indices) == 0:
            return []
        return [np.array(self.embeddings[i], dtype=dtype) for i in indices]


# ---------------------------------------------------------------------------
# EMemStore — high-level store combining EDU + Argument + Session storage
# ---------------------------------------------------------------------------


class EMemStore:
    """High-level EMem storage backing.

    Manages three content stores:
    - **edu_store**: EDU records with embeddings.
    - **argument_store**: Argument/entity records with embeddings.
    - **session_store**: Session records (no embeddings).

    Also provides access to nanobot's standard file-based memory files
    (MEMORY.md, history.jsonl, SOUL.md, USER.md) for compatibility.
    """

    def __init__(
        self,
        workspace: Path,
        embedding_model: Any | None = None,
        batch_size: int = 32,
        algo_name: str | None = None,
    ):
        from nanobot.utils.gitstore import GitStore

        self.workspace = workspace

        if algo_name:
            self._algo_name = algo_name
            self.memory_dir = workspace / "memory" / algo_name
        else:
            self._algo_name = None
            self.memory_dir = workspace / "memory"
        self.emem_dir = self.memory_dir / "emem"
        self.emem_dir.mkdir(parents=True, exist_ok=True)

        # Standard nanobot file paths (for interoperability)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        if algo_name:
            self.soul_file = self.memory_dir / "SOUL.md"
            self.user_file = self.memory_dir / "USER.md"
        else:
            self.soul_file = workspace / "SOUL.md"
            self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"

        # Git integration for line age tracking and auto-commit
        self._git = GitStore(
            workspace,
            tracked_files=[
                f"memory/{algo_name}/SOUL.md" if algo_name else "SOUL.md",
                f"memory/{algo_name}/USER.md" if algo_name else "USER.md",
                f"memory/{algo_name}/MEMORY.md" if algo_name else "memory/MEMORY.md",
            ],
        )

        # Migrate legacy shared files if needed
        if algo_name:
            self._migrate_from_legacy()

        # EDU store with embeddings
        self.edu_store = ContentStore[EDURecord](
            db_dir=self.emem_dir / "edu_storage",
            namespace="edu",
            batch_size=batch_size,
            embedding_model=embedding_model,
            text_extraction_fn=lambda e: e.text,
            enable_embeddings=embedding_model is not None,
        )

        # Argument store with embeddings
        self.argument_store = ContentStore[ArgumentRecord](
            db_dir=self.emem_dir / "argument_storage",
            namespace="argument",
            batch_size=batch_size,
            embedding_model=embedding_model,
            text_extraction_fn=lambda a: a.text,
            enable_embeddings=embedding_model is not None,
        )

        # Session store (no embeddings)
        self.session_store = ContentStore[SessionRecord](
            db_dir=self.emem_dir / "session_storage",
            namespace="session",
            batch_size=batch_size,
            embedding_model=None,
            text_extraction_fn=None,
            enable_embeddings=False,
        )

    def _migrate_from_legacy(self) -> None:
        """Migrate data from the legacy shared location to the algorithm-specific dir."""
        from nanobot.memory.migrate import maybe_migrate_legacy_files
        old_memory_dir = self.workspace / "memory"
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=self.workspace,
            files=[
                "MEMORY.md",
                "history.jsonl",
                "SOUL.md",
                "USER.md",
                ".cursor",
                ".dream_cursor",
            ],
            dirs=["emem"],
        )

    # -- Standard file access (interop with naive_memory tools) ---------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- History helpers -----------------------------------------------------

    def append_history(self, entry: str) -> int:
        from datetime import datetime

        from nanobot.utils.helpers import strip_think

        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = strip_think(entry.rstrip()) or entry.rstrip()
        record = {"cursor": cursor, "timestamp": ts, "content": content}
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def _next_cursor(self) -> int:
        if self._cursor_file.exists():
            try:
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
            except (ValueError, OSError):
                pass
        last = self._read_last_entry()
        if last and last.get("cursor"):
            return last["cursor"] + 1
        return 1

    def _read_last_entry(self) -> dict[str, Any] | None:
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [l for l in data.split("\n") if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _read_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        return [e for e in self._read_entries() if e.get("cursor", 0) > since_cursor]

    def compact_history(self, max_entries: int = 1000) -> None:
        if max_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= max_entries:
            return
        kept = entries[-max_entries:]
        self._write_entries(kept)

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            try:
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    @property
    def git(self) -> Any:
        """Return the GitStore for line age tracking and auto-commit."""
        return self._git

    def raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to history.jsonl without LLM summarization."""
        from nanobot.memory.naive_memory.store import MemoryStore

        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{MemoryStore._format_messages(messages)}"
        )
        from loguru import logger as _logger

        _logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )
