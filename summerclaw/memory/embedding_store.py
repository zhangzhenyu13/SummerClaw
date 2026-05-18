"""EmbeddingStore — numpy binary chunked storage for embeddings.

A foundational memory module usable by any memory algorithm.  Stores
embeddings as float32 arrays in ``.npy`` files, keeping each file under
1 GB with automatic chunking.  Content/metadata remains in the algorithm's
own format; only the vectors live here, keyed by ``mem_id``.

File layout (configurable via *prefix*)::

    memory/
    ├── {prefix}_index.json          # mem_id → {chunk, row} mapping
    ├── {prefix}_000.npy
    ├── {prefix}_001.npy
    └── ...

Each ``.npy`` stores a 2-D float32 array of shape ``(N_chunk, dim)``.

Usage::

    from summerclaw.memory.embedding_store import EmbeddingStore

    es = EmbeddingStore(memory_dir, prefix="hindsight_embeddings")
    es.add("mem-1", [0.1, 0.2, 0.3])
    ids, matrix = es.get_all_embeddings()
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# ============================================================================
# NumPy batch cosine similarity
# ============================================================================


def batch_cosine_np(query: "np.ndarray", candidates: "np.ndarray") -> "np.ndarray":
    """NumPy batch cosine similarity.

    Args:
        query: (D,) float32 array.
        candidates: (N, D) float32 array.

    Returns:
        (N,) float32 similarity scores.
    """
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(len(candidates), dtype=np.float32)
    c_norms = np.linalg.norm(candidates, axis=1)
    denom = q_norm * c_norms
    denom[denom == 0] = 1e-10
    return np.dot(candidates, query) / denom


# ============================================================================
# EmbeddingStore
# ============================================================================


class EmbeddingStore:
    """NumPy binary storage for embeddings with chunked file management.

    Stores embeddings as float32 arrays in ``.npy`` files, keeping each file
    under 1 GB.  Designed as a shared foundation usable by any memory algorithm.

    Parameters:
        memory_dir: Directory where chunk files and index are stored.
        prefix: File-name prefix for index and chunk files
                (default ``"embeddings"`` → ``embeddings_index.json``,
                ``embeddings_000.npy``, …).
    """

    _MAX_CHUNK_BYTES = 1024 * 1024 * 1024  # 1 GB per chunk file

    def __init__(self, memory_dir: Path, *, prefix: str = "embeddings") -> None:
        self._dir = memory_dir
        self._prefix = prefix
        self._index_path = memory_dir / f"{prefix}_index.json"

        # mem_id → {"chunk": int, "row": int}
        self._index: dict[str, dict[str, int]] = {}
        # chunk_idx → [(row, mem_id), ...]  (reverse lookup for fast iteration)
        self._chunk_rows: dict[int, list[tuple[int, str]]] = {}
        # chunk_idx → row_count (including deleted/gap rows = array length)
        self._chunk_sizes: dict[int, int] = {}

        self._dim: int | None = None
        self._max_rows_per_chunk: int | None = None
        self._dirty_chunks: set[int] = set()

        self._load_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index = data.get("index", {})
            self._dim = data.get("dim")
            self._max_rows_per_chunk = data.get("max_rows_per_chunk")
            self._chunk_sizes = {
                int(k): v for k, v in data.get("chunk_sizes", {}).items()
            }
            # Rebuild reverse map
            self._chunk_rows = {}
            for mem_id, entry in self._index.items():
                chunk_idx = entry["chunk"]
                row = entry["row"]
                self._chunk_rows.setdefault(chunk_idx, []).append((row, mem_id))
        except Exception:
            logger.exception("EmbeddingStore: failed to load index; starting fresh")
            self._index = {}
            self._chunk_rows = {}
            self._chunk_sizes = {}
            self._dim = None
            self._max_rows_per_chunk = None

    def _save_index(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "index": self._index,
            "dim": self._dim,
            "max_rows_per_chunk": self._max_rows_per_chunk,
            "chunk_sizes": {str(k): v for k, v in self._chunk_sizes.items()},
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._index_path)

    # ------------------------------------------------------------------
    # Chunk file helpers
    # ------------------------------------------------------------------

    def _chunk_path(self, chunk_idx: int) -> Path:
        return self._dir / f"{self._prefix}_{chunk_idx:03d}.npy"

    def _compute_max_rows(self, dim: int) -> int:
        """How many rows fit in one chunk under the 1 GB limit."""
        row_bytes = dim * 4  # float32 = 4 bytes
        header_estimate = 128
        return max(1, (self._MAX_CHUNK_BYTES - header_estimate) // row_bytes)

    def _load_chunk(self, chunk_idx: int) -> "np.ndarray":
        path = self._chunk_path(chunk_idx)
        if path.exists():
            return np.load(path)
        return np.empty((0, self._dim or 0), dtype=np.float32)

    def _save_chunk(self, chunk_idx: int, arr: "np.ndarray") -> None:
        path = self._chunk_path(chunk_idx)
        np.save(path, arr)
        self._chunk_sizes[chunk_idx] = len(arr)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, mem_id: str, embedding: list[float]) -> None:
        """Store an embedding vector, auto-creating chunks as needed."""
        if not embedding:
            return
        emb_arr = np.array(embedding, dtype=np.float32)
        dim = emb_arr.shape[0]

        # Initialise dimension on first write
        if self._dim is None:
            self._dim = dim
            self._max_rows_per_chunk = self._compute_max_rows(dim)

        if dim != self._dim:
            logger.error(
                "EmbeddingStore: dimension mismatch — expected {}, got {} for {}",
                self._dim, dim, mem_id,
            )
            return

        # Remove old entry if present (update case)
        self.remove(mem_id)

        # Find or create chunk
        chunk_idx = self._find_available_chunk()
        chunk = self._load_chunk(chunk_idx)
        row = len(chunk)

        # Append to chunk
        if len(chunk) > 0:
            chunk = np.vstack([chunk, emb_arr.reshape(1, -1)])
        else:
            chunk = emb_arr.reshape(1, -1)
        self._save_chunk(chunk_idx, chunk)
        self._dirty_chunks.discard(chunk_idx)

        # Update index
        self._index[mem_id] = {"chunk": chunk_idx, "row": row}
        self._chunk_rows.setdefault(chunk_idx, []).append((row, mem_id))
        self._chunk_sizes[chunk_idx] = len(chunk)

        self._save_index()

    def _find_available_chunk(self) -> int:
        """Find a chunk with room, or return index of a new chunk."""
        max_rows = self._max_rows_per_chunk or 100_000
        for chunk_idx in sorted(self._chunk_sizes.keys()):
            if self._chunk_sizes[chunk_idx] < max_rows:
                return chunk_idx
        # All full — create new chunk
        if not self._chunk_sizes:
            return 0
        return max(self._chunk_sizes.keys()) + 1

    def get(self, mem_id: str) -> list[float] | None:
        """Retrieve a single embedding as a list of floats."""
        entry = self._index.get(mem_id)
        if entry is None:
            return None
        chunk = self._load_chunk(entry["chunk"])
        row = entry["row"]
        if row >= len(chunk):
            return None
        return chunk[row].tolist()

    def remove(self, mem_id: str) -> None:
        """Remove an embedding from the index (lazy — file is not compacted)."""
        entry = self._index.pop(mem_id, None)
        if entry is None:
            return
        chunk_idx = entry["chunk"]
        rows = self._chunk_rows.get(chunk_idx, [])
        self._chunk_rows[chunk_idx] = [(r, mid) for r, mid in rows if mid != mem_id]
        self._save_index()

    # ------------------------------------------------------------------
    # Bulk retrieval for search
    # ------------------------------------------------------------------

    def get_all_embeddings(self) -> tuple[list[str], "np.ndarray"]:
        """Return all (mem_id, embedding) pairs for batch operations.

        Returns:
            (mem_ids, matrix) where *matrix* is (N, D) float32.
            If no embeddings exist, returns ([], empty (0,0) array).
        """
        if not self._index:
            return [], np.empty((0, 0), dtype=np.float32)

        all_ids: list[str] = []
        all_embs: list["np.ndarray"] = []

        for chunk_idx in sorted(self._chunk_rows.keys()):
            rows = self._chunk_rows[chunk_idx]
            if not rows:
                continue
            chunk = self._load_chunk(chunk_idx)
            for row, mem_id in rows:
                if row < len(chunk):
                    all_ids.append(mem_id)
                    all_embs.append(chunk[row])

        if all_embs:
            return all_ids, np.stack(all_embs)
        return [], np.empty((0, self._dim or 0), dtype=np.float32)

    @property
    def count(self) -> int:
        """Number of stored embeddings."""
        return len(self._index)

    def get_embedding_count(self) -> int:
        """Alias for :attr:`count` (backward compatibility)."""
        return self.count

    # ------------------------------------------------------------------
    # Migration: bulk import from old JSON format
    # ------------------------------------------------------------------

    def import_from_dict(self, memories: dict[str, dict]) -> int:
        """Import embeddings from in-memory records (old JSON format).

        Reads ``embedding`` field from each record, stores to numpy chunks,
        and removes the field from the dict in-place.

        Returns the number of embeddings migrated.
        """
        count = 0
        for mem_id, rec in memories.items():
            emb = rec.get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                self.add(mem_id, emb)
                rec.pop("embedding", None)
                count += 1
        if count:
            self.save()
            logger.info(
                "EmbeddingStore: migrated {} embeddings from old JSON format", count,
            )
        return count

    def save(self) -> None:
        """Flush any pending writes (index is saved by add/remove already)."""
        for _chunk_idx in list(self._dirty_chunks):
            self._dirty_chunks.discard(_chunk_idx)
        self._save_index()

    def delete_all_chunks(self) -> None:
        """Remove all chunk files (used for teardown/reset)."""
        import glob as _glob
        pattern = str(self._dir / f"{self._prefix}_*.npy")
        for path in _glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass
        idx_tmp = self._index_path.with_suffix(".json.tmp")
        if idx_tmp.exists():
            try:
                os.remove(idx_tmp)
            except OSError:
                pass