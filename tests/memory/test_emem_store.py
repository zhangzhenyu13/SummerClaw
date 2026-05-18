"""Comprehensive tests for EMem memory store — ContentStore, EDU/argument/session records."""

import json
from pathlib import Path

from nanobot.memory.emem_memory.datatypes import (
    ArgumentRecord,
    EDURecord,
    SessionRecord,
    compute_mdhash_id,
    min_max_normalize,
)
from nanobot.memory.emem_memory.store import ContentStore, EMemStore

import numpy as np


# ============================================================================
# ContentStore tests (without embeddings — simpler)
# ============================================================================


class TestContentStoreWithoutEmbeddings:
    """Test ContentStore with embeddings disabled."""

    def test_insert_and_get_content(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids = cs.insert_content(["Hello world"])
        assert len(ids) == 1

        assert cs.get_content(ids[0]) == "Hello world"

    def test_get_row(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids = cs.insert_content(["Test content"])
        row = cs.get_row(ids[0])
        assert row["hash_id"] == ids[0]
        assert row["content"] == "Test content"

    def test_get_rows(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids = cs.insert_content(["A", "B", "C"])
        rows = cs.get_rows(ids)
        assert len(rows) == 3

    def test_get_all_ids(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        cs.insert_content(["X", "Y", "Z"])
        ids = cs.get_all_ids()
        assert len(ids) == 3

    def test_get_all_id_to_rows(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        cs.insert_content(["A", "B"])
        rows = cs.get_all_id_to_rows()
        assert len(rows) == 2

    def test_get_all_contents(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        cs.insert_content(["X", "Y"])
        contents = cs.get_all_contents()
        assert contents == ["X", "Y"]

    def test_duplicate_insert_skipped(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids1 = cs.insert_content(["Hello world"])
        ids2 = cs.insert_content(["Hello world"])
        assert len(ids1) == 1
        assert len(ids2) == 0  # duplicate

    def test_delete_content(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids = cs.insert_content(["A", "B", "C"])
        cs.delete([ids[1]])  # delete "B"

        all_ids = cs.get_all_ids()
        assert len(all_ids) == 2

    def test_persistence(self, tmp_path):
        cs1 = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids = cs1.insert_content(["Persistent data"])

        cs2 = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        assert cs2.get_content(ids[0]) == "Persistent data"

    def test_insert_empty_list(self, tmp_path):
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            enable_embeddings=False,
        )
        ids = cs.insert_content([])
        assert ids == []


# ============================================================================
# ContentStore tests with mock embeddings
# ============================================================================


class MockEmbeddingModel:
    def batch_encode(self, texts, instruction=None):
        return [[0.1 * (i + 1) for i in range(len(t))] for t in texts]


class TestContentStoreWithEmbeddings:
    """Test ContentStore with embeddings enabled."""

    def test_insert_generates_embeddings(self, tmp_path):
        mock_model = MockEmbeddingModel()
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            embedding_model=mock_model,
            text_extraction_fn=lambda s: s,
            enable_embeddings=True,
        )
        ids = cs.insert_content(["Hello"])
        assert len(ids) == 1

        emb = cs.get_embedding(ids[0])
        assert emb is not None
        assert emb.ndim == 1

    def test_get_embeddings_batch(self, tmp_path):
        mock_model = MockEmbeddingModel()
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            embedding_model=mock_model,
            text_extraction_fn=lambda s: s,
            enable_embeddings=True,
        )
        ids = cs.insert_content(["A", "B", "C"])
        embs = cs.get_embeddings(ids)
        assert len(embs) == 3

    def test_get_embedding_nonexistent(self, tmp_path):
        mock_model = MockEmbeddingModel()
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            embedding_model=mock_model,
            text_extraction_fn=lambda s: s,
            enable_embeddings=True,
        )
        assert cs.get_embedding("nonexistent") is None

    def test_insert_strings_backward_compat(self, tmp_path):
        mock_model = MockEmbeddingModel()
        cs = ContentStore[str](
            db_dir=tmp_path,
            namespace="test",
            embedding_model=mock_model,
            text_extraction_fn=lambda s: s,
            enable_embeddings=True,
        )
        ids = cs.insert_strings(["Test string"])
        assert len(ids) == 1


# ============================================================================
# ContentStore with EDURecords
# ============================================================================


class TestContentStoreWithEDURecords:
    def test_insert_edu_records(self, tmp_path):
        mock_model = MockEmbeddingModel()
        cs = ContentStore[EDURecord](
            db_dir=tmp_path,
            namespace="edu",
            embedding_model=mock_model,
            text_extraction_fn=lambda e: e.text,
            enable_embeddings=True,
        )
        edu = EDURecord(
            edu_id=EDURecord.compute_id("Python is great"),
            text="Python is great",
            source_speakers=["user"],
        )
        ids = cs.insert_content([edu])
        assert len(ids) == 1

    def test_insert_argument_records(self, tmp_path):
        mock_model = MockEmbeddingModel()
        cs = ContentStore[ArgumentRecord](
            db_dir=tmp_path,
            namespace="argument",
            embedding_model=mock_model,
            text_extraction_fn=lambda a: a.text,
            enable_embeddings=True,
        )
        arg = ArgumentRecord(
            arg_id=ArgumentRecord.compute_id("Python"),
            text="Python",
        )
        ids = cs.insert_content([arg])
        assert len(ids) == 1


# ============================================================================
# EMemStore tests
# ============================================================================


class TestEMemStoreBasics:
    def test_store_creation(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.memory_dir == tmp_path / "memory"
        assert s.emem_dir == tmp_path / "memory" / "emem"

    def test_memory_read_write(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.read_memory() == ""
        s.write_memory("Test memory content")
        assert "Test memory content" in s.read_memory()

    def test_soul_read_write(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.read_soul() == ""
        s.write_soul("Soul text")
        assert "Soul text" in s.read_soul()

    def test_user_read_write(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.read_user() == ""
        s.write_user("User text")
        assert "User text" in s.read_user()

    def test_memory_context(self, tmp_path):
        s = EMemStore(tmp_path)
        s.write_memory("Long-term facts")
        ctx = s.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "Long-term facts" in ctx

    def test_memory_context_empty(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.get_memory_context() == ""


class TestEMemStoreHistory:
    def test_append_history(self, tmp_path):
        s = EMemStore(tmp_path)
        c1 = s.append_history("Entry 1")
        c2 = s.append_history("Entry 2")
        assert c1 == 1
        assert c2 == 2

    def test_read_unprocessed_history(self, tmp_path):
        s = EMemStore(tmp_path)
        s.append_history("A")
        s.append_history("B")
        s.append_history("C")

        entries = s.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2

    def test_compact_history(self, tmp_path):
        s = EMemStore(tmp_path)
        for i in range(10):
            s.append_history(f"Entry {i}")

        s.compact_history(max_entries=3)
        entries = s._read_entries()
        assert len(entries) == 3

    def test_dream_cursor(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.get_last_dream_cursor() == 0
        s.set_last_dream_cursor(5)
        assert s.get_last_dream_cursor() == 5

    def test_raw_archive(self, tmp_path):
        s = EMemStore(tmp_path)
        msgs = [{"role": "user", "content": "test"}]
        s.raw_archive(msgs)
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW]" in entries[0]["content"]


class TestEMemStorePathIsolation:
    def test_isolated_directory(self, tmp_path):
        s = EMemStore(tmp_path, algo_name="emem_memory")
        assert s.memory_dir == tmp_path / "memory" / "emem_memory"
        assert s.emem_dir == tmp_path / "memory" / "emem_memory" / "emem"

    def test_legacy_compat(self, tmp_path):
        s = EMemStore(tmp_path)
        assert s.memory_dir == tmp_path / "memory"
        assert s.emem_dir == tmp_path / "memory" / "emem"

    def test_legacy_migration(self, tmp_path):
        old_dir = tmp_path / "memory"
        old_dir.mkdir(parents=True)
        (old_dir / "MEMORY.md").write_text("legacy mem")

        s = EMemStore(tmp_path, algo_name="emem_test")
        mem_file = s.memory_dir / "MEMORY.md"
        if mem_file.exists():
            assert mem_file.read_text() == "legacy mem"


# ============================================================================
# Data types tests
# ============================================================================


class TestEDURecord:
    def test_compute_id_deterministic(self):
        id1 = EDURecord.compute_id("Hello world")
        id2 = EDURecord.compute_id("Hello world")
        assert id1 == id2

    def test_compute_id_different_for_different_text(self):
        id1 = EDURecord.compute_id("Hello")
        id2 = EDURecord.compute_id("World")
        assert id1 != id2

    def test_to_context_string(self):
        from datetime import datetime
        edu = EDURecord(
            edu_id="test-1",
            text="Python is great",
            source_speakers=["user"],
            timestamp=datetime(2025, 1, 15, 14, 30),
        )
        ctx = edu.to_context_string(date_format="iso")
        assert "2025-01-15T14:30:00" in ctx
        assert "Python is great" in ctx
        assert "user" in ctx

    def test_to_context_string_no_timestamp(self):
        edu = EDURecord(
            edu_id="test-1",
            text="No date",
            source_speakers=[],
        )
        ctx = edu.to_context_string()
        assert "unknown date" in ctx
        assert "Unknown" in ctx


class TestArgumentRecord:
    def test_compute_id(self):
        arg_id = ArgumentRecord.compute_id("entity_name")
        assert arg_id.startswith("argument-")

    def test_compute_id_deterministic(self):
        id1 = ArgumentRecord.compute_id("John")
        id2 = ArgumentRecord.compute_id("John")
        assert id1 == id2


class TestSessionRecord:
    def test_compute_id(self):
        sid = SessionRecord.compute_id("session-key")
        assert sid.startswith("session-")

    def test_default_turns(self):
        session = SessionRecord(session_id="test-1")
        assert session.turns == []


# ============================================================================
# Utility functions
# ============================================================================


class TestUtilityFunctions:
    def test_compute_mdhash_id(self):
        h = compute_mdhash_id("test content", prefix="pref-")
        assert h.startswith("pref-")
        assert len(h) == len("pref-") + 32  # MD5 hex digest is 32 chars

    def test_compute_mdhash_id_deterministic(self):
        h1 = compute_mdhash_id("hello")
        h2 = compute_mdhash_id("hello")
        assert h1 == h2

    def test_min_max_normalize(self):
        arr = np.array([1.0, 2.0, 3.0])
        norm = min_max_normalize(arr)
        assert np.isclose(norm[0], 0.0)
        assert np.isclose(norm[2], 1.0)

    def test_min_max_normalize_all_same(self):
        arr = np.array([5.0, 5.0, 5.0])
        norm = min_max_normalize(arr)
        assert np.allclose(norm, np.ones_like(arr))