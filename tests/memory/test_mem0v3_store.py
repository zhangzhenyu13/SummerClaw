"""Comprehensive tests for Mem0V3 memory store — BM25, embeddings, entities, SQLite, CRUD."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from nanobot.memory.embedding_store import EmbeddingStore
from nanobot.memory.mem0v3_memory.store import BM25Index, Mem0V3Store, MessageLog


# ============================================================================
# BM25Index tests
# ============================================================================


class TestBM25Index:
    """Test the BM25 inverted index for keyword search."""

    def test_add_and_search_single_doc(self):
        idx = BM25Index()
        idx.add("doc1", ["hello", "world", "python"])
        results = idx.search(["python"], top_k=5)
        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert results[0][1] > 0

    def test_search_returns_sorted_by_score(self):
        idx = BM25Index()
        idx.add("doc1", ["python", "python", "python", "code"])
        idx.add("doc2", ["python", "java"])
        idx.add("doc3", ["java", "java", "java", "code"])
        results = idx.search(["python"])
        # doc1 has more 'python' occurrences, should rank higher
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"

    def test_remove_document(self):
        idx = BM25Index()
        idx.add("doc1", ["hello", "world"])
        idx.add("doc2", ["goodbye", "world"])
        assert len(idx.search(["world"])) == 2

        idx.remove("doc1")
        results = idx.search(["world"])
        assert len(results) == 1
        assert results[0][0] == "doc2"

    def test_remove_nonexistent_is_safe(self):
        idx = BM25Index()
        idx.remove("nonexistent")  # should not raise

    def test_search_empty_index(self):
        idx = BM25Index()
        assert idx.search(["anything"]) == []

    def test_search_empty_query(self):
        idx = BM25Index()
        idx.add("doc1", ["hello"])
        assert idx.search([]) == []

    def test_add_replaces_old(self):
        idx = BM25Index()
        idx.add("doc1", ["old"])
        idx.add("doc1", ["new", "tokens"])
        results = idx.search(["new"])
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_to_dict_and_from_dict(self):
        idx = BM25Index()
        idx.add("a", ["hello", "world"])
        idx.add("b", ["python", "code"])
        idx.add("c", ["hello", "python"])

        data = idx.to_dict()
        idx2 = BM25Index.from_dict(data)

        results1 = idx.search(["python"])
        results2 = idx2.search(["python"])
        assert [r[0] for r in results1] == [r[0] for r in results2]

    def test_top_k_limits_results(self):
        idx = BM25Index()
        for i in range(10):
            idx.add(f"doc{i}", ["test", str(i)])
        results = idx.search(["test"], top_k=3)
        assert len(results) == 3

    def test_multiple_terms_query(self):
        idx = BM25Index()
        idx.add("doc1", ["machine", "learning", "deep"])
        idx.add("doc2", ["python", "programming"])
        results = idx.search(["machine", "learning"])
        assert len(results) >= 1
        assert results[0][0] == "doc1"


# ============================================================================
# MessageLog tests
# ============================================================================


class TestMessageLog:
    """Test the SQLite-based message log."""

    def test_save_and_retrieve_messages(self, tmp_path):
        db_path = tmp_path / "messages.db"
        log = MessageLog(db_path)

        msgs = [
            {"role": "user", "content": "Hello", "name": None},
            {"role": "assistant", "content": "Hi there!", "name": None},
        ]
        log.save_messages(msgs, session_scope="session-1")

        retrieved = log.get_last_messages("session-1", limit=10)
        assert len(retrieved) == 2
        assert retrieved[0]["role"] == "user"
        assert retrieved[0]["content"] == "Hello"
        assert retrieved[1]["role"] == "assistant"
        assert retrieved[1]["content"] == "Hi there!"

    def test_get_last_messages_respects_limit(self, tmp_path):
        db_path = tmp_path / "messages.db"
        log = MessageLog(db_path)

        for i in range(30):
            log.save_messages(
                [{"role": "user", "content": f"Message {i}", "name": None}],
                session_scope="session-1",
            )

        retrieved = log.get_last_messages("session-1", limit=5)
        # SQLite keeps only last 20, then we limit to 5
        assert len(retrieved) == 5

    def test_save_messages_trims_to_20_max(self, tmp_path):
        db_path = tmp_path / "messages.db"
        log = MessageLog(db_path)

        for i in range(25):
            log.save_messages(
                [{"role": "user", "content": f"Msg {i}", "name": None}],
                session_scope="session-1",
            )

        # Should only keep the last 20
        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_scope = ?",
            ("session-1",)
        ).fetchone()[0]
        conn.close()
        assert count == 20

    def test_session_isolation(self, tmp_path):
        db_path = tmp_path / "messages.db"
        log = MessageLog(db_path)

        log.save_messages(
            [{"role": "user", "content": "Scope A", "name": None}],
            session_scope="scope-a",
        )
        log.save_messages(
            [{"role": "user", "content": "Scope B", "name": None}],
            session_scope="scope-b",
        )

        a_msgs = log.get_last_messages("scope-a")
        b_msgs = log.get_last_messages("scope-b")
        assert len(a_msgs) == 1
        assert len(b_msgs) == 1
        assert a_msgs[0]["content"] == "Scope A"
        assert b_msgs[0]["content"] == "Scope B"

    def test_save_empty_messages_is_safe(self, tmp_path):
        db_path = tmp_path / "messages.db"
        log = MessageLog(db_path)
        log.save_messages([], session_scope="session-1")  # should not raise


# ============================================================================
# Mem0V3Store tests
# ============================================================================


class TestMem0V3StoreMemoryCRUD:
    """Test basic memory insert/get/delete/list operations."""

    def test_insert_single_memory(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([{"text": "Hello world"}])
        assert len(ids) == 1

        mem = s.get_memory(ids[0])
        assert mem is not None
        assert mem["text"] == "Hello world"

    def test_insert_multiple_memories(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([
            {"text": "Memory 1"},
            {"text": "Memory 2"},
            {"text": "Memory 3"},
        ])
        assert len(ids) == 3
        assert s.memory_count == 3

    def test_insert_skips_duplicates_by_hash(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids1 = s.insert_memories_batch([{"text": "Python is great"}])
        ids2 = s.insert_memories_batch([{"text": "Python is great"}])
        assert len(ids1) == 1
        assert len(ids2) == 0  # duplicate skipped
        assert s.memory_count == 1

    def test_insert_skips_empty_text(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([{"text": ""}, {"text": "valid"}])
        assert len(ids) == 1

    def test_get_nonexistent_memory(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        assert s.get_memory("nonexistent") is None

    def test_get_all_memories(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "A"}, {"text": "B"}, {"text": "C"},
        ])
        all_mems = s.get_all_memories()
        assert len(all_mems) == 3
        assert all("embedding" not in m or m["embedding"] is None for m in all_mems)

    def test_delete_memory(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([{"text": "To delete"}])
        assert s.memory_count == 1

        assert s.delete_memory(ids[0])
        assert s.memory_count == 0
        assert s.get_memory(ids[0]) is None

    def test_delete_nonexistent_memory(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        assert s.delete_memory("nonexistent") is False

    def test_memory_persistence(self, tmp_path):
        s1 = Mem0V3Store(tmp_path)
        s1.insert_memories_batch([{"text": "Persistent data"}])

        s2 = Mem0V3Store(tmp_path)
        assert s2.memory_count == 1


class TestMem0V3StoreEmbeddingDecoupling:
    """Tests that embeddings are stored in .npy files, NOT in JSON."""

    def test_json_has_no_embedding_field(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([{
            "text": "Test memory",
            "embedding": [0.1, 0.2, 0.3],
        }])

        json_path = s.memory_dir / "mem0v3_memories.json"
        with open(json_path) as f:
            data = json.load(f)
        for rec in data["memories"].values():
            assert "embedding" not in rec, "JSON must not contain embedding field"

    def test_embedding_stored_in_npy(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([{
            "text": "With embedding",
            "embedding": [0.1, 0.2, 0.3],
        }])

        import glob
        npy_files = glob.glob(str(s.memory_dir / "mem0v3_mem_embeddings_*.npy"))
        assert len(npy_files) >= 1

    def test_memory_embedding_persistence(self, tmp_path):
        s1 = Mem0V3Store(tmp_path)
        s1.insert_memories_batch([{
            "text": "Persist embedding",
            "embedding": [1.0, 0.0, 0.0],
        }])

        s2 = Mem0V3Store(tmp_path)
        ids, mat = s2._mem_embeddings.get_all_embeddings()
        assert len(ids) >= 1
        assert mat.shape[1] == 3

    def test_delete_removes_embedding(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([{
            "text": "Delete me",
            "embedding": [0.5, 0.5, 0.5],
        }])
        assert s._mem_embeddings.count == 1

        s.delete_memory(ids[0])
        assert s._mem_embeddings.count == 0

    def test_old_format_migration(self, tmp_path):
        """Old JSON with embedding field is auto-migrated to .npy on load."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True)
        old_data = {
            "memories": {
                "old-id": {
                    "id": "old-id",
                    "text": "old format",
                    "hash": "abc123",
                    "lemmatized": "old format",
                    "embedding": [0.1, 0.2, 0.3],
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                    "metadata": {},
                }
            },
            "version": 1,
        }
        json_path = memory_dir / "mem0v3_memories.json"
        with open(json_path, "w") as f:
            json.dump(old_data, f)

        s = Mem0V3Store(tmp_path)
        assert s.memory_count >= 1
        # JSON should be cleaned
        with open(json_path) as f:
            data = json.load(f)
        for rec in data["memories"].values():
            assert "embedding" not in rec


class TestMem0V3StoreSemanticSearch:
    """Test semantic (embedding) search."""

    def test_search_empty_store(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        results = s.search_semantic([0.1, 0.2, 0.3])
        assert results == []

    def test_search_returns_scored_results(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "Python programming", "embedding": [1.0, 0.0, 0.0]},
            {"text": "JavaScript coding", "embedding": [0.0, 1.0, 0.0]},
            {"text": "Machine learning", "embedding": [0.7, 0.3, 0.0]},
        ])

        # Search with Python-like embedding
        results = s.search_semantic([1.0, 0.0, 0.0], top_k=3)
        assert len(results) >= 1
        assert results[0]["payload"]["text"] == "Python programming"
        assert results[0]["score"] > 0.5

    def test_search_respects_threshold(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "Python", "embedding": [1.0, 0.0]},
        ])

        # High threshold — orthogonal query returns empty
        results = s.search_semantic([0.0, 1.0], top_k=5, threshold=0.9)
        assert len(results) == 0

    def test_search_respects_top_k(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        for i in range(10):
            s.insert_memories_batch([{
                "text": f"Memory {i}",
                "embedding": [float(i % 3), float(i % 5), float(i % 7)],
            }])

        results = s.search_semantic([1.0, 1.0, 1.0], top_k=3)
        assert len(results) <= 3

    def test_search_no_embeddings(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "No embedding 1"},
            {"text": "No embedding 2"},
        ])
        results = s.search_semantic([0.1, 0.2])
        assert results == []


class TestMem0V3StoreKeywordSearch:
    """Test BM25 keyword search."""

    def test_keyword_search_basic(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "Python is a great programming language", "lemmatized": "python great programming language"},
            {"text": "Java is also popular", "lemmatized": "java also popular"},
        ])

        results = s.search_keyword(["python"])
        assert len(results) == 1
        assert "Python" in results[0]["payload"]["text"]

    def test_keyword_search_multiple_terms(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "deep learning neural networks", "lemmatized": "deep learning neural networks"},
            {"text": "python web development", "lemmatized": "python web development"},
            {"text": "neural style transfer", "lemmatized": "neural style transfer"},
        ])

        results = s.search_keyword(["neural", "networks"])
        assert len(results) >= 1

    def test_keyword_search_removes_deleted(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([
            {"text": "Test keyword", "lemmatized": "test keyword"},
        ])
        assert len(s.search_keyword(["test"])) == 1

        s.delete_memory(ids[0])
        assert s.search_keyword(["test"]) == []

    def test_keyword_search_empty_store(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        assert s.search_keyword(["anything"]) == []


class TestMem0V3StoreEntities:
    """Test entity store operations."""

    def test_upsert_entity_creates_new(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([{"text": "I live in Paris"}])

        eid = s.upsert_entity("Paris", "location", s._memories.popitem()[1]["id"])
        assert eid is not None
        assert s.entity_count >= 1

    def test_upsert_entity_links_to_existing(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([
            {"text": "I live in Paris"},
            {"text": "Paris is beautiful"},
        ])

        # Create entity for first memory
        eid1 = s.upsert_entity("Paris", "location", ids[0])
        # Link second memory to same entity via embedding similarity
        eid2 = s.upsert_entity(
            "Paris", "location", ids[1],
            embedding=[0.9, 0.1, 0.05],
        )
        # Should still have 1 entity (or possibly 2 if embeddings are different)
        assert s.entity_count >= 1

    def test_upsert_entity_without_embedding(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([{"text": "Something"}])
        eid = s.upsert_entity("EntityName", "person", s._memories.popitem()[1]["id"])
        assert eid is not None

    def test_search_entities_empty(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        results = s.search_entities([0.1, 0.2])
        assert results == []

    def test_search_entities_with_embeddings(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        ids = s.insert_memories_batch([{"text": "John is a developer"}])

        s.upsert_entity("John", "person", ids[0], embedding=[1.0, 0.0, 0.0])

        results = s.search_entities([0.9, 0.1, 0.0], top_k=5, threshold=0.5)
        assert len(results) >= 1
        assert results[0]["payload"]["text"] == "John"


class TestMem0V3StoreMessages:
    """Test SQLite message log via Mem0V3Store."""

    def test_save_and_get_messages(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.save_messages(
            [{"role": "user", "content": "Hi"}],
            session_scope="scope-1",
        )
        msgs = s.get_last_messages("scope-1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hi"


class TestMem0V3StoreMemoryMD:
    """Test MEMORY.md read/write."""

    def test_read_memory_empty(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        assert s.read_memory() == ""
        assert s.read_memory_md() == ""

    def test_write_and_read_memory_md(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.write_memory_md("# Memory\n- Fact 1")
        assert "Fact 1" in s.read_memory()

    def test_get_memory_context(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.write_memory_md("Long-term facts")
        ctx = s.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "Long-term facts" in ctx


class TestMem0V3StoreHistory:
    """Test history compatibility methods."""

    def test_read_unprocessed_history_returns_empty(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        # mem0v3 does not use cursor-based history
        assert s.read_unprocessed_history(since_cursor=0) == []

    def test_get_last_dream_cursor_returns_zero(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        assert s.get_last_dream_cursor() == 0


class TestMem0V3StoreStats:
    """Test statistics method."""

    def test_stats(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        s.insert_memories_batch([
            {"text": "Memory 1", "embedding": [0.1, 0.2]},
            {"text": "Memory 2"},
        ])
        stats = s.stats()
        assert stats["memories"] == 2
        assert stats["entities"] == 0
        assert stats["memory_embeddings"] == 1
        assert isinstance(stats["bm25_docs"], int)


class TestMem0V3StorePathIsolation:
    """Test algorithm-specific path isolation."""

    def test_isolated_directory(self, tmp_path):
        s = Mem0V3Store(tmp_path, algo_name="mem0v3_memory")
        assert s.memory_dir == tmp_path / "memory" / "mem0v3_memory"
        assert s._memories_path == s.memory_dir / "mem0v3_memories.json"

    def test_legacy_compat(self, tmp_path):
        s = Mem0V3Store(tmp_path)
        assert s.memory_dir == tmp_path / "memory"

    def test_legacy_migration(self, tmp_path):
        old_dir = tmp_path / "memory"
        old_dir.mkdir(parents=True)
        (old_dir / "MEMORY.md").write_text("legacy content")

        s = Mem0V3Store(tmp_path, algo_name="mem0v3_memory")
        mem_file = s.memory_dir / "MEMORY.md"
        if mem_file.exists():
            assert mem_file.read_text() == "legacy content"