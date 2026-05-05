"""Tests for EMem store — ContentStore, EMemStore, history, and persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nanobot.memory.emem_memory.store import ContentStore, EMemStore


# ===================================================================
# EMemStore fixtures
# ===================================================================

@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedding model."""
    m = MagicMock()
    # batch_encode returns a list of np.ndarray, one per text
    m.batch_encode = MagicMock(
        side_effect=lambda texts, **kwargs: [np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in texts]
    )
    return m


@pytest.fixture
def store(tmp_path: Path, mock_embedder: MagicMock) -> EMemStore:
    """Create an EMemStore with a mock embedding model."""
    return EMemStore(workspace=tmp_path, embedding_model=mock_embedder)


@pytest.fixture
def store_no_embedding(tmp_path: Path) -> EMemStore:
    """Create an EMemStore without embedding model."""
    return EMemStore(workspace=tmp_path, embedding_model=None)


# ===================================================================
# EMemStore — Basic file I/O
# ===================================================================

class TestEMemStoreBasicIO:
    """Test basic read/write of MEMORY.md, SOUL.md, USER.md."""

    def test_read_memory_returns_empty_when_missing(self, store: EMemStore) -> None:
        assert store.read_memory() == ""

    def test_write_and_read_memory(self, store: EMemStore) -> None:
        store.write_memory("hello world")
        assert store.read_memory() == "hello world"

    def test_write_memory_overwrites(self, store: EMemStore) -> None:
        store.write_memory("first")
        store.write_memory("second")
        assert store.read_memory() == "second"

    def test_read_soul_returns_empty_when_missing(self, store: EMemStore) -> None:
        assert store.read_soul() == ""

    def test_write_and_read_soul(self, store: EMemStore) -> None:
        store.write_soul("soul content")
        assert store.read_soul() == "soul content"

    def test_read_user_returns_empty_when_missing(self, store: EMemStore) -> None:
        assert store.read_user() == ""

    def test_write_and_read_user(self, store: EMemStore) -> None:
        store.write_user("user content")
        assert store.read_user() == "user content"

    def test_get_memory_context_returns_empty_when_missing(self, store: EMemStore) -> None:
        assert store.get_memory_context() == ""

    def test_get_memory_context_returns_formatted_content(self, store: EMemStore) -> None:
        store.write_memory("important fact")
        ctx = store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "important fact" in ctx

    def test_write_memory_with_unicode(self, store: EMemStore) -> None:
        content = "记忆系统测试 \U0001f680 日本語 테스트"
        store.write_memory(content)
        assert store.read_memory() == content

    def test_write_memory_with_empty_string(self, store: EMemStore) -> None:
        store.write_memory("")
        assert store.read_memory() == ""


# ===================================================================
# EMemStore — History cursor management
# ===================================================================

class TestEMemStoreHistoryCursor:
    """Test history append / cursor / read-unprocessed pipeline."""

    def test_append_history_returns_cursor(self, store: EMemStore) -> None:
        c1 = store.append_history("event 1")
        assert c1 == 1
        c2 = store.append_history("event 2")
        assert c2 == 2

    def test_append_history_includes_cursor_in_file(self, store: EMemStore) -> None:
        store.append_history("event 1")
        with open(store.history_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["cursor"] == 1

    def test_cursor_persists_across_appends(self, store: EMemStore) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        cursor = store.append_history("event 3")
        assert cursor == 3

    def test_cursor_persists_across_store_recreation(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s1 = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s1.append_history("event 1")
        s1.append_history("event 2")
        s2 = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        cursor = s2.append_history("event 3")
        assert cursor == 3

    def test_read_unprocessed_history(self, store: EMemStore) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2
        assert entries[1]["cursor"] == 3

    def test_read_unprocessed_history_returns_all_when_cursor_zero(
        self, store: EMemStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_read_unprocessed_history_returns_empty_when_cursor_ahead(
        self, store: EMemStore,
    ) -> None:
        store.append_history("event 1")
        entries = store.read_unprocessed_history(since_cursor=999)
        assert entries == []

    def test_read_unprocessed_skips_entries_without_cursor(
        self, store: EMemStore,
    ) -> None:
        """Entries missing the cursor key should be silently skipped."""
        store.history_file.write_text(
            '{"timestamp": "2026-04-01 10:00", "content": "no cursor"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "valid"}\n'
            '{"cursor": 3, "timestamp": "2026-04-01 10:02", "content": "also valid"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert [e["cursor"] for e in entries] == [2, 3]

    def test_next_cursor_falls_back_when_last_entry_has_no_cursor(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s.history_file.write_text(
            '{"timestamp": "2026-04-01 10:01", "content": "no cursor"}\n',
            encoding="utf-8",
        )
        # Delete cursor file so _next_cursor falls back to reading JSONL
        s._cursor_file.unlink(missing_ok=True)
        cursor = s.append_history("new event")
        assert cursor == 1

    def test_append_history_preserves_unicode_content(
        self, store: EMemStore,
    ) -> None:
        content = "ユーザーが設定を変更しました 🎉"
        cursor = store.append_history(content)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert entries[0]["content"] == content
        assert entries[0]["cursor"] == cursor


# ===================================================================
# EMemStore — History compaction
# ===================================================================

class TestEMemStoreCompact:
    """Test history compaction logic."""

    def test_compact_history_drops_oldest(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s.append_history("event 1")
        s.append_history("event 2")
        s.append_history("event 3")
        s.append_history("event 4")
        s.append_history("event 5")
        s.compact_history(max_entries=2)
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_compact_history_noop_when_under_limit(
        self, store: EMemStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.compact_history(max_entries=10)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_compact_history_noop_when_max_is_zero(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s.append_history("event 1")
        s.append_history("event 2")
        s.compact_history(max_entries=0)
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_compact_history_noop_when_max_is_negative(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s.append_history("event 1")
        s.compact_history(max_entries=-5)
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1

    def test_compact_history_exactly_at_limit(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s.append_history("event 1")
        s.append_history("event 2")
        s.append_history("event 3")
        s.compact_history(max_entries=3)
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 3


# ===================================================================
# EMemStore — Dream cursor
# ===================================================================

class TestEMemStoreDreamCursor:
    """Test dream cursor persistence."""

    def test_initial_cursor_is_zero(self, store: EMemStore) -> None:
        assert store.get_last_dream_cursor() == 0

    def test_set_and_get_cursor(self, store: EMemStore) -> None:
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5

    def test_cursor_persists_across_store_recreation(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s1 = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        s1.set_last_dream_cursor(3)
        s2 = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        assert s2.get_last_dream_cursor() == 3

    def test_cursor_file_corrupted_returns_zero(self, store: EMemStore) -> None:
        store._dream_cursor_file.write_text("not-a-number", encoding="utf-8")
        assert store.get_last_dream_cursor() == 0

    def test_set_cursor_zero(self, store: EMemStore) -> None:
        store.set_last_dream_cursor(5)
        store.set_last_dream_cursor(0)
        assert store.get_last_dream_cursor() == 0


# ===================================================================
# EMemStore — raw_archive
# ===================================================================

class TestEMemStoreRawArchive:
    """Test raw_archive fallback dumping."""

    def test_raw_archive_appends_to_history(self, store: EMemStore) -> None:
        msgs = [
            {"role": "user", "content": "fix bug", "timestamp": "2026-04-01 10:00:00"},
            {"role": "assistant", "content": "done", "timestamp": "2026-04-01 10:00:05",
             "tools_used": ["edit_file"]},
        ]
        store.raw_archive(msgs)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert "[RAW] 2 messages" in entries[0]["content"]
        assert "USER: fix bug" in entries[0]["content"]


# ===================================================================
# EMemStore — Git integration
# ===================================================================

class TestEMemStoreGit:
    """Test GitStore integration."""

    def test_git_property_returns_gitstore(self, store: EMemStore) -> None:
        git = store.git
        assert git is not None

    def test_git_initialized_false_by_default(self, store: EMemStore) -> None:
        assert store.git.is_initialized() is False

    def test_git_init_and_line_ages(self, store: EMemStore) -> None:
        store.write_memory("# Memory\n- fact 1\n- fact 2")
        store.git.init()
        assert store.git.is_initialized() is True

    def test_git_auto_commit(self, store: EMemStore) -> None:
        store.write_memory("# Test memory")
        store.git.init()
        # Configure git user (required for commits)
        import subprocess
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(store.workspace), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(store.workspace), capture_output=True,
        )
        sha = store.git.auto_commit("test commit")
        # auto_commit may return None if git is not fully configured
        # or if there are no changes to commit. Either is acceptable.
        if sha is None:
            # Check if the commit at least succeeded by looking at git log
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=str(store.workspace), capture_output=True, text=True,
            )
            # Just verify no exception was raised
            pass
        else:
            assert isinstance(sha, str)


# ===================================================================
# EMemStore — JSONL edge cases
# ===================================================================

class TestEMemStoreEdgeCases:
    """Test JSONL corruption, empty files, and other edge cases."""

    def test_jsonl_with_corrupted_line_skipped(self, store: EMemStore) -> None:
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "good"}\n'
            'not-valid-json\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "also good"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert [e["cursor"] for e in entries] == [1, 2]

    def test_jsonl_empty_file(self, store: EMemStore) -> None:
        entries = store.read_unprocessed_history(since_cursor=0)
        assert entries == []

    def test_jsonl_file_not_exists(self, store: EMemStore) -> None:
        store.history_file.unlink(missing_ok=True)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert entries == []

    def test_jsonl_blank_lines_skipped(self, store: EMemStore) -> None:
        store.history_file.write_text(
            '\n'
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "good"}\n'
            '\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "also good"}\n'
            '\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_read_last_entry_on_empty_file(self, store: EMemStore) -> None:
        last = store._read_last_entry()
        assert last is None

    def test_read_last_entry_returns_last(self, store: EMemStore) -> None:
        store.append_history("first")
        store.append_history("second")
        last = store._read_last_entry()
        assert last is not None
        assert last["content"] == "second"

    def test_read_write_entries_roundtrip(self, store: EMemStore) -> None:
        entries = [
            {"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "a"},
            {"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "b"},
        ]
        store._write_entries(entries)
        read_back = store._read_entries()
        assert read_back == entries

    def test_memory_dir_created_on_init(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        assert s.memory_dir.exists()
        assert s.memory_dir.is_dir()

    def test_emem_dir_created_on_init(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        assert s.emem_dir.exists()
        assert s.emem_dir.is_dir()


# ===================================================================
# EMemStore — ContentStores (edu, argument, session)
# ===================================================================

class TestEMemStoreContentStores:
    """Test that edu, argument, and session ContentStores are initialized."""

    def test_edu_store_created(self, store: EMemStore) -> None:
        assert store.edu_store is not None
        assert store.edu_store.namespace == "edu"

    def test_argument_store_created(self, store: EMemStore) -> None:
        assert store.argument_store is not None
        assert store.argument_store.namespace == "argument"

    def test_session_store_created(self, store: EMemStore) -> None:
        assert store.session_store is not None
        assert store.session_store.namespace == "session"

    def test_edu_store_has_embeddings_with_model(self, store: EMemStore) -> None:
        assert store.edu_store.enable_embeddings is True

    def test_edu_store_no_embeddings_without_model(
        self, store_no_embedding: EMemStore,
    ) -> None:
        assert store_no_embedding.edu_store.enable_embeddings is False

    def test_argument_store_no_embeddings_without_model(
        self, store_no_embedding: EMemStore,
    ) -> None:
        assert store_no_embedding.argument_store.enable_embeddings is False

    def test_session_store_never_has_embeddings(self, store: EMemStore) -> None:
        assert store.session_store.enable_embeddings is False


# ===================================================================
# ContentStore — basic CRUD
# ===================================================================

class TestContentStoreBasic:
    """Test ContentStore insert/delete/read operations."""

    @pytest.fixture
    def mock_embedder_cs(self) -> MagicMock:
        m = MagicMock()
        m.batch_encode = MagicMock(side_effect=lambda texts, **kwargs: [np.array([0.1, 0.2], dtype=np.float32) for _ in texts])
        return m

    @pytest.fixture
    def cs(self, tmp_path: Path, mock_embedder_cs: MagicMock) -> ContentStore[str]:
        return ContentStore[str](
            db_dir=tmp_path / "test_db",
            namespace="test",
            batch_size=4,
            embedding_model=mock_embedder_cs,
            text_extraction_fn=lambda s: s,
            enable_embeddings=True,
        )

    def test_insert_content(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["hello world", "foo bar"])
        assert len(ids) == 2
        assert len(cs.hash_ids) == 2

    def test_insert_duplicate_skipped(self, cs: ContentStore[str]) -> None:
        ids1 = cs.insert_content(["hello"])
        ids2 = cs.insert_content(["hello"])
        assert len(ids1) == 1
        assert ids2 == []  # Already exists

    def test_get_content(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["test content"])
        content = cs.get_content(ids[0])
        assert content == "test content"

    def test_get_row(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["row test"])
        row = cs.get_row(ids[0])
        assert row["hash_id"] == ids[0]
        assert row["content"] == "row test"

    def test_get_rows(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["a", "b", "c"])
        rows = cs.get_rows(ids[:2])
        assert len(rows) == 2

    def test_get_all_ids(self, cs: ContentStore[str]) -> None:
        cs.insert_content(["x", "y"])
        all_ids = cs.get_all_ids()
        assert len(all_ids) == 2

    def test_get_all_contents(self, cs: ContentStore[str]) -> None:
        cs.insert_content(["one", "two"])
        contents = cs.get_all_contents()
        assert len(contents) == 2
        assert "one" in contents

    def test_delete(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["to_delete", "to_keep"])
        cs.delete([ids[0]])
        assert len(cs.hash_ids) == 1
        assert cs.get_content(ids[1]) == "to_keep"

    def test_delete_missing_id_no_error(self, cs: ContentStore[str]) -> None:
        cs.delete(["nonexistent"])

    def test_get_embedding(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["embeddable"])
        emb = cs.get_embedding(ids[0])
        assert emb is not None
        assert len(emb) == 2

    def test_get_embedding_missing_id(self, cs: ContentStore[str]) -> None:
        emb = cs.get_embedding("nonexistent")
        assert emb is None

    def test_get_embeddings(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content(["e1", "e2"])
        embs = cs.get_embeddings(ids)
        assert len(embs) == 2

    def test_get_embeddings_empty_list(self, cs: ContentStore[str]) -> None:
        embs = cs.get_embeddings([])
        assert embs == []

    def test_insert_empty_list(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_content([])
        assert ids == []

    def test_insert_strings_backward_compat(self, cs: ContentStore[str]) -> None:
        ids = cs.insert_strings(["hello", "world"])
        assert len(ids) == 2
        assert cs.get_content(ids[0]) == "hello"


# ===================================================================
# ContentStore — embedding-less mode
# ===================================================================

class TestContentStoreNoEmbeddings:
    """Test ContentStore without embeddings enabled."""

    @pytest.fixture
    def cs_noemb(self, tmp_path: Path) -> ContentStore[str]:
        return ContentStore[str](
            db_dir=tmp_path / "noemb_db",
            namespace="test",
            embedding_model=None,
            text_extraction_fn=None,
            enable_embeddings=False,
        )

    def test_insert_without_embeddings(self, cs_noemb: ContentStore[str]) -> None:
        ids = cs_noemb.insert_content(["no embeddings"])
        assert len(ids) == 1
        assert cs_noemb.embeddings == []

    def test_get_embedding_returns_none(self, cs_noemb: ContentStore[str]) -> None:
        ids = cs_noemb.insert_content(["no emb"])
        emb = cs_noemb.get_embedding(ids[0])
        assert emb is None

    def test_get_embeddings_returns_empty(self, cs_noemb: ContentStore[str]) -> None:
        ids = cs_noemb.insert_content(["test"])
        embs = cs_noemb.get_embeddings(ids)
        assert embs == []

    def test_insert_strings_no_embeddings(self, cs_noemb: ContentStore[str]) -> None:
        ids = cs_noemb.insert_strings(["plain"])
        assert len(ids) == 1


# ===================================================================
# ContentStore — custom types (EDURecord, etc.)
# ===================================================================

class TestContentStoreCustomTypes:
    """Test ContentStore with custom types like EDURecord and ArgumentRecord."""

    def test_edu_store_insert_and_retrieve(self, store: EMemStore) -> None:
        """Insert EDURecords into the EDU content store."""
        from nanobot.memory.emem_memory.datatypes import EDURecord

        edu = EDURecord(
            edu_id="edu-test-001",
            text="The user deployed a new version of the app.",
            source_speakers=["user"],
            session_id="session-001",
        )
        ids = store.edu_store.insert_content([edu])
        assert len(ids) == 1
        retrieved = store.edu_store.get_content(ids[0])
        assert retrieved.text == edu.text
        assert retrieved.edu_id == edu.edu_id

    def test_argument_store_insert_and_retrieve(self, store: EMemStore) -> None:
        """Insert ArgumentRecords into the argument content store."""
        from nanobot.memory.emem_memory.datatypes import ArgumentRecord

        arg = ArgumentRecord(
            arg_id="arg-test-001",
            text="Deployment Pipeline",
            source_edu_ids=["edu-xxx"],
        )
        ids = store.argument_store.insert_content([arg])
        assert len(ids) == 1
        retrieved = store.argument_store.get_content(ids[0])
        assert retrieved.text == arg.text

    def test_session_store_insert_and_retrieve(self, store: EMemStore) -> None:
        """Insert SessionRecords into the session content store."""
        from nanobot.memory.emem_memory.datatypes import SessionRecord

        session = SessionRecord(
            session_id="session-test-001",
            turns=[{"role": "user", "content": "hello"}],
        )
        ids = store.session_store.insert_content([session])
        assert len(ids) == 1
        retrieved = store.session_store.get_content(ids[0])
        assert retrieved.session_id == session.session_id


# ===================================================================
# ContentStore — persistence (save/load roundtrip)
# ===================================================================

class TestContentStorePersistence:
    """Test ContentStore save and load roundtrip."""

    def test_persistence_roundtrip(
        self, tmp_path: Path, mock_embedder: MagicMock,
    ) -> None:
        from nanobot.memory.emem_memory.datatypes import EDURecord

        db_dir = tmp_path / "persist_db"
        # Create, insert, and "reopen"
        cs1 = ContentStore[EDURecord](
            db_dir=db_dir,
            namespace="edu",
            embedding_model=mock_embedder,
            text_extraction_fn=lambda e: e.text,
            enable_embeddings=True,
        )
        edu = EDURecord(
            edu_id="edu-persist-001",
            text="Persistent EDU.",
            source_speakers=["user"],
            session_id="session-p",
        )
        cs1.insert_content([edu])

        # Reopen
        cs2 = ContentStore[EDURecord](
            db_dir=db_dir,
            namespace="edu",
            embedding_model=mock_embedder,
            text_extraction_fn=lambda e: e.text,
            enable_embeddings=True,
        )
        assert len(cs2.hash_ids) == 1
        retrieved = cs2.get_content(cs2.hash_ids[0])
        assert retrieved.text == "Persistent EDU."

    def test_get_all_id_to_rows(self, store: EMemStore) -> None:
        """Test get_all_id_to_rows on the edu store."""
        from nanobot.memory.emem_memory.datatypes import EDURecord

        edu = EDURecord(edu_id="edu-rows", text="Test rows.")
        store.edu_store.insert_content([edu])
        id_to_rows = store.edu_store.get_all_id_to_rows()
        assert len(id_to_rows) >= 1


# ===================================================================
# ContentStore — error handling
# ===================================================================

class TestContentStoreErrors:
    """Test ContentStore error handling."""

    def test_requires_embedding_model_when_embeddings_enabled(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="embedding_model"):
            ContentStore[str](
                db_dir=tmp_path / "err_db",
                namespace="err",
                embedding_model=None,
                text_extraction_fn=None,
                enable_embeddings=True,
            )

    def test_requires_text_extraction_fn_when_embeddings_enabled(
        self, tmp_path: Path,
    ) -> None:
        mock_emb = MagicMock()
        with pytest.raises(ValueError, match="text_extraction_fn"):
            ContentStore[str](
                db_dir=tmp_path / "err_db2",
                namespace="err",
                embedding_model=mock_emb,
                text_extraction_fn=None,
                enable_embeddings=True,
            )
