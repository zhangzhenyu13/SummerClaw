"""Tests for NemoriStore — file-based storage layer.

Covers message buffer, episode CRUD, semantic memory CRUD, text search,
vector search (cosine similarity), and resilience against malformed data.
"""

import json
from pathlib import Path

import pytest

from summerclaw.memory.nemori_memory.models import Episode, Message, SemanticMemory
from summerclaw.memory.nemori_memory.store import NemoriStore


@pytest.fixture
def store(tmp_path):
    return NemoriStore(Path(tmp_path), backend="file")


# ────────────────────────────────────────────────────────────────────────────
# Message buffer
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriStoreMessageBuffer:
    """Message buffer push / get / mark / compact."""

    def test_push_and_get_unprocessed(self, store):
        msg = Message(role="user", content="hello")
        store.push_messages([msg])
        assert store.count_unprocessed() == 1
        results = store.get_unprocessed_messages()
        assert len(results) == 1
        assert results[0].content == "hello"

    def test_push_multiple_messages(self, store):
        msgs = [
            Message(role="user", content="a"),
            Message(role="user", content="b"),
            Message(role="user", content="c"),
        ]
        store.push_messages(msgs)
        assert store.count_unprocessed() == 3

    def test_mark_processed_removes_from_unprocessed(self, store):
        msg = Message(role="user", content="to remove")
        store.push_messages([msg])
        store.mark_messages_processed([msg.message_id])
        assert store.count_unprocessed() == 0
        assert store.get_unprocessed_messages() == []

    def test_mark_processed_partial(self, store):
        m1 = Message(role="user", content="keep")
        m2 = Message(role="user", content="remove")
        store.push_messages([m1, m2])
        store.mark_messages_processed([m2.message_id])
        results = store.get_unprocessed_messages()
        assert len(results) == 1
        assert results[0].content == "keep"

    def test_mark_processed_unknown_ids_noop(self, store):
        msg = Message(role="user", content="x")
        store.push_messages([msg])
        store.mark_messages_processed(["nonexistent"])
        assert store.count_unprocessed() == 1

    def test_compact_buffer_removes_processed(self, store):
        m1 = Message(role="user", content="keep")
        m2 = Message(role="user", content="remove")
        store.push_messages([m1, m2])
        store.mark_messages_processed([m2.message_id])
        store.compact_buffer()
        results = store.get_unprocessed_messages()
        assert len(results) == 1
        assert results[0].content == "keep"

    def test_buffer_empty_initial(self, store):
        assert store.count_unprocessed() == 0
        assert store.get_unprocessed_messages() == []


# ────────────────────────────────────────────────────────────────────────────
# Episode CRUD
# ────────────────────────────────────────────────────────────────────────────


def _ep(user_id="u1", title="T", content="C", agent_id="default", **kw):
    return Episode(
        user_id=user_id, agent_id=agent_id, title=title,
        content=content, source_messages=[], **kw,
    )


class TestNemoriStoreEpisodeCRUD:
    """Episode save / get / list / delete."""

    def test_save_and_list(self, store):
        ep = _ep(title="Test", content="content")
        store.save_episode(ep)
        results = store.list_episodes("u1")
        assert len(results) == 1
        assert results[0].title == "Test"

    def test_list_filters_by_user(self, store):
        store.save_episode(_ep(user_id="u1", title="U1"))
        store.save_episode(_ep(user_id="u2", title="U2"))
        assert len(store.list_episodes("u1")) == 1
        assert len(store.list_episodes("u2")) == 1

    def test_list_filters_by_agent(self, store):
        store.save_episode(_ep(agent_id="a1", title="A1"))
        store.save_episode(_ep(agent_id="a2", title="A2"))
        assert len(store.list_episodes("u1", agent_id="a1")) == 1
        assert len(store.list_episodes("u1", agent_id="a2")) == 1

    def test_list_limits_results(self, store):
        for i in range(10):
            store.save_episode(_ep(title=f"E{i}"))
        assert len(store.list_episodes("u1", limit=5)) == 5

    def test_get_by_id(self, store):
        ep = _ep(title="Find Me")
        store.save_episode(ep)
        found = store.get_episode(ep.id, "u1")
        assert found is not None
        assert found.title == "Find Me"

    def test_get_by_id_returns_none_for_wrong_user(self, store):
        ep = _ep(title="X")
        store.save_episode(ep)
        assert store.get_episode(ep.id, "u2") is None

    def test_get_by_id_returns_none_for_missing(self, store):
        assert store.get_episode("nonexistent", "u1") is None

    def test_get_batch(self, store):
        ep1 = _ep(title="A")
        ep2 = _ep(title="B")
        store.save_episode(ep1)
        store.save_episode(ep2)
        results = store.get_episodes_batch([ep1.id, ep2.id], "u1")
        titles = {e.title for e in results}
        assert titles == {"A", "B"}

    def test_get_batch_skips_missing_ids(self, store):
        ep1 = _ep(title="A")
        store.save_episode(ep1)
        results = store.get_episodes_batch([ep1.id, "nonexistent"], "u1")
        assert len(results) == 1

    def test_delete_episode(self, store):
        ep = _ep(title="Del")
        store.save_episode(ep)
        store.delete_episode(ep.id)
        assert store.list_episodes("u1") == []

    def test_delete_episodes_by_user(self, store):
        store.save_episode(_ep(user_id="u1", title="A"))
        store.save_episode(_ep(user_id="u1", title="B"))
        store.save_episode(_ep(user_id="u2", title="C"))
        store.delete_episodes_by_user("u1")
        assert store.list_episodes("u1") == []
        assert len(store.list_episodes("u2")) == 1

    def test_save_updates_existing(self, store):
        ep = _ep(title="Original")
        store.save_episode(ep)
        ep.title = "Updated"
        store.save_episode(ep)
        found = store.get_episode(ep.id, "u1")
        assert found.title == "Updated"


# ────────────────────────────────────────────────────────────────────────────
# Semantic memory CRUD
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriStoreSemanticCRUD:
    """Semantic memory save / get / list / delete."""

    def test_save_and_list(self, store):
        sm = SemanticMemory(user_id="u1", content="fact", memory_type="identity")
        store.save_semantic(sm)
        results = store.list_semantics("u1")
        assert len(results) == 1
        assert results[0].content == "fact"

    def test_list_filters_by_type(self, store):
        store.save_semantic(SemanticMemory(user_id="u1", content="a", memory_type="identity"))
        store.save_semantic(SemanticMemory(user_id="u1", content="b", memory_type="preference"))
        assert len(store.list_semantics("u1", memory_type="identity")) == 1

    def test_save_batch(self, store):
        sms = [
            SemanticMemory(user_id="u1", content="f1", memory_type="identity"),
            SemanticMemory(user_id="u1", content="f2", memory_type="goal"),
        ]
        store.save_semantic_batch(sms)
        assert len(store.list_semantics("u1")) == 2

    def test_delete_semantic(self, store):
        sm = SemanticMemory(user_id="u1", content="del", memory_type="identity")
        store.save_semantic(sm)
        store.delete_semantic(sm.id)
        assert store.list_semantics("u1") == []

    def test_delete_semantics_by_user(self, store):
        store.save_semantic(SemanticMemory(user_id="u1", content="a", memory_type="identity"))
        store.save_semantic(SemanticMemory(user_id="u2", content="b", memory_type="identity"))
        store.delete_semantics_by_user("u1")
        assert store.list_semantics("u1") == []
        assert len(store.list_semantics("u2")) == 1

    def test_get_by_id(self, store):
        sm = SemanticMemory(user_id="u1", content="findme", memory_type="identity")
        store.save_semantic(sm)
        found = store.get_semantic(sm.id, "u1")
        assert found is not None
        assert found.content == "findme"

    def test_get_batch(self, store):
        s1 = SemanticMemory(user_id="u1", content="a", memory_type="identity")
        s2 = SemanticMemory(user_id="u1", content="b", memory_type="identity")
        store.save_semantic(s1)
        store.save_semantic(s2)
        results = store.get_semantics_batch([s1.id, s2.id], "u1")
        assert len(results) == 2


# ────────────────────────────────────────────────────────────────────────────
# Text search
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriStoreTextSearch:
    """Keyword-based text search for episodes and semantic memories."""

    def test_search_episodes_by_text(self, store):
        store.save_episode(_ep(title="Python", content="learning Python"))
        store.save_episode(_ep(title="Java", content="learning Java"))
        results = store.search_episodes_by_text("u1", "default", "Python", 10)
        assert len(results) == 1
        assert results[0].title == "Python"

    def test_search_episodes_case_insensitive(self, store):
        store.save_episode(_ep(title="Python", content="learning"))
        results = store.search_episodes_by_text("u1", "default", "python", 10)
        assert len(results) == 1

    def test_search_episodes_filters_by_user(self, store):
        store.save_episode(_ep(user_id="u1", title="Python"))
        store.save_episode(_ep(user_id="u2", title="Python"))
        results = store.search_episodes_by_text("u1", "default", "Python", 10)
        assert len(results) == 1

    def test_search_episodes_respects_top_k(self, store):
        for i in range(5):
            store.save_episode(_ep(title=f"Python {i}"))
        assert len(store.search_episodes_by_text("u1", "default", "Python", 2)) == 2

    def test_search_episodes_no_match(self, store):
        store.save_episode(_ep(title="Python"))
        results = store.search_episodes_by_text("u1", "default", "Rust", 10)
        assert results == []

    def test_search_semantics_by_text(self, store):
        store.save_semantic(SemanticMemory(user_id="u1", content="likes Python", memory_type="preference"))
        store.save_semantic(SemanticMemory(user_id="u1", content="works at Google", memory_type="identity"))
        results = store.search_semantics_by_text("u1", "default", "Python", 10)
        assert len(results) == 1
        assert results[0].content == "likes Python"


# ────────────────────────────────────────────────────────────────────────────
# Vector search
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriStoreVectorSearch:
    """Cosine similarity vector search."""

    def test_cosine_similarity_identical(self):
        sim = NemoriStore._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        sim = NemoriStore._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_empty_vectors(self):
        assert NemoriStore._cosine_similarity([], []) == 0.0
        assert NemoriStore._cosine_similarity([1.0], []) == 0.0

    def test_cosine_similarity_mismatched_length(self):
        assert NemoriStore._cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_cosine_similarity_zero_norm(self):
        assert NemoriStore._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_search_episodes_by_vector(self, store):
        ep = _ep(
            title="T", content="C",
            embedding=[1.0, 0.0, 0.0],
        )
        store.save_episode(ep)
        results = store.search_episodes_by_vector(
            [1.0, 0.0, 0.0], "u1", "default", top_k=5,
        )
        assert len(results) == 1
        assert results[0]["id"] == ep.id
        assert results[0]["score"] == pytest.approx(1.0)

    def test_search_episodes_by_vector_no_embeddings(self, store):
        ep = _ep(title="T", content="C")  # no embedding
        store.save_episode(ep)
        results = store.search_episodes_by_vector([1.0, 0.0], "u1", "default", 5)
        assert results == []

    def test_search_semantics_by_vector(self, store):
        sm = SemanticMemory(
            user_id="u1", content="f", memory_type="identity",
            embedding=[0.0, 1.0],
        )
        store.save_semantic(sm)
        results = store.search_semantics_by_vector([0.0, 1.0], "u1", "default", 5)
        assert len(results) == 1


# ────────────────────────────────────────────────────────────────────────────
# Resilience
# ────────────────────────────────────────────────────────────────────────────


class TestNemoriStoreResilience:
    """Edge cases and error resilience."""

    def test_episode_from_dict_handles_json_string_metadata(self, store):
        ep = _ep(title="T", content="C", metadata={"key": "value"})
        store.save_episode(ep)
        raw = store._episodes._read()
        assert len(raw) == 1
        restored = Episode.from_dict(raw[0])
        assert restored.metadata == {"key": "value"}

    def test_mark_processed_empty_list_noop(self, store):
        store.mark_messages_processed([])  # should not raise

    def test_episode_get_batch_empty_ids(self, store):
        assert store.get_episodes_batch([], "u1") == []

    def test_semantic_get_batch_empty_ids(self, store):
        assert store.get_semantics_batch([], "u1") == []

    def test_backend_file_is_default(self, store):
        assert store._backend == "file"

    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            NemoriStore(Path(tmp_path), backend="invalid")

    def test_store_creates_data_dir(self, tmp_path):
        store = NemoriStore(Path(tmp_path), backend="file")
        data_dir = tmp_path / "memory" / "nemori"
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_store_data_persists_across_instances(self, tmp_path):
        s1 = NemoriStore(Path(tmp_path))
        s1.save_episode(_ep(title="P"))
        s2 = NemoriStore(Path(tmp_path))
        assert len(s2.list_episodes("u1")) == 1

    def test_episode_handles_malformed_source_messages(self, store):
        """Episodes with corrupted source_messages should not crash listing."""
        store._episodes._write([{
            "id": "bad-ep",
            "user_id": "u1",
            "agent_id": "default",
            "title": "Bad Ep",
            "content": "content",
            "source_messages": "not-a-list",
            "metadata": {},
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }])
        episodes = store.list_episodes("u1")
        assert len(episodes) == 1
        assert isinstance(episodes[0].source_messages, list)  # from_dict normalizes

    def test_buffer_handles_malformed_lines(self, store):
        """Malformed JSONL lines should be silently skipped in buffer reads."""
        buffer_file = store.memory_dir / "message_buffer.jsonl"
        buffer_file.write_text(
            "not json\n"
            + json.dumps({
                "message_id": "ok-1",
                "role": "user",
                "content": "good",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "metadata": {},
                "processed": False,
            }) + "\n",
            encoding="utf-8",
        )
        results = store.get_unprocessed_messages()
        assert len(results) == 1
        assert results[0].content == "good"
