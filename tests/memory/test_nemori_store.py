"""Comprehensive tests for Nemori memory store — episodes, semantic memories, buffer, vector search."""

import json
from pathlib import Path

import numpy as np

from summerclaw.memory.nemori_memory.models import Episode, Message, SemanticMemory
from summerclaw.memory.nemori_memory.store import NemoriStore, _FileBufferStore, _FileStore


# ============================================================================
# _FileStore tests
# ============================================================================


class TestFileStore:
    def test_save_and_get_all(self, tmp_path):
        fs = _FileStore(tmp_path / "test.json")
        fs.save({"id": "1", "name": "Alice"})
        fs.save({"id": "2", "name": "Bob"})

        all_items = fs.get_all()
        assert len(all_items) == 2
        names = {i["name"] for i in all_items}
        assert names == {"Alice", "Bob"}

    def test_get_by_id(self, tmp_path):
        fs = _FileStore(tmp_path / "test.json")
        fs.save({"id": "abc", "value": 42})
        assert fs.get_by_id("abc")["value"] == 42
        assert fs.get_by_id("nonexistent") is None

    def test_get_batch(self, tmp_path):
        fs = _FileStore(tmp_path / "test.json")
        fs.save({"id": "a", "val": 1})
        fs.save({"id": "b", "val": 2})
        fs.save({"id": "c", "val": 3})

        batch = fs.get_batch(["a", "c"])
        assert len(batch) == 2
        assert {b["val"] for b in batch} == {1, 3}

    def test_update_existing(self, tmp_path):
        fs = _FileStore(tmp_path / "test.json")
        fs.save({"id": "1", "name": "Old"})
        fs.save({"id": "1", "name": "New"})

        all_items = fs.get_all()
        assert len(all_items) == 1
        assert all_items[0]["name"] == "New"

    def test_delete(self, tmp_path):
        fs = _FileStore(tmp_path / "test.json")
        fs.save({"id": "1"})
        fs.save({"id": "2"})
        fs.delete("1")

        all_items = fs.get_all()
        assert len(all_items) == 1
        assert all_items[0]["id"] == "2"

    def test_delete_by_filter(self, tmp_path):
        fs = _FileStore(tmp_path / "test.json")
        fs.save({"id": "1", "user_id": "u1"})
        fs.save({"id": "2", "user_id": "u2"})
        fs.save({"id": "3", "user_id": "u1"})

        fs.delete_by_filter(user_id="u1")
        all_items = fs.get_all()
        assert len(all_items) == 1
        assert all_items[0]["user_id"] == "u2"

    def test_empty_store_get_all(self, tmp_path):
        fs = _FileStore(tmp_path / "new.json")
        assert fs.get_all() == []

    def test_thread_safety_basic(self, tmp_path):
        """Basic write/read integrity."""
        fs = _FileStore(tmp_path / "thread.json")
        for i in range(100):
            fs.save({"id": str(i), "val": i})

        items = fs.get_all()
        assert len(items) == 100


# ============================================================================
# _FileBufferStore tests
# ============================================================================


class TestFileBufferStore:
    def test_push_and_get_unprocessed(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        msg = Message(role="user", content="Hello")
        buf.push([msg])

        unprocessed = buf.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0].role == "user"
        assert unprocessed[0].content == "Hello"

    def test_push_multiple_messages(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        msgs = [
            Message(role="user", content="A"),
            Message(role="assistant", content="B"),
            Message(role="user", content="C"),
        ]
        buf.push(msgs)
        assert buf.count_unprocessed() == 3

    def test_mark_processed(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        msgs = [
            Message(role="user", content="A"),
            Message(role="user", content="B"),
        ]
        buf.push(msgs)
        assert buf.count_unprocessed() == 2

        buf.mark_processed([msgs[0].message_id])
        assert buf.count_unprocessed() == 1

    def test_delete_processed(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        msgs = [
            Message(role="user", content="A"),
            Message(role="user", content="B"),
        ]
        buf.push(msgs)
        buf.mark_processed([msgs[0].message_id])
        buf.delete_processed()

        unprocessed = buf.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0].content == "B"

    def test_mark_processed_empty_list(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        buf.push([Message(role="user", content="test")])
        buf.mark_processed([])  # should not crash
        assert buf.count_unprocessed() == 1

    def test_count_unprocessed_empty(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        assert buf.count_unprocessed() == 0

    def test_get_unprocessed_empty(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        assert buf.get_unprocessed() == []

    def test_message_with_metadata(self, tmp_path):
        buf = _FileBufferStore(tmp_path / "buffer.jsonl")
        msg = Message(role="user", content="test", metadata={"key": "val"})
        buf.push([msg])

        unprocessed = buf.get_unprocessed()
        assert unprocessed[0].metadata == {"key": "val"}


# ============================================================================
# NemoriStore tests
# ============================================================================


class TestNemoriStoreEpisodes:
    def test_save_and_get_episode(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        ep = Episode(
            user_id="user1",
            title="Test Episode",
            content="This is a test episode",
            source_messages=[],
        )
        s.save_episode(ep)

        retrieved = s.get_episode(ep.id, user_id="user1")
        assert retrieved is not None
        assert retrieved.title == "Test Episode"
        assert retrieved.content == "This is a test episode"

    def test_save_episode_with_embedding(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        ep = Episode(
            user_id="user1",
            title="With Embedding",
            content="Content with embedding",
            source_messages=[],
            embedding=[0.1, 0.2, 0.3],
        )
        s.save_episode(ep)

        # Embedding should be in the numpy store
        assert s._ep_embeddings.count >= 1

    def test_get_episode_wrong_user(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        ep = Episode(
            user_id="user1",
            title="Private",
            content="Private content",
            source_messages=[],
        )
        s.save_episode(ep)
        assert s.get_episode(ep.id, user_id="user2") is None

    def test_list_episodes(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        for i in range(5):
            ep = Episode(
                user_id="user1",
                title=f"Episode {i}",
                content=f"Content {i}",
                source_messages=[],
            )
            s.save_episode(ep)

        episodes = s.list_episodes("user1")
        assert len(episodes) == 5
        # Sorted by created_at descending
        titles = [e.title for e in episodes]
        assert titles == ["Episode 4", "Episode 3", "Episode 2", "Episode 1", "Episode 0"]

    def test_list_episodes_respects_limit(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        for i in range(10):
            s.save_episode(Episode(
                user_id="user1", title=f"Ep {i}",
                content=f"Content {i}", source_messages=[],
            ))

        episodes = s.list_episodes("user1", limit=3)
        assert len(episodes) == 3

    def test_get_episodes_batch(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        ep1 = Episode(user_id="user1", title="A", content="A", source_messages=[])
        ep2 = Episode(user_id="user1", title="B", content="B", source_messages=[])
        s.save_episode(ep1)
        s.save_episode(ep2)

        batch = s.get_episodes_batch([ep1.id, ep2.id], user_id="user1")
        assert len(batch) == 2

    def test_delete_episode(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        ep = Episode(user_id="user1", title="To Delete", content="...", source_messages=[])
        s.save_episode(ep)
        assert s.get_episode(ep.id, user_id="user1") is not None

        s.delete_episode(ep.id)
        assert s.get_episode(ep.id, user_id="user1") is None

    def test_delete_episodes_by_user(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        for i in range(3):
            s.save_episode(Episode(
                user_id="user1", title=f"Ep {i}",
                content=f"Content {i}", source_messages=[],
            ))

        s.delete_episodes_by_user("user1")
        assert s.list_episodes("user1") == []

    def test_search_episodes_by_text(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        s.save_episode(Episode(
            user_id="user1", title="Python Guide",
            content="Python programming language tutorial", source_messages=[],
        ))
        s.save_episode(Episode(
            user_id="user1", title="Java Guide",
            content="Java enterprise development", source_messages=[],
        ))

        results = s.search_episodes_by_text("user1", "default", "python", top_k=5)
        assert len(results) == 1
        assert results[0].title == "Python Guide"


class TestNemoriStoreSemantics:
    def test_save_and_get_semantic(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        mem = SemanticMemory(
            user_id="user1",
            content="User prefers dark mode",
            memory_type="preference",
        )
        s.save_semantic(mem)

        retrieved = s.get_semantic(mem.id, user_id="user1")
        assert retrieved is not None
        assert retrieved.content == "User prefers dark mode"
        assert retrieved.memory_type == "preference"

    def test_save_semantic_batch(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        mems = [
            SemanticMemory(user_id="user1", content="Fact 1", memory_type="fact"),
            SemanticMemory(user_id="user1", content="Fact 2", memory_type="fact"),
            SemanticMemory(user_id="user1", content="Preference 1", memory_type="preference"),
        ]
        s.save_semantic_batch(mems)

        facts = s.list_semantics("user1", memory_type="fact")
        assert len(facts) == 2

        prefs = s.list_semantics("user1", memory_type="preference")
        assert len(prefs) == 1

    def test_list_semantics_all(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        s.save_semantic(SemanticMemory(user_id="user1", content="A", memory_type="fact"))
        s.save_semantic(SemanticMemory(user_id="user1", content="B", memory_type="fact"))

        all_mems = s.list_semantics("user1")
        assert len(all_mems) == 2

    def test_get_semantics_batch(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        m1 = SemanticMemory(user_id="user1", content="A", memory_type="fact")
        m2 = SemanticMemory(user_id="user1", content="B", memory_type="fact")
        s.save_semantic_batch([m1, m2])

        batch = s.get_semantics_batch([m1.id, m2.id], user_id="user1")
        assert len(batch) == 2

    def test_delete_semantic(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        mem = SemanticMemory(user_id="user1", content="Delete me", memory_type="fact")
        s.save_semantic(mem)
        assert s.get_semantic(mem.id, user_id="user1") is not None

        s.delete_semantic(mem.id)
        assert s.get_semantic(mem.id, user_id="user1") is None

    def test_delete_semantics_by_user(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        for i in range(3):
            s.save_semantic(SemanticMemory(
                user_id="user1", content=f"Mem {i}", memory_type="fact",
            ))

        s.delete_semantics_by_user("user1")
        assert s.list_semantics("user1") == []

    def test_search_semantics_by_text(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        s.save_semantic(SemanticMemory(
            user_id="user1", content="User lives in Paris", memory_type="fact",
        ))
        s.save_semantic(SemanticMemory(
            user_id="user1", content="User works remotely", memory_type="fact",
        ))

        results = s.search_semantics_by_text("user1", "default", "Paris", top_k=5)
        assert len(results) == 1
        assert "Paris" in results[0].content


class TestNemoriStoreBuffer:
    def test_push_and_get_messages(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        s.push_messages(msgs)

        unprocessed = s.get_unprocessed_messages()
        assert len(unprocessed) == 2
        assert s.count_unprocessed() == 2

    def test_mark_processed(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        msgs = [Message(role="user", content="Test")]
        s.push_messages(msgs)
        s.mark_messages_processed([m.message_id for m in msgs])
        assert s.count_unprocessed() == 0

    def test_compact_buffer(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        msgs = [
            Message(role="user", content="Keep"),
            Message(role="user", content="Discard"),
        ]
        s.push_messages(msgs)
        s.mark_messages_processed([msgs[1].message_id])
        s.compact_buffer()

        unprocessed = s.get_unprocessed_messages()
        assert len(unprocessed) == 1
        assert unprocessed[0].content == "Keep"


class TestNemoriStoreVectorSearch:
    def test_search_episodes_by_vector(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        ep = Episode(
            user_id="user1",
            title="Python Episode",
            content="About Python",
            source_messages=[],
            embedding=[1.0, 0.0, 0.0],
        )
        s.save_episode(ep)

        results = s.search_episodes_by_vector(
            [0.9, 0.1, 0.0], user_id="user1", agent_id="default", top_k=5,
        )
        assert len(results) == 1
        assert results[0]["score"] > 0.5

    def test_search_episodes_by_vector_empty_store(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        results = s.search_episodes_by_vector(
            [0.1, 0.2], user_id="user1", agent_id="default", top_k=5,
        )
        assert results == []

    def test_search_semantics_by_vector(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        mem = SemanticMemory(
            user_id="user1",
            content="Dark mode preference",
            memory_type="preference",
            embedding=[0.0, 1.0, 0.0],
        )
        s.save_semantic(mem)

        results = s.search_semantics_by_vector(
            [0.1, 0.9, 0.0], user_id="user1", agent_id="default", top_k=5,
        )
        assert len(results) == 1
        assert results[0]["score"] > 0.5


class TestNemoriStoreMemoryContext:
    def test_empty_context(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        assert s.get_memory_context() == ""

    def test_context_with_episodes(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        s.save_episode(Episode(
            user_id="default", agent_id="default",
            title="Test Episode", content="Content here",
            source_messages=[],
        ))
        ctx = s.get_memory_context()
        assert "Recent Episodes" in ctx
        assert "Test Episode" in ctx

    def test_context_with_semantics(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        s.save_semantic(SemanticMemory(
            user_id="default", agent_id="default",
            content="User likes coffee",
            memory_type="preference",
        ))
        ctx = s.get_memory_context()
        assert "Semantic Knowledge" in ctx
        assert "User likes coffee" in ctx

    def test_context_for_specific_user(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        s.save_semantic(SemanticMemory(
            user_id="u1", agent_id="default",
            content="User 1 data",
            memory_type="fact",
        ))
        ctx = s.get_memory_context_for("u1")
        assert "User 1 data" in ctx


class TestNemoriStoreLegacyCompat:
    def test_read_memory(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        assert s.read_memory() == ""

    def test_read_unprocessed_history_empty(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        assert s.read_unprocessed_history() == []

    def test_dream_cursor(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        assert s.get_last_dream_cursor() == 0
        s.set_last_dream_cursor(42)
        assert s.get_last_dream_cursor() == 42

    def test_dream_cursor_persists(self, tmp_path):
        s1 = NemoriStore(tmp_path, backend="file")
        s1.set_last_dream_cursor(100)
        s2 = NemoriStore(tmp_path, backend="file")
        assert s2.get_last_dream_cursor() == 100


class TestNemoriStorePathIsolation:
    def test_isolated_directory(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file", algo_name="nemori_memory")
        assert s._episodes._path.parent == tmp_path / "memory" / "nemori_memory"

    def test_legacy_compat_no_algo_name(self, tmp_path):
        s = NemoriStore(tmp_path, backend="file")
        assert s.memory_dir == tmp_path / "memory" / "nemori"


class TestMessageDomainModel:
    def test_text_content_string(self):
        msg = Message(role="user", content="Plain text message")
        assert msg.text_content() == "Plain text message"

    def test_text_content_multimodal(self):
        msg = Message(role="user", content=[
            {"type": "text", "text": "Look at this:"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ])
        assert msg.text_content() == "Look at this: [image]"

    def test_has_images(self):
        msg = Message(role="user", content="No image")
        assert not msg.has_images()

        msg2 = Message(role="user", content=[
            {"type": "image_url", "image_url": {"url": "http://x.com/a.png"}},
        ])
        assert msg2.has_images()

    def test_image_urls(self):
        msg = Message(role="user", content=[
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "http://a.com/1.png"}},
            {"type": "image_url", "image_url": {"url": "http://a.com/2.png"}},
        ])
        urls = msg.image_urls()
        assert len(urls) == 2

    def test_from_dict(self):
        data = {
            "message_id": "test-123",
            "role": "user",
            "content": "Hello world",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metadata": {"k": "v"},
        }
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.metadata == {"k": "v"}

    def test_from_summerclaw_message(self):
        nb_msg = {"role": "assistant", "content": "Done", "tools_used": ["read_file"]}
        msg = Message.from_summerclaw_message(nb_msg)
        assert msg.role == "assistant"
        assert msg.metadata["tools_used"] == ["read_file"]