"""Tests for Nemori domain models — Message, Episode, SemanticMemory."""

import json
from datetime import datetime, timezone

import pytest

from nanobot.memory.nemori_memory.models import Episode, Message, SemanticMemory


# ────────────────────────────────────────────────────────────────────────────
# Message
# ────────────────────────────────────────────────────────────────────────────


class TestMessage:
    """Message model tests."""

    def test_create_simple_text_message(self):
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert isinstance(msg.timestamp, datetime)
        assert msg.message_id  # auto-generated

    def test_create_multimodal_message(self):
        content = [
            {"type": "text", "text": "Look at this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role="user", content=content)
        assert msg.has_images() is True
        assert msg.image_urls() == ["https://example.com/img.png"]

    def test_text_content_extracts_text_parts(self):
        msg = Message(
            role="user",
            content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}],
        )
        assert msg.text_content() == "Hello World"

    def test_text_content_strips_images_by_default(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "HW"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        )
        assert msg.text_content(include_placeholders=False) == "HW"
        assert msg.text_content(include_placeholders=True) == "HW [image]"

    def test_text_content_string_content(self):
        msg = Message(role="user", content="plain text")
        assert msg.text_content() == "plain text"

    def test_has_images_string_content(self):
        msg = Message(role="user", content="plain text")
        assert msg.has_images() is False

    def test_image_urls_string_content(self):
        msg = Message(role="user", content="plain text")
        assert msg.image_urls() == []

    def test_to_dict_includes_all_fields(self):
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msg = Message(
            role="assistant",
            content="test",
            timestamp=ts,
            metadata={"tool": "search"},
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "test"
        assert d["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert d["metadata"] == {"tool": "search"}
        assert d["message_id"] == msg.message_id

    def test_from_dict_roundtrip(self):
        original = Message(role="user", content="hello")
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.message_id == original.message_id

    def test_from_dict_parses_iso_timestamp(self):
        d = {
            "role": "user",
            "content": "hi",
            "timestamp": "2025-06-15T08:30:00+00:00",
            "message_id": "abc-123",
        }
        msg = Message.from_dict(d)
        assert msg.timestamp.year == 2025
        assert msg.timestamp.month == 6

    def test_from_dict_handles_missing_timestamp(self):
        d = {"role": "user", "content": "hi"}
        msg = Message.from_dict(d)
        assert isinstance(msg.timestamp, datetime)

    def test_from_dict_handles_missing_message_id(self):
        d = {"role": "user", "content": "hi"}
        msg = Message.from_dict(d)
        assert msg.message_id  # auto-generated

    def test_from_nanobot_message_basic(self):
        d = {"role": "assistant", "content": "response"}
        msg = Message.from_nanobot_message(d)
        assert msg.role == "assistant"
        assert msg.content == "response"

    def test_from_nanobot_message_with_tools(self):
        d = {
            "role": "assistant",
            "content": "done",
            "timestamp": "2025-01-01T12:00:00",
            "tools_used": ["read_file", "edit_file"],
        }
        msg = Message.from_nanobot_message(d)
        assert msg.metadata["tools_used"] == ["read_file", "edit_file"]

    def test_from_nanobot_message_invalid_timestamp(self):
        d = {"role": "user", "content": "hi", "timestamp": "not-a-date"}
        msg = Message.from_nanobot_message(d)
        assert isinstance(msg.timestamp, datetime)  # fallback to now

    def test_custom_message_id(self):
        msg = Message(role="user", content="test", message_id="custom-123")
        assert msg.message_id == "custom-123"


# ────────────────────────────────────────────────────────────────────────────
# Episode
# ────────────────────────────────────────────────────────────────────────────


class TestEpisode:
    """Episode model tests."""

    def test_create_episode(self):
        ep = Episode(
            user_id="u1", title="Test Episode", content="Something happened.",
            source_messages=[],
        )
        assert ep.user_id == "u1"
        assert ep.title == "Test Episode"
        assert ep.agent_id == "default"
        assert ep.id  # auto-generated
        assert ep.embedding is None
        assert ep.metadata == {}
        assert ep.source_messages == []

    def test_to_dict_roundtrip(self):
        ts = datetime(2025, 3, 1, tzinfo=timezone.utc)
        ep = Episode(
            user_id="u1",
            title="Title",
            content="Content",
            source_messages=[{"role": "user", "content": "hi"}],
            agent_id="agent1",
            metadata={"key": "value"},
            created_at=ts,
            updated_at=ts,
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)
        assert restored.user_id == "u1"
        assert restored.title == "Title"
        assert restored.content == "Content"
        assert restored.agent_id == "agent1"
        assert restored.metadata == {"key": "value"}
        assert restored.id == ep.id

    def test_from_dict_handles_missing_fields(self):
        d = {"user_id": "u1", "title": "T", "content": "C"}
        ep = Episode.from_dict(d)
        assert ep.user_id == "u1"
        assert ep.agent_id == "default"
        assert ep.source_messages == []
        assert isinstance(ep.created_at, datetime)

    def test_from_dict_parses_iso_timestamps(self):
        d = {
            "user_id": "u1",
            "title": "T",
            "content": "C",
            "created_at": "2025-04-01T10:00:00+00:00",
            "updated_at": "2025-04-02T10:00:00+00:00",
        }
        ep = Episode.from_dict(d)
        assert ep.created_at.month == 4
        assert ep.updated_at.day == 2

    def test_custom_id(self):
        ep = Episode(user_id="u1", title="T", content="C", id="custom-ep-1", source_messages=[])
        assert ep.id == "custom-ep-1"

    def test_embedding_stored(self):
        ep = Episode(
            user_id="u1", title="T", content="C", source_messages=[],
            embedding=[0.1, 0.2, 0.3],
        )
        assert ep.embedding == [0.1, 0.2, 0.3]


# ────────────────────────────────────────────────────────────────────────────
# SemanticMemory
# ────────────────────────────────────────────────────────────────────────────


class TestSemanticMemory:
    """SemanticMemory model tests."""

    def test_create_semantic_memory(self):
        sm = SemanticMemory(
            user_id="u1", content="User likes Python", memory_type="preference",
        )
        assert sm.user_id == "u1"
        assert sm.content == "User likes Python"
        assert sm.memory_type == "preference"
        assert sm.agent_id == "default"
        assert sm.confidence == 1.0
        assert sm.source_episode_id is None

    def test_to_dict_roundtrip(self):
        sm = SemanticMemory(
            user_id="u1",
            content="fact",
            memory_type="identity",
            agent_id="agent1",
            source_episode_id="ep-1",
            confidence=0.95,
            metadata={"src": "nemori"},
        )
        d = sm.to_dict()
        restored = SemanticMemory.from_dict(d)
        assert restored.user_id == "u1"
        assert restored.content == "fact"
        assert restored.memory_type == "identity"
        assert restored.confidence == 0.95
        assert restored.source_episode_id == "ep-1"

    def test_from_dict_with_knowledge_type_alias(self):
        """from_dict handles legacy 'knowledge_type' key."""
        d = {"user_id": "u1", "content": "fact", "knowledge_type": "goal"}
        sm = SemanticMemory.from_dict(d)
        assert sm.memory_type == "goal"

    def test_from_dict_missing_type_defaults_to_empty(self):
        d = {"user_id": "u1", "content": "fact"}
        sm = SemanticMemory.from_dict(d)
        assert sm.memory_type == ""

    def test_embedding_stored(self):
        sm = SemanticMemory(
            user_id="u1", content="f", memory_type="identity",
            embedding=[0.5, 0.6],
        )
        assert sm.embedding == [0.5, 0.6]


# ────────────────────────────────────────────────────────────────────────────
# JSON serialization
# ────────────────────────────────────────────────────────────────────────────


class TestJsonSerialization:
    """Verify all models can be serialized to JSON and back."""

    def test_message_json_roundtrip(self):
        msg = Message(role="user", content="hello")
        d = msg.to_dict()
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        msg2 = Message.from_dict(reloaded)
        assert msg2.role == "user"
        assert msg2.content == "hello"

    def test_episode_json_roundtrip(self):
        ep = Episode(user_id="u1", title="T", content="C", source_messages=[])
        d = ep.to_dict()
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        ep2 = Episode.from_dict(reloaded)
        assert ep2.title == "T"

    def test_semantic_json_roundtrip(self):
        sm = SemanticMemory(user_id="u1", content="fact", memory_type="identity")
        d = sm.to_dict()
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        sm2 = SemanticMemory.from_dict(reloaded)
        assert sm2.content == "fact"
