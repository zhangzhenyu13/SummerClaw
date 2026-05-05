"""Tests for EpisodeGenerator — LLM-powered episode narrative generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.memory.nemori_memory.models import Episode, Message
from nanobot.memory.nemori_memory.episode_generator import EpisodeGenerator


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def generator(mock_provider):
    return EpisodeGenerator(mock_provider, "test-model")


# ────────────────────────────────────────────────────────────────────────────
# Basic generation
# ────────────────────────────────────────────────────────────────────────────


class TestEpisodeGenerator:
    """Episode generator tests."""

    @pytest.mark.asyncio
    async def test_generate_episode(self, generator, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '{"title": "Test Title", "content": "A test episode.", "timestamp": "2025-03-15T14:30:00"}'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [Message(role="user", content="hello")]
        episode = await generator.generate("u1", "agent1", msgs, "topic shift")

        assert isinstance(episode, Episode)
        assert episode.user_id == "u1"
        assert episode.agent_id == "agent1"
        assert episode.title == "Test Title"
        assert episode.content == "A test episode."
        assert episode.metadata["boundary_reason"] == "topic shift"

    @pytest.mark.asyncio
    async def test_generate_parses_timestamp(self, generator, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '{"title": "T", "content": "C", "timestamp": "2025-06-15T08:30:00+00:00"}'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [Message(role="user", content="x")]
        episode = await generator.generate("u1", "default", msgs, "test")
        assert episode.created_at.year == 2025
        assert episode.created_at.month == 6
        assert episode.created_at.day == 15

    @pytest.mark.asyncio
    async def test_generate_fallback_on_error(self, generator, mock_provider):
        mock_provider.chat_with_retry.side_effect = RuntimeError("LLM error")
        msgs = [Message(role="user", content="hello")]
        episode = await generator.generate("u1", "default", msgs, "error")
        assert isinstance(episode, Episode)
        assert episode.metadata.get("fallback") is True
        assert "hello" in episode.content

    @pytest.mark.asyncio
    async def test_generate_fallback_on_invalid_json(self, generator, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = "not valid json"
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [Message(role="user", content="hello")]
        episode = await generator.generate("u1", "default", msgs, "test")
        assert isinstance(episode, Episode)
        assert episode.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_generate_multimodal_message(self, generator, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '{"title": "Image chat", "content": "Discussed images.", "timestamp": "2025-01-01T12:00:00"}'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            ),
        ]
        episode = await generator.generate("u1", "default", msgs, "image")
        assert isinstance(episode, Episode)
        assert episode.title == "Image chat"

    @pytest.mark.asyncio
    async def test_generate_json_with_markdown_fences(self, generator, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '```json\n{"title": "T", "content": "C", "timestamp": "2025-01-01T00:00:00"}\n```'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [Message(role="user", content="x")]
        episode = await generator.generate("u1", "default", msgs, "test")
        assert episode.title == "T"
        assert episode.content == "C"

    @pytest.mark.asyncio
    async def test_generate_source_messages_preserved(self, generator, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '{"title": "T", "content": "C", "timestamp": "2025-01-01T00:00:00"}'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [Message(role="user", content="hello")]
        episode = await generator.generate("u1", "default", msgs, "test")
        assert len(episode.source_messages) == 1
        assert episode.source_messages[0]["role"] == "user"
        assert episode.source_messages[0]["content"] == "hello"


# ────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ────────────────────────────────────────────────────────────────────────────


class TestEpisodeGeneratorJsonParsing:
    """JSON response parsing from generator."""

    def test_parse_response_plain(self):
        from nanobot.memory.nemori_memory.episode_generator import EpisodeGenerator
        gen = EpisodeGenerator(MagicMock(), "x")
        result = gen._parse_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_response_with_fence(self):
        from nanobot.memory.nemori_memory.episode_generator import EpisodeGenerator
        gen = EpisodeGenerator(MagicMock(), "x")
        result = gen._parse_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}
