"""Tests for BatchSegmenter — LLM-powered message batch segmentation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from summerclaw.memory.nemori_memory.models import Message
from summerclaw.memory.nemori_memory.segmenter import BatchSegmenter


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def segmenter(mock_provider):
    return BatchSegmenter(mock_provider, "test-model")


# ────────────────────────────────────────────────────────────────────────────
# Basic segmentation
# ────────────────────────────────────────────────────────────────────────────


class TestBatchSegmenter:
    """Segmenter basic functionality."""

    @pytest.mark.asyncio
    async def test_segment_empty_messages(self, segmenter):
        result = await segmenter.segment([])
        assert result == []

    @pytest.mark.asyncio
    async def test_segment_returns_groups(self, segmenter, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = '{"episodes": [{"indices": [1, 2, 3], "topic": "greeting"}]}'
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [
            Message(role="user", content="hello"),
            Message(role="user", content="how are you"),
            Message(role="assistant", content="I am fine"),
        ]
        result = await segmenter.segment(msgs)
        assert len(result) == 1
        assert result[0]["topic"] == "greeting"
        assert len(result[0]["messages"]) == 3

    @pytest.mark.asyncio
    async def test_segment_multiple_groups(self, segmenter, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '{"episodes": ['
            '  {"indices": [1, 2], "topic": "greeting"},'
            '  {"indices": [3, 4], "topic": "question"}'
            ']}'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="what is Python?"),
            Message(role="assistant", content="a language"),
        ]
        result = await segmenter.segment(msgs)
        assert len(result) == 2
        assert result[0]["topic"] == "greeting"
        assert result[1]["topic"] == "question"

    @pytest.mark.asyncio
    async def test_segment_fallback_on_llm_error(self, segmenter, mock_provider):
        mock_provider.chat_with_retry.side_effect = RuntimeError("LLM error")
        msgs = [Message(role="user", content="hello")]
        result = await segmenter.segment(msgs)
        assert len(result) == 1
        assert result[0]["topic"] == "conversation"
        assert result[0]["messages"] == msgs

    @pytest.mark.asyncio
    async def test_segment_fallback_on_empty_json(self, segmenter, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = '{"episodes": []}'
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [Message(role="user", content="hello")]
        result = await segmenter.segment(msgs)
        assert len(result) == 1
        assert result[0]["topic"] == "conversation"

    @pytest.mark.asyncio
    async def test_segment_parse_json_with_markdown_fences(self, segmenter, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = '```json\n{"episodes": [{"indices": [1, 2], "topic": "test"}]}\n```'
        mock_provider.chat_with_retry.return_value = mock_resp

        msgs = [
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
        ]
        result = await segmenter.segment(msgs)
        assert len(result) == 1
        assert result[0]["topic"] == "test"


# ────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ────────────────────────────────────────────────────────────────────────────


class TestSegmenterJsonParsing:
    """JSON response parsing."""

    def test_parse_json_plain(self):
        result = BatchSegmenter._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_fence(self):
        result = BatchSegmenter._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_with_fence_no_trailing(self):
        result = BatchSegmenter._parse_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_invalid(self):
        with pytest.raises(Exception):
            BatchSegmenter._parse_json("not json")
