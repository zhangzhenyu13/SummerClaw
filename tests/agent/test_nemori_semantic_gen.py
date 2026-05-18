"""Tests for SemanticGenerator — Predict-Calibrate knowledge extraction."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from summerclaw.memory.nemori_memory.models import Episode, SemanticMemory
from summerclaw.memory.nemori_memory.semantic_generator import SemanticGenerator, _extract_text


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def gen(mock_provider):
    return SemanticGenerator(mock_provider, "test-model", enable_prediction_correction=True)


def _ep(**kw):
    defaults = {"user_id": "u1", "title": "T", "content": "C", "source_messages": [{"role": "user", "content": "hello"}]}
    defaults.update(kw)
    return Episode(**defaults)


# ────────────────────────────────────────────────────────────────────────────
# Direct Extraction
# ────────────────────────────────────────────────────────────────────────────


class TestSemanticGeneratorDirectExtraction:
    """Direct extraction when no existing semantics."""

    @pytest.mark.asyncio
    async def test_extract_without_existing(self, gen, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = '{"statements": ["User likes Python"]}'
        mock_provider.chat_with_retry.return_value = mock_resp

        episode = _ep()
        results = await gen.generate("u1", "default", episode, [])

        assert len(results) == 1
        assert results[0].content == "User likes Python"
        assert results[0].source_episode_id == episode.id

    @pytest.mark.asyncio
    async def test_extract_multiple_statements(self, gen, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '{"statements": ["User likes Python", "User works at Google"]}'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        episode = _ep()
        results = await gen.generate("u1", "default", episode, [])

        assert len(results) == 2
        contents = {r.content for r in results}
        assert "User likes Python" in contents
        assert "User works at Google" in contents

    @pytest.mark.asyncio
    async def test_extract_empty_statements(self, gen, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = '{"statements": []}'
        mock_provider.chat_with_retry.return_value = mock_resp

        episode = _ep()
        results = await gen.generate("u1", "default", episode, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_fallback_on_error(self, gen, mock_provider):
        mock_provider.chat_with_retry.side_effect = RuntimeError("LLM error")
        episode = _ep()
        results = await gen.generate("u1", "default", episode, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_markdown_fence(self, gen, mock_provider):
        mock_resp = MagicMock()
        mock_resp.content = (
            '```json\n{"statements": ["User likes hiking"]}\n```'
        )
        mock_provider.chat_with_retry.return_value = mock_resp

        episode = _ep()
        results = await gen.generate("u1", "default", episode, [])
        assert len(results) == 1
        assert results[0].content == "User likes hiking"


# ────────────────────────────────────────────────────────────────────────────
# Predict-Calibrate
# ────────────────────────────────────────────────────────────────────────────


class TestSemanticGeneratorPredictCalibrate:
    """Predict-Calibrate two-step extraction."""

    @pytest.mark.asyncio
    async def test_predict_calibrate_with_existing(self, gen, mock_provider):
        """When existing semantics present, use Predict-Calibrate path."""
        existing = [SemanticMemory(user_id="u1", content="User knows Python", memory_type="identity")]

        predict_resp = MagicMock()
        predict_resp.content = "Predicted: user discusses Python."
        extract_resp = MagicMock()
        extract_resp.content = '{"statements": ["User uses Django"]}'

        mock_provider.chat_with_retry.side_effect = [predict_resp, extract_resp]

        episode = Episode(
            user_id="u1", title="Python chat", content="Discussed Python web frameworks",
            source_messages=[
                {"role": "user", "content": "I use Django for web dev"},
                {"role": "assistant", "content": "That's great"},
            ],
        )
        results = await gen.generate("u1", "default", episode, existing)

        assert len(results) == 1
        assert results[0].content == "User uses Django"
        assert mock_provider.chat_with_retry.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_fails_falls_back_to_direct(self, gen, mock_provider):
        """If predict step fails, fall back to direct extraction."""
        existing = [SemanticMemory(user_id="u1", content="User knows Python", memory_type="identity")]

        extract_resp = MagicMock()
        extract_resp.content = '{"statements": ["User likes Python"]}'
        predict_error = RuntimeError("Predict error")

        mock_provider.chat_with_retry.side_effect = [predict_error, extract_resp]

        episode = _ep()
        results = await gen.generate("u1", "default", episode, existing)

        assert len(results) == 1
        assert results[0].content == "User likes Python"

    @pytest.mark.asyncio
    async def test_extract_step_fails_returns_empty(self, gen, mock_provider):
        """If extract step fails in P-C mode, return empty."""
        existing = [SemanticMemory(user_id="u1", content="X", memory_type="identity")]

        predict_resp = MagicMock()
        predict_resp.content = "Prediction"
        extract_error = RuntimeError("Extract error")

        mock_provider.chat_with_retry.side_effect = [predict_resp, extract_error]

        episode = _ep()
        results = await gen.generate("u1", "default", episode, existing)
        assert results == []

    @pytest.mark.asyncio
    async def test_pc_disabled_uses_direct(self, mock_provider):
        """When prediction_correction disabled, always use direct extraction."""
        gen_disabled = SemanticGenerator(mock_provider, "test-model", enable_prediction_correction=False)
        existing = [SemanticMemory(user_id="u1", content="X", memory_type="identity")]

        mock_resp = MagicMock()
        mock_resp.content = '{"statements": ["Direct fact"]}'
        mock_provider.chat_with_retry.return_value = mock_resp

        results = await gen_disabled.generate("u1", "default", _ep(), existing)
        assert len(results) == 1
        assert results[0].content == "Direct fact"
        assert mock_provider.chat_with_retry.call_count == 1  # only direct


# ────────────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────────────


class TestSemanticGeneratorClassification:
    """Knowledge type classification."""

    def test_identity(self):
        assert SemanticGenerator._classify_type("User's name is John") == "identity"
        assert SemanticGenerator._classify_type("works at Google as engineer") == "identity"

    def test_preference(self):
        assert SemanticGenerator._classify_type("likes Python") == "preference"
        assert SemanticGenerator._classify_type("prefers dark mode") == "preference"
        assert SemanticGenerator._classify_type("favorite book is Dune") == "preference"

    def test_relationship(self):
        assert SemanticGenerator._classify_type("family lives in Boston") == "relationship"
        assert SemanticGenerator._classify_type("wife is a doctor") == "relationship"

    def test_goal(self):
        assert SemanticGenerator._classify_type("wants to learn Rust") == "goal"
        assert SemanticGenerator._classify_type("plan to visit Tokyo") == "goal"

    def test_belief(self):
        assert SemanticGenerator._classify_type("believes in open source") == "belief"

    def test_habit(self):
        assert SemanticGenerator._classify_type("always starts day with coffee") == "habit"
        assert SemanticGenerator._classify_type("routine exercise every morning") == "habit"

    def test_default_identity(self):
        assert SemanticGenerator._classify_type("Some random fact") == "identity"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_extract_text_string(self):
        assert _extract_text({"content": "hello"}) == "hello"

    def test_extract_text_array(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert _extract_text({"content": content}) == "hello world"

    def test_extract_text_with_image(self):
        content = [
            {"type": "text", "text": "Look"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]
        assert _extract_text({"content": content}) == "Look [image]"

    def test_extract_text_empty(self):
        assert _extract_text({}) == ""
