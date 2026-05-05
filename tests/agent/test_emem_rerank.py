"""Tests for EMem rerankers — EDUReranker and ArgumentReranker."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.memory.emem_memory.rerank import ArgumentReranker, EDUReranker


# ===================================================================
# Common fixtures
# ===================================================================

@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLMProvider."""
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


def _make_llm_response(content: str, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.finish_reason = finish_reason
    return resp


# ===================================================================
# EDUReranker tests
# ===================================================================

class TestEDUReranker:
    """Test EDUReranker rerank() and _parse_edu_response."""

    @pytest.fixture
    def reranker(self, mock_provider: MagicMock) -> EDUReranker:
        return EDUReranker(provider=mock_provider, model="test-model")

    async def test_rerank_selects_relevant_edus(
        self, reranker: EDUReranker, mock_provider: MagicMock,
    ) -> None:
        candidates = [
            "Alice deployed the app on Tuesday.",
            "Bob likes ice cream.",
            "The deployment used Kubernetes.",
        ]
        indices = [10, 20, 30]
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"selected_edus": [
                "Alice deployed the app on Tuesday.",
                "The deployment used Kubernetes.",
            ]}),
        )
        filtered_idx, filtered_items, meta = await reranker.rerank(
            query="Tell me about the deployment",
            candidate_items=candidates,
            candidate_indices=indices,
        )
        # Fuzzy matching should match the two deployment-related EDUs
        assert len(filtered_idx) >= 1
        assert meta["num_candidates"] == 3

    async def test_rerank_empty_candidates(
        self, reranker: EDUReranker,
    ) -> None:
        filtered_idx, filtered_items, meta = await reranker.rerank(
            query="query",
            candidate_items=[],
            candidate_indices=[],
        )
        assert filtered_idx == []
        assert filtered_items == []
        assert meta["num_candidates"] == 0
        assert meta["num_selected"] == 0

    async def test_rerank_llm_call_failed(
        self, reranker: EDUReranker, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.side_effect = Exception("API down")
        candidates = ["Some EDU"]
        indices = [5]
        filtered_idx, filtered_items, meta = await reranker.rerank(
            query="query",
            candidate_items=candidates,
            candidate_indices=indices,
        )
        # Should fall back to full candidates on error
        assert filtered_idx == indices
        assert filtered_items == candidates
        assert "error" in meta

    async def test_rerank_llm_error_response(
        self, reranker: EDUReranker, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            "", finish_reason="error",
        )
        candidates = ["Some EDU"]
        indices = [5]
        filtered_idx, filtered_items, meta = await reranker.rerank(
            query="query",
            candidate_items=candidates,
            candidate_indices=indices,
        )
        # Should fall back to full candidates
        assert filtered_idx == indices
        assert "num_selected" in meta

    async def test_rerank_no_matches_fallback(
        self, reranker: EDUReranker, mock_provider: MagicMock,
    ) -> None:
        """When LLM selects EDUs that don't fuzzy-match, fallback to all."""
        candidates = ["Original EDU text"]
        indices = [42]
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"selected_edus": ["Completely different text"]}),
        )
        filtered_idx, filtered_items, meta = await reranker.rerank(
            query="query",
            candidate_items=candidates,
            candidate_indices=indices,
        )
        # Should fallback
        assert filtered_idx == indices
        assert meta.get("fallback") is True

    async def test_rerank_with_max_limit(
        self, reranker: EDUReranker, mock_provider: MagicMock,
    ) -> None:
        candidates = ["A"] * 10
        indices = list(range(10))
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"selected_edus": ["A"]}),
        )
        filtered_idx, filtered_items, _ = await reranker.rerank(
            query="query",
            candidate_items=candidates,
            candidate_indices=indices,
            max_after_rerank=3,
        )
        assert len(filtered_idx) <= 3


# ===================================================================
# EDUReranker — parse
# ===================================================================

class TestEDURerankerParse:
    """Test EDUReranker._parse_edu_response."""

    def test_parse_json_with_selected_edus(self) -> None:
        content = json.dumps({"selected_edus": ["EDU A", "EDU B"]})
        result = EDUReranker._parse_edu_response(content)
        assert result == ["EDU A", "EDU B"]

    def test_parse_markdown_fence_removed(self) -> None:
        content = '```json\n{"selected_edus": ["Markdown EDU"]}\n```'
        result = EDUReranker._parse_edu_response(content)
        assert result == ["Markdown EDU"]

    def test_parse_find_json_in_text(self) -> None:
        content = 'some text {"selected_edus": ["Found"]} trailing'
        result = EDUReranker._parse_edu_response(content)
        assert result == ["Found"]

    def test_parse_invalid_json_returns_empty(self) -> None:
        result = EDUReranker._parse_edu_response("not json")
        assert result == []

    def test_parse_empty_string(self) -> None:
        result = EDUReranker._parse_edu_response("")
        assert result == []


# ===================================================================
# ArgumentReranker tests
# ===================================================================

class TestArgumentReranker:
    """Test ArgumentReranker rerank() and _parse_arg_response."""

    @pytest.fixture
    def reranker(self, mock_provider: MagicMock) -> ArgumentReranker:
        return ArgumentReranker(provider=mock_provider, model="test-model")

    async def test_rerank_selects_relevant_arguments(
        self, reranker: ArgumentReranker, mock_provider: MagicMock,
    ) -> None:
        candidate_args = ["Alice", "Ice Cream", "Kubernetes"]
        candidate_keys = ["arg-1", "arg-2", "arg-3"]
        candidate_scores = [0.9, 0.3, 0.85]
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"selected_arguments": ["Alice", "Kubernetes"]}),
        )
        filtered_keys, filtered_args, filtered_scores, meta = await reranker.rerank(
            query="What did Alice deploy?",
            candidate_arguments=candidate_args,
            candidate_arg_keys=candidate_keys,
            candidate_arg_scores=candidate_scores,
        )
        # Fuzzy matching should select at least one
        assert meta["num_candidates"] == 3
        assert len(filtered_args) <= 3

    async def test_rerank_empty_candidates(
        self, reranker: ArgumentReranker,
    ) -> None:
        result = await reranker.rerank(
            query="query",
            candidate_arguments=[],
            candidate_arg_keys=[],
            candidate_arg_scores=[],
        )
        assert result == ([], [], [], {"num_candidates": 0, "num_selected": 0})

    async def test_rerank_llm_call_failed(
        self, reranker: ArgumentReranker, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.side_effect = Exception("API down")
        result = await reranker.rerank(
            query="query",
            candidate_arguments=["Alice"],
            candidate_arg_keys=["arg-1"],
            candidate_arg_scores=[0.9],
        )
        # Should fall back
        assert result[0] == ["arg-1"]
        assert result[1] == ["Alice"]

    async def test_rerank_llm_error_response(
        self, reranker: ArgumentReranker, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            "", finish_reason="error",
        )
        result = await reranker.rerank(
            query="query",
            candidate_arguments=["Alice"],
            candidate_arg_keys=["arg-1"],
            candidate_arg_scores=[0.9],
        )
        assert result[0] == ["arg-1"]

    async def test_rerank_with_max_limit(
        self, reranker: ArgumentReranker, mock_provider: MagicMock,
    ) -> None:
        candidate_args = ["A"] * 10
        candidate_keys = [f"arg-{i}" for i in range(10)]
        candidate_scores = [0.9] * 10
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"selected_arguments": ["A"]}),
        )
        filtered_keys, _, _, _ = await reranker.rerank(
            query="query",
            candidate_arguments=candidate_args,
            candidate_arg_keys=candidate_keys,
            candidate_arg_scores=candidate_scores,
            max_after_rerank=3,
        )
        assert len(filtered_keys) <= 3


# ===================================================================
# ArgumentReranker — parse
# ===================================================================

class TestArgumentRerankerParse:
    """Test ArgumentReranker._parse_arg_response."""

    def test_parse_json_with_selected_arguments(self) -> None:
        content = json.dumps({"selected_arguments": ["Alice", "Bob"]})
        result = ArgumentReranker._parse_arg_response(content)
        assert result == ["Alice", "Bob"]

    def test_parse_markdown_fence_removed(self) -> None:
        content = '```\n{"selected_arguments": ["Entity"]}\n```'
        result = ArgumentReranker._parse_arg_response(content)
        assert result == ["Entity"]

    def test_parse_invalid_json_returns_empty(self) -> None:
        result = ArgumentReranker._parse_arg_response("invalid")
        assert result == []

    def test_parse_empty_string(self) -> None:
        result = ArgumentReranker._parse_arg_response("")
        assert result == []
