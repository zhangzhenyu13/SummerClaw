"""Unit tests for SkillOpt lr_autonomous module."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.lr_autonomous import (
    _coerce_nonnegative_int,
    decide_autonomous_learning_rate,
)


class TestCoerceNonnegativeInt:
    def test_positive_int(self):
        assert _coerce_nonnegative_int(5) == 5

    def test_zero(self):
        assert _coerce_nonnegative_int(0) == 0

    def test_negative_clamped(self):
        assert _coerce_nonnegative_int(-3) == 0

    def test_float_integer(self):
        assert _coerce_nonnegative_int(4.0) == 4

    def test_float_non_integer(self):
        # P7: float non-integer falls through to regex (official behavior)
        # str(4.5) = "4.5" → regex extracts 4
        assert _coerce_nonnegative_int(4.5) == 4

    def test_bool_returns_none(self):
        assert _coerce_nonnegative_int(True) is None
        assert _coerce_nonnegative_int(False) is None

    def test_string_int(self):
        assert _coerce_nonnegative_int("7") == 7

    def test_string_with_number(self):
        assert _coerce_nonnegative_int("lr=3") == 3

    def test_empty_string(self):
        assert _coerce_nonnegative_int("") is None

    def test_none(self):
        assert _coerce_nonnegative_int(None) is None

    def test_negative_string(self):
        assert _coerce_nonnegative_int("-5") == 0


from collections import namedtuple

MockResponse = namedtuple("MockResponse", ["content"])


def _make_mock_provider(response_text: str):
    """Create a mock provider that returns the given text."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MockResponse(content=response_text))
    return provider


class TestDecideAutonomousLearningRate:
    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        """LLM returns valid JSON with learning_rate."""
        response = json.dumps({
            "learning_rate": 3,
            "reasoning": "moderate failures detected",
            "confidence": "medium",
            "risk_notes": [],
        })
        provider = _make_mock_provider(response)
        merged_patch = {"edits": [{"op": "append", "content": "edit1"}] * 5}

        result = await decide_autonomous_learning_rate(
            provider, "test-model",
            skill_content="# Skill",
            merged_patch=merged_patch,
            update_mode="patch",
            rollout_hard=0.3,
            rollout_soft=0.4,
            rollout_n=5,
            system_prompt="Test system prompt",
        )
        assert result["learning_rate"] == 3
        assert result["fallback"] is False
        assert result["available_update_items"] == 5

    @pytest.mark.asyncio
    async def test_learning_rate_clamped_to_available(self):
        """LLM returns LR larger than available items."""
        response = json.dumps({"learning_rate": 100})
        provider = _make_mock_provider(response)
        merged_patch = {"edits": [{"op": "append", "content": "e1"}] * 2}

        result = await decide_autonomous_learning_rate(
            provider, "test-model",
            skill_content="# Skill",
            merged_patch=merged_patch,
            update_mode="patch",
            rollout_hard=0.5,
            rollout_soft=0.5,
            rollout_n=5,
            system_prompt="Test",
        )
        assert result["learning_rate"] == 2
        assert result["clamped"] is True

    @pytest.mark.asyncio
    async def test_invalid_response_fallback(self):
        """LLM returns invalid content -> fallback to 0."""
        provider = _make_mock_provider("not valid json at all")
        merged_patch = {"edits": [{"op": "append", "content": "e1"}]}

        result = await decide_autonomous_learning_rate(
            provider, "test-model",
            skill_content="# Skill",
            merged_patch=merged_patch,
            update_mode="patch",
            rollout_hard=0.5,
            rollout_soft=0.5,
            rollout_n=5,
            system_prompt="Test",
        )
        assert result["fallback"] is True
        assert result["learning_rate"] == 0

    @pytest.mark.asyncio
    async def test_llm_exception_fallback(self):
        """LLM call raises exception -> fallback."""
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API error"))
        merged_patch = {"edits": [{"op": "append", "content": "e1"}]}

        result = await decide_autonomous_learning_rate(
            provider, "test-model",
            skill_content="# Skill",
            merged_patch=merged_patch,
            update_mode="patch",
            rollout_hard=0.5,
            rollout_soft=0.5,
            rollout_n=5,
            system_prompt="Test",
        )
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_empty_merged_patch(self):
        """No items available -> LR=0 regardless."""
        response = json.dumps({"learning_rate": 5})
        provider = _make_mock_provider(response)
        merged_patch = {"edits": []}

        result = await decide_autonomous_learning_rate(
            provider, "test-model",
            skill_content="# Skill",
            merged_patch=merged_patch,
            update_mode="patch",
            rollout_hard=0.5,
            rollout_soft=0.5,
            rollout_n=5,
            system_prompt="Test",
        )
        assert result["learning_rate"] == 0
        assert result["available_update_items"] == 0
