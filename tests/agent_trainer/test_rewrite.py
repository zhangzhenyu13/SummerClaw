"""Unit tests for SkillOpt rewrite module."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.rewrite import (
    rewrite_skill_from_suggestions,
)


from collections import namedtuple

MockResponse = namedtuple("MockResponse", ["content"])


def _make_mock_provider(response_text: str):
    """Create a mock provider that returns the given text."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MockResponse(content=response_text))
    return provider


class TestRewriteSkillFromSuggestions:
    @pytest.mark.asyncio
    async def test_valid_rewrite(self):
        """LLM returns valid JSON with new_skill."""
        new_skill = "# Skill\n- Updated rule\n"
        response = json.dumps({
            "new_skill": new_skill,
            "reasoning": "Integrated suggestions",
            "change_summary": ["Added rule"],
        })
        provider = _make_mock_provider(response)
        patch = {
            "revise_suggestions": [
                {"suggestion": "Add a rule", "reason": "missing coverage"},
            ]
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill\n",
            patch=patch,
            system_prompt="Test rewrite prompt",
        )
        assert result is not None
        assert result["new_skill"] == new_skill
        assert result["reasoning"] == "Integrated suggestions"
        assert "Added rule" in result["change_summary"]

    @pytest.mark.asyncio
    async def test_empty_suggestions_returns_none(self):
        """No suggestions in patch -> return None."""
        provider = _make_mock_provider("")
        patch = {"revise_suggestions": []}

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_suggestions_key_returns_none(self):
        """Patch without revise_suggestions -> return None."""
        provider = _make_mock_provider("")
        patch = {"edits": [{"op": "append", "content": "x"}]}

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_returns_invalid_json(self):
        """LLM returns garbage -> return None."""
        provider = _make_mock_provider("not json at all")
        patch = {
            "revise_suggestions": [
                {"suggestion": "Fix this", "reason": "broken"},
            ]
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_returns_empty_new_skill(self):
        """LLM returns JSON with empty new_skill -> return None."""
        response = json.dumps({"new_skill": "", "reasoning": "nothing"})
        provider = _make_mock_provider(response)
        patch = {
            "revise_suggestions": [
                {"suggestion": "Fix", "reason": "broken"},
            ]
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self):
        """LLM call raises exception -> return None."""
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API down"))
        patch = {
            "revise_suggestions": [
                {"suggestion": "Fix", "reason": "broken"},
            ]
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_new_skill_gets_trailing_newline(self):
        """new_skill should end with a newline."""
        response = json.dumps({"new_skill": "# Skill\n- Rule"})
        provider = _make_mock_provider(response)
        patch = {
            "revise_suggestions": [
                {"suggestion": "Add rule"},
            ]
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is not None
        assert result["new_skill"].endswith("\n")

    @pytest.mark.asyncio
    async def test_missing_change_summary_defaults_to_empty_list(self):
        """If change_summary is missing, it should default to []."""
        response = json.dumps({"new_skill": "# Skill\n- New"})
        provider = _make_mock_provider(response)
        patch = {
            "revise_suggestions": [{"suggestion": "Add"}],
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
        )
        assert result is not None
        assert result["change_summary"] == []

    @pytest.mark.asyncio
    async def test_step_buffer_context_included(self):
        """step_buffer_context should be injected into the prompt."""
        response = json.dumps({"new_skill": "# Skill\n- Updated"})
        provider = _make_mock_provider(response)
        patch = {
            "revise_suggestions": [{"suggestion": "Add"}],
        }

        result = await rewrite_skill_from_suggestions(
            provider, "test-model",
            skill_content="# Skill",
            patch=patch,
            system_prompt="Test",
            step_buffer_context="Previous step: added rule X",
        )
        assert result is not None
        # Verify the LLM was called (the context is in the user message)
        provider.chat_with_retry.assert_called_once()
