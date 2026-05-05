"""Tests for task complexity evaluator (nanobot.agent.complexity).

Covers:
  Phase 1 — regex pre-filter (_is_trivially_simple)
  Phase 2 — LLM classifier (LLMComplexityEvaluator.evaluate)
  Legacy — deprecated is_complex_task
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.complexity import (
    LLMComplexityEvaluator,
    _is_trivially_simple,
    is_complex_task,
)
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMResponse

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


# ============================================================================
# Phase 1 — regex pre-filter tests
# ============================================================================

class TestPhase1TrivialMessages:
    """_is_trivially_simple should catch obviously trivial messages."""

    # ---- very short ----
    def test_empty(self) -> None:
        assert _is_trivially_simple("") is True

    def test_whitespace(self) -> None:
        assert _is_trivially_simple("   ") is True

    def test_short_message(self) -> None:
        assert _is_trivially_simple("a" * 14) is True

    def test_exactly_threshold(self) -> None:
        assert _is_trivially_simple("a" * 15) is False  # 15 >= threshold

    # ---- greetings ----
    def test_hi(self) -> None:
        assert _is_trivially_simple("hi") is True

    def test_hello(self) -> None:
        assert _is_trivially_simple("hello") is True

    def test_good_morning(self) -> None:
        assert _is_trivially_simple("good morning") is True

    def test_how_are_you(self) -> None:
        assert _is_trivially_simple("how are you") is True

    def test_long_greeting_not_trivial(self) -> None:
        # A long message starting with a greeting is NOT trivially simple
        long_hello = (
            "hello, I need you to help me analyze a complex issue "
            "with the database migration that keeps failing"
        )
        assert _is_trivially_simple(long_hello) is False

    # ---- thanks / acks ----
    def test_thanks(self) -> None:
        assert _is_trivially_simple("thanks") is True

    def test_ok(self) -> None:
        assert _is_trivially_simple("ok") is True

    def test_got_it(self) -> None:
        assert _is_trivially_simple("got it") is True

    def test_sure(self) -> None:
        assert _is_trivially_simple("sure") is True

    def test_long_ack_not_trivial(self) -> None:
        assert _is_trivially_simple(
            "ok thanks, but actually I also need you to refactor the cache layer"
        ) is False

    # ---- yes/no ----
    def test_yes(self) -> None:
        assert _is_trivially_simple("yes") is True

    def test_no(self) -> None:
        assert _is_trivially_simple("no") is True

    def test_proceed(self) -> None:
        assert _is_trivially_simple("proceed") is True

    def test_go_ahead(self) -> None:
        assert _is_trivially_simple("go ahead") is True


class TestPhase1NonTrivialMessages:
    """Messages that should pass Phase 1 and go to LLM."""

    def test_factual_question(self) -> None:
        assert _is_trivially_simple("What is the capital of France?") is False

    def test_code_question(self) -> None:
        assert _is_trivially_simple("how do i print hello world in python") is False

    def test_single_command(self) -> None:
        assert _is_trivially_simple("list files in current directory") is False

    def test_build_task(self) -> None:
        assert _is_trivially_simple("Build a complete REST API") is False

    def test_complex_analysis(self) -> None:
        assert _is_trivially_simple(
            "First analyze the codebase, then refactor the database layer"
        ) is False


# ============================================================================
# Deprecated is_complex_task — now a simple wrapper
# ============================================================================

class TestDeprecatedIsComplexTask:
    """The deprecated is_complex_task returns True for anything that
    passes Phase 1 (conservative — assumes complex without LLM)."""

    def test_trivial_returns_false(self) -> None:
        assert is_complex_task("hi") is False
        assert is_complex_task("thanks") is False
        assert is_complex_task("") is False

    def test_non_trivial_returns_true(self) -> None:
        # Without LLM verification, defaults to True (conservative)
        assert is_complex_task("What is the capital of France?") is True
        assert is_complex_task("Build a REST API") is True
        assert is_complex_task("list files in current directory") is True


# ============================================================================
# Phase 2 — LLM evaluator tests
# ============================================================================

def _make_evaluator():
    """Create an LLMComplexityEvaluator with a mock provider."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = MagicMock()
    provider.generation.max_tokens = 4096
    evaluator = LLMComplexityEvaluator(
        provider=provider,
        model="test-model",
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    )
    return evaluator, provider


class TestLLMEvaluatorPhase1ShortCircuit:
    """Phase 1 (regex) should short-circuit before any LLM call."""

    @pytest.mark.asyncio
    async def test_empty_task(self) -> None:
        evaluator, provider = _make_evaluator()
        result = await evaluator.evaluate("")
        assert result is False
        # No LLM call was made
        provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_greeting(self) -> None:
        evaluator, provider = _make_evaluator()
        result = await evaluator.evaluate("hi")
        assert result is False
        provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_thanks(self) -> None:
        evaluator, provider = _make_evaluator()
        result = await evaluator.evaluate("thanks")
        assert result is False
        provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_acknowledgment(self) -> None:
        evaluator, provider = _make_evaluator()
        result = await evaluator.evaluate("ok got it")
        assert result is False
        provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only(self) -> None:
        evaluator, provider = _make_evaluator()
        result = await evaluator.evaluate("   ")
        assert result is False
        provider.chat_with_retry.assert_not_called()


class TestLLMEvaluatorPhase2Classification:
    """Phase 2: LLM-based classification tests."""

    @pytest.mark.asyncio
    async def test_llm_returns_complex(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="COMPLEX",
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 1},
        ))
        result = await evaluator.evaluate(
            "Build a complete REST API with authentication and database"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_llm_returns_simple(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="SIMPLE",
            tool_calls=[],
            usage={"prompt_tokens": 80, "completion_tokens": 1},
        ))
        result = await evaluator.evaluate("what is the capital of France?")
        assert result is False

    @pytest.mark.asyncio
    async def test_llm_returns_lowercase(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="simple",
            tool_calls=[],
            usage={"prompt_tokens": 50, "completion_tokens": 1},
        ))
        result = await evaluator.evaluate("tell me a joke")
        assert result is False

    @pytest.mark.asyncio
    async def test_llm_returns_with_extra_text(self) -> None:
        """LLM might return extra text like 'COMPLEX - multi-step task'."""
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="COMPLEX. The task requires planning.",
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 5},
        ))
        result = await evaluator.evaluate(
            "First analyze the logs, then fix the bug"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_llm_returns_with_punctuation(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="SIMPLE!",
            tool_calls=[],
            usage={"prompt_tokens": 50, "completion_tokens": 1},
        ))
        result = await evaluator.evaluate("what time is it")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_stripped_content_defaults_complex(self) -> None:
        """Content that strips to empty (just whitespace) should default.
        The runner's retry logic turns whitespace-only into a fallback
        response, so we go through the normal path here — whitespace is
        not truly "empty" to the runner.  We test this via runner mock."""
        evaluator, provider = _make_evaluator()

        import nanobot.agent.runner as runner_mod
        from nanobot.agent.runner import AgentRunResult

        async def mock_run(self, spec):
            return AgentRunResult(
                final_content="",  # empty → strip → no first word
                messages=spec.initial_messages,
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(runner_mod.AgentRunner, "run", mock_run)
            result = await evaluator.evaluate("Build a REST API with auth")
        assert result is True  # empty after strip → conservative fallback

    @pytest.mark.asyncio
    async def test_unexpected_response_treated_as_simple(self) -> None:
        """Any response not starting with COMPLEX is treated as SIMPLE."""
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="The task seems straightforward, just a simple question.",
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 10},
        ))
        result = await evaluator.evaluate("what is python?")
        assert result is False  # doesn't start with COMPLEX → SIMPLE


class TestLLMEvaluatorErrorHandling:
    """LLM call failures should default to COMPLEX (conservative)."""

    @pytest.mark.asyncio
    async def test_llm_call_raises_exception(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("network error"))
        result = await evaluator.evaluate("Build a REST API")
        assert result is True  # fallback to COMPLEX

    @pytest.mark.asyncio
    async def test_llm_stop_reason_error(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[],
            usage={},
        ))
        # Set stop_reason by patching the runner
        import nanobot.agent.runner as runner_mod
        from nanobot.agent.runner import AgentRunResult

        async def mock_run(self, spec):
            return AgentRunResult(
                final_content=None,
                messages=spec.initial_messages,
                stop_reason="error",
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(runner_mod.AgentRunner, "run", mock_run)
            result = await evaluator.evaluate("Build a REST API")
        assert result is True  # fallback to COMPLEX

    @pytest.mark.asyncio
    async def test_llm_none_content_runner_error(self) -> None:
        """When runner returns final_content=None due to error, default to COMPLEX."""
        evaluator, provider = _make_evaluator()

        import nanobot.agent.runner as runner_mod
        from nanobot.agent.runner import AgentRunResult

        async def mock_run(self, spec):
            return AgentRunResult(
                final_content=None,
                messages=spec.initial_messages,
                stop_reason="error",
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(runner_mod.AgentRunner, "run", mock_run)
            result = await evaluator.evaluate("Build a REST API")
        assert result is True  # fallback to COMPLEX


# ============================================================================
# Real-world classification scenarios
# ============================================================================

class TestRealWorldClassification:
    """Test that the LLM evaluator correctly classifies representative tasks.
    Uses mocked LLM responses to verify the full pipeline."""

    @pytest.mark.asyncio
    async def test_simple_greeting_phase1(self) -> None:
        evaluator, _ = _make_evaluator()
        assert await evaluator.evaluate("hi") is False
        assert await evaluator.evaluate("hello there") is False

    @pytest.mark.asyncio
    async def test_simple_thanks_phase1(self) -> None:
        evaluator, _ = _make_evaluator()
        assert await evaluator.evaluate("thanks") is False
        assert await evaluator.evaluate("ok") is False

    @pytest.mark.asyncio
    async def test_simple_question_llm_simple(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="SIMPLE", tool_calls=[], usage={},
        ))
        assert await evaluator.evaluate("what is the capital of France?") is False
        # Should have called LLM (Phase 1 didn't catch it)
        provider.chat_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_complex_build_llm_complex(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="COMPLEX", tool_calls=[], usage={},
        ))
        assert await evaluator.evaluate(
            "Build a complete REST API with authentication"
        ) is True
        provider.chat_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_step_llm_complex(self) -> None:
        evaluator, provider = _make_evaluator()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="COMPLEX", tool_calls=[], usage={},
        ))
        assert await evaluator.evaluate(
            "First analyze the error, then fix it, finally add tests"
        ) is True
