"""Integration tests verifying the planning auto-skip mechanism end-to-end.

Tests that complex tasks enter the planner while simple tasks skip it,
using a mocked AgentLoop with execution_mode='auto' and the LLM-based
complexity evaluator.

Uses loguru's own StringIO sink to capture logs.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from nanobot.agent.complexity import LLMComplexityEvaluator, _is_trivially_simple
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMResponse

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_loguru():
    """Add a StringIO sink to loguru and return the buffer + handler_id."""
    buf = io.StringIO()
    handler_id = logger.add(buf, level="INFO", format="{message}")
    return buf, handler_id


def _make_loop(tmp_path: Path, *, execution_mode: str = "auto"):
    """Create a minimal AgentLoop with mocked dependencies.

    The complexity evaluator's LLM call is also mocked — it returns
    COMPLEX for any non-trivial message by default (conservative).
    """
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = MagicMock()
    provider.generation.max_tokens = 4096

    # Default LLM response: COMPLEX (conservative)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="COMPLEX",
        tool_calls=[],
        usage={"prompt_tokens": 100, "completion_tokens": 1},
    ))

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr, \
         patch("nanobot.memory.naive_memory.consolidator.Consolidator"), \
         patch("nanobot.memory.naive_memory.dream.Dream"), \
         patch("nanobot.memory.naive_memory.auto_compact.AutoCompact"):
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            execution_mode=execution_mode,
            max_iterations=2,
        )
    return loop, bus, provider


# ---------------------------------------------------------------------------
# Phase 1 integration — trivially simple messages skip entirely
# ---------------------------------------------------------------------------

class TestPhase1Integration:
    """Messages caught by Phase 1 regex never reach the LLM."""

    @pytest.mark.asyncio
    async def test_short_greeting_skips_planning_and_llm(self, tmp_path):
        """'hi' is caught by Phase 1 — no planning, no LLM call."""
        loop, bus, provider = _make_loop(tmp_path)

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="hi",
                session_key="test:greeting",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" not in log_text
        assert "LLMComplexityEvaluator: Phase 1 → SIMPLE" in log_text
        # Phase 1 short-circuited → no LLM call for complexity
        # (The provider.chat_with_retry was still called for the agent loop
        # response, but NOT for complexity evaluation)

    @pytest.mark.asyncio
    async def test_thanks_skips_planning(self, tmp_path):
        """'thanks' is trivially simple — skip everything."""
        loop, bus, provider = _make_loop(tmp_path)

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="thanks",
                session_key="test:thanks",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" not in log_text
        assert "LLMComplexityEvaluator: Phase 1 → SIMPLE" in log_text

    @pytest.mark.asyncio
    async def test_ok_skips_planning(self, tmp_path):
        """'ok' is trivially simple."""
        loop, bus, provider = _make_loop(tmp_path)

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="ok",
                session_key="test:ok",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" not in log_text
        assert "LLMComplexityEvaluator: Phase 1 → SIMPLE" in log_text

    @pytest.mark.asyncio
    async def test_yes_skips_planning(self, tmp_path):
        """'yes' is trivially simple."""
        loop, bus, provider = _make_loop(tmp_path)

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="yes",
                session_key="test:yes",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" not in log_text


# ---------------------------------------------------------------------------
# Phase 2 integration — LLM classifies non-trivial messages
# ---------------------------------------------------------------------------

class TestPhase2Integration:
    """Non-trivial messages pass Phase 1 and are classified by the LLM."""

    @pytest.mark.asyncio
    async def test_llm_complex_triggers_planning(self, tmp_path):
        """LLM returns COMPLEX → planning runs."""
        loop, bus, provider = _make_loop(tmp_path)

        call_count = [0]

        async def chat_with_retry(*, messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Complexity evaluator: COMPLEX
                return LLMResponse(
                    content="COMPLEX",
                    tool_calls=[],
                    usage={"prompt_tokens": 100, "completion_tokens": 1},
                )
            elif call_count[0] == 2:
                # Planner response
                return LLMResponse(
                    content="## Execution Plan\n\n**Goal**: Build the feature\n\n"
                            "### Tasks\n1. **Read code** [sequential]\n"
                            "2. **Implement** [sequential]\n",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 30},
                )
            else:
                # Agent response
                return LLMResponse(
                    content="Done implementing.",
                    tool_calls=[],
                    usage={"prompt_tokens": 80, "completion_tokens": 10},
                )

        provider.chat_with_retry = chat_with_retry

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="Build a complete REST API with authentication and database",
                session_key="test:complex_llm",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" in log_text, (
            f"LLM said COMPLEX, planning should run. Logs: {log_text[:500]}"
        )
        assert "LLMComplexityEvaluator: LLM → COMPLEX" in log_text
        assert call_count[0] >= 3, f"Expected ≥3 calls (evaluator+planner+agent), got {call_count[0]}"

    @pytest.mark.asyncio
    async def test_llm_simple_skips_planning(self, tmp_path):
        """LLM returns SIMPLE → planning is skipped."""
        loop, bus, provider = _make_loop(tmp_path)

        call_count = [0]

        async def chat_with_retry(*, messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Complexity evaluator: SIMPLE
                return LLMResponse(
                    content="SIMPLE",
                    tool_calls=[],
                    usage={"prompt_tokens": 80, "completion_tokens": 1},
                )
            else:
                # Direct agent response (no planner)
                return LLMResponse(
                    content="The capital of France is Paris.",
                    tool_calls=[],
                    usage={"prompt_tokens": 60, "completion_tokens": 10},
                )

        provider.chat_with_retry = chat_with_retry

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="what is the capital of France?",
                session_key="test:simple_llm",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" not in log_text, (
            "LLM said SIMPLE, planning should NOT run"
        )
        assert "LLMComplexityEvaluator: LLM → SIMPLE" in log_text
        assert "planning skipped" in log_text.lower()
        assert call_count[0] == 2, (
            f"Expected 2 calls (evaluator + agent), got {call_count[0]}"
        )

    @pytest.mark.asyncio
    async def test_llm_error_defaults_complex(self, tmp_path):
        """LLM call fails → conservative fallback to COMPLEX."""
        loop, bus, provider = _make_loop(tmp_path)

        call_count = [0]

        async def chat_with_retry(*, messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Complexity evaluator: simulate error
                raise RuntimeError("LLM unavailable")
            elif call_count[0] == 2:
                return LLMResponse(
                    content="## Execution Plan\n\n**Goal**: Fix issue\n"
                            "### Tasks\n1. **Check logs**\n",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 20},
                )
            else:
                return LLMResponse(
                    content="Done.", tool_calls=[], usage={},
                )

        provider.chat_with_retry = chat_with_retry

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="fix the production bug",
                session_key="test:error_fallback",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        # Should fallback to COMPLEX → planning runs
        log_lower = log_text.lower()
        assert "defaulting to complex" in log_lower, (
            f"Error should fallback to COMPLEX. Logs: {log_text[:500]}"
        )
        assert "TaskPlanner: plan generated" in log_text, (
            f"Error should fallback to COMPLEX. Logs: {log_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_planning_disabled_when_mode_simple(self, tmp_path):
        """When execution_mode='simple', no evaluation happens at all."""
        loop, bus, provider = _make_loop(tmp_path, execution_mode="simple")

        buf, hid = _capture_loguru()
        try:
            result = await loop.process_direct(
                content="Build a complete REST API with database and auth",
                session_key="test:no_plan",
            )
        finally:
            logger.remove(hid)

        assert result is not None
        log_text = buf.getvalue()
        assert "TaskPlanner: plan generated" not in log_text
        assert "LLMComplexityEvaluator" not in log_text


# ---------------------------------------------------------------------------
# Phase 1 direct tests (redundant with test_complexity.py but fast sanity)
# ---------------------------------------------------------------------------

class TestPhase1Direct:
    """Verify _is_trivially_simple directly."""

    def test_trivial_greetings(self):
        assert _is_trivially_simple("hi") is True
        assert _is_trivially_simple("hello there") is True
        assert _is_trivially_simple("good morning") is True

    def test_trivial_thanks(self):
        assert _is_trivially_simple("thanks") is True
        assert _is_trivially_simple("ok") is True
        assert _is_trivially_simple("got it") is True

    def test_non_trivial_pass_through(self):
        assert _is_trivially_simple("What is the capital of France?") is False
        assert _is_trivially_simple("Build a REST API") is False
        assert _is_trivially_simple("how do I print in Python") is False

    def test_empty_and_whitespace(self):
        assert _is_trivially_simple("") is True
        assert _is_trivially_simple("   ") is True
