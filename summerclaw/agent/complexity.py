"""Task complexity evaluator for Plan-and-Solve auto-skip.

Two-phase hybrid architecture:
  Phase 1 — regex pre-filter (zero-cost, instant):
    Catches obviously trivial messages: very short, greetings, thanks, acks.
    These bypass the LLM entirely — no API call, no latency.

  Phase 2 — LLM reasoning (accurate, minimal tokens):
    For all remaining messages, a single-turn LLM call classifies the task as
    COMPLEX or SIMPLE.  The system prompt provides clear criteria; the model
    responds with exactly one word.  Typical cost: ~200 input tokens + 1 output.

Design principles:
- **Zero cost for trivial** — Phase 1 catches ~30% of messages at zero API cost.
- **LLM accuracy for the rest** — Phase 2 uses semantic reasoning, eliminating
  the brittleness of regex keyword matching.
- **Conservative fallback** — any LLM error defaults to COMPLEX so that a
  planning-worthy task is never incorrectly skipped.
- **Fast** — Phase 1 is O(n), Phase 2 is a single no-tool LLM turn.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from summerclaw.agent.context import ContextBuilder
from summerclaw.agent.runner import AgentRunner, AgentRunSpec
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# Phase 1 — regex pre-filter (zero-cost, instant)
# ---------------------------------------------------------------------------

_SHORT_MESSAGE_THRESHOLD = 15  # characters after stripping

_GREETING_PATTERNS: list[str] = [
    r"^(hi|hello|hey|hiya|howdy)\b",
    r"^(good\s+(morning|afternoon|evening|night|day))\b",
    r"^(yo|sup|hola|aloha)\b",
    r"^(what'?s\s+up|how\s+(are|is)\s+you|how\s+it\s+going)",
]

_THANKS_PATTERNS: list[str] = [
    r"^(thanks?|thx|thank\s+you|ty|tyvm|cheers|appreciate\s+(it|that))",
    r"^(awesome|great|nice|perfect|wonderful|cool)\b",
    r"^(ok|okay|k|kk|fine|sure|alright|got\s+it|gotcha|noted)\b",
]

_ACK_PATTERNS: list[str] = [
    r"^(yes|no|yep|nope|yeah|nah|maybe|perhaps)\b",
    r"^(go\s+ahead|proceed|continue|do\s+it)\b",
    r"^(i\s+see|understood|ok\s+got\s+it)\b",
]


def _is_trivially_simple(text: str) -> bool:
    """Return True when *text* is almost certainly a trivial conversational
    message that does NOT need planning.  This is intentionally narrow —
    only the most obvious patterns are short-circuited here."""
    if len(text) < _SHORT_MESSAGE_THRESHOLD:
        return True

    lower = text.lower()

    for pat in _GREETING_PATTERNS:
        if re.search(pat, lower) and len(text) < 40:
            return True

    for pat in _THANKS_PATTERNS:
        if re.search(pat, lower) and len(text) < 40:
            return True

    for pat in _ACK_PATTERNS:
        if re.search(pat, lower) and len(text) < 30:
            return True

    return False


# ---------------------------------------------------------------------------
# Phase 2 — LLM reasoning (accurate, minimal tokens)
# ---------------------------------------------------------------------------

class LLMComplexityEvaluator:
    """Decide whether a task requires planning using an LLM.

    Runs a single-turn, no-tool LLM call with a focused classification prompt.
    The model is asked to output exactly one word: ``COMPLEX`` or ``SIMPLE``.

    Usage::

        evaluator = LLMComplexityEvaluator(provider=provider, model="gpt-4o")
        is_complex = await evaluator.evaluate("Build a REST API with auth")
        # → True  (COMPLEX)
    """

    def __init__(
        self,
        provider: "LLMProvider",
        model: str,
        max_tool_result_chars: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tool_result_chars = max_tool_result_chars
        self._runner = AgentRunner(provider)

    async def evaluate(self, task: str, channel: str | None = None, chat_id: str | None = None) -> bool:
        """Return True when *task* should go through the planner.

        Phase 1 (regex) runs first.  If the message is trivially simple,
        returns False immediately with zero API cost.  Otherwise delegates
        to Phase 2 (LLM).

        Args:
            task: The user message text.
            channel: Optional origin channel (for runtime context in prompt).
            chat_id: Optional chat ID (for runtime context in prompt).

        Returns:
            True → complex, should plan.  False → simple, skip planning.
        """
        text = task.strip()
        if not text:
            return False

        # --- Phase 1: regex pre-filter (zero-cost) ---
        if _is_trivially_simple(text):
            logger.info(
                "LLMComplexityEvaluator: Phase 1 → SIMPLE (regex pre-filter, "
                "{} chars): {}", len(text), text[:80],
            )
            return False

        # --- Phase 2: LLM classification ---
        logger.info(
            "LLMComplexityEvaluator: Phase 1 passed, asking LLM "
            "(task {} chars): {}", len(text), text[:120],
        )

        time_ctx = ContextBuilder._build_runtime_context(channel, chat_id)
        system_prompt = render_template(
            "agent/complexity_classifier_system.md",
            time_ctx=time_ctx,
        )

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            result = await self._runner.run(
                AgentRunSpec(
                    initial_messages=messages,
                    tools=ToolRegistry(),        # no tools
                    model=self._model,
                    max_iterations=1,            # single turn
                    max_tool_result_chars=self._max_tool_result_chars,
                    error_message=None,
                )
            )
        except Exception as exc:
            logger.warning(
                "LLMComplexityEvaluator: LLM call failed ({}) — "
                "defaulting to COMPLEX (conservative)", exc,
            )
            return True

        if result.stop_reason == "error" or result.final_content is None:
            logger.warning(
                "LLMComplexityEvaluator: LLM returned stop_reason={} — "
                "defaulting to COMPLEX (conservative)", result.stop_reason,
            )
            return True

        raw = result.final_content.strip().upper()
        if not raw:
            logger.warning(
                "LLMComplexityEvaluator: LLM returned empty content — "
                "defaulting to COMPLEX (conservative)",
            )
            return True
        # Parse the first meaningful word
        first_word = raw.split(None, 1)[0] if raw else ""
        first_word = first_word.rstrip(".,;:!?()[]{}\"'")

        if first_word == "COMPLEX":
            logger.info(
                "LLMComplexityEvaluator: LLM → COMPLEX "
                "(task {} chars)", len(text),
            )
            return True

        logger.info(
            "LLMComplexityEvaluator: LLM → SIMPLE "
            "(raw={!r}, task {} chars)", raw[:20], len(text),
        )
        return False


# ---------------------------------------------------------------------------
# Legacy sync interface — kept for backwards compatibility and testing
# ---------------------------------------------------------------------------

def is_complex_task(task: str) -> bool:
    """**DEPRECATED** — use ``LLMComplexityEvaluator.evaluate()`` instead.

    Pure regex-based heuristic, retained for backwards compatibility
    with tests that cannot make async LLM calls.  Returns True for any
    message that passes the Phase 1 pre-filter (conservative: never
    incorrectly skip planning without LLM verification).
    """
    text = task.strip()
    if not text or _is_trivially_simple(text):
        return False
    # Without LLM, default to COMPLEX (conservative — see design principles)
    return True
