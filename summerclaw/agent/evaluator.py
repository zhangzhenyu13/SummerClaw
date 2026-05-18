"""Task evaluator for plan-and-solve closed-loop mode.

Evaluates execution results against the original goal and returns a
graded decision: PASS, LOCAL_REPLAN, or GLOBAL_REPLAN.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from loguru import logger

from summerclaw.agent.context import ContextBuilder
from summerclaw.agent.runner import AgentRunner, AgentRunSpec
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.providers.base import LLMProvider
from summerclaw.utils.prompt_templates import render_template


class EvalDecision(str, Enum):
    """Graded evaluation decision for closed-loop replanning."""

    PASS = "PASS"
    LOCAL_REPLAN = "LOCAL_REPLAN"
    GLOBAL_REPLAN = "GLOBAL_REPLAN"


class EvalResult:
    """Result of a single evaluation call."""

    def __init__(self, decision: EvalDecision, hint: str = "") -> None:
        self.decision = decision
        self.hint = hint

    def __repr__(self) -> str:
        return f"EvalResult(decision={self.decision}, hint={self.hint[:80]!r})"


class TaskEvaluator:
    """Quantitatively evaluates execution results to decide whether replanning is needed.

    The evaluator runs a single LLM call (no tools) and parses the response to
    determine one of three decisions:

    - ``PASS``         — goal was achieved; continue without replanning.
    - ``LOCAL_REPLAN`` — partial failure; a targeted adjustment to a specific
                         sub-task branch is sufficient.
    - ``GLOBAL_REPLAN``— overall approach is fundamentally wrong or the
                         environment changed; a full plan rebuild is required.

    On any parse failure the evaluator conservatively returns ``PASS`` to avoid
    triggering unnecessary replan iterations.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        max_tool_result_chars: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tool_result_chars = max_tool_result_chars
        self._runner = AgentRunner(provider)

    async def evaluate(
        self,
        task: str,
        execution_result: str,
        plan_text: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> EvalResult:
        """Evaluate the execution result against the original task.

        Args:
            task: The original user goal / message.
            execution_result: The agent's final response after executing the plan.
            plan_text: The plan that was used (optional, improves evaluation accuracy).
            channel: Origin channel (used for runtime context timestamp).
            chat_id: Origin chat ID.

        Returns:
            An :class:`EvalResult` with one of PASS / LOCAL_REPLAN / GLOBAL_REPLAN.
        """
        time_ctx = ContextBuilder._build_runtime_context(channel, chat_id)
        system_prompt = render_template(
            "agent/evaluator_system.md",
            time_ctx=time_ctx,
        )

        user_content_parts: list[str] = [f"**Original Goal**:\n{task}\n"]
        if plan_text:
            user_content_parts.append(f"**Execution Plan Used**:\n{plan_text}\n")
        user_content_parts.append(f"**Execution Result**:\n{execution_result}")
        user_content = "\n".join(user_content_parts)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.debug(
            "TaskEvaluator: evaluating result for task ({} chars)",
            len(task),
        )

        result = await self._runner.run(
            AgentRunSpec(
                initial_messages=messages,
                tools=ToolRegistry(),  # no tools during evaluation
                model=self._model,
                max_iterations=1,     # single-turn pure text output
                max_tool_result_chars=self._max_tool_result_chars,
                error_message=None,
            )
        )

        if result.stop_reason == "error" or result.final_content is None:
            logger.warning(
                "TaskEvaluator: evaluation failed (stop_reason={}), defaulting to PASS",
                result.stop_reason,
            )
            return EvalResult(EvalDecision.PASS, "Evaluation failed; proceeding with current result.")

        raw = result.final_content.strip()
        logger.debug("TaskEvaluator: raw response ({} chars): {}", len(raw), raw[:300])

        # Parse decision from the first line of the response.
        first_line = raw.split("\n", 1)[0].strip().upper()
        if "GLOBAL_REPLAN" in first_line:
            decision = EvalDecision.GLOBAL_REPLAN
        elif "LOCAL_REPLAN" in first_line:
            decision = EvalDecision.LOCAL_REPLAN
        elif "PASS" in first_line:
            decision = EvalDecision.PASS
        else:
            # Fall back to scanning the full response body
            upper = raw.upper()
            if "GLOBAL_REPLAN" in upper:
                decision = EvalDecision.GLOBAL_REPLAN
            elif "LOCAL_REPLAN" in upper:
                decision = EvalDecision.LOCAL_REPLAN
            else:
                # Conservative default: do not trigger unnecessary replanning
                decision = EvalDecision.PASS

        hint_preview = raw[:200] + ("..." if len(raw) > 200 else "")
        logger.info(
            "TaskEvaluator: decision={} hint={}",
            decision.value,
            hint_preview,
        )
        return EvalResult(decision, raw)
