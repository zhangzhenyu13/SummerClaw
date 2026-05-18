"""Task planner for plan-and-solve mode.

Generates a structured execution plan before the main agent loop runs,
enabling explicit plan-then-execute behaviour (Plan and Solve pattern).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from summerclaw.agent.context import ContextBuilder
from summerclaw.agent.runner import AgentRunner, AgentRunSpec
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.providers.base import LLMProvider
from summerclaw.utils.prompt_templates import render_template


class TaskPlanner:
    """Generates a structured task plan via a single LLM call.

    The planner runs *before* the main ReAct loop.  It receives the user
    message, produces a human-readable structured plan (task list with
    dependency annotations), and returns it as a string that is injected
    into the main agent's context.

    The planner itself uses no tools — it is a pure single-turn LLM call
    (``max_iterations=1``, empty ``ToolRegistry``).
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        max_tool_result_chars: int,
        max_subagent_depth: int = 0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tool_result_chars = max_tool_result_chars
        self._max_subagent_depth = max_subagent_depth
        self._runner = AgentRunner(provider)

    async def plan(
        self,
        task: str,
        channel: str | None = None,
        chat_id: str | None = None,
        context_summary: str | None = None,
        search_info: str | None = None,
    ) -> str | None:
        """Generate a structured plan for *task*.

        Args:
            task: The user message / goal to plan for.
            channel: Origin channel (used for runtime context timestamp).
            chat_id: Origin chat ID.
            context_summary: Optional condensed session summary to give the
                planner awareness of prior conversation context.
            search_info: Optional purified web search information to inject
                into the planner context (from SearchEnhancedPlanner).

        Returns:
            A markdown-formatted plan string, or ``None`` if planning fails.
        """
        time_ctx = ContextBuilder._build_runtime_context(channel, chat_id)
        system_prompt = render_template(
            "agent/planner_system.md",
            time_ctx=time_ctx,
            max_subagent_depth=self._max_subagent_depth,
            search_info=search_info,
        )

        user_content: str = task
        if context_summary:
            user_content = (
                f"[Prior conversation summary]\n{context_summary}\n\n"
                f"[Current goal]\n{task}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.debug("TaskPlanner: generating plan for task ({} chars)\n  task: {}", len(task), task[:200])

        result = await self._runner.run(
            AgentRunSpec(
                initial_messages=messages,
                tools=ToolRegistry(),   # no tools during planning
                model=self._model,
                max_iterations=1,       # single turn — pure text output
                max_tool_result_chars=self._max_tool_result_chars,
                error_message=None,
            )
        )

        if result.stop_reason == "error" or result.final_content is None:
            logger.warning("TaskPlanner: planning failed (stop_reason={})", result.stop_reason)
            return None

        plan_text = result.final_content.strip()
        _preview = plan_text[:1000] + ("\n... (truncated)" if len(plan_text) > 1000 else "")
        logger.info("TaskPlanner: plan generated ({} chars)\n{}", len(plan_text), _preview)
        return plan_text

    async def replan(
        self,
        task: str,
        previous_plan: str,
        feedback: str,
        mode: str = "local",
        channel: str | None = None,
        chat_id: str | None = None,
        context_summary: str | None = None,
        search_info: str | None = None,
    ) -> str | None:
        """Generate a revised plan based on execution feedback.

        Args:
            task: The original user goal.
            previous_plan: The plan that was previously executed.
            feedback: Evaluator's hint about what went wrong / what to adjust.
            mode: ``"local"`` for targeted branch adjustment (inherits depth),
                  ``"global"`` for full DAG rebuild (resets depth to 1).
            channel: Origin channel (used for runtime context timestamp).
            chat_id: Origin chat ID.
            context_summary: Optional condensed session summary.
            search_info: Optional purified web search information (fresh or
                cached) to inject into the replanning context.

        Returns:
            A revised markdown-formatted plan string, or ``None`` if replanning fails.
        """
        time_ctx = ContextBuilder._build_runtime_context(channel, chat_id)
        system_prompt = render_template(
            "agent/planner_system.md",
            time_ctx=time_ctx,
            max_subagent_depth=self._max_subagent_depth,
            replan_mode=mode,
            previous_plan=previous_plan,
            feedback=feedback,
            search_info=search_info,
        )

        user_content_parts: list[str] = []
        if context_summary:
            user_content_parts.append(f"[Prior conversation summary]\n{context_summary}\n")
        user_content_parts.append(f"[Original Goal]\n{task}")
        user_content = "\n".join(user_content_parts)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.debug(
            "TaskPlanner.replan: generating {} plan for task ({} chars)",
            mode, len(task),
        )

        result = await self._runner.run(
            AgentRunSpec(
                initial_messages=messages,
                tools=ToolRegistry(),   # no tools during replanning
                model=self._model,
                max_iterations=1,       # single turn — pure text output
                max_tool_result_chars=self._max_tool_result_chars,
                error_message=None,
            )
        )

        if result.stop_reason == "error" or result.final_content is None:
            logger.warning(
                "TaskPlanner.replan: {} replanning failed (stop_reason={})",
                mode, result.stop_reason,
            )
            return None

        plan_text = result.final_content.strip()
        _preview = plan_text[:1000] + ("\n... (truncated)" if len(plan_text) > 1000 else "")
        logger.info(
            "TaskPlanner.replan: {} plan generated ({} chars)\n{}",
            mode, len(plan_text), _preview,
        )
        return plan_text
