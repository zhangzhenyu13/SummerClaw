"""Subagent manager for background task execution."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from summerclaw.agent.hook import AgentHook, AgentHookContext
from summerclaw.agent.task_persistence import PersistedTask, TaskRegistry
from summerclaw.utils.prompt_templates import render_template
from summerclaw.agent.runner import AgentRunSpec, AgentRunner
from summerclaw.agent.skills import BUILTIN_SKILLS_DIR
from summerclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.agent.tools.search import GlobTool, GrepTool
from summerclaw.agent.tools.shell import ExecTool
from summerclaw.agent.tools.web import WebFetchTool, WebSearchTool
from summerclaw.bus.events import InboundMessage, OutboundMessage
from summerclaw.bus.queue import MessageBus
from summerclaw.config.schema import BrowserToolsConfig, ExecToolConfig, WebToolsConfig
from summerclaw.providers.base import LLMProvider

if TYPE_CHECKING:
    from summerclaw.proxy.pool import ProxyPool


_SUBAGENT_MAX_ITERATIONS = 15


@dataclass(slots=True)
class SubagentStatus:
    """Real-time status of a running subagent."""

    task_id: str
    label: str
    task_description: str
    started_at: float          # time.monotonic()
    phase: str = "initializing"  # initializing | awaiting_tools | tools_completed | final_response | done | error
    iteration: int = 0
    tool_events: list = field(default_factory=list)   # [{name, status, detail}, ...]
    usage: dict = field(default_factory=dict)          # token usage
    stop_reason: str | None = None
    error: str | None = None


class _SubagentHook(AgentHook):
    """Hook for subagent execution — logs tool calls and updates status."""

    def __init__(self, task_id: str, status: SubagentStatus | None = None) -> None:
        super().__init__()
        self._task_id = task_id
        self._status = status

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tool_call in context.tool_calls:
            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
            logger.debug(
                "Subagent [{}] executing: {} with arguments: {}",
                self._task_id, tool_call.name, args_str,
            )

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._status is None:
            return
        self._status.iteration = context.iteration
        self._status.tool_events = list(context.tool_events)
        self._status.usage = dict(context.usage)
        if context.error:
            self._status.error = str(context.error)


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        max_tool_result_chars: int,
        model: str | None = None,
        web_config: "WebToolsConfig | None" = None,
        browser_config: "BrowserToolsConfig | None" = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        disabled_skills: list[str] | None = None,
        max_subagent_depth: int = 0,
        task_registry: TaskRegistry | None = None,
        proxy_pool: ProxyPool | None = None,
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.web_config = web_config or WebToolsConfig()
        self.browser_config = browser_config
        self.max_tool_result_chars = max_tool_result_chars
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.disabled_skills = set(disabled_skills or [])
        self.max_subagent_depth = max_subagent_depth
        self.task_registry = task_registry
        self.runner = AgentRunner(provider)
        self._proxy_pool = proxy_pool
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_statuses: dict[str, SubagentStatus] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        current_depth: int = 0,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
        )
        self._task_statuses[task_id] = status

        # Persist task metadata to disk so it survives a process restart.
        if self.task_registry is not None:
            persisted = PersistedTask(
                task_id=task_id,
                task_type="subagent",
                label=display_label,
                description=task,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                session_key=session_key,
                depth=current_depth,
                started_at=TaskRegistry.now_iso(),
            )
            self.task_registry.save(persisted)

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, status, current_depth)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        _registry = self.task_registry

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            self._task_statuses.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]
            # Remove persisted entry — task completed (success or failure).
            if _registry is not None:
                _registry.remove(task_id)

        bg_task.add_done_callback(_cleanup)

        logger.info(
            "Spawned subagent [{}] depth={}/{} label='{}' task: {}",
            task_id, current_depth, self.max_subagent_depth, display_label,
            task[:200] + ("..." if len(task) > 200 else ""),
        )
        asyncio.create_task(self._push_channel(
            origin_channel, origin_chat_id,
            f"🔄 **Subagent [{task_id}] started** (depth {current_depth}/{self.max_subagent_depth})\n"
            f"> {display_label}",
        ))
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        status: SubagentStatus,
        current_depth: int = 0,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(
            "Subagent [{}] starting task (depth={}/{}): {}",
            task_id, current_depth, self.max_subagent_depth, label,
        )

        async def _on_checkpoint(payload: dict) -> None:
            status.phase = payload.get("phase", status.phase)
            status.iteration = payload.get("iteration", status.iteration)

        try:
            # Depth control (RDC logic): only register SpawnTool when depth < max_depth.
            # Subagents at max depth run in atomic mode (no further spawning).
            can_recurse = current_depth < self.max_subagent_depth
            tools = ToolRegistry()
            allowed_dir = self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
            extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
            tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
            tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(GlobTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(GrepTool(workspace=self.workspace, allowed_dir=allowed_dir))
            if self.exec_config.enable:
                tools.register(ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                ))
            if self.web_config.enable:
                tools.register(WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy, proxy_pool=self._proxy_pool))
                tools.register(WebFetchTool(proxy=self.web_config.proxy, proxy_pool=self._proxy_pool))
            if self.browser_config is not None and self.browser_config.enable:
                from summerclaw.agent.tools.browser import BrowserFetchTool, BrowserSearchTool
                browser_proxy = {"server": self.browser_config.proxy} if self.browser_config.proxy else None
                tools.register(BrowserSearchTool(timeout=self.browser_config.timeout, proxy=browser_proxy, proxy_pool=self._proxy_pool))
                tools.register(BrowserFetchTool(timeout=self.browser_config.timeout, proxy=browser_proxy, proxy_pool=self._proxy_pool))
            # Recursion gate: register SpawnTool only when further nesting is allowed
            if can_recurse:
                from summerclaw.agent.tools.spawn import SpawnTool
                child_spawn = SpawnTool(manager=self, current_depth=current_depth + 1)
                child_spawn.set_context(origin["channel"], origin["chat_id"])
                tools.register(child_spawn)
                logger.debug(
                    "Subagent [{}] depth={}/{}: SpawnTool registered for recursive decomposition",
                    task_id, current_depth, self.max_subagent_depth,
                )
            system_prompt = self._build_subagent_prompt(current_depth=current_depth)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            result = await self.runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=tools,
                model=self.model,
                max_iterations=_SUBAGENT_MAX_ITERATIONS,
                max_tool_result_chars=self.max_tool_result_chars,
                hook=_SubagentHook(task_id, status),
                max_iterations_message="Task completed but no final response was generated.",
                error_message=None,
                fail_on_tool_error=True,
                checkpoint_callback=_on_checkpoint,
            ))
            status.phase = "done"
            status.stop_reason = result.stop_reason

            if result.stop_reason == "tool_error":
                status.tool_events = list(result.tool_events)
                _error_detail = self._format_partial_progress(result)
                logger.error(
                    "Subagent [{}] depth={}/{} failed: tool error after {} calls\n  detail: {}",
                    task_id, current_depth, self.max_subagent_depth,
                    len(result.tools_used), _error_detail[:300],
                )
                await self._push_channel(
                    origin["channel"], origin["chat_id"],
                    f"❌ **Subagent [{task_id}] failed** "
                    f"(tool error, depth {current_depth}/{self.max_subagent_depth})\n"
                    f"> {label}\n"
                    f"{_error_detail[:500]}",
                )
                await self._announce_result(
                    task_id, label, task, _error_detail, origin, "error",
                )
            elif result.stop_reason == "error":
                _error_msg = result.error or "Error: subagent execution failed."
                logger.error(
                    "Subagent [{}] depth={}/{} failed: model/runtime error: {}",
                    task_id, current_depth, self.max_subagent_depth, _error_msg,
                )
                await self._push_channel(
                    origin["channel"], origin["chat_id"],
                    f"❌ **Subagent [{task_id}] failed** "
                    f"(model error, depth {current_depth}/{self.max_subagent_depth})\n"
                    f"> {label}\n"
                    f"Error: {_error_msg[:500]}",
                )
                await self._announce_result(
                    task_id, label, task, _error_msg, origin, "error",
                )
            elif result.stop_reason == "max_iterations":
                _timeout_msg = (
                    result.final_content
                    or f"Task reached the maximum number of iterations ({_SUBAGENT_MAX_ITERATIONS}) without completing."
                )
                logger.warning(
                    "Subagent [{}] depth={}/{} timed out: max iterations reached ({} tool calls)",
                    task_id, current_depth, self.max_subagent_depth, len(result.tools_used),
                )
                await self._push_channel(
                    origin["channel"], origin["chat_id"],
                    f"⏱️ **Subagent [{task_id}] timed out** "
                    f"(max iterations, depth {current_depth}/{self.max_subagent_depth}, "
                    f"{len(result.tools_used)} tool calls)\n"
                    f"> {label}",
                )
                await self._announce_result(
                    task_id, label, task, _timeout_msg, origin, "error",
                )
            elif result.stop_reason in ("empty_final_response",):
                _empty_msg = result.error or "Subagent produced no final response."
                logger.warning(
                    "Subagent [{}] depth={}/{} returned empty response ({} tool calls)",
                    task_id, current_depth, self.max_subagent_depth, len(result.tools_used),
                )
                await self._push_channel(
                    origin["channel"], origin["chat_id"],
                    f"⚠️ **Subagent [{task_id}] returned no response** "
                    f"(depth {current_depth}/{self.max_subagent_depth})\n"
                    f"> {label}\n"
                    f"{_empty_msg[:300]}",
                )
                await self._announce_result(
                    task_id, label, task, _empty_msg, origin, "error",
                )
            else:
                final_result = result.final_content or "Task completed but no final response was generated."
                _result_preview = final_result[:300] + ("..." if len(final_result) > 300 else "")
                logger.info(
                    "Subagent [{}] depth={}/{} completed ({} tool calls)\n  result: {}",
                    task_id, current_depth, self.max_subagent_depth,
                    len(result.tools_used), _result_preview,
                )
                await self._push_channel(
                    origin["channel"], origin["chat_id"],
                    f"✅ **Subagent [{task_id}] completed** (depth {current_depth}/{self.max_subagent_depth}, "
                    f"{len(result.tools_used)} tool calls)\n> {label}",
                )
                await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            status.phase = "error"
            status.error = str(e)
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._push_channel(
                origin["channel"], origin["chat_id"],
                f"❌ **Subagent [{task_id}] failed** (depth {current_depth}/{self.max_subagent_depth})\n"
                f"> {label}\n"
                f"Error: {str(e)[:500]}",
            )
            await self._announce_result(task_id, label, task, f"Error: {e}", origin, "error")

    async def _push_channel(
        self,
        channel: str,
        chat_id: str,
        content: str,
    ) -> None:
        """Push a progress message directly to the originating channel."""
        try:
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=content,
                metadata={"_progress": True},
            ))
        except Exception as _e:
            logger.warning("_push_channel failed (channel={} chat_id={}): {}", channel, chat_id, _e)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=task,
            result=result,
        )

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    @staticmethod
    def _format_partial_progress(result) -> str:
        completed = [e for e in result.tool_events if e["status"] == "ok"]
        failure = next((e for e in reversed(result.tool_events) if e["status"] == "error"), None)
        lines: list[str] = []
        if completed:
            lines.append("Completed steps:")
            for event in completed[-3:]:
                lines.append(f"- {event['name']}: {event['detail']}")
        if failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {failure['name']}: {failure['detail']}")
        if result.error and not failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {result.error}")
        return "\n".join(lines) or (result.error or "Error: subagent execution failed.")

    def _build_subagent_prompt(self, current_depth: int = 0) -> str:
        """Build a focused system prompt for the subagent."""
        from summerclaw.agent.context import ContextBuilder
        from summerclaw.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        skills_summary = SkillsLoader(
            self.workspace,
            disabled_skills=self.disabled_skills,
        ).build_skills_summary()
        return render_template(
            "agent/subagent_system.md",
            time_ctx=time_ctx,
            workspace=str(self.workspace),
            skills_summary=skills_summary or "",
            current_depth=current_depth,
            max_depth=self.max_subagent_depth,
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_all_statuses(self) -> list[SubagentStatus]:
        """Return status snapshots for all currently running subagents."""
        return list(self._task_statuses.values())

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    def get_running_count_by_session(self, session_key: str) -> int:
        """Return the number of currently running subagents for a session."""
        tids = self._session_tasks.get(session_key, set())
        return sum(
            1 for tid in tids
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        )
