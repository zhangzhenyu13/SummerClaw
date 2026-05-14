"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.complexity import LLMComplexityEvaluator, is_complex_task
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.task_persistence import PersistedTask, TaskRegistry
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.notebook import NotebookEditTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GlobTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.self import MyTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.ask_user import ASK_USER_PENDING, AskUserTool
from nanobot.agent.tools.browser_control import (
    BrowserExecuteJSTool,
    BrowserManager,
    BrowserNavigateTool,
    BrowserSnapshotTool,
)
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.document import extract_documents
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ProxyPoolConfig, SkillAutogenConfig, ToolsConfig, WebToolsConfig
    from nanobot.cron.service import CronService
    from nanobot.proxy.pool import ProxyPool


UNIFIED_SESSION_KEY = "unified:default"


class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._loop._current_iteration = context.iteration

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    async def after_iteration(self, context: AgentHookContext) -> None:
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


# ---------------------------------------------------------------------------
# Mode prefix parsing — detects /simple, /plan, /search-plan, /auto prefixes
# ---------------------------------------------------------------------------

_MODE_PREFIX_MAP: dict[str, str] = {
    "/simple": "simple",
    "/plan": "plan",
    "/search-plan": "search-plan",
    "/auto": "auto",
}
# Sorted by length descending so /search-plan is checked before /plan etc.
_MODE_PREFIX_KEYS = sorted(_MODE_PREFIX_MAP.keys(), key=len, reverse=True)


def _parse_mode_prefix(text: str) -> tuple[str, str]:
    """Detect mode prefix in message text and return (mode, remaining_text).

    Prefixes can be followed by:
    - A space (e.g. ``/plan Build a REST API``)
    - A non-ASCII-letter character like a Chinese character (e.g. ``/plan那个``)
    - End of string (e.g. ``/plan``)

    ASCII letters immediately after the prefix (e.g. ``/planetary``) are
    NOT treated as a mode prefix to avoid false positives.

    Returns:
        (mode, text) — mode is the resolved mode name (empty string if no
        prefix found), text is the task content with the prefix stripped.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "", text

    for prefix in _MODE_PREFIX_KEYS:
        if stripped == prefix:
            return _MODE_PREFIX_MAP[prefix], text

        if not stripped.startswith(prefix):
            continue

        after = stripped[len(prefix):]
        # Guard: if an ASCII letter immediately follows the prefix
        # (e.g. /planetary), it is NOT a mode prefix.
        if after and after[0].isalpha() and after[0].isascii():
            continue

        # Allowed: whitespace, Chinese/emoji/punctuation, or any
        # non-ASCII-letter character right after the prefix.
        rest = after.lstrip()
        if not rest:
            return _MODE_PREFIX_MAP[prefix], text
        return _MODE_PREFIX_MAP[prefix], rest

    return "", text


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        execution_mode: str = "auto",
        max_subagent_depth: int = 0,
        max_replan_iterations: int = 0,
        search_enhanced_planning_config: "SearchEnhancedPlanningConfig | None" = None,
        tools_config: "ToolsConfig | None" = None,
        skill_autogen_config: "SkillAutogenConfig | None" = None,
        memory_algorithm_name: str = "naive_memory",
        embedding_config: Any = None,
        max_injections_per_turn: int = _MAX_INJECTIONS_PER_TURN,
        max_injection_cycles: int = 5,
        proxy_pool_config: "ProxyPoolConfig | None" = None,
    ):
        from nanobot.config.schema import BrowserToolsConfig, ExecToolConfig, ProxyPoolConfig, ToolsConfig, WebToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self._last_inbound: "InboundMessage | None" = None
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.max_injections_per_turn = max_injections_per_turn
        self.max_injection_cycles = max_injection_cycles
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.browser_config: BrowserToolsConfig = _tc.browser
        self._browser_manager: BrowserManager | None = None
        self._proxy_pool_config = proxy_pool_config
        self._proxy_pool: ProxyPool | None = None
        # Create ProxyPool instance early so tools can reference it
        if self._proxy_pool_config is not None and self._proxy_pool_config.enabled:
            from nanobot.proxy.pool import ProxyPool
            self._proxy_pool = ProxyPool(self._proxy_pool_config)
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.sessions = session_manager or SessionManager(workspace)

        # Build memory components via the configured algorithm
        from nanobot.memory import _HAS_EMEM, _HAS_REME
        from nanobot.memory.registry import MemoryRegistry
        from nanobot.memory.naive_memory import NaiveMemoryAlgorithm
        from nanobot.memory.nemori_memory import NemoriMemoryAlgorithm
        from nanobot.memory.layerga_memory import LayergaMemoryAlgorithm
        from nanobot.memory.mem0v3_memory import Mem0V3MemoryAlgorithm
        from nanobot.memory.supermemory_memory import SupermemoryMemoryAlgorithm
        from nanobot.memory.hindsight_memory import HindsightMemoryAlgorithm
        from nanobot.memory.mastra_om_memory import MastraOMMemoryAlgorithm

        _REME_INSTALL_MSG = (
            "ReMe memory algorithm 'remem_memory' is configured but the 'reme' "
            "package is not installed.\n"
            "Install via:  git clone https://github.com/agentscope-ai/ReMe.git\n"
            "              cd ReMe && pip install -e \".[light]\"\n"
            "Falling back to 'naive_memory'."
        )

        _original_algorithm_name = memory_algorithm_name
        if memory_algorithm_name == "remem_memory" and not _HAS_REME:
            logger.warning(_REME_INSTALL_MSG)
            memory_algorithm_name = "naive_memory"

        self.memory_algorithm_name = memory_algorithm_name

        _mem_registry = MemoryRegistry()
        _mem_registry.register(NaiveMemoryAlgorithm())
        _mem_registry.register(NemoriMemoryAlgorithm())
        _mem_registry.register(LayergaMemoryAlgorithm())
        _mem_registry.register(Mem0V3MemoryAlgorithm())
        _mem_registry.register(SupermemoryMemoryAlgorithm())
        _mem_registry.register(HindsightMemoryAlgorithm())
        _mem_registry.register(MastraOMMemoryAlgorithm())
        if _HAS_REME:
            from nanobot.memory.remem_memory import ReMeMemoryAlgorithm

            _mem_registry.register(ReMeMemoryAlgorithm())
        if _HAS_EMEM:
            from nanobot.memory.emem_memory import EMemMemoryAlgorithm

            _mem_registry.register(EMemMemoryAlgorithm())
        _mem_algo = _mem_registry.get(memory_algorithm_name)
        _mem = _mem_algo.build(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=None,  # set below after context is built
            get_tool_definitions=None,  # set below after tools registered
            max_completion_tokens=provider.generation.max_tokens,
            session_ttl_minutes=session_ttl_minutes,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=self.max_tool_result_chars,
            annotate_line_ages=True,
            embedding_config=embedding_config,
        )
        self.consolidator = _mem.consolidator
        self.dream = _mem.dream
        self.auto_compact = _mem.auto_compact

        self.context = ContextBuilder(
            workspace, timezone=timezone, disabled_skills=disabled_skills,
            memory_store=_mem.store,
        )
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self._task_registry = TaskRegistry(workspace)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            browser_config=self.browser_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
            max_subagent_depth=max_subagent_depth,
            task_registry=self._task_registry,
            proxy_pool=self._proxy_pool,
        )
        # Execution mode: resolve effective mode (backward-compatible with legacy plan_and_solve)
        self._execution_mode = self._resolve_execution_mode(execution_mode)
        self._max_replan_iterations = max_replan_iterations

        # Planner / ComplexityEvaluator / SearchEnhancedPlanner creation
        # Always create the planner — it is needed when any mode may trigger
        # planning (auto, plan, search-plan).  Per-message prefixes (/plan,
        # /search-plan) can override the config default at runtime, so the
        # planner must exist even when the default config says "simple".
        _needs_complexity = self._execution_mode == "auto"

        # Always instantiate the TaskPlanner — per-message prefixes may
        # override the config default, so the planner must always be available.
        from nanobot.agent.planner import TaskPlanner
        self._planner: TaskPlanner | None = TaskPlanner(
            provider=provider,
            model=self.model,
            max_tool_result_chars=self.max_tool_result_chars,
            max_subagent_depth=max_subagent_depth,
        )

        if _needs_complexity:
            self._complexity_evaluator: LLMComplexityEvaluator | None = (
                LLMComplexityEvaluator(
                    provider=provider,
                    model=self.model,
                    max_tool_result_chars=self.max_tool_result_chars,
                )
            )
        else:
            self._complexity_evaluator = None

        # TaskEvaluator for closed-loop replanning: only when planner exists and replan enabled
        if self._planner is not None and max_replan_iterations > 0:
            from nanobot.agent.evaluator import TaskEvaluator
            self._evaluator: TaskEvaluator | None = TaskEvaluator(
                provider=provider,
                model=self.model,
                max_tool_result_chars=self.max_tool_result_chars,
            )
        else:
            self._evaluator = None

        # SearchEnhancedPlanner (plan + web search augmentation)
        # Always create it when the planner exists — per-message prefixes
        # (/search-plan) can override the config default at runtime.
        self._search_planner: "SearchEnhancedPlanner | None" = None
        if self._planner is not None:
            from nanobot.agent.search_planner import SearchEnhancedPlanner
            _sep_cfg = search_enhanced_planning_config
            self._search_planner = SearchEnhancedPlanner(
                planner=self._planner,
                provider=provider,
                model=self.model,
                max_tool_result_chars=self.max_tool_result_chars,
                tools=self.tools,
                max_results=_sep_cfg.max_results if _sep_cfg else 5,
                timeout=_sep_cfg.timeout if _sep_cfg else 15,
                max_purified_chars=_sep_cfg.max_purified_chars if _sep_cfg else 2000,
                search_on_replan=_sep_cfg.search_on_replan if _sep_cfg else False,
                web_search_backend=_sep_cfg.web_search_backend if _sep_cfg else "auto",
                reasoning_effort=provider.generation.reasoning_effort,
            )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # Tracks the message currently being processed per session: key -> (preview, started_at)
        self._active_processing: dict[str, tuple[str, float]] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        # Skill-Autogen: background tool-call-threshold-triggered skill generation
        # (distinct from Dream — Dream consolidates memory files, Skill-Autogen creates SKILL.md)
        self._skill_autogen_config = skill_autogen_config
        self.skill_autogen: "SkillAutogen | None" = None  # type: ignore[name-defined]
        if skill_autogen_config is not None and skill_autogen_config.enable:
            from nanobot.agent.skill_autogen import SkillAutogen
            self.skill_autogen = SkillAutogen(
                store=self.context.memory,
                provider=provider,
                model=skill_autogen_config.model_override or self.model,
                workspace=workspace,
                max_iterations=skill_autogen_config.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                memory_algorithm_name=self.memory_algorithm_name,
            )
        self._register_default_tools()
        # Patch consolidator callbacks now that context and tools are ready
        if self.consolidator is not None:
            self.consolidator._build_messages = self.context.build_messages
            self.consolidator._get_tool_definitions = self.tools.get_definitions
        if _tc.my.enable:
            self.tools.register(MyTool(loop=self, modify_allowed=_tc.my.allow_set))
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)
        # Log active execution mode and feature status
        _features = [f"mode={self._execution_mode}"]
        if self._planner is not None:
            _features.append("plan-and-solve")
            if self._evaluator is not None:
                _features.append(f"evaluation(max_replan={max_replan_iterations})")
            if self._search_planner is not None:
                _features.append("search-enhanced-planning")
        if self.skill_autogen is not None:
            _features.append(
                f"skill-autogen(interval={skill_autogen_config.nudge_interval})"
            )
        logger.info(
            "AgentLoop ready: model={} features=[{}]",
            self.model, ", ".join(_features) if _features else "none",
        )
        logger.info(
            "Memory algorithm: {} (available: {})",
            self.memory_algorithm_name,
            ", ".join(_mem_registry.list()),
        )

    @staticmethod
    def _make_browser_proxy(proxy_url: str | None) -> dict | None:
        """Convert a proxy URL string to a Playwright proxy dict."""
        if proxy_url:
            return {"server": proxy_url}
        return None

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy, proxy_pool=self._proxy_pool)
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy, proxy_pool=self._proxy_pool))
        if self.browser_config.enable:
            from nanobot.agent.tools.browser import BrowserFetchTool, BrowserSearchTool
            browser_proxy = self._make_browser_proxy(self.browser_config.proxy)
            self.tools.register(BrowserSearchTool(timeout=self.browser_config.timeout, proxy=browser_proxy, proxy_pool=self._proxy_pool))
            self.tools.register(BrowserFetchTool(timeout=self.browser_config.timeout, proxy=browser_proxy, proxy_pool=self._proxy_pool))
            # Stateful browser control tools (navigate / snapshot / execute JS)
            _enable_control = getattr(self.browser_config, "enable_control", False)
            if _enable_control:
                if self._browser_manager is None:
                    self._browser_manager = BrowserManager(proxy=browser_proxy, proxy_pool=self._proxy_pool)
                self.tools.register(BrowserNavigateTool(manager=self._browser_manager))
                self.tools.register(BrowserSnapshotTool(manager=self._browser_manager))
                self.tools.register(BrowserExecuteJSTool(manager=self._browser_manager))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(AskUserTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron", "my", "ask_user"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from nanobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hook: AgentHook = (
            CompositeHook([loop_hook] + self._extra_hooks) if self._extra_hooks else loop_hook
        )

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int | None = None) -> list[dict[str, Any]]:
            """Non-blocking drain of follow-up messages from the pending queue."""
            if pending_queue is None:
                return []
            if limit is None:
                limit = self.max_injections_per_turn
            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    pending_msg = pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = extract_documents(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                runtime_ctx = self.context._build_runtime_context(
                    pending_msg.channel,
                    pending_msg.chat_id,
                    self.context.timezone,
                )
                if isinstance(user_content, str):
                    merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
                else:
                    merged = [{"type": "text", "text": runtime_ctx}] + user_content
                items.append({"role": "user", "content": merged})
            return items

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            workspace=self.workspace,
            session_key=session.key if session else None,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            progress_callback=on_progress,
            checkpoint_callback=_checkpoint,
            injection_callback=_drain_pending,
            max_injections_per_turn=self.max_injections_per_turn,
            max_injection_cycles=self.max_injection_cycles,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        # Start the proxy pool if enabled
        if self._proxy_pool is not None:
            await self._proxy_pool.start()
        logger.info("Agent loop started")
        # Notify channels about tasks that were interrupted when the process
        # last exited.  A small delay lets channels finish their startup
        # handshake before we push messages to the outbound bus.
        self._schedule_background(self._notify_interrupted_tasks())

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                if self.auto_compact is not None:
                    self.auto_compact.check_expired(
                        self._schedule_background,
                        active_session_keys=self._pending_queues.keys(),
                    )
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            effective_key = self._effective_session_key(msg)
            # If this session already has an active pending queue (i.e. a task
            # is processing this session), route the message there for mid-turn
            # injection instead of creating a competing task.
            if effective_key in self._pending_queues:
                pending_msg = msg
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                try:
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    logger.warning(
                        "Pending queue full for session {}, falling back to queued task",
                        effective_key,
                    )
                else:
                    logger.info(
                        "Routed follow-up message to pending queue for session {}",
                        effective_key,
                    )
                    continue
            # Compute the effective session key before dispatching
            # This ensures /stop command can find tasks correctly when unified session is enabled
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        session_key = self._effective_session_key(msg)
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        # Register a pending queue so follow-up messages for this session are
        # routed here (mid-turn injection) instead of spawning a new task.
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending
        preview = msg.content.strip()[:80]
        self._active_processing[session_key] = (preview, time.monotonic())

        try:
            async with lock, gate:
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # Split one answer into distinct stream segments.
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1

                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    ))
        finally:
            # Drain any messages still in the pending queue and re-publish
            # them to the bus so they are processed as fresh inbound messages
            # rather than silently lost.
            self._active_processing.pop(session_key, None)
            queue = self._pending_queues.pop(session_key, None)
            if queue is not None:
                leftover = 0
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self.bus.publish_inbound(item)
                    leftover += 1
                if leftover:
                    logger.info(
                        "Re-published {} leftover message(s) to bus for session {}",
                        leftover, session_key,
                    )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        # Schedule proxy pool shutdown
        if self._proxy_pool is not None:
            self._schedule_background(self._proxy_pool.stop())
        logger.info("Agent loop stopping")

    @staticmethod
    def _resolve_execution_mode(execution_mode: str) -> str:
        """Resolve the effective execution mode, normalizing legacy values.

        Legacy 'plan_and_solve' boolean is no longer passed to __init__.
        All callers pass execution_mode as a string from config.
        """
        mode = (execution_mode or "auto").strip().lower()
        if mode in ("simple", "plan", "search-plan", "auto"):
            return mode
        logger.warning(
            "Unknown execution_mode '{}', falling back to 'auto'", execution_mode,
        )
        return "auto"

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            if self._restore_pending_user_turn(session):
                self.sessions.save(session)

            if self.auto_compact is not None:
                session, pending = self.auto_compact.prepare_session(session, key)
            else:
                pending = None

            await self.consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"

            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                session_summary=pending,
                current_role=current_role,
            )
            final_content, _, all_msgs, _sys_stop_reason, _ = await self._run_agent_loop(
                messages, session=session, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            if _sys_stop_reason == "max_iterations":
                logger.info(
                    "max_iterations reached for {}; consolidating session synchronously",
                    key,
                )
                await self.consolidator.maybe_consolidate_by_tokens(session)
            else:
                self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        # Extract document text from media at the processing boundary so all
        # channels benefit without format-specific logic in ContextBuilder.
        if msg.media:
            new_content, image_only = extract_documents(msg.content, msg.media)
            msg = dataclasses.replace(msg, content=new_content, media=image_only)

        self._last_inbound = msg
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

        if self.auto_compact is not None:
            session, pending = self.auto_compact.prepare_session(session, key)
        else:
            pending = None

        # Parse mode prefix (/simple, /plan, /search-plan, /auto) before command dispatch.
        # Mode prefixes are NOT commands — they modify how the task is executed.
        # If only a prefix is given without a task, show a usage hint.
        if isinstance(msg.content, str):
            prefix_mode, mode_stripped = _parse_mode_prefix(msg.content)
            if prefix_mode:
                if mode_stripped == msg.content:
                    # No task after the prefix
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=(
                            f"Usage: `/{prefix_mode} <task>`\n\n"
                            f"Available modes: `/simple`, `/plan`, `/search-plan`, `/auto`\n"
                            f"Default mode (config): `{self._execution_mode}`\n\n"
                            f"Example: `/{prefix_mode} Build a REST API with auth`"
                        ),
                    )
                msg = dataclasses.replace(msg, content=mode_stripped)
                self._per_message_mode = prefix_mode
                logger.info(
                    "ExecutionMode: per-message prefix '/{}', overriding config default '{}'",
                    prefix_mode, self._execution_mode,
                )
            else:
                self._per_message_mode = ""
        else:
            self._per_message_mode = ""

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        await self.consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)

        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            session_summary=pending,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Persist the triggering user message immediately, before running the
        # agent loop. If the process is killed mid-turn (OOM, SIGKILL, self-
        # restart, etc.), the existing runtime_checkpoint preserves the
        # in-flight assistant/tool state but NOT the user message itself, so
        # the user's prompt is silently lost on recovery. Saving it up front
        # makes recovery possible from the session log alone.
        user_persisted_early = False
        if isinstance(msg.content, str) and msg.content.strip():
            session.add_message("user", msg.content)
            self._mark_pending_user_turn(session)
            self.sessions.save(session)
            user_persisted_early = True

        # Execution mode decision: resolve the effective mode for this message.
        # Per-message prefix (/simple, /plan, /search-plan, /auto) overrides config default.
        effective_mode = self._per_message_mode or self._execution_mode

        # Determine whether to plan and whether to search based on the effective mode.
        #   simple      → skip planning entirely, go straight to _run_agent_loop
        #   plan        → always use TaskPlanner (no web search)
        #   search-plan → always use SearchEnhancedPlanner (plan + web search)
        #   auto        → use LLMComplexityEvaluator; if complex, optionally search-enhanced
        _should_plan = effective_mode != "simple" and self._planner is not None
        _should_search = (
            effective_mode == "search-plan"
            or (
                effective_mode == "auto"
                and self._search_planner is not None
            )
        )

        if _should_plan and effective_mode != "auto":
            # Forced plan/search-plan mode: bypass complexity evaluator, always plan.
            pass
        elif effective_mode == "auto" and self._planner is not None:
            # Auto mode: let the complexity evaluator decide.
            _should_plan = isinstance(msg.content, str) and await self._complexity_evaluator.evaluate(
                msg.content, channel=msg.channel, chat_id=msg.chat_id,
            )
            logger.info(
                "LLMComplexityEvaluator: result → {} ({} planning)",
                "COMPLEX" if _should_plan else "SIMPLE",
                "triggering" if _should_plan else "skipping",
            )
            if _should_plan and self._search_planner is not None:
                _should_search = True

        # _plan_progress unifies the two progress callbacks:
        #   - process_direct (single CLI msg): on_progress=_cli_progress → printed directly
        #   - _dispatch (bus-driven: interactive/serve): on_progress=None → _bus_progress → bus
        _plan_progress = on_progress or _bus_progress

        # Send mode hint to channel and log
        _mode_source = "prefix" if self._per_message_mode else "config"
        _mode_labels: dict[str, str] = {
            "simple": "⚡ Simple mode — executing directly",
            "plan": "📋 Plan mode — generating execution plan...",
            "search-plan": "🔍 Search-plan mode — searching web + generating plan...",
            "auto": "🤔 Auto mode — evaluating task complexity...",
        }
        _mode_hint = _mode_labels.get(effective_mode, f"Mode: {effective_mode}")
        await _plan_progress(_mode_hint)
        logger.info(
            "ExecutionMode: mode={}, should_plan={}, should_search={} (source={})",
            effective_mode, _should_plan, _should_search, _mode_source,
        )

        if _should_plan:
            plan_text: str | None = None
            _search_info_cache: str | None = None  # cached purified info for GLOBAL_REPLAN reuse
            try:
                if _should_search and self._search_planner is not None:
                    # Search-enhanced planning: decide → search agent → plan
                    # In search-plan mode, force search (skip the decide step internally).
                    _force_search = (effective_mode == "search-plan")
                    logger.info(
                        "SearchEnhancedPlanner: starting (force_search={}, task={} chars)",
                        _force_search, len(msg.content),
                    )
                    sep_result = await self._search_planner.plan(
                        task=msg.content,
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        context_summary=pending,
                        progress_callback=_plan_progress,
                        force_search=_force_search,
                    )
                    plan_text = sep_result.plan_text
                    _search_info_cache = sep_result.purified_info
                else:
                    logger.info("TaskPlanner: generating execution plan (no web search)")
                    await _plan_progress("⏳ Generating execution plan...")
                    plan_text = await self._planner.plan(
                        task=msg.content,
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        context_summary=pending,
                    )
                if plan_text:
                    logger.info(
                        "Plan-and-Solve: plan injected (session={}, {} chars)",
                        key, len(plan_text),
                    )
                    plan_preview = plan_text[:200].replace("\n", "\\n")
                    logger.info("Plan-and-Solve: plan preview: {}", plan_preview)
                    await _plan_progress("📋 **Execution Plan**\n\n" + plan_text)
            except Exception as _plan_err:
                logger.warning("Plan-and-Solve: planning failed, continuing without plan: {}", _plan_err)

            replan_count = 0
            while True:
                # Build run_messages: base messages + optional plan injection
                if plan_text:
                    plan_injection = {
                        "role": "system",
                        "content": (
                            "[Plan-and-Solve] The following execution plan was generated "
                            "for the user's request. Follow this plan — use the `spawn` tool "
                            "to delegate independent subtasks in parallel, and execute "
                            "sequential tasks in order.\n\n" + plan_text
                        ),
                    }
                    run_messages = initial_messages + [plan_injection]
                else:
                    run_messages = initial_messages

                # Only the last iteration receives the real pending_queue so that
                # mid-turn injections are not lost and do not race with replanning.
                is_last_iteration = (
                    self._evaluator is None
                    or replan_count >= self._max_replan_iterations
                )
                final_content, _, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
                    run_messages,
                    on_progress=on_progress or _bus_progress,
                    on_stream=on_stream if is_last_iteration else None,
                    on_stream_end=on_stream_end if is_last_iteration else None,
                    session=session,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    message_id=msg.metadata.get("message_id"),
                    pending_queue=pending_queue if is_last_iteration else None,
                )

                # Circuit-breaker: no evaluator or limit reached → exit loop
                if is_last_iteration:
                    break

                # Evaluate execution result
                try:
                    await _plan_progress("🔍 Evaluating execution result...")
                    eval_result = await self._evaluator.evaluate(
                        task=msg.content,
                        execution_result=final_content or "",
                        plan_text=plan_text,
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                    )
                except Exception as _eval_err:
                    logger.warning(
                        "Plan-and-Solve: evaluation failed, exiting loop: {}", _eval_err
                    )
                    break

                from nanobot.agent.evaluator import EvalDecision
                if eval_result.decision == EvalDecision.PASS:
                    logger.info(
                        "Plan-and-Solve: evaluator decision=PASS, loop complete (session={})",
                        key,
                    )
                    break

                # Replanning needed
                replan_count += 1
                replan_mode = (
                    "global" if eval_result.decision == EvalDecision.GLOBAL_REPLAN else "local"
                )
                logger.info(
                    "Plan-and-Solve: evaluator decision={}, triggering {} replan "
                    "(iteration {}/{}, session={})",
                    eval_result.decision.value, replan_mode,
                    replan_count, self._max_replan_iterations, key,
                )
                # Notify channel about evaluation decision (only on non-PASS)
                _hint_preview = eval_result.hint[:120] + ("..." if len(eval_result.hint) > 120 else "")
                await _plan_progress(
                    f"📊 **Evaluation: {eval_result.decision.value}** — {_hint_preview}"
                )
                await _plan_progress(
                    f"🔄 **Replanning ({replan_mode}, iteration "
                    f"{replan_count}/{self._max_replan_iterations})**..."
                )
                try:
                    if self._search_planner is not None:
                        # Search-enhanced replanning: may re-search on GLOBAL_REPLAN
                        sep_replan = await self._search_planner.replan(
                            task=msg.content,
                            previous_plan=plan_text or "",
                            feedback=eval_result.hint,
                            mode=replan_mode,
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            context_summary=pending,
                            existing_search_info=_search_info_cache,
                            progress_callback=_plan_progress,
                        )
                        new_plan = sep_replan.plan_text
                        # Update cached search info if fresh info was retrieved
                        if sep_replan.purified_info is not None:
                            _search_info_cache = sep_replan.purified_info
                    else:
                        new_plan = await self._planner.replan(
                            task=msg.content,
                            previous_plan=plan_text or "",
                            feedback=eval_result.hint,
                            mode=replan_mode,
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            context_summary=pending,
                        )
                except Exception as _replan_err:
                    logger.warning(
                        "Plan-and-Solve: replan failed, exiting loop: {}", _replan_err
                    )
                    break

                if new_plan is None:
                    logger.warning(
                        "Plan-and-Solve: replan returned None, exiting loop (session={})", key
                    )
                    break

                plan_text = new_plan
                await _plan_progress("📋 **Revised Execution Plan**\n\n" + plan_text)
                # Loop continues with the new plan
        else:
            # No plan-and-solve, or task is simple (auto-skipped by LLMComplexityEvaluator).
            if self._planner is not None:
                logger.info(
                    "Plan-and-Solve: planning skipped — task classified as simple "
                    "(session={}, {} chars): {}",
                    key, len(msg.content.strip()), msg.content[:120],
                )
                await (on_progress or _bus_progress)(
                    "⏩ Task classified as simple — executing directly"
                )
            # Standard single-pass execution
            final_content, _, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
                initial_messages,
                on_progress=on_progress or _bus_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                session=session,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_id=msg.metadata.get("message_id"),
                pending_queue=pending_queue,
            )

        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        # Skip the already-persisted user message when saving the turn
        save_skip = 1 + len(history) + (1 if user_persisted_early else 0)
        self._save_turn(session, all_msgs, save_skip)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        if stop_reason == "max_iterations":
            # The session now contains hundreds of tool-call messages that triggered
            # the limit. Consolidate synchronously so the very next turn starts with
            # a compact history instead of re-loading the same bloated context and
            # immediately hitting the limit again.
            logger.info(
                "max_iterations reached for {}; consolidating session synchronously",
                key,
            )
            await self.consolidator.maybe_consolidate_by_tokens(session)
        else:
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

        # Skill-Autogen: check whether cumulative tool-call threshold is reached and
        # schedule a background review pass to extract reusable skills from this turn.
        # This runs independently of Dream and only when explicitly enabled.
        if self.skill_autogen is not None and self._skill_autogen_config is not None:
            n_tools = sum(
                1 for m in all_msgs
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            self.skill_autogen.increment_tool_count(n_tools)
            if self.skill_autogen.should_trigger(self._skill_autogen_config.nudge_interval):
                self.skill_autogen.reset_count()
                logger.info(
                    "SkillAutogen: threshold reached ({} cumulative tool calls), scheduling review",
                    self._skill_autogen_config.nudge_interval,
                )
                _ch, _cid = msg.channel, msg.chat_id
                _msgs_snapshot = list(all_msgs)

                async def _autogen_and_notify(
                    _msgs=_msgs_snapshot, _channel=_ch, _chat_id=_cid
                ) -> None:
                    created = await self.skill_autogen.run(_msgs)  # type: ignore[union-attr]
                    if created:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=_channel,
                            chat_id=_chat_id,
                            content="[Skill-Autogen] 后台提炼完成，新技能已保存到 workspace/skills/",
                        ))

                self._schedule_background(_autogen_and_notify())

        # When follow-up messages were injected mid-turn, a later natural
        # language reply may address those follow-ups and should not be
        # suppressed just because MessageTool was used earlier in the turn.
        # However, if the turn falls back to the empty-final-response
        # placeholder, suppress it when the real user-visible output already
        # came from MessageTool.
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason != "error":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the entire runtime-context block (including any session summary).
                    # The block is bounded by _RUNTIME_CONTEXT_TAG and _RUNTIME_CONTEXT_END.
                    end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                    end_pos = content.find(end_marker)
                    if end_pos >= 0:
                        after = content[end_pos + len(end_marker):].lstrip("\n")
                        if after:
                            entry["content"] = after
                        else:
                            continue
                    else:
                        # Fallback: no end marker found, strip the tag prefix
                        after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG):].lstrip("\n")
                        if after_tag.strip():
                            entry["content"] = after_tag
                        else:
                            continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """Close a turn that only persisted the user message before crashing.

        If the session is awaiting a user resume/discard decision
        (``_INTERRUPTED_TASK_ID_KEY`` is set), do NOT add the error
        message — leave it for the user to decide via /tasks commands.
        """
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        # If we already notified the user and are waiting for their decision,
        # preserve the pending state so the /tasks commands can act on it.
        if session.metadata.get(self._INTERRUPTED_TASK_ID_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    # ------------------------------------------------------------------
    # Interrupted-task metadata keys
    # ------------------------------------------------------------------
    _INTERRUPTED_TASK_ID_KEY = "interrupted_task_id"

    async def _notify_interrupted_tasks(self) -> None:
        """Scan for tasks interrupted during the previous run and push notifications.

        Called once at startup from ``run()``.  A short delay gives channels
        time to finish their connection handshake before we publish.
        """
        await asyncio.sleep(3)

        # --- 1. subagent tasks from the persistent registry ----------------
        interrupted_subagents = self._task_registry.list_all()
        if interrupted_subagents:
            # Group by (origin_channel, origin_chat_id) to produce one
            # notification per conversation.
            from collections import defaultdict
            groups: dict[tuple[str, str], list[PersistedTask]] = defaultdict(list)
            for pt in interrupted_subagents:
                groups[(pt.origin_channel, pt.origin_chat_id)].append(pt)

            for (channel, chat_id), tasks in groups.items():
                lines = ["⚠️ **Service restarted** — the following background tasks were interrupted:\n"]
                for pt in tasks:
                    lines.append(f"• `[{pt.task_id}]` **{pt.label}** ({pt.elapsed_description()})")
                lines.append("")
                for pt in tasks:
                    lines.append(
                        f"• `/tasks resume {pt.task_id}` — re-run"
                        f"  |  `/tasks discard {pt.task_id}` — discard"
                    )
                content = "\n".join(lines)
                try:
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=channel,
                        chat_id=chat_id,
                        content=content,
                        metadata={"_progress": True},
                    ))
                    logger.info(
                        "Notified {}:{} about {} interrupted subagent task(s)",
                        channel, chat_id, len(tasks),
                    )
                except Exception as e:
                    logger.warning("_notify_interrupted_tasks: publish failed: {}", e)

        # --- 2. main-session tasks with pending_user_turn ------------------
        try:
            all_sessions = self.sessions.list_all_sessions()
        except Exception as e:
            logger.warning("_notify_interrupted_tasks: list_all_sessions failed: {}", e)
            return

        for session in all_sessions:
            if not session.metadata.get(self._PENDING_USER_TURN_KEY):
                continue
            # Already notified (has an interrupted_task_id set).
            if session.metadata.get(self._INTERRUPTED_TASK_ID_KEY):
                continue
            # Try to extract channel + chat_id from the session key.
            if ":" not in session.key:
                continue
            channel, chat_id = session.key.split(":", 1)
            # Skip synthetic / non-channel sessions (unified, sdk, cli).
            if channel in ("cli", "sdk", "unified", "system"):
                continue

            # Find the last user message to show as a preview.
            last_user_content = ""
            for msg in reversed(session.messages):
                if msg.get("role") == "user":
                    raw_content = msg.get("content", "")
                    if isinstance(raw_content, str):
                        last_user_content = raw_content.strip()[:120]
                    break

            # Generate a short task ID and store it so /tasks can locate it.
            short_id = TaskRegistry.make_task_id()
            session.metadata[self._INTERRUPTED_TASK_ID_KEY] = short_id
            self.sessions.save(session)

            preview = f'"{last_user_content}"' if last_user_content else "(unknown)"
            content = (
                f"⚠️ **Service restarted** — your last request was interrupted:\n"
                f"{preview}\n\n"
                f"• `/tasks resume {short_id}` — re-run the request\n"
                f"• `/tasks discard {short_id}` — discard\n"
            )
            try:
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=content,
                    metadata={"_progress": True},
                ))
                logger.info(
                    "Notified {}:{} about interrupted main-session task (id={})",
                    channel, chat_id, short_id,
                )
            except Exception as e:
                logger.warning("_notify_interrupted_tasks: publish failed for session {}: {}", session.key, e)

    async def _resume_session_task(self, short_id: str) -> str:
        """Resume a main-session task by re-injecting the pending user message.

        Args:
            short_id: The ``interrupted_task_id`` stored in session metadata.

        Returns:
            A human-readable result string.
        """
        # Find the session that owns this short_id.
        try:
            all_sessions = self.sessions.list_all_sessions()
        except Exception as e:
            return f"Error scanning sessions: {e}"

        target: Session | None = None
        for session in all_sessions:
            if session.metadata.get(self._INTERRUPTED_TASK_ID_KEY) == short_id:
                target = session
                break

        if target is None:
            return "Task not found or already completed."

        # Find the last user message.
        last_user_msg: dict | None = None
        for msg in reversed(target.messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break

        if last_user_msg is None:
            return "No pending user message found for this task."

        # Clear interrupted markers so _restore_pending_user_turn works normally.
        target.metadata.pop(self._INTERRUPTED_TASK_ID_KEY, None)
        # Keep pending_user_turn set — it will be cleared by the normal flow
        # after the agent processes the message.
        self.sessions.save(target)

        # Extract channel/chat_id from session key.
        if ":" not in target.key:
            return "Cannot determine channel from session key."
        channel, chat_id = target.key.split(":", 1)

        content = last_user_msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        inbound = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            session_key_override=target.key,
        )
        await self.bus.publish_inbound(inbound)
        logger.info("Resumed main-session task (id={}, session={})", short_id, target.key)
        return f"Task resumed. Re-running your previous request."

    async def _discard_session_task(self, short_id: str) -> str:
        """Discard a main-session task, marking it as errored.

        Args:
            short_id: The ``interrupted_task_id`` stored in session metadata.

        Returns:
            A human-readable result string.
        """
        try:
            all_sessions = self.sessions.list_all_sessions()
        except Exception as e:
            return f"Error scanning sessions: {e}"

        target: Session | None = None
        for session in all_sessions:
            if session.metadata.get(self._INTERRUPTED_TASK_ID_KEY) == short_id:
                target = session
                break

        if target is None:
            return "Task not found or already completed."

        # Clear the interrupted marker so the normal recovery path takes over.
        target.metadata.pop(self._INTERRUPTED_TASK_ID_KEY, None)
        self.sessions.save(target)

        # Now let _restore_pending_user_turn do its work (adds error message).
        if self._restore_pending_user_turn(target):
            self.sessions.save(target)

        logger.info("Discarded main-session task (id={}, session={})", short_id, target.key)
        return "Task discarded."

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id,
            content=content, media=media or [],
        )
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )
