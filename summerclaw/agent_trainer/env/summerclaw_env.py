"""SummerClaw environment adapter — bridges ReACT agent + memory system.

Core principle: **runtime environment consistency + data flow isolation**.

The training agent must have the exact same runtime environment as the
online agent (tools, memory algorithm type, model config), but with a
completely isolated data space — using ``train-outputs/<alg>-<task>/``
as its independent workspace.

Isolation strategy:
- Independent workspace: train-outputs/<alg>-<task>/
- Independent memory store: memory algorithm uses the training workspace
- Independent session: training session_key isolated from online
- Training agent hermes/dream OFF (online agent unaffected)

Consistency:
- Tool set: identical ToolRegistry configuration
- Memory algorithm type: same as online (e.g. nemori → nemori)
- Model config: reuse online agent's provider/model
- ContextBuilder: same build logic, only memory store points to isolated dir
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import uuid

# Regex for extracting content inside <answer>...</answer> tags.
# Used by both exact_match and llm_judge scorers to normalize predicted
# output and candidate answers before comparison.
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from summerclaw.agent.context import ContextBuilder
from summerclaw.agent.hook import AgentHook, AgentHookContext
from summerclaw.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.agent_trainer.types import RolloutResult
from summerclaw.memory import (
    MemoryAlgorithm,
    MemoryComponents,
    MemoryRegistry,
)
from summerclaw.memory.naive_memory import NaiveMemoryAlgorithm
from summerclaw.memory.nemori_memory import NemoriMemoryAlgorithm
from summerclaw.memory.layerga_memory import LayergaMemoryAlgorithm
from summerclaw.memory.mem0v3_memory import Mem0V3MemoryAlgorithm
from summerclaw.memory.supermemory_memory import SupermemoryMemoryAlgorithm
from summerclaw.memory.hindsight_memory import HindsightMemoryAlgorithm
from summerclaw.memory.mastra_om_memory import MastraOMMemoryAlgorithm


class _EvalStreamingHook(AgentHook):
    """Hook that enables streaming for eval rollouts and logs tool calls.

    DashScope's ``enable_thinking=True`` (and other thinking models)
    require streaming mode to function correctly. Without streaming,
    the API may hang indefinitely waiting for thinking tokens to be
    delivered. This hook ensures eval uses the same streaming path
    as the main agent.

    Additionally, it logs every tool invocation and result through
    loguru so the trainer's dashboard log sink (which filters on
    ``agent_trainer``) can display them in the Gradio log window.
    """

    def __init__(self, session_key: str = "") -> None:
        super().__init__()
        self._session_key = session_key

    def wants_streaming(self) -> bool:
        return True

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        # Optionally print chunks in debug mode
        if os.environ.get("SUMMERCLAW_DEBUG_LLM"):
            import sys
            print(delta, end="", flush=True)
            sys.stdout.flush()

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if os.environ.get("SUMMERCLAW_DEBUG_LLM"):
            print()  # newline after streamed content

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Log each tool call before execution."""
        import json as _json
        for tc in context.tool_calls:
            args_str = _json.dumps(tc.arguments, ensure_ascii=False)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            logger.info(
                "[ROLLOUT-TOOL] iter={} calling {}(args={}) session={}",
                context.iteration, tc.name, args_str, self._session_key,
            )

    async def after_iteration(self, context: AgentHookContext) -> None:
        """Log tool results after execution."""
        for evt in context.tool_events:
            name = evt.get("name", "?")
            status = evt.get("status", "?")
            detail = evt.get("detail", "")
            if status == "ok":
                logger.info(
                    "[ROLLOUT-TOOL] iter={} {} completed: {} session={}",
                    context.iteration, name, detail[:120], self._session_key,
                )
            else:
                logger.warning(
                    "[ROLLOUT-TOOL] iter={} {} {}: {} session={}",
                    context.iteration, name, status, detail[:120], self._session_key,
                )


def _register_all_memory_algorithms(registry: MemoryRegistry) -> None:
    """Register all built-in memory algorithms into the registry."""
    for algo_cls in (
        NaiveMemoryAlgorithm,
        NemoriMemoryAlgorithm,
        LayergaMemoryAlgorithm,
        Mem0V3MemoryAlgorithm,
        SupermemoryMemoryAlgorithm,
        HindsightMemoryAlgorithm,
        MastraOMMemoryAlgorithm,
    ):
        try:
            registry.register(algo_cls())
        except Exception:
            pass  # already registered or unavailable


class _NullMemoryStore:
    """No-op memory store used when memory algorithm is disabled (null).

    All reads return empty strings / empty lists, so ``ContextBuilder``
    builds a system prompt containing only identity + bootstrap files +
    skills — no memory section and no recent history.

    The conversation context (tool loop) is independently bounded by
    ``max_tool_iterations`` which is set to 20 when memory is disabled.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        # Point to a non-existent file so ContextBuilder's
        # _is_template_content check treats memory as empty.
        self.memory_file = workspace / "memory" / ".null" / "MEMORY.md"

    def get_memory_context(self) -> str:
        return ""

    def read_memory(self) -> str:
        return ""

    def read_unprocessed_history(self, since_cursor: int = 0) -> list:
        return []

    def get_last_dream_cursor(self) -> int:
        return 0

    def read_user(self) -> str:
        return ""

    def read_soul(self) -> str:
        return ""


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return content
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", content, re.DOTALL)
    if match:
        return content[match.end():].strip()
    return content


class SummerClawEnvAdapter:
    """Bridge SummerClaw's ReACT agent to the trainer environment interface.

    Wraps :class:`AgentRunner` with isolated workspace, real memory system,
    and consistent tool/model configuration.
    """

    def __init__(
        self,
        provider: Any,
        model: str,
        workspace: Path,
        train_out_dir: str | Path,
        memory_algorithm_name: str = "naive_memory",
        context_window_tokens: int = 65536,
        max_tool_iterations: int = 200,
        max_tool_result_chars: int = 16000,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        workers: int = 0,
        tool_registry: ToolRegistry | None = None,
    ):
        """Initialize the environment adapter with isolated workspace.

        Parameters
        ----------
        provider : LLMProvider
            The LLM provider (same as online agent).
        model : str
            Model name (same as online agent).
        workspace : Path
            The online agent's workspace (read-only reference).
        train_out_dir : str | Path
            Isolated training output directory (train-outputs/<alg>-<task>/).
        memory_algorithm_name : str
            Memory algorithm name (same type as online agent).
        context_window_tokens : int
            Context window size (same as online agent).
        max_tool_iterations : int
            Max tool loop iterations (same as online agent).
        max_tool_result_chars : int
            Max chars per tool result (same as online agent).
        temperature : float
            LLM temperature (same as online agent).
        max_tokens : int
            Max completion tokens (same as online agent).
        workers : int
            Max concurrent rollout workers.
            ``0`` (default) = auto-derive as 80%% of the provider's
            ``max_concurrency`` (min 1, fallback 4).
        tool_registry : ToolRegistry | None
            Pre-built tool registry. If None, builds a default one.
        """
        self.provider = provider
        self.model = model
        self.online_workspace = Path(workspace)
        self.train_workspace = Path(train_out_dir)
        self._workspace_ready = False
        self._memory_algorithm_name = memory_algorithm_name

        self.context_window_tokens = context_window_tokens
        # When memory is disabled, bound the tool loop to the last 20 turns
        # (matching the "only recent 20 conversation turns" semantics).
        if not memory_algorithm_name or memory_algorithm_name.lower() in ("null", "none"):
            self.max_tool_iterations = min(max_tool_iterations, 20)
            logger.debug("Memory disabled: max_tool_iterations capped at {}", self.max_tool_iterations)
        else:
            self.max_tool_iterations = max_tool_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
        # workers: 0 = auto-derive 80% of provider.max_concurrency
        if workers <= 0:
            provider_max = getattr(provider, "max_concurrency", 0) or 0
            if provider_max > 0:
                self.workers = max(1, int(provider_max * 0.8))
            else:
                self.workers = 4  # safe fallback
        else:
            self.workers = workers

        # Per-item rollout timeout (seconds).  A single rollout may involve
        # multiple LLM calls (ReACT tool loop).  120s is generous for simple
        # QA; increase for tasks that require many tool iterations.
        self.rollout_timeout_s: int = int(
            os.environ.get("SUMMERCLAW_ROLLOUT_TIMEOUT_S", "120")
        )

        # Memory and context are built lazily in _ensure_workspace()
        # to avoid creating directories (memory/, etc.) at boot time.
        self._mem: MemoryComponents | None = None
        self._context: ContextBuilder | None = None

        # Eval read-only mode: temporarily swaps in NullMemoryStore during
        # test/val evaluation so no memory content is written.
        self._eval_readonly: bool = False
        self._saved_mem: MemoryComponents | None = None
        self._saved_context: ContextBuilder | None = None

        # 3. Build tool registry (consistent with online agent)
        self._tools = tool_registry or ToolRegistry()

        # 4. Build runner
        self._runner = AgentRunner(provider)

        # Save original tool registry for filtering
        self._original_tools = self._tools

        # 5. Custom scorer cache (loaded lazily when scorer='custom')
        self._custom_scorer_fn: Callable[[dict, str], float] | None = None

    def _build_memory_components(self, algorithm_name: str) -> MemoryComponents:
        """Build memory components using the isolated training workspace.

        Memory algorithm type matches the online agent, but data is stored
        in the training workspace directory.
        Hermes/dream are NOT activated for the training agent.

        When *algorithm_name* is empty, ``"null"`` or ``"none"`` (case-insensitive),
        memory is fully disabled — returns a ``MemoryComponents`` with all fields
        set to ``None``.
        """
        # Handle disabled memory
        if not algorithm_name or algorithm_name.lower() in ("null", "none"):
            logger.info("Memory algorithm disabled — skipping memory build")
            # Use _NullMemoryStore so ContextBuilder doesn't fall back to a real MemoryStore
            null_store = _NullMemoryStore(self.train_workspace)
            return MemoryComponents(store=null_store, consolidator=None, dream=None, auto_compact=None)

        registry = MemoryRegistry()
        _register_all_memory_algorithms(registry)

        try:
            algo: MemoryAlgorithm = registry.get(algorithm_name)
        except KeyError:
            logger.warning(
                "Memory algorithm '{}' not found, falling back to naive_memory",
                algorithm_name,
            )
            algo = NaiveMemoryAlgorithm()

        # Build with training workspace — memory data stays isolated
        # Note: we pass minimal build params; dream won't be activated
        try:
            components = algo.build(
                workspace=self.train_workspace,
                provider=self.provider,
                model=self.model,
                sessions=None,  # no session manager for training
                context_window_tokens=self.context_window_tokens,
                build_messages=lambda *a, **kw: [],
                get_tool_definitions=lambda: [],
                max_completion_tokens=self.max_tokens,
                session_ttl_minutes=0,
                max_batch_size=20,
                max_iterations=15,
                max_tool_result_chars=self.max_tool_result_chars,
                annotate_line_ages=False,
            )
        except Exception as exc:
            logger.warning(
                "Failed to build memory algorithm '{}': {}; "
                "falling back to naive_memory",
                algorithm_name, exc,
            )
            fallback = NaiveMemoryAlgorithm()
            components = fallback.build(
                workspace=self.train_workspace,
                provider=self.provider,
                model=self.model,
                sessions=None,
                context_window_tokens=self.context_window_tokens,
                build_messages=lambda *a, **kw: [],
                get_tool_definitions=lambda: [],
                max_completion_tokens=self.max_tokens,
                session_ttl_minutes=0,
                max_batch_size=20,
                max_iterations=15,
                max_tool_result_chars=self.max_tool_result_chars,
                annotate_line_ages=False,
            )

        return components

    def _ensure_workspace(self) -> None:
        """Lazily create the training workspace and build memory/context.

        Called before any rollout that writes to ``train_workspace``.
        Safe to call multiple times — only creates on first call or
        after ``train_workspace`` is reassigned by ``TrainerEngine._ensure_out_dir``.
        """
        if self._workspace_ready:
            return
        self.train_workspace.mkdir(parents=True, exist_ok=True)
        # Build memory components now that the directory exists
        self._mem = self._build_memory_components(self._memory_algorithm_name)
        # Build/rebuild context builder with correct workspace
        self._context = ContextBuilder(
            self.train_workspace,
            memory_store=self._mem.store,
        )
        self._workspace_ready = True
        logger.debug("Training workspace ensured: {}", self.train_workspace)

    def reconfigure_for_task(
        self,
        memory_algorithm_name: str | None,
        enabled_tools: list[str] | None,
    ) -> None:
        """Reconfigure memory algorithm and tool set for a specific task.

        Called by ``_apply_yaml_to_engine`` before training starts so that
        the task's YAML-configured ``memory_algorithm`` and ``enabled_tools``
        take effect instead of the agent defaults.

        Parameters
        ----------
        memory_algorithm_name : str | None
            Algorithm name from YAML.  ``None``, ``"null"``, ``"none"`` or
            empty string all disable memory.
        enabled_tools : list[str] | None
            List of tool names to keep.  ``None`` or empty list means keep
            all tools from the original registry.
        """
        # Normalize memory algorithm name
        normalized = (
            "null"
            if memory_algorithm_name in (None, "", "null", "none")
            else str(memory_algorithm_name)
        )
        if normalized != self._memory_algorithm_name:
            logger.info(
                "Reconfiguring memory: {} → {}",
                self._memory_algorithm_name, normalized,
            )
            self._memory_algorithm_name = normalized
            # Reset workspace so memory is rebuilt with new algorithm
            self._workspace_ready = False
            self._mem = None
            self._context = None
            # Re-apply max_tool_iterations cap
            if normalized == "null":
                self.max_tool_iterations = min(self.max_tool_iterations, 20)
            # (Don't increase it back if previously capped — the original
            #  value is unknown after cap; the caller can set it explicitly.)
        else:
            logger.info("Memory algorithm: {} (unchanged)", normalized)

        # Filter tools
        all_tool_names = sorted(self._original_tools.tool_names)
        if enabled_tools is not None and len(enabled_tools) > 0:
            filtered = ToolRegistry()
            found: list[str] = []
            skipped: list[str] = []
            for name in enabled_tools:
                tool = self._original_tools.get(name)
                if tool is not None:
                    filtered.register(tool)
                    found.append(name)
                else:
                    skipped.append(name)
            self._tools = filtered
            logger.info(
                "Enabled tools ({}/{}): [{}]",
                len(found), len(all_tool_names), ", ".join(found),
            )
            if skipped:
                logger.warning("Tools not found in registry, skipped: [{}]", ", ".join(skipped))
        else:
            self._tools = self._original_tools
            logger.info(
                "Using all tools ({}): [{}]",
                len(all_tool_names), ", ".join(all_tool_names),
            )

    def _build_system_prompt(self, skill_content: str) -> str:
        """Build system prompt with isolated memory + injected skill content."""
        # Get base system prompt from context builder (uses isolated memory)
        try:
            base_prompt = self._context.build_system_prompt()
        except Exception as exc:
            logger.warning("Failed to build base system prompt: {}; using minimal", exc)
            base_prompt = "You are a helpful AI assistant."

        # Inject skill content for optimization
        if skill_content:
            stripped = _strip_frontmatter(skill_content)
            return (
                f"{base_prompt}\n\n"
                f"---\n\n"
                f"# Active Skill\n\n{stripped}\n"
            )
        return base_prompt

    async def rollout_one(
        self,
        item: dict,
        skill_content: str,
        *,
        epoch: int = 0,
        step: int = 0,
    ) -> RolloutResult:
        """Execute one agent rollout for a single training item.

        1. Build system prompt (isolated memory + skill injection)
        2. Build user prompt (from item's question/task)
        3. Call AgentRunner.run(spec)
        4. Extract hard/soft score from result
        5. Capture trajectory for reflect
        """
        self._ensure_workspace()
        item_id = str(item.get("id", uuid.uuid4().hex[:8]))
        session_key = f"trainer:epoch{epoch:02d}:step{step:03d}:{item_id}"

        # Build messages
        system_prompt = self._build_system_prompt(skill_content)
        question = item.get("question", "")
        context = item.get("context", "")
        user_text = f"{context}\n\n{question}" if context else question

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # ── Debug: print message sizes ──────────────────────────────
        import os as _os
        if _os.environ.get("SUMMERCLAW_DEBUG_LLM"):
            tool_defs = self._tools.get_definitions() if hasattr(self._tools, 'get_definitions') else []
            print(
                f"\n[ROLLOUT-DEBUG] item={item_id} | "
                f"system_prompt={len(system_prompt)} chars | "
                f"user_text={len(user_text)} chars | "
                f"tools={len(tool_defs)} | "
                f"model={self.model} | "
                f"temperature={self.temperature} | "
                f"max_tokens={self.max_tokens}",
                flush=True,
            )
            # Print first 500 chars of system prompt for inspection
            print(f"[ROLLOUT-DEBUG] system_prompt head:\n{system_prompt[:500]}\n", flush=True)

        # Build spec
        # IMPORTANT: Use streaming hook to match main agent behavior.
        # Explicitly set reasoning_effort="minimal" to disable thinking mode.
        # Thinking models (Qwen enable_thinking) can hang on non-interactive
        # eval workloads. The main agent uses thinking for interactive chat,
        # but eval doesn't need it.
        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self._tools,
            model=self.model,
            max_iterations=self.max_tool_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            workspace=self.train_workspace,
            session_key=session_key,
            context_window_tokens=self.context_window_tokens,
            hook=_EvalStreamingHook(session_key=session_key),
            reasoning_effort="minimal",
        )

        # Execute
        try:
            if _os.environ.get("SUMMERCLAW_DEBUG_LLM"):
                print(f"[ROLLOUT-DEBUG] about to call AgentRunner.run() for item={item_id}", flush=True)
            result = await self._runner.run(spec)
        except Exception as exc:
            logger.error("Rollout failed for item {}: {}", item_id, exc)
            return RolloutResult(
                id=item_id,
                hard=0,
                soft=0.0,
                fail_reason=str(exc),
                question=question,
                task_description=item.get("task_description", ""),
            )

        # Score the result
        hard, soft = await self.score_result(result, item)

        # Extract trajectory from messages
        trajectory = result.messages or []

        return RolloutResult(
            id=item_id,
            hard=hard,
            soft=soft,
            n_turns=len([m for m in trajectory if m.get("role") == "assistant"]),
            fail_reason="" if hard else (result.error or "incorrect_answer"),
            task_type=item.get("task_type", ""),
            task_description=item.get("task_description", ""),
            predicted_answer=result.final_content or "",
            question=question,
            reference_text=item.get("expected", ""),
            trajectory=[
                {k: v for k, v in m.items() if k != "content" or isinstance(v, str)}
                for m in trajectory
            ],
        )

    async def rollout_batch(
        self,
        items: list[dict],
        skill_content: str,
        *,
        epoch: int = 0,
        step: int = 0,
        phase_label: str = "ROLLOUT",
    ) -> list[RolloutResult]:
        """Execute a batch of rollouts with controlled concurrency.

        Concurrency priority:
          1. ``SUMMERCLAW_DEBUG_LLM=1``  → serial (workers=1)
          2. ``self.workers``            → trainer-specific setting
             (defaults to 80%% of provider's ``maxConcurrency``)
          3. provider ``max_concurrency`` → global system setting
          4. fallback: 4

        Logs progress as each item completes so the dashboard can show
        that evaluation / rollout is still making progress.
        """
        import os as _os
        # Force serial execution when debug mode is on
        debug = _os.environ.get("SUMMERCLAW_DEBUG_LLM")
        if debug:
            effective_workers = 1
        elif self.workers > 0:
            # self.workers takes priority (trainer-specific, default 80% of maxConcurrency)
            effective_workers = self.workers
        else:
            provider_max = getattr(self.provider, "max_concurrency", 0) or 0
            effective_workers = provider_max if provider_max > 0 else 4
        semaphore = asyncio.Semaphore(effective_workers)
        if debug:
            logger.warning("[DEBUG] rollout_batch running with workers=1 (serial mode)")
        else:
            source = "trainer-workers" if self.workers > 0 else "provider-maxConcurrency"
            logger.info(
                "[{}] rollout_batch concurrency={} ({})",
                phase_label, effective_workers, source,
            )
        total = len(items)
        done_count = 0
        lock = asyncio.Lock()
        # In debug mode, disable rollout timeout so slow models aren't killed
        timeout_s = 0 if debug else self.rollout_timeout_s

        async def _run_one(item: dict) -> RolloutResult:
            nonlocal done_count
            item_id = str(item.get("id", "?"))
            async with semaphore:
                try:
                    if timeout_s > 0:
                        result = await asyncio.wait_for(
                            self.rollout_one(
                                item, skill_content, epoch=epoch, step=step,
                            ),
                            timeout=timeout_s,
                        )
                    else:
                        # Debug mode: no timeout, let it run to completion
                        result = await self.rollout_one(
                            item, skill_content, epoch=epoch, step=step,
                        )
                except asyncio.TimeoutError:
                    logger.error(
                        "[{}] item={} timed out after {}s",
                        phase_label, item_id, timeout_s,
                    )
                    result = RolloutResult(
                        id=item_id,
                        hard=0,
                        soft=0.0,
                        fail_reason=f"rollout_timeout_{timeout_s}s",
                        question=item.get("question", ""),
                    )
            async with lock:
                done_count += 1
                logger.info(
                    "[{}] progress {}/{} (item={} {})",
                    phase_label, done_count, total, item_id,
                    "TIMEOUT" if result.fail_reason.startswith("rollout_timeout") else "done",
                )
            return result

        return await asyncio.gather(*[_run_one(item) for item in items])

    def _set_eval_readonly(self) -> None:
        """Switch to a no-op memory store for test/val evaluation.

        Prevents any memory writes (and reads) during evaluation so the
        memory store is not polluted with eval artifacts.  Call
        :meth:`_clear_eval_readonly` to restore the original store.
        """
        if self._eval_readonly:
            return  # already in eval mode
        if self._mem is None or isinstance(self._mem.store, _NullMemoryStore):
            return  # no real store to protect
        self._saved_mem = self._mem
        self._saved_context = self._context
        null_store = _NullMemoryStore(self.train_workspace)
        self._mem = MemoryComponents(
            store=null_store, consolidator=None, dream=None, auto_compact=None,
        )
        self._context = ContextBuilder(
            self.train_workspace, memory_store=null_store,
        )
        self._eval_readonly = True
        logger.info("[EVAL] memory switched to read-only (NullMemoryStore)")

    def _clear_eval_readonly(self) -> None:
        """Restore the original memory store after evaluation."""
        if not self._eval_readonly:
            return
        if self._saved_mem is not None:
            self._mem = self._saved_mem
            self._saved_mem = None
        if self._saved_context is not None:
            self._context = self._saved_context
            self._saved_context = None
        self._eval_readonly = False
        logger.info("[EVAL] memory restored to original store")

    def _load_custom_scorer(self) -> Callable[[dict, str], float] | None:
        """Load custom-scorer.py from the training output directory.

        The script must define a top-level function:
            def score(sample: dict, predicted: str) -> float

        Returns the score function, or None if loading fails.
        """
        if self._custom_scorer_fn is not None:
            return self._custom_scorer_fn

        scorer_path = self.train_workspace / "custom-scorer.py"
        if not scorer_path.exists():
            logger.warning("custom-scorer.py not found at: {}", scorer_path)
            return None

        try:
            spec = importlib.util.spec_from_file_location("custom_scorer", str(scorer_path))
            if spec is None or spec.loader is None:
                logger.warning("Cannot load custom-scorer.py spec from: {}", scorer_path)
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "score", None)
            if fn is None or not callable(fn):
                logger.warning("custom-scorer.py must define a callable 'score(sample, predicted)'")
                return None
            self._custom_scorer_fn = fn
            logger.info("Loaded custom scorer from: {}", scorer_path)
            return fn
        except Exception as exc:
            logger.error("Failed to load custom-scorer.py: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Answer-tag parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer_content(text: str) -> str:
        """Extract the inner content from ``<answer>...</answer>`` tags.

        If one or more ``<answer>`` tags are found, their inner text is
        returned (joined by newline for multiple tags).  Otherwise the
        original ``text`` is returned unchanged — this keeps backward
        compatibility for data that does not use the tag convention.
        """
        matches = _ANSWER_TAG_RE.findall(text)
        if matches:
            return "\n".join(m.strip() for m in matches)
        return text.strip()

    # ------------------------------------------------------------------
    # Scoring implementations
    # ------------------------------------------------------------------

    def _score_exact_match(self, predicted: str, answers: list[str]) -> tuple[int, float]:
        """Score by checking predicted against any candidate answer.

        Both ``predicted`` and each candidate answer are first parsed for
        ``<answer>...</answer>`` tags; the inner content is used for
        comparison so that surrounding boilerplate does not interfere.

        Returns (hard: 0/1, soft: 0.0-1.0).
        """
        # Normalize predicted: prefer tag-extracted content, else raw text
        pred_text = self._extract_answer_content(predicted).strip().lower()
        if not pred_text:
            pred_text = predicted.strip().lower()

        if not answers:
            has_output = bool(pred_text)
            return (1 if has_output else 0, 0.5 if has_output else 0.0)

        best_soft = 0.0
        for ans in answers:
            # Normalize candidate answer: prefer tag-extracted content
            ans_text = self._extract_answer_content(str(ans)).strip().lower()
            if not ans_text:
                continue

            # Exact substring match (extracted content in extracted predicted)
            if ans_text in pred_text:
                return (1, 1.0)

            # Also try raw substring match as a fallback (tag-wrapped answer
            # inside raw predicted, or vice versa)
            if str(ans).strip().lower() in predicted.strip().lower():
                return (1, 1.0)

            # Word overlap on extracted content
            ans_words = set(ans_text.split())
            pred_words = set(pred_text.split())
            if ans_words:
                overlap = ans_words & pred_words
                soft = len(overlap) / len(ans_words)
                best_soft = max(best_soft, soft)

        hard = 1 if best_soft >= 0.8 else 0
        return (hard, best_soft)

    async def _score_llm_judge(
        self, predicted: str, answers: list[str], question: str = "",
    ) -> tuple[int, float]:
        """Use the LLM to judge whether ``predicted`` matches any candidate.

        Both predicted and candidate answers are first parsed for
        ``<answer>...</answer>`` tags so the LLM only sees the core content.

        Returns (hard: 0/1, soft: 0.0-1.0).
        """
        pred_text = self._extract_answer_content(predicted) or predicted.strip()
        if not pred_text:
            return (0, 0.0)

        if not answers:
            has_output = bool(pred_text)
            return (1 if has_output else 0, 0.5 if has_output else 0.0)

        # Build candidate list for the judge prompt
        candidates = []
        for ans in answers:
            c = self._extract_answer_content(str(ans)) or str(ans).strip()
            if c:
                candidates.append(c)
        if not candidates:
            return (0, 0.0)

        candidates_str = "\n".join(f"  - {c}" for c in candidates)
        question_ctx = f"\nQuestion: {question}" if question else ""

        judge_prompt = (
            "You are a strict answer judge.\n"
            f"{question_ctx}\n\n"
            f"Predicted answer:\n  {pred_text}\n\n"
            f"Accepted candidate answers (any one is correct):\n{candidates_str}\n\n"
            "Decide if the predicted answer is semantically equivalent to "
            "ANY of the candidate answers. Minor wording differences are "
            "acceptable as long as the core meaning / value is the same.\n\n"
            "Reply with ONLY a JSON object in the format:\n"
            '{"match": true/false, "confidence": 0.0-1.0, "reason": "..."}\n'
            "Do not include any other text."
        )

        messages = [{"role": "user", "content": judge_prompt}]
        try:
            resp = await self.provider.chat_with_retry(
                messages=messages,
                model=self.model,
                temperature=0.0,
                max_tokens=256,
            )
            content = (resp.content or "").strip()
            # Parse the JSON response
            import json as _json
            # Try to find JSON in the response (strip markdown fences if present)
            json_str = content
            if "```" in json_str:
                # Extract between ```json ... ``` or ``` ... ```
                m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
                if m:
                    json_str = m.group(1).strip()
            data = _json.loads(json_str)
            match = bool(data.get("match", False))
            confidence = float(data.get("confidence", 1.0 if match else 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = data.get("reason", "")
            logger.debug(
                "llm_judge result: match={} confidence={} reason={}",
                match, confidence, reason,
            )
            hard = 1 if match else 0
            soft = confidence if match else confidence * 0.3
            return (hard, soft)
        except Exception as exc:
            logger.warning(
                "llm_judge LLM call failed: {} — falling back to exact_match", exc,
            )
            return self._score_exact_match(predicted, answers)

    async def score_result(
        self,
        result: AgentRunResult,
        item: dict,
    ) -> tuple[int, float]:
        """Score an agent result against candidate answers.

        Returns (hard: 0/1, soft: 0.0-1.0).

        Scoring methods:
        - exact_match (default): parses <answer> tags then does
          case-insensitive substring / word-overlap matching
        - llm_judge: uses the LLM to semantically judge predicted vs
          candidate answers (falls back to exact_match on failure)
        - custom: loads custom-scorer.py from train_out_dir
        """
        predicted = (result.final_content or "").strip()
        scorer = str(item.get("scorer", "exact_match")).strip().lower()

        # Resolve answers: prefer 'answers' list, fall back to 'expected' for compat
        answers = item.get("answers")
        if answers is None:
            expected = item.get("expected")
            if expected is not None:
                answers = [str(expected)]
            else:
                answers = []

        # Custom scorer
        if scorer == "custom":
            fn = self._load_custom_scorer()
            if fn is not None:
                try:
                    score_val = fn(item, predicted)
                    score_val = float(score_val)
                    score_val = max(0.0, min(1.0, score_val))
                    hard = 1 if score_val >= 0.5 else 0
                    return (hard, score_val)
                except Exception as exc:
                    logger.warning("Custom scorer raised error: {} — falling back to exact_match", exc)
            # Fallback if custom scorer unavailable
            return self._score_exact_match(predicted, answers)

        # llm_judge: LLM-based semantic scoring
        if scorer == "llm_judge":
            question = item.get("question", "")
            return await self._score_llm_judge(predicted, answers, question=question)

        return self._score_exact_match(predicted, answers)
