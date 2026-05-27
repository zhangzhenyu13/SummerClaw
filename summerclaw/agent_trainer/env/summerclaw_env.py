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
        workers: int = 4,
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
        self.max_tool_iterations = max_tool_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
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

        # 3. Build tool registry (consistent with online agent)
        self._tools = tool_registry or ToolRegistry()

        # 4. Build runner
        self._runner = AgentRunner(provider)

        # 5. Custom scorer cache (loaded lazily when scorer='custom')
        self._custom_scorer_fn: Callable[[dict, str], float] | None = None

    def _build_memory_components(self, algorithm_name: str) -> MemoryComponents:
        """Build memory components using the isolated training workspace.

        Memory algorithm type matches the online agent, but data is stored
        in the training workspace directory.
        Hermes/dream are NOT activated for the training agent.
        """
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
        hard, soft = self.score_result(result, item)

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

        Logs progress as each item completes so the dashboard can show
        that evaluation / rollout is still making progress.
        """
        import os as _os
        # Force serial execution when debug mode is on
        debug = _os.environ.get("SUMMERCLAW_DEBUG_LLM")
        effective_workers = 1 if debug else self.workers
        semaphore = asyncio.Semaphore(effective_workers)
        if effective_workers == 1:
            logger.warning("[DEBUG] rollout_batch running with workers=1 (serial mode)")
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

    def _score_exact_match(self, predicted: str, answers: list[str]) -> tuple[int, float]:
        """Score by checking predicted against any candidate answer.

        Returns (hard: 0/1, soft: 0.0-1.0).
        """
        predicted_lower = predicted.strip().lower()
        if not answers:
            has_output = bool(predicted_lower)
            return (1 if has_output else 0, 0.5 if has_output else 0.0)

        best_soft = 0.0
        for ans in answers:
            ans_lower = str(ans).strip().lower()
            if not ans_lower:
                continue
            # Exact substring match
            if ans_lower in predicted_lower:
                return (1, 1.0)
            # Word overlap
            ans_words = set(ans_lower.split())
            predicted_words = set(predicted_lower.split())
            if ans_words:
                overlap = ans_words & predicted_words
                soft = len(overlap) / len(ans_words)
                best_soft = max(best_soft, soft)

        hard = 1 if best_soft >= 0.8 else 0
        return (hard, best_soft)

    def score_result(
        self,
        result: AgentRunResult,
        item: dict,
    ) -> tuple[int, float]:
        """Score an agent result against candidate answers.

        Returns (hard: 0/1, soft: 0.0-1.0).

        Scoring methods:
        - exact_match (default): case-insensitive substring match against any answer
        - llm_judge: TODO (falls back to exact_match)
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

        # llm_judge: TODO — fall back to exact_match
        if scorer == "llm_judge":
            logger.debug("llm_judge not yet implemented — falling back to exact_match")

        return self._score_exact_match(predicted, answers)
