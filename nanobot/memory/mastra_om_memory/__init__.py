"""MastraOM memory algorithm — Observational Memory for nanobot.

Based on Mastra's Observational Memory (OM) system that achieves SOTA on
LongMemEval (94.87% with gpt-5-mini). The core insight: two background
agents — an Observer and a Reflector — maintain a dense observation log
that replaces raw message history as it grows.

Architecture:
    - Observer: converts raw messages into structured observations
    - Reflector: condenses observations when they exceed token threshold
    - The main Agent sees observations + recent unobserved messages

Key advantages over naive memory:
    - Stable, predictable context window (prompt-cacheable)
    - Observations are dense and information-rich
    - No per-turn dynamic retrieval needed
    - Temporal anchoring preserves when things happened

Dream Mode:
    The MastraOMDream runs on cron (default: every 2h). Phase 1 analyzes
    history + observations; Phase 2 edits MEMORY.md/SOUL.md/USER.md via
    AgentRunner and optionally generates dreamed-* skills.

Hermes Mode:
    The MastraOMConsolidator.extract_and_store() provides the Hermes-Autogen
    integration point, using the Observer to extract facts from recent
    conversation before skill generation.

Usage::

    from nanobot.memory import MemoryRegistry
    from nanobot.memory.mastra_om_memory import MastraOMMemoryAlgorithm

    registry = MemoryRegistry()
    registry.register(MastraOMMemoryAlgorithm())
    algo = registry.get("mastra_om_memory")
    components = algo.build(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.memory.base import MemoryAlgorithm, MemoryComponents
from nanobot.memory.mastra_om_memory.store import MastraOMStore
from nanobot.memory.mastra_om_memory.consolidator import MastraOMConsolidator
from nanobot.memory.mastra_om_memory.dream import MastraOMDream
from nanobot.memory.mastra_om_memory.auto_compact import MastraOMAutoCompact

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


class MastraOMMemoryAlgorithm(MemoryAlgorithm):
    """Mastra Observational Memory algorithm.

    Implements the full OM pipeline:

    1. **MastraOMStore** — file-based storage (OBSERVATIONS.md, history.jsonl, etc.)
    2. **MastraOMConsolidator** — Observer/Reflector pipeline triggered by token budget
    3. **MastraOMDream** — offline cron-scheduled deep processing (Phase 1 + 2)
    4. **MastraOMAutoCompact** — idle session compression via Observer

    Key features:
    - Observer converts messages → observations when message tokens > 30k
    - Reflector condenses observations when observation tokens > 40k
    - Stable context window with prompt-cacheable observations prefix
    - Compatible with Dream and Hermes modes
    - No external dependencies beyond nanobot standard stack

    Configuration uses nanobot's standard config:
    - ``dream``: Dream schedule config (interval, model_override, max_batch_size)
    - ``skill_autogen.enable``: Hermes-Autogen mid-turn skill distillation
    - ``session_ttl_minutes``: AutoCompact idle threshold
    """

    name = "mastra_om_memory"

    def build(
        self,
        workspace: Path,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages: Any,
        get_tool_definitions: Any,
        max_completion_tokens: int,
        session_ttl_minutes: int,
        max_batch_size: int,
        max_iterations: int,
        max_tool_result_chars: int,
        annotate_line_ages: bool,
        embedding_config: Any = None,
    ) -> MemoryComponents:
        # 1. Store
        store = MastraOMStore(workspace, algo_name=self.name)

        # 2. Consolidator (Observer/Reflector pipeline)
        consolidator = MastraOMConsolidator(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
            # OM-specific thresholds
            message_tokens_threshold=30_000,
            observation_tokens_threshold=40_000,
        )

        # 3. Dream
        dream = MastraOMDream(
            store=store,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=self.name,
        )

        # 4. AutoCompact (optional)
        auto_compact = None
        if session_ttl_minutes > 0:
            auto_compact = MastraOMAutoCompact(
                sessions=sessions,
                consolidator=consolidator,
                session_ttl_minutes=session_ttl_minutes,
            )

        return MemoryComponents(
            store=store,
            consolidator=consolidator,
            dream=dream,
            auto_compact=auto_compact,
        )
