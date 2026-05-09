"""Hindsight memory algorithm — file-based memory with built-in TEMPR retrieval.

The Hindsight memory algorithm wraps the naive file-based storage
(MEMORY.md, history.jsonl, SOUL.md, USER.md) and adds a **local TEMPR**
multi-strategy memory engine — zero external dependencies, no server needed.

TEMPR stands for Temporal + Embedding + Metadata + Probabilistic + Relational,
a five-engine fusion retrieval system implemented entirely locally:
- Keyword search via BM25
- Semantic search via provider.embed() + cosine similarity
- Temporal decay scoring
- Relational context boosting

Key features:
- **Built-in TEMPR**: Local multi-strategy memory retrieval (no server needed)
- **Hermes mode**: Mid-turn skill distillation with fact extraction + TEMPR retain
- **Dream mode**: Offline cron-scheduled deep processing with TEMPR reflect
- **Embedding optional**: Semantic search gracefully degrades if no embedding provider
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.memory.base import MemoryAlgorithm, MemoryComponents
from nanobot.memory.hindsight_memory.auto_compact import HindsightAutoCompact
from nanobot.memory.hindsight_memory.consolidator import HindsightConsolidator
from nanobot.memory.hindsight_memory.dream import HindsightDream
from nanobot.memory.hindsight_memory.store import HindsightStore

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


class HindsightMemoryAlgorithm(MemoryAlgorithm):
    """Hindsight memory algorithm — file-based + built-in local TEMPR engine.

    Builds a complete memory pipeline:
    1. HindsightStore — naive file store + local TEMPR memory bank
    2. HindsightConsolidator — token-budget consolidation + TEMPR retention
    3. HindsightDream — offline cron processing + TEMPR-backed analysis
    4. HindsightAutoCompact — idle session compression
    """

    name = "hindsight_memory"

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
        # Determine embedding model — prefer embedding_config.model, fall back
        # to the chat model (most providers use same API key for embeddings).
        embedding_model = model
        if embedding_config is not None and getattr(embedding_config, "model", None):
            embedding_model = embedding_config.model

        # 1. Store — file-based + local TEMPR memory bank
        store = HindsightStore(
            workspace,
            provider=provider,
            embedding_model=embedding_model,
        )

        # 2. Consolidator — token-budget + TEMPR retention
        consolidator = HindsightConsolidator(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
            hindsight_store=store,
        )

        # 3. Dream — offline processing + TEMPR-backed analysis
        dream = HindsightDream(
            store=store,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            hindsight_store=store,
        )

        # 4. AutoCompact — idle session compression
        auto_compact = HindsightAutoCompact(
            sessions=sessions,
            consolidator=consolidator,
            session_ttl_minutes=session_ttl_minutes,
            hindsight_store=store,
        )

        return MemoryComponents(
            store=store,
            consolidator=consolidator,
            dream=dream,
            auto_compact=auto_compact,
        )
