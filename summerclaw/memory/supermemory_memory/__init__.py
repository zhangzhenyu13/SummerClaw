"""Supermemory memory algorithm — chunk-based memory with relational versioning and temporal grounding.

Implements the Supermemory research architecture (https://supermemory.ai/research/):
- Chunk-based ingestion & Contextual Memories
- Relational Versioning (updates, extends, derives)
- Temporal Grounding (documentDate, eventDate)
- Hybrid Search (semantic search on memories + source chunk injection)
- Session-Based Processing

Fully local — no external API dependency. Uses the same file-based storage
pattern as naive_memory with additional memory graph (memory_graph.json) and
chunk storage (chunks/ directory).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from summerclaw.memory.base import MemoryAlgorithm, MemoryComponents
from summerclaw.memory.supermemory_memory.store import SupermemoryStore
from summerclaw.memory.supermemory_memory.consolidator import SupermemoryConsolidator
from summerclaw.memory.supermemory_memory.dream import SupermemoryDream
from summerclaw.memory.supermemory_memory.auto_compact import SupermemoryAutoCompact

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import SessionManager


class SupermemoryMemoryAlgorithm(MemoryAlgorithm):
    """Supermemory algorithm — SOTA agent memory with relational versioning.

    Key features:
    - Chunk-based memory generation: decomposes conversations into semantic blocks,
      generates atomic memories with contextual reference resolution.
    - Relational versioning: tracks updates (state mutation), extends (refinement),
      and derives (inference) relationships between memories.
    - Temporal grounding: dual-layer timestamps (documentDate + eventDate).
    - Hybrid search: semantic search on atomic memories returns source chunks.
    - Version chains: when facts change, old versions are preserved as history.
    """

    name = "supermemory_memory"

    def build(
        self,
        workspace: Path,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages,
        get_tool_definitions,
        max_completion_tokens: int,
        session_ttl_minutes: int,
        max_batch_size: int,
        max_iterations: int,
        max_tool_result_chars: int,
        annotate_line_ages: bool,
        embedding_config: Any = None,
    ) -> MemoryComponents:
        # Determine embedding model: prefer embedding_config.model, fall back to chat model
        embedding_model = model
        if embedding_config is not None and getattr(embedding_config, "model", None):
            embedding_model = embedding_config.model

        # Create the Supermemory store with graph, chunks, and relationship tracking
        store = SupermemoryStore(workspace, algo_name=self.name)

        # Chunk-based consolidator with memory graph integration and embedding support
        consolidator = SupermemoryConsolidator(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
            embedding_model=embedding_model,
        )

        # Dream processor with graph context for relational versioning
        dream = SupermemoryDream(
            store=store,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=self.name,
        )

        # Auto-compact for idle session compression
        auto_compact = SupermemoryAutoCompact(
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
