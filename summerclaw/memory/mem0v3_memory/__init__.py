"""Mem0V3 memory algorithm — token-efficient ADD-only memory for summerclaw.

Based on the mem0 v3 algorithm (April 2026) by the Mem0 team.
Read the announcement at https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm

Core innovations over the old two-pass approach:

1. **Single-pass ADD-only extraction**: One LLM call, no UPDATE/DELETE.
   Every fact becomes an independent record. Changes coexist with old facts.

2. **Agent-generated facts are first-class**: Both user and assistant messages
   are extracted with equal weight, closing the agent memory blind spot.

3. **Entity linking**: Each memory is analyzed for entities (proper nouns,
   quoted text, compound phrases). Entities are embedded and linked to
   memories, enabling entity-aware retrieval.

4. **Multi-signal retrieval**: Three parallel scoring passes — semantic
   similarity, BM25 keyword matching, and entity matching — fused into
   a combined score.

5. **Keyword normalization**: Lemmatization for verb form normalization
   (attending/attends/attended → attend).

Usage::

    from summerclaw.memory import MemoryRegistry
    from summerclaw.memory.mem0v3_memory import Mem0V3MemoryAlgorithm

    registry = MemoryRegistry()
    registry.register(Mem0V3MemoryAlgorithm())
    algo = registry.get("mem0v3_memory")
    components = algo.build(...)

Hermes Mode:
    The Mem0V3Consolidator also serves as the Hermes-Autogen extraction
    engine. When ``skill_autogen.enable: true``, the mid-turn skill
    distillation triggers ``consolidator.extract_and_store()`` on the
    recent conversation to capture facts before skill generation.

Dream Mode:
    The Mem0V3Dream runs on cron (default: every 2h). Phase 1 analyzes
    MEMORY.md + vector memories; Phase 2 edits MEMORY.md via AgentRunner
    and optionally generates dreamed-* skills.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from summerclaw.memory.base import MemoryAlgorithm, MemoryComponents
from summerclaw.memory.mem0v3_memory.store import Mem0V3Store
from summerclaw.memory.mem0v3_memory.consolidator import Mem0V3Consolidator
from summerclaw.memory.mem0v3_memory.dream import Mem0V3Dream
from summerclaw.memory.mem0v3_memory.auto_compact import Mem0V3AutoCompact

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import SessionManager


class Mem0V3MemoryAlgorithm(MemoryAlgorithm):
    """mem0 v3 memory algorithm — token-efficient ADD-only memory.

    Implements the full mem0 v3 pipeline:

    1. **Mem0V3Store** — file-based vector store with entity index and BM25
    2. **Mem0V3Consolidator** — single-pass ADD-only LLM extraction + hash dedup
    3. **Mem0V3Dream** — offline cron-scheduled deep processing (Phase 1 + 2)
    4. **Mem0V3AutoCompact** — idle session compression

    Key features:
    - ADD-only extraction preserves full state change history
    - Entity linking enables entity-aware multi-signal retrieval
    - Zero external dependencies beyond summerclaw standard stack
    - Compatible with Dream and Hermes modes
    - Embedding-based semantic search via provider.embed()

    Configuration is controlled via summerclaw's standard config:
    - ``embedding``: Embedding model config (model, provider, batch_size)
    - ``dream``: Dream schedule config (interval, model_override, max_batch_size, etc.)
    - ``skill_autogen.enable``: Hermes-Autogen mid-turn skill distillation
    - ``session_ttl_minutes``: AutoCompact idle threshold
    """

    name = "mem0v3_memory"

    def __init__(self):
        super().__init__()

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
        """Build and return all mem0 v3 memory components.

        Args:
            workspace: Agent workspace path.
            provider: LLM provider (must support embed() for vector search).
            model: Model name for LLM calls.
            sessions: Session manager.
            context_window_tokens: Total context window size.
            build_messages: Callable for building messages.
            get_tool_definitions: Callable for getting tool definitions.
            max_completion_tokens: Max completion tokens budget.
            session_ttl_minutes: Auto-compact idle threshold (0 = disabled).
            max_batch_size: Dream max batch size.
            max_iterations: Dream max tool iterations.
            max_tool_result_chars: Max chars for tool results.
            annotate_line_ages: Whether Dream annotates line ages.
            embedding_config: Optional EmbeddingConfig for embedding model.
        """
        # ------------------------------------------------------------------
        # 1. Store — file-based vector store with entity index
        # ------------------------------------------------------------------
        store = Mem0V3Store(workspace=workspace, algo_name=self.name)

        # ------------------------------------------------------------------
        # 2. Consolidator — single-pass ADD-only extraction
        # ------------------------------------------------------------------
        # Determine embedding model: prefer embedding_config.model, fall back to chat model.
        # The provider's embed() method handles API routing (OpenAI-compatible vs native).
        embedding_model = model
        if embedding_config is not None and getattr(embedding_config, "model", None):
            embedding_model = embedding_config.model

        consolidator = Mem0V3Consolidator(
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

        # ------------------------------------------------------------------
        # 3. Dream — offline cron-scheduled deep processing
        # ------------------------------------------------------------------
        # Check for dream model override
        dream_model = model
        # Dream model override is typically passed via config but not through
        # build() params directly. We use the default model.
        dream = Mem0V3Dream(
            store=store,
            provider=provider,
            model=dream_model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=self.name,
        )

        # ------------------------------------------------------------------
        # 4. AutoCompact — idle session compression
        # ------------------------------------------------------------------
        auto_compact = None
        if session_ttl_minutes > 0:
            auto_compact = Mem0V3AutoCompact(
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
