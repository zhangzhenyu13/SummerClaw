"""Nemori memory algorithm — self-organising long-term memory for nanobot.

Based on nemori (https://github.com/nemori-ai/nemori).

Implements two coupled control loops:
  1. Two-Step Alignment:
     - Boundary Alignment: LLM-powered topic segmentation
     - Representation Alignment: episode narrative generation
  2. Predict-Calibrate Learning:
     - Predict: hypothesise from existing semantic knowledge
     - Calibrate: extract high-value facts from discrepancies

Usage::

    from nanobot.memory import MemoryRegistry
    from nanobot.memory.nemori_memory import NemoriMemoryAlgorithm

    registry = MemoryRegistry()
    registry.register(NemoriMemoryAlgorithm())
    algo = registry.get("nemori_memory")
    components = algo.build(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.memory.base import MemoryAlgorithm, MemoryComponents
from nanobot.memory.nemori_memory.consolidator import NemoriConsolidator
from nanobot.memory.nemori_memory.dream import NemoriDream
from nanobot.memory.nemori_memory.episode_generator import EpisodeGenerator
from nanobot.memory.nemori_memory.merger import EpisodeMerger
from nanobot.memory.nemori_memory.search import UnifiedSearch
from nanobot.memory.nemori_memory.segmenter import BatchSegmenter
from nanobot.memory.nemori_memory.semantic_generator import SemanticGenerator
from nanobot.memory.nemori_memory.store import NemoriStore

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


class NemoriMemoryAlgorithm(MemoryAlgorithm):
    """Nemori memory algorithm — self-organising long-term memory.

    Defaults to file-based storage (zero extra dependencies).
    Set ``backend="postgres"`` in config to use PostgreSQL + Qdrant.

    Key features:
    - LLM-powered batch segmentation into topic-coherent episodes
    - Structured episode generation with temporal anchoring
    - Predict-Calibrate semantic knowledge extraction
    - Episode merging to avoid duplication
    - Unified search across episodes + semantic memories
    """

    name = "nemori_memory"

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
        # Storage layer
        store = NemoriStore(workspace, backend="file")

        # Pipeline components
        segmenter = BatchSegmenter(provider, model)
        episode_gen = EpisodeGenerator(provider, model)
        semantic_gen = SemanticGenerator(
            provider, model, enable_prediction_correction=True
        )
        merger = EpisodeMerger(provider, model, store)

        # Orchestrator (consolidator)
        consolidator = NemoriConsolidator(
            store=store,
            segmenter=segmenter,
            episode_generator=episode_gen,
            semantic_generator=semantic_gen,
            merger=merger,
            buffer_size_min=2,
            batch_threshold=10,
            episode_min_messages=2,
            enable_semantic=True,
            enable_merging=True,
        )

        # Search
        search = UnifiedSearch(store)

        # Dream (cron-scheduled deep processing + skill generation)
        dream = NemoriDream(
            store=store,
            search=search,
            provider=provider,
            model=model,
            workspace=workspace,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
        )

        return MemoryComponents(
            store=store,
            consolidator=consolidator,
            dream=dream,
            auto_compact=None,  # nemori handles compaction via buffer cleanup
        )
