"""EMem memory algorithm — structured conversational memory with EDU extraction.

EMem (Elementary Discourse Unit Memory) provides:

- **EDU extraction**: LLM-based decomposition of conversation turns into atomic
  propositions (EDUs) with event types, triggers, and role-argument pairs.
- **Dense retrieval**: Embedding-based KNN search for relevant EDUs.
- **LLM rerank**: Semantic filtering of candidate EDUs and arguments.
- **Heterogeneous graph**: Session-EDU-Argument graph with optional Personalized
  PageRank (PPR) for associative recall.
- **Token-budget consolidation**: Online compression with EDU archiving.
- **Dream processing**: Offline cron-scheduled memory processing with graph updates.

Optional dependencies (``pip install summerclaw-ai[emem]``):
- ``igraph`` for fast PPR.
- ``sentence-transformers`` for local embedding models.
- ``torch``, ``scipy`` (scipy fallback for PPR when igraph unavailable).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from summerclaw.memory.base import MemoryAlgorithm, MemoryComponents
from summerclaw.memory.emem_memory.auto_compact import EMemAutoCompact
from summerclaw.memory.emem_memory.consolidator import EMemConsolidator
from summerclaw.memory.emem_memory.datatypes import EMemConfig
from summerclaw.memory.emem_memory.dream import EMemDream
from summerclaw.memory.emem_memory.edu_extractor import EDUExtractor
from summerclaw.memory.emem_memory.embedding import EMemEmbedder
from summerclaw.memory.emem_memory.graph import EMemGraph
from summerclaw.memory.emem_memory.rerank import ArgumentReranker, EDUReranker
from summerclaw.memory.emem_memory.store import EMemStore

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import SessionManager


class EMemMemoryAlgorithm(MemoryAlgorithm):
    """EMem memory algorithm — structured memory with EDU extraction and graph retrieval.

    Builds a complete memory pipeline:
    1. EMemEmbedder — embedding generation (OpenAI API or local).
    2. EMemStore — persistent storage for EDUs, arguments, and sessions.
    3. EDUExtractor — LLM-based EDU extraction from conversations.
    4. EDUReranker + ArgumentReranker — LLM-based candidate filtering.
    5. EMemGraph — heterogeneous graph with optional PPR.
    6. EMemConsolidator — token-budget consolidation with EDU archiving.
    7. EMemDream — offline cron processing with graph updates.
    8. EMemAutoCompact — idle session compression.

    Configuration is controlled via :class:`EMemConfig`:
    - ``skip_ppr``: If True, skip PPR graph propagation (dense-only mode).
    - ``linking_top_k``: Number of top candidates to keep after linking.
    - ``retrieval_top_k``: Number of results to retrieve per query.
    - ``damping``: PPR damping factor (0–1).
    """

    name = "emem_memory"

    def __init__(self, config: EMemConfig | None = None):
        super().__init__()
        self.config = config or EMemConfig()

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
        """Build and return all EMem memory components.

        Assembles the full EMem pipeline from embedder through graph to
        consolidation and dream processing.
        """
        import os

        # 1. Embedding model — prefer explicit embedding_config, fall back to provider/env
        if embedding_config is not None:
            embedder = EMemEmbedder.from_config(
                embedding_config,
                provider=provider,
                fallback_api_key=getattr(provider, "api_key", None),
                fallback_api_base=getattr(provider, "api_base", None),
            )
        else:
            embedder = EMemEmbedder(
                model_name=None,  # defaults to text-embedding-3-small
                api_key=getattr(provider, "api_key", None) or os.environ.get("OPENAI_API_KEY"),
                api_base=getattr(provider, "api_base", None) or os.environ.get("OPENAI_BASE_URL"),
                batch_size=self.config.embedding_batch_size,
                normalize=self.config.embedding_return_as_normalized,
                provider=provider,
            )

        # 2. EMem store (EDU + argument + session persistence)
        store = EMemStore(
            workspace=workspace,
            embedding_model=embedder,
            batch_size=self.config.embedding_batch_size,
            algo_name=self.name,
        )

        # 3. EDU extractor (LLM-based OpenIE)
        edu_extractor = EDUExtractor(
            provider=provider,
            model=model,
            extract_events=not self.config.skip_edu_context_gen,
            skip_context_gen=self.config.skip_edu_context_gen,
        )

        # 4. Graph (heterogeneous Session-EDU-Argument with PPR)
        graph = EMemGraph(
            working_dir=store.emem_dir,
            directed=False,
            force_rebuild=self.config.force_reindex,
        )

        # 5. Rerankers (LLM-based candidate filtering)
        edu_reranker = EDUReranker(provider=provider, model=model)
        arg_reranker = ArgumentReranker(provider=provider, model=model)

        # 6. Consolidator (token-budget + EDU archiving)
        consolidator = EMemConsolidator(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            edu_extractor=edu_extractor,
            emem_store=store,
            max_completion_tokens=max_completion_tokens,
        )

        # 7. Dream (offline cron processing + graph updates)
        dream = EMemDream(
            store=store,
            provider=provider,
            model=model,
            edu_extractor=edu_extractor,
            emem_store=store,
            emem_graph=graph,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=self.name,
        )

        # 8. AutoCompact (idle session compression)
        auto_compact = EMemAutoCompact(
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
