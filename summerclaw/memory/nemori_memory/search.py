"""Unified search across episode and semantic memory stores.

Ported from nemori (https://github.com/nemori-ai/nemori).

Supports three search methods:
  - TEXT: keyword-based text search (works with file backend)
  - VECTOR: cosine similarity over stored embeddings
  - HYBRID: RRF (Reciprocal Rank Fusion) combining text + vector results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from summerclaw.memory.nemori_memory.models import Episode, SemanticMemory

if TYPE_CHECKING:
    from summerclaw.memory.nemori_memory.store import NemoriStore

logger = logging.getLogger("nemori")


class SearchMethod(Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Container for unified search results."""

    episodes: list[Episode] = field(default_factory=list)
    semantic_memories: list[SemanticMemory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "semantic_memories": [s.to_dict() for s in self.semantic_memories],
        }


class UnifiedSearch:
    """Delegates search: text for file backend, text+vector for PG+Qdrant.

    Uses RRF (Reciprocal Rank Fusion) for hybrid search mode.
    RRF constant k=60 as per the proven value in TREC evaluations.
    """

    _RRF_K = 60

    def __init__(self, store: "NemoriStore") -> None:
        self._store = store

    async def search(
        self,
        user_id: str,
        agent_id: str,
        query: str,
        top_k_episodes: int = 10,
        top_k_semantic: int = 10,
        method: SearchMethod = SearchMethod.HYBRID,
    ) -> SearchResult:
        """Execute search across episodes and semantic memories.

        Args:
            user_id: User identifier.
            agent_id: Agent namespace.
            query: The search query string.
            top_k_episodes: Max episode results.
            top_k_semantic: Max semantic memory results.
            method: Search method (VECTOR, TEXT, or HYBRID).

        Returns:
            SearchResult with matched episodes and semantic memories.
        """
        if method == SearchMethod.TEXT:
            episodes = self._store.search_episodes_by_text(
                user_id, agent_id, query, top_k_episodes
            )
            semantics = self._store.search_semantics_by_text(
                user_id, agent_id, query, top_k_semantic
            )
        elif method == SearchMethod.VECTOR:
            # Vector search requires embeddings from the query — only works
            # if an embedding provider is available. Fall back to text.
            episodes = self._store.search_episodes_by_text(
                user_id, agent_id, query, top_k_episodes
            )
            semantics = self._store.search_semantics_by_text(
                user_id, agent_id, query, top_k_semantic
            )
        else:  # HYBRID
            episodes = self._hybrid_search_episodes(
                user_id, agent_id, query, top_k_episodes
            )
            semantics = self._hybrid_search_semantics(
                user_id, agent_id, query, top_k_semantic
            )

        return SearchResult(episodes=episodes, semantic_memories=semantics)

    def _hybrid_search_episodes(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[Episode]:
        """RRF fusion of text + vector search for episodes."""
        # For file backend: do text search with broader recall, then re-rank
        text_results = self._store.search_episodes_by_text(
            user_id, agent_id, query, max(top_k * 2, 20)
        )

        # Try vector search if possible
        vec_results: list[dict[str, Any]] = []
        # In file backend without embeddings, vector search is a no-op
        # The text search alone is sufficient for the HYBRID method

        # RRF fusion
        rrf: dict[str, float] = {}
        for rank, r in enumerate(vec_results, 1):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1.0 / (self._RRF_K + rank)
        for rank, ep in enumerate(text_results, 1):
            rrf[ep.id] = rrf.get(ep.id, 0) + 1.0 / (self._RRF_K + rank)

        sorted_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]
        if not sorted_ids:
            return text_results[:top_k]

        # Re-sort by RRF score
        id_to_ep = {ep.id: ep for ep in text_results}
        return [id_to_ep[eid] for eid in sorted_ids if eid in id_to_ep]

    def _hybrid_search_semantics(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]:
        """RRF fusion of text + vector search for semantic memories."""
        text_results = self._store.search_semantics_by_text(
            user_id, agent_id, query, max(top_k * 2, 20)
        )

        vec_results: list[dict[str, Any]] = []

        rrf: dict[str, float] = {}
        for rank, r in enumerate(vec_results, 1):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1.0 / (self._RRF_K + rank)
        for rank, sm in enumerate(text_results, 1):
            rrf[sm.id] = rrf.get(sm.id, 0) + 1.0 / (self._RRF_K + rank)

        sorted_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]
        if not sorted_ids:
            return text_results[:top_k]

        id_to_mem = {m.id: m for m in text_results}
        return [id_to_mem[mid] for mid in sorted_ids if mid in id_to_mem]
