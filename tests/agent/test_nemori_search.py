"""Tests for UnifiedSearch — text, vector, and hybrid search over episodes and semantics."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.memory.nemori_memory.models import Episode, SemanticMemory
from nanobot.memory.nemori_memory.search import SearchMethod, SearchResult, UnifiedSearch
from nanobot.memory.nemori_memory.store import NemoriStore


@pytest.fixture
def store(tmp_path):
    return NemoriStore(Path(tmp_path), backend="file")


@pytest.fixture
def search(store):
    return UnifiedSearch(store)


# ────────────────────────────────────────────────────────────────────────────
# SearchResult
# ────────────────────────────────────────────────────────────────────────────


class TestSearchResult:
    """SearchResult dataclass tests."""

    def test_empty_result(self):
        sr = SearchResult()
        assert sr.episodes == []
        assert sr.semantic_memories == []

    def test_to_dict(self):
        ep = Episode(user_id="u1", title="T", content="C", source_messages=[])
        sm = SemanticMemory(user_id="u1", content="fact", memory_type="identity")
        sr = SearchResult(episodes=[ep], semantic_memories=[sm])
        d = sr.to_dict()
        assert len(d["episodes"]) == 1
        assert len(d["semantic_memories"]) == 1
        assert d["episodes"][0]["title"] == "T"
        assert d["semantic_memories"][0]["content"] == "fact"


# ────────────────────────────────────────────────────────────────────────────
# UnifiedSearch — TEXT mode
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedSearchText:
    """Text-mode search delegates to store text search."""

    @pytest.mark.asyncio
    async def test_text_search_episodes(self, search, store):
        ep = Episode(user_id="u1", title="Python", content="learning", source_messages=[])
        store.save_episode(ep)
        result = await search.search("u1", "default", "Python", method=SearchMethod.TEXT)
        assert len(result.episodes) == 1
        assert result.episodes[0].title == "Python"

    @pytest.mark.asyncio
    async def test_text_search_semantics(self, search, store):
        store.save_semantic(SemanticMemory(user_id="u1", content="likes Python", memory_type="preference"))
        result = await search.search("u1", "default", "Python", method=SearchMethod.TEXT)
        assert len(result.semantic_memories) == 1
        assert result.semantic_memories[0].content == "likes Python"

    @pytest.mark.asyncio
    async def test_text_search_no_results(self, search, store):
        result = await search.search("u1", "default", "xyz", method=SearchMethod.TEXT)
        assert result.episodes == []
        assert result.semantic_memories == []

    @pytest.mark.asyncio
    async def test_text_search_respects_top_k(self, search, store):
        for i in range(5):
            store.save_episode(Episode(user_id="u1", title=f"Python {i}", content=f"learn {i}", source_messages=[]))
        result = await search.search("u1", "default", "Python", top_k_episodes=2, method=SearchMethod.TEXT)
        assert len(result.episodes) == 2


# ────────────────────────────────────────────────────────────────────────────
# UnifiedSearch — VECTOR mode (falls back to text for file backend)
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedSearchVector:
    """Vector mode falls back to text when no PG+Qdrant backend."""

    @pytest.mark.asyncio
    async def test_vector_search_falls_back_to_text(self, search, store):
        ep = Episode(user_id="u1", title="Search", content="find", source_messages=[])
        store.save_episode(ep)
        result = await search.search("u1", "default", "Search", method=SearchMethod.VECTOR)
        assert len(result.episodes) == 1


# ────────────────────────────────────────────────────────────────────────────
# UnifiedSearch — HYBRID mode
# ────────────────────────────────────────────────────────────────────────────


class TestUnifiedSearchHybrid:
    """HYBRID mode uses RRF fusion of text + vector."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, search, store):
        ep = Episode(user_id="u1", title="Testing", content="test content", source_messages=[])
        store.save_episode(ep)
        result = await search.search("u1", "default", "Testing", method=SearchMethod.HYBRID)
        assert len(result.episodes) == 1
        assert result.episodes[0].title == "Testing"

    @pytest.mark.asyncio
    async def test_hybrid_default_method(self, search, store):
        """Default search method is HYBRID."""
        ep = Episode(user_id="u1", title="X", content="y", source_messages=[])
        store.save_episode(ep)
        result = await search.search("u1", "default", "X")
        assert len(result.episodes) == 1

    @pytest.mark.asyncio
    async def test_hybrid_returns_text_results_when_no_vector_results(self, search, store):
        """When no vector results, RRF still returns text results."""
        for i in range(3):
            store.save_episode(Episode(user_id="u1", title=f"Python {i}", content=f"Python {i}", source_messages=[]))
        result = await search.search("u1", "default", "Python", method=SearchMethod.HYBRID)
        assert len(result.episodes) == 3

    @pytest.mark.asyncio
    async def test_hybrid_search_no_results(self, search, store):
        result = await search.search("u1", "default", "no_match", method=SearchMethod.HYBRID)
        assert result.episodes == []


# ────────────────────────────────────────────────────────────────────────────
# RRF constant
# ────────────────────────────────────────────────────────────────────────────


class TestRRF:
    """RRF (Reciprocal Rank Fusion) constant verification."""

    def test_rrf_constant(self):
        assert UnifiedSearch._RRF_K == 60
