"""Tests for EpisodeMerger — LLM-powered episode deduplication merge logic."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.memory.nemori_memory.merger import EpisodeMerger
from nanobot.memory.nemori_memory.models import Episode
from nanobot.memory.nemori_memory.store import NemoriStore


@pytest.fixture
def store(tmp_path):
    return NemoriStore(Path(tmp_path), backend="file")


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def merger(mock_provider, store):
    return EpisodeMerger(mock_provider, "test-model", store)


def _ep(**kw):
    defaults = {"user_id": "u1", "title": "T", "content": "C", "source_messages": [{"role": "user", "content": "hello"}]}
    defaults.update(kw)
    return Episode(**defaults)


# ────────────────────────────────────────────────────────────────────────────
# Merge — no candidates
# ────────────────────────────────────────────────────────────────────────────


class TestEpisodeMergerNoCandidates:
    """When no similar episodes exist, check_and_merge returns False."""

    @pytest.mark.asyncio
    async def test_no_candidates(self, merger):
        ep = _ep()
        merged, result, old_id = await merger.check_and_merge(ep, "default")
        assert merged is False
        assert result is None
        assert old_id is None

    @pytest.mark.asyncio
    async def test_only_self_in_store(self, merger, store):
        ep = _ep()
        store.save_episode(ep)
        merged, result, old_id = await merger.check_and_merge(ep, "default")
        assert merged is False


# ────────────────────────────────────────────────────────────────────────────
# Merge — decision
# ────────────────────────────────────────────────────────────────────────────


class TestEpisodeMergerDecision:
    """LLM merge decision tests."""

    @pytest.mark.asyncio
    async def test_merge_approved(self, merger, store, mock_provider):
        ep_new = _ep(title="New")
        ep_old = _ep(title="Old")
        store.save_episode(ep_old)

        # Mock: find similar returns old episode
        # Mock: LLM decision says "merge"
        decision_resp = MagicMock()
        decision_resp.content = json.dumps({
            "decision": "merge",
            "merge_target_id": ep_old.id,
            "reason": "same topic",
        })

        # Mock: merge content LLM call
        content_resp = MagicMock()
        content_resp.content = json.dumps({
            "title": "Merged",
            "content": "merged content",
            "timestamp": "2025-01-01T00:00:00",
        })

        mock_provider.chat_with_retry.side_effect = [decision_resp, content_resp]

        # Fill store with both episodes
        store.save_episode(ep_new)

        merged, result, old_id = await merger.check_and_merge(ep_new, "default")
        assert merged is True
        assert result is not None
        assert old_id == ep_old.id
        assert result.title == "Merged"

    @pytest.mark.asyncio
    async def test_merge_rejected(self, merger, store, mock_provider):
        ep_new = _ep(title="New")
        ep_old = _ep(title="Old")
        store.save_episode(ep_old)

        decision_resp = MagicMock()
        decision_resp.content = json.dumps({
            "decision": "new",
            "merge_target_id": None,
            "reason": "different topics",
        })
        mock_provider.chat_with_retry.return_value = decision_resp

        merged, result, old_id = await merger.check_and_merge(ep_new, "default")
        assert merged is False
        assert result is None

    @pytest.mark.asyncio
    async def test_merge_target_not_found(self, merger, store, mock_provider):
        """LLM returns a target_id that doesn't exist in candidates."""
        ep_new = _ep(title="New")

        decision_resp = MagicMock()
        decision_resp.content = json.dumps({
            "decision": "merge",
            "merge_target_id": "nonexistent-ep",
            "reason": "test",
        })
        mock_provider.chat_with_retry.return_value = decision_resp

        merged, result, old_id = await merger.check_and_merge(ep_new, "default")
        assert merged is False

    @pytest.mark.asyncio
    async def test_merge_llm_error_fallback(self, merger, store, mock_provider):
        """If LLM call fails, return (False, None, None)."""
        ep_new = _ep(title="New")
        ep_old = _ep(title="Old")
        store.save_episode(ep_old)
        mock_provider.chat_with_retry.side_effect = RuntimeError("LLM error")

        merged, result, old_id = await merger.check_and_merge(ep_new, "default")
        assert merged is False
        assert result is None
        assert old_id is None

    @pytest.mark.asyncio
    async def test_merge_content_fallback_on_llm_error(self, merger, store, mock_provider):
        """If merge content LLM fails, use simple concatenation fallback."""
        ep_new = _ep(title="New")
        ep_old = _ep(title="Old")
        store.save_episode(ep_old)

        decision_resp = MagicMock()
        decision_resp.content = json.dumps({
            "decision": "merge",
            "merge_target_id": ep_old.id,
            "reason": "same topic",
        })
        content_error = RuntimeError("Content LLM error")
        mock_provider.chat_with_retry.side_effect = [decision_resp, content_error]

        merged, result, old_id = await merger.check_and_merge(ep_new, "default")
        assert merged is True
        assert result is not None
        assert "Merged:" in result.title  # fell back to concatenation

    @pytest.mark.asyncio
    async def test_merge_preserves_nemori_data(self, merger, store, mock_provider):
        """Merged episode should preserve source_messages and metadata."""
        ep_new = _ep(title="New", content="new content")
        ep_old = _ep(title="Old", content="old content")
        store.save_episode(ep_old)

        decision_resp = MagicMock()
        decision_resp.content = json.dumps({
            "decision": "merge",
            "merge_target_id": ep_old.id,
            "reason": "same",
        })
        content_resp = MagicMock()
        content_resp.content = json.dumps({
            "title": "Merged Episode",
            "content": "combined",
            "timestamp": "2025-01-01T00:00:00",
        })
        mock_provider.chat_with_retry.side_effect = [decision_resp, content_resp]

        merged, result, old_id = await merger.check_and_merge(ep_new, "default")
        assert merged is True
        assert "merged_from" in result.metadata
        assert result.metadata["merged_from"] == [ep_old.id, ep_new.id]
        # source_messages combined
        assert len(result.source_messages) == 2


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────


class TestEpisodeMergerConfig:
    """Configuration defaults."""

    def test_default_similarity_threshold(self, merger):
        assert merger._similarity_threshold == 0.85

    def test_default_merge_top_k(self, merger):
        assert merger._merge_top_k == 5

    def test_custom_config(self, mock_provider, store):
        m = EpisodeMerger(mock_provider, "model", store, similarity_threshold=0.9, merge_top_k=3)
        assert m._similarity_threshold == 0.9
        assert m._merge_top_k == 3
