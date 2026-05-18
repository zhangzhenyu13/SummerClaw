"""Tests for Mem0V3 consolidator embedding integration with provider.embed().

Verifies that the consolidator correctly delegates embedding calls to
``provider.embed()`` and handles edge cases (empty results, errors, etc.).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from summerclaw.memory.mem0v3_memory.consolidator import Mem0V3Consolidator
from summerclaw.memory.mem0v3_memory.store import Mem0V3Store


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_store(tmp_path) -> Mem0V3Store:
    """Create a Mem0V3Store in a temp workspace."""
    store = Mem0V3Store(workspace=tmp_path)
    return store


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLMProvider that supports embed()."""
    provider = MagicMock()
    provider.embed.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
    ]
    provider.chat = MagicMock()
    return provider


@pytest.fixture
def consolidator(mock_store, mock_provider) -> Mem0V3Consolidator:
    """Create a Mem0V3Consolidator with mock dependencies."""
    cons = Mem0V3Consolidator(
        store=mock_store,
        provider=mock_provider,
        model="qwen-plus",
        sessions=None,
        context_window_tokens=128000,
        build_messages=None,
        get_tool_definitions=None,
        max_completion_tokens=4096,
        embedding_model="tongyi-embedding-vision-flash",
    )
    return cons


# ===================================================================
# _embed() — single text embedding
# ===================================================================


class TestConsolidatorEmbed:
    """Tests for Mem0V3Consolidator._embed()."""

    def test_embed_calls_provider_with_embedding_model(self, consolidator, mock_provider) -> None:
        """_embed() calls provider.embed() with the configured embedding_model."""
        mock_provider.embed.reset_mock()
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]

        result = consolidator._embed("hello world")

        mock_provider.embed.assert_called_once_with(
            ["hello world"],
            "tongyi-embedding-vision-flash",
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_returns_none_on_empty_result(self, consolidator, mock_provider) -> None:
        """_embed() returns None when provider returns empty list."""
        mock_provider.embed.return_value = []

        result = consolidator._embed("test")
        assert result is None

    def test_embed_returns_none_on_not_implemented(self, consolidator, mock_provider) -> None:
        """_embed() returns None when provider raises NotImplementedError."""
        mock_provider.embed.side_effect = NotImplementedError("no embeddings")

        result = consolidator._embed("test")
        assert result is None

    def test_embed_returns_none_on_exception(self, consolidator, mock_provider) -> None:
        """_embed() returns None and logs warning when provider raises exception."""
        mock_provider.embed.side_effect = RuntimeError("API error")

        result = consolidator._embed("test")
        assert result is None

    def test_embed_falls_back_to_chat_model_when_no_embedding_config(self, mock_store, mock_provider) -> None:
        """When embedding_model is None, falls back to the chat model."""
        cons = Mem0V3Consolidator(
            store=mock_store,
            provider=mock_provider,
            model="qwen-plus",
            sessions=None,
            context_window_tokens=128000,
            build_messages=None,
            get_tool_definitions=None,
            max_completion_tokens=4096,
            # embedding_model not specified → defaults to model
        )
        mock_provider.embed.return_value = [[0.5, 0.6]]

        cons._embed("test")
        mock_provider.embed.assert_called_with(["test"], "qwen-plus")


# ===================================================================
# _embed_batch() — batch embedding
# ===================================================================


class TestConsolidatorEmbedBatch:
    """Tests for Mem0V3Consolidator._embed_batch()."""

    def test_embed_batch_returns_dict(self, consolidator, mock_provider) -> None:
        """_embed_batch() returns {text: embedding} dict."""
        mock_provider.embed.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        result = consolidator._embed_batch(["hello", "world"])

        assert result == {
            "hello": [0.1, 0.2],
            "world": [0.3, 0.4],
        }
        mock_provider.embed.assert_called_with(
            ["hello", "world"],
            "tongyi-embedding-vision-flash",
        )

    def test_embed_batch_empty_input(self, consolidator, mock_provider) -> None:
        """_embed_batch() returns empty dict for empty input."""
        mock_provider.embed.reset_mock()

        result = consolidator._embed_batch([])
        assert result == {}
        mock_provider.embed.assert_not_called()

    def test_embed_batch_falls_back_to_single_on_error(self, consolidator, mock_provider) -> None:
        """When batch fails, falls back to calling _embed() individually."""
        mock_provider.embed.side_effect = RuntimeError("batch error")

        with patch.object(consolidator, "_embed") as mock_single_embed:
            mock_single_embed.side_effect = lambda text: {
                "hello": [0.1, 0.2],
                "world": [0.3, 0.4],
            }.get(text)

            result = consolidator._embed_batch(["hello", "world"])
            assert result == {"hello": [0.1, 0.2], "world": [0.3, 0.4]}
            assert mock_single_embed.call_count == 2

    def test_embed_batch_handles_partial_failures(self, consolidator, mock_provider) -> None:
        """Some texts may fail in fallback — results contain only successful ones."""
        mock_provider.embed.side_effect = RuntimeError("batch error")

        with patch.object(consolidator, "_embed") as mock_single_embed:
            mock_single_embed.side_effect = lambda text: (
                [0.1, 0.2] if text == "hello" else None
            )

            result = consolidator._embed_batch(["hello", "world"])
            assert result == {"hello": [0.1, 0.2]}

    def test_embed_batch_mismatched_lengths_handled_gracefully(
        self, consolidator, mock_provider,
    ) -> None:
        """If provider returns fewer embeddings than texts, mismatched texts are silently dropped."""
        mock_provider.embed.return_value = [[0.1, 0.2]]  # only 1 for 3 texts

        result = consolidator._embed_batch(["a", "b", "c"])
        assert result == {"a": [0.1, 0.2]}


# ===================================================================
# store interface methods for ContextBuilder
# ===================================================================


class TestStoreContextBuilderInterface:
    """Verify Mem0V3Store implements all ContextBuilder-required methods."""

    def test_get_memory_context(self, tmp_path) -> None:
        """get_memory_context() returns formatted string."""
        store = Mem0V3Store(workspace=tmp_path)
        ctx = store.get_memory_context()
        assert isinstance(ctx, str)

    def test_get_memory_context_with_md(self, tmp_path) -> None:
        """get_memory_context() includes MEMORY.md content when present."""
        store = Mem0V3Store(workspace=tmp_path)
        store.write_memory_md("User likes pizza")
        ctx = store.get_memory_context()
        assert "User likes pizza" in ctx

    def test_read_memory(self, tmp_path) -> None:
        """read_memory() is an alias for read_memory_md()."""
        store = Mem0V3Store(workspace=tmp_path)
        store.write_memory_md("test content")
        assert store.read_memory() == "test content"

    def test_read_unprocessed_history(self, tmp_path) -> None:
        """read_unprocessed_history() returns empty list for mem0v3."""
        store = Mem0V3Store(workspace=tmp_path)
        result = store.read_unprocessed_history(since_cursor=0)
        assert result == []

    def test_get_last_dream_cursor(self, tmp_path) -> None:
        """get_last_dream_cursor() returns 0 for mem0v3."""
        store = Mem0V3Store(workspace=tmp_path)
        assert store.get_last_dream_cursor() == 0
