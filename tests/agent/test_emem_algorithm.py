"""Tests for EMemMemoryAlgorithm — integration of all EMem components."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from nanobot.memory.base import MemoryComponents
from nanobot.memory.emem_memory import EMemMemoryAlgorithm
from nanobot.memory.emem_memory.auto_compact import EMemAutoCompact
from nanobot.memory.emem_memory.consolidator import EMemConsolidator
from nanobot.memory.emem_memory.datatypes import EMemConfig
from nanobot.memory.emem_memory.dream import EMemDream
from nanobot.memory.emem_memory.store import EMemStore


# ===================================================================
# EMemMemoryAlgorithm — name and basic properties
# ===================================================================

class TestEMemMemoryAlgorithmBasic:
    """Test algorithm name, config, and basic properties."""

    def test_algorithm_name(self) -> None:
        algo = EMemMemoryAlgorithm()
        assert algo.name == "emem_memory"

    def test_default_config(self) -> None:
        algo = EMemMemoryAlgorithm()
        assert algo.config is not None
        assert isinstance(algo.config, EMemConfig)

    def test_custom_config(self) -> None:
        config = EMemConfig(linking_top_k=10, damping=0.8)
        algo = EMemMemoryAlgorithm(config=config)
        assert algo.config.linking_top_k == 10
        assert algo.config.damping == 0.8


# ===================================================================
# EMemMemoryAlgorithm — build()
# ===================================================================

class TestEMemMemoryAlgorithmBuild:
    """Test the build() method assembling all components."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        p = MagicMock()
        p.generation = MagicMock()
        p.generation.temperature = 0.7
        p.generation.max_tokens = 4096
        p.api_key = "test-openai-api-key"
        p.api_base = "https://api.test.com"
        return p

    @pytest.fixture
    def mock_sessions(self) -> MagicMock:
        sm = MagicMock()
        sm.save = MagicMock()
        sm.invalidate = MagicMock()
        sm.list_sessions = MagicMock(return_value=[])
        return sm

    def test_build_returns_memory_components(
        self, tmp_path, mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        algo = EMemMemoryAlgorithm()

        # Patch EMemEmbedder to avoid OpenAI API key requirement during test
        with patch(
            "nanobot.memory.emem_memory.embedding.EMemEmbedder",
            autospec=True,
        ) as mock_embedder_cls:
            mock_embedder = MagicMock()
            mock_embedder.batch_encode.return_value = []
            mock_embedder_cls.return_value = mock_embedder

            components = algo.build(
                workspace=tmp_path,
                provider=mock_provider,
                model="test-model",
                sessions=mock_sessions,
                context_window_tokens=128_000,
                build_messages=MagicMock(),
                get_tool_definitions=MagicMock(),
                max_completion_tokens=4096,
                session_ttl_minutes=15,
                max_batch_size=20,
                max_iterations=10,
                max_tool_result_chars=16000,
                annotate_line_ages=True,
            )

            assert isinstance(components, MemoryComponents)

    def test_build_components_have_correct_types(
        self, tmp_path, mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        algo = EMemMemoryAlgorithm()

        with patch(
            "nanobot.memory.emem_memory.embedding.EMemEmbedder",
            autospec=True,
        ) as mock_embedder_cls:
            mock_embedder = MagicMock()
            mock_embedder.batch_encode.return_value = []
            mock_embedder_cls.return_value = mock_embedder

            components = algo.build(
                workspace=tmp_path,
                provider=mock_provider,
                model="test-model",
                sessions=mock_sessions,
                context_window_tokens=128_000,
                build_messages=MagicMock(),
                get_tool_definitions=MagicMock(),
                max_completion_tokens=4096,
                session_ttl_minutes=15,
                max_batch_size=20,
                max_iterations=10,
                max_tool_result_chars=16000,
                annotate_line_ages=True,
            )

            assert isinstance(components.store, EMemStore)
            assert isinstance(components.consolidator, EMemConsolidator)
            assert isinstance(components.dream, EMemDream)
            assert isinstance(components.auto_compact, EMemAutoCompact)

    def test_auto_compact_is_not_none(
        self, tmp_path, mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        """auto_compact should not be None — parity with naive_memory."""
        algo = EMemMemoryAlgorithm()

        with patch(
            "nanobot.memory.emem_memory.embedding.EMemEmbedder",
            autospec=True,
        ) as mock_embedder_cls:
            mock_embedder = MagicMock()
            mock_embedder.batch_encode.return_value = []
            mock_embedder_cls.return_value = mock_embedder

            components = algo.build(
                workspace=tmp_path,
                provider=mock_provider,
                model="test-model",
                sessions=mock_sessions,
                context_window_tokens=128_000,
                build_messages=MagicMock(),
                get_tool_definitions=MagicMock(),
                max_completion_tokens=4096,
                session_ttl_minutes=15,
                max_batch_size=20,
                max_iterations=10,
                max_tool_result_chars=16000,
                annotate_line_ages=True,
            )

            assert components.auto_compact is not None

    def test_build_respects_parameters(
        self, tmp_path, mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        """build() should pass parameters through to components."""
        algo = EMemMemoryAlgorithm()

        with patch(
            "nanobot.memory.emem_memory.embedding.EMemEmbedder",
            autospec=True,
        ) as mock_embedder_cls:
            mock_embedder = MagicMock()
            mock_embedder.batch_encode.return_value = []
            mock_embedder_cls.return_value = mock_embedder

            components = algo.build(
                workspace=tmp_path,
                provider=mock_provider,
                model="custom-model",
                sessions=mock_sessions,
                context_window_tokens=64_000,
                build_messages=MagicMock(),
                get_tool_definitions=MagicMock(),
                max_completion_tokens=2048,
                session_ttl_minutes=30,
                max_batch_size=50,
                max_iterations=5,
                max_tool_result_chars=8000,
                annotate_line_ages=False,
            )

            assert components.consolidator.model == "custom-model"
            assert components.consolidator.context_window_tokens == 64_000
            assert components.consolidator.max_completion_tokens == 2048
            assert components.dream.max_batch_size == 50
            assert components.dream.max_iterations == 5
            assert components.dream.max_tool_result_chars == 8000
            assert components.dream.annotate_line_ages is False
            assert components.auto_compact._ttl == 30

    def test_build_with_minimal_provider(
        self, tmp_path, mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        """build() should work even if provider lacks generation attrs."""
        algo = EMemMemoryAlgorithm()
        # Remove generation attribute
        del mock_provider.generation
        mock_provider.api_key = "test-key"
        mock_provider.api_base = None

        with patch(
            "nanobot.memory.emem_memory.embedding.EMemEmbedder",
            autospec=True,
        ) as mock_embedder_cls:
            mock_embedder = MagicMock()
            mock_embedder.batch_encode.return_value = []
            mock_embedder_cls.return_value = mock_embedder

            components = algo.build(
                workspace=tmp_path,
                provider=mock_provider,
                model="test-model",
                sessions=mock_sessions,
                context_window_tokens=128_000,
                build_messages=MagicMock(),
                get_tool_definitions=MagicMock(),
                max_completion_tokens=4096,
                session_ttl_minutes=15,
                max_batch_size=20,
                max_iterations=10,
                max_tool_result_chars=16000,
                annotate_line_ages=True,
            )

            assert isinstance(components, MemoryComponents)

    def test_build_with_custom_config(
        self, tmp_path, mock_provider: MagicMock, mock_sessions: MagicMock,
    ) -> None:
        """build() should respect EMemConfig settings."""
        config = EMemConfig(
            skip_ppr=True,
            skip_edu_context_gen=True,
            embedding_batch_size=8,
            embedding_return_as_normalized=False,
        )
        algo = EMemMemoryAlgorithm(config=config)

        with patch(
            "nanobot.memory.emem_memory.embedding.EMemEmbedder",
            autospec=True,
        ) as mock_embedder_cls:
            mock_embedder = MagicMock()
            mock_embedder.batch_encode.return_value = []
            mock_embedder_cls.return_value = mock_embedder

            components = algo.build(
                workspace=tmp_path,
                provider=mock_provider,
                model="test-model",
                sessions=mock_sessions,
                context_window_tokens=128_000,
                build_messages=MagicMock(),
                get_tool_definitions=MagicMock(),
                max_completion_tokens=4096,
                session_ttl_minutes=15,
                max_batch_size=20,
                max_iterations=10,
                max_tool_result_chars=16000,
                annotate_line_ages=True,
            )

            assert isinstance(components, MemoryComponents)


# ===================================================================
# EMemMemoryAlgorithm — registry
# ===================================================================

class TestEMemMemoryAlgorithmRegistry:
    """Test registry integration."""

    def test_algorithm_registers_in_registry(self) -> None:
        from nanobot.memory.registry import MemoryRegistry

        registry = MemoryRegistry()
        registry.register(EMemMemoryAlgorithm())
        algo = registry.get("emem_memory")
        assert algo is not None
        assert algo.name == "emem_memory"

    def test_algorithm_overrides_registry(self) -> None:
        from nanobot.memory.registry import MemoryRegistry

        registry = MemoryRegistry()
        algo1 = EMemMemoryAlgorithm(EMemConfig(damping=0.5))
        registry.register(algo1)
        algo2 = EMemMemoryAlgorithm(EMemConfig(damping=0.9))
        registry.register(algo2)
        retrieved = registry.get("emem_memory")
        assert retrieved.config.damping == 0.9
