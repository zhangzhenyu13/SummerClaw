"""Tests for NemoriMemoryAlgorithm — MemoryAlgorithm integration.

Verifies that build() constructs proper MemoryComponents with the correct
types for all sub-components.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from summerclaw.memory import MemoryComponents
from summerclaw.memory.nemori_memory import NemoriMemoryAlgorithm
from summerclaw.memory.nemori_memory.consolidator import NemoriConsolidator
from summerclaw.memory.nemori_memory.dream import NemoriDream
from summerclaw.memory.nemori_memory.store import NemoriStore


@pytest.fixture
def mock_provider():
    p = MagicMock()
    return p


@pytest.fixture
def algo():
    return NemoriMemoryAlgorithm()


class TestNemoriMemoryAlgorithm:
    """NemoriMemoryAlgorithm — build and component verification."""

    def test_name_is_nemori_memory(self, algo):
        assert algo.name == "nemori_memory"

    def test_build_produces_required_components(self, algo, tmp_path, mock_provider):
        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="test-model",
            sessions=MagicMock(),
            context_window_tokens=100000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=0,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=False,
        )

        assert isinstance(components, MemoryComponents)
        assert isinstance(components.store, NemoriStore)
        assert isinstance(components.consolidator, NemoriConsolidator)
        assert isinstance(components.dream, NemoriDream)
        assert components.auto_compact is None

    def test_build_store_uses_file_backend(self, algo, tmp_path, mock_provider):
        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="test-model",
            sessions=MagicMock(),
            context_window_tokens=100000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=0,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=False,
        )
        assert components.store._backend == "file"

    def test_build_creates_data_directory(self, algo, tmp_path, mock_provider):
        algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="test-model",
            sessions=MagicMock(),
            context_window_tokens=100000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=0,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=False,
        )
        data_dir = tmp_path / "memory" / "nemori_memory"
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_build_consolidator_has_correct_config(self, algo, tmp_path, mock_provider):
        components = algo.build(
            workspace=tmp_path,
            provider=mock_provider,
            model="test-model",
            sessions=MagicMock(),
            context_window_tokens=100000,
            build_messages=MagicMock(),
            get_tool_definitions=MagicMock(),
            max_completion_tokens=4096,
            session_ttl_minutes=0,
            max_batch_size=20,
            max_iterations=10,
            max_tool_result_chars=16000,
            annotate_line_ages=False,
        )
        c = components.consolidator
        assert c._buffer_size_min == 2
        assert c._batch_threshold == 10
        assert c._episode_min_messages == 2
        assert c._enable_semantic is True
        assert c._enable_merging is True

    def test_register_in_memory_registry(self, algo):
        """NemoriMemoryAlgorithm should be registrable in MemoryRegistry."""
        from summerclaw.memory.registry import MemoryRegistry
        registry = MemoryRegistry()
        registry.register(algo)
        assert "nemori_memory" in registry.list()
        retrieved = registry.get("nemori_memory")
        assert retrieved.name == "nemori_memory"
