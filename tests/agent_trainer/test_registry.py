"""Unit tests for agent_trainer registry module."""
from __future__ import annotations

import pytest

from summerclaw.agent_trainer.base import BaseAlgorithm
from summerclaw.agent_trainer.registry import (
    _REGISTRY,
    algorithm,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


class _DummyAlgorithm(BaseAlgorithm):
    name = "dummy"

    async def rollout(self, env, skill, items, out_dir):
        return []

    async def reflect(self, results, skill, out_dir):
        return []

    async def aggregate(self, patches, skill):
        from summerclaw.agent_trainer.types import Patch
        return Patch()

    async def select(self, patch, budget, skill):
        return patch

    async def update(self, skill, patch):
        return skill, []

    async def evaluate(self, env, skill, items, out_dir):
        return 0.0


class TestRegistry:
    def test_register_and_get(self):
        register_algorithm("test_dummy", _DummyAlgorithm)
        cls = get_algorithm("test_dummy")
        assert cls is _DummyAlgorithm

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown algorithm"):
            get_algorithm("nonexistent_xyz")

    def test_duplicate_register_same_class(self):
        register_algorithm("test_dup", _DummyAlgorithm)
        register_algorithm("test_dup", _DummyAlgorithm)  # idempotent
        assert get_algorithm("test_dup") is _DummyAlgorithm

    def test_duplicate_register_different_class_raises(self):
        register_algorithm("test_conflict", _DummyAlgorithm)

        class OtherAlgo(_DummyAlgorithm):
            pass

        with pytest.raises(ValueError, match="already registered"):
            register_algorithm("test_conflict", OtherAlgo)

    def test_list_algorithms(self):
        register_algorithm("alpha_test", _DummyAlgorithm)
        names = list_algorithms()
        assert "alpha_test" in names

    def test_decorator(self):
        @algorithm("decorated_test")
        class DecoratedAlgo(BaseAlgorithm):
            name = "decorated_test"

            async def rollout(self, env, skill, items, out_dir):
                return []

            async def reflect(self, results, skill, out_dir):
                return []

            async def aggregate(self, patches, skill):
                from summerclaw.agent_trainer.types import Patch
                return Patch()

            async def select(self, patch, budget, skill):
                return patch

            async def update(self, skill, patch):
                return skill, []

            async def evaluate(self, env, skill, items, out_dir):
                return 0.0

        assert get_algorithm("decorated_test") is DecoratedAlgo
        assert DecoratedAlgo.name == "decorated_test"
