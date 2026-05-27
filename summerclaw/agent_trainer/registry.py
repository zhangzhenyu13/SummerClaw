"""Algorithm registry — name-to-class mapping for pluggable algorithms.

Supports two registration patterns:
1. Direct: ``register_algorithm("skillopt", SkillOptAlgorithm)``
2. Decorator: ``@algorithm("skillopt")``

Usage::

    from summerclaw.agent_trainer.registry import algorithm, get_algorithm

    @algorithm("skillopt")
    class SkillOptAlgorithm(BaseAlgorithm):
        ...

    cls = get_algorithm("skillopt")
    algo = cls()
"""
from __future__ import annotations

from typing import Type

from summerclaw.agent_trainer.base import BaseAlgorithm

_REGISTRY: dict[str, Type[BaseAlgorithm]] = {}


def register_algorithm(name: str, cls: Type[BaseAlgorithm]) -> None:
    """Register an algorithm class under the given name."""
    if name in _REGISTRY:
        existing = _REGISTRY[name]
        if existing is cls:
            return  # idempotent re-registration
        raise ValueError(
            f"Algorithm '{name}' already registered as {existing.__name__}; "
            f"cannot re-register as {cls.__name__}"
        )
    _REGISTRY[name] = cls


def get_algorithm(name: str) -> Type[BaseAlgorithm]:
    """Look up an algorithm class by name.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {available}"
        )
    return cls


def list_algorithms() -> list[str]:
    """Return sorted list of registered algorithm names."""
    return sorted(_REGISTRY)


def algorithm(name: str):
    """Decorator that registers a BaseAlgorithm subclass under *name*.

    Usage::

        @algorithm("skillopt")
        class SkillOptAlgorithm(BaseAlgorithm):
            ...
    """

    def decorator(cls: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
        cls.name = name
        register_algorithm(name, cls)
        return cls

    return decorator
