"""Memory algorithm registry — maps algorithm names to MemoryAlgorithm instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory.base import MemoryAlgorithm


class MemoryRegistry:
    """Registry for memory algorithms.

    Usage::

        registry = MemoryRegistry()
        registry.register(NaiveMemoryAlgorithm())

        algo = registry.get("naive_memory")
        components = algo.build(...)
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, "MemoryAlgorithm"] = {}

    def register(self, algorithm: "MemoryAlgorithm") -> None:
        """Register a memory algorithm.

        Args:
            algorithm: A MemoryAlgorithm instance with a unique name.
        """
        name = algorithm.name
        if name in self._algorithms:
            logger.warning(
                "Memory algorithm '{}' already registered, overwriting", name
            )
        self._algorithms[name] = algorithm
        logger.debug("Registered memory algorithm: {}", name)

    def get(self, name: str) -> "MemoryAlgorithm":
        """Look up a memory algorithm by name.

        Args:
            name: The algorithm name (e.g. ``"naive_memory"``).

        Returns:
            The registered MemoryAlgorithm instance.

        Raises:
            KeyError: If no algorithm is registered under *name*.
        """
        if name not in self._algorithms:
            available = ", ".join(sorted(self._algorithms.keys())) or "(none)"
            raise KeyError(
                f"Unknown memory algorithm '{name}'. "
                f"Available: {available}"
            )
        return self._algorithms[name]

    def list(self) -> list[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(self._algorithms.keys())

    @property
    def default_name(self) -> str:
        """Return the name of the default algorithm.

        The default algorithm is always ``"naive_memory"``.
        """
        return "naive_memory"
