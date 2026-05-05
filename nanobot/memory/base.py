"""Memory algorithm abstract base and component container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


@dataclass
class MemoryComponents:
    """Container for all components produced by a MemoryAlgorithm.

    Attributes:
        store: The memory store (file I/O layer).
        consolidator: The consolidator (online token-budget compression).
        dream: The dream processor (offline cron-scheduled deep processing).
        auto_compact: Optional auto-compact (idle session compression).
    """

    store: Any
    consolidator: Any
    dream: Any
    auto_compact: Any | None = None


class MemoryAlgorithm(ABC):
    """Abstract base class for memory algorithms.

    Each memory algorithm is responsible for producing a complete set of
    memory components (store, consolidator, dream, auto_compact).  Different
    algorithms can implement different storage backends, consolidation
    strategies, or dream processing pipelines.

    Subclasses must set a unique ``name`` class attribute and implement
    ``build()``.
    """

    name: str

    @abstractmethod
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
        """Build and return all memory components for this algorithm.

        Args:
            workspace: The agent workspace path.
            provider: The LLM provider.
            model: The model name.
            sessions: The session manager.
            context_window_tokens: Total context window size in tokens.
            build_messages: Callable for building messages.
            get_tool_definitions: Callable for getting tool definitions.
            max_completion_tokens: Max completion tokens budget.
            session_ttl_minutes: Session TTL for auto-compact.
            max_batch_size: Dream max batch size.
            max_iterations: Dream max tool iterations.
            max_tool_result_chars: Max chars for tool results.
            annotate_line_ages: Whether Dream annotates line ages.
            embedding_config: Optional EmbeddingConfig for embedding model
                (used by memory algorithms that require embeddings, e.g. EMem).

        Returns:
            A fully-initialized MemoryComponents container.
        """
        ...
