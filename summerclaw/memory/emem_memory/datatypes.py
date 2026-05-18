"""EMem data models — EDU, Session, and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import md5
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core EDU data types
# ---------------------------------------------------------------------------


@dataclass
class EDURecord:
    """An Elementary Discourse Unit — atomic proposition from conversation.

    Attributes:
        edu_id: Unique hash identifier for this EDU.
        text: The EDU text content.
        source_speakers: List of speaker names who contributed to this EDU.
        timestamp: When the source conversation turn occurred.
        session_id: The session this EDU belongs to.
        event_type: Optional event type classification.
        event_triggers: Optional list of trigger words/phrases.
        event_role_argument_pairs: Optional list of {"role": ..., "argument": ...} dicts.
        context_text: Optional surrounding context text.
        metadata: Arbitrary additional metadata.
    """

    edu_id: str
    text: str
    source_speakers: list[str] = field(default_factory=list)
    timestamp: datetime | None = None
    session_id: str = ""
    event_type: str | None = None
    event_triggers: list[str] | None = None
    event_role_argument_pairs: list[dict[str, str]] | None = None
    context_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def compute_id(text: str, prefix: str = "edu-") -> str:
        """Compute a deterministic hash ID for an EDU from its text."""
        return prefix + md5(text.encode()).hexdigest()

    def to_context_string(self, date_format: str = "iso") -> str:
        """Format EDU as a context string with metadata for QA/retrieval.

        Args:
            date_format: Format style — "iso" or "locomo".
        """
        speakers_str = ", ".join(self.source_speakers) if self.source_speakers else "Unknown"
        if self.timestamp:
            if date_format == "locomo":
                date_str = self.timestamp.strftime("%-I:%M %p on %-d %B, %Y")
            else:
                date_str = self.timestamp.isoformat()
        else:
            date_str = "unknown date"
        return (
            f"[Source date: {date_str} - Speakers: {speakers_str}] "
            f'"{self.text}"'
        )


@dataclass
class ArgumentRecord:
    """An argument/entity node extracted from EDUs.

    Attributes:
        arg_id: Unique hash identifier.
        text: The argument text (e.g. entity name, value).
        source_edu_ids: EDU IDs that reference this argument.
    """

    arg_id: str
    text: str
    source_edu_ids: list[str] = field(default_factory=list)

    @staticmethod
    def compute_id(text: str) -> str:
        return "argument-" + md5(text.encode()).hexdigest()


@dataclass
class SessionRecord:
    """Represents a conversation session (a batch of turns).

    Attributes:
        session_id: Unique session identifier (hash).
        turns: List of turn dicts with role/content/timestamp.
        summary: Optional session summary text.
        date: Optional session date.
    """

    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    date: datetime | None = None

    @staticmethod
    def compute_id(key: str) -> str:
        return "session-" + md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EMemConfig:
    """Configuration for EMem memory algorithm.

    Attributes:
        linking_top_k: Number of top EDUs/arguments to keep after linking.
        retrieval_top_k: Number of EDUs to retrieve per query.
        qa_top_k: Number of top EDUs fed to QA model.
        damping: PPR damping factor (0–1). Higher = more teleportation.
        synonymy_edge_topk: K for KNN synonymy edge construction.
        synonymy_edge_sim_threshold: Similarity threshold for synonymy edges.
        synonymy_edge_query_batch_size: Query batch size for KNN.
        synonymy_edge_key_batch_size: Key batch size for KNN.
        skip_ppr: If True, skip PPR graph propagation (EMem mode).
        skip_edu_context_gen: If True, skip context generation for EDUs.
        passage_node_weight: Weight multiplier for session nodes in PPR.
        embedding_batch_size: Batch size for embedding model calls.
        embedding_return_as_normalized: Whether to normalize embeddings.
        force_reindex: If True, rebuild all indices from scratch.
        date_format_type: Date format style ("iso" or "locomo").
    """

    linking_top_k: int = 5
    retrieval_top_k: int = 200
    qa_top_k: int = 5
    damping: float = 0.5
    synonymy_edge_topk: int = 2047
    synonymy_edge_sim_threshold: float = 0.8
    synonymy_edge_query_batch_size: int = 1000
    synonymy_edge_key_batch_size: int = 10000
    skip_ppr: bool = False
    skip_edu_context_gen: bool = True
    passage_node_weight: float = 0.05
    embedding_batch_size: int = 16
    embedding_return_as_normalized: bool = True
    force_reindex: bool = False
    date_format_type: str = "iso"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash of content with optional prefix."""
    return prefix + md5(content.encode()).hexdigest()


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1].

    Returns an array of ones if all values are identical.
    """
    min_val = np.min(x)
    max_val = np.max(x)
    rng = max_val - min_val
    if rng == 0:
        return np.ones_like(x, dtype=np.float64)
    return (x - min_val) / rng
