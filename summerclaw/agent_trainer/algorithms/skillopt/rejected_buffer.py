"""SkillOpt Rejected Buffer — track rejected edits to avoid repetition.

When the validation gate rejects a candidate skill, the applied edits and
their score delta are stored in a circular buffer.  Subsequent Reflect
and Aggregate stages receive this context so the LLM can avoid
generating similar (likely-failing) edits again.

Design notes
------------
- The buffer is **epoch-local** (per the official SkillOpt paper): it is
  cleared at the start of each epoch via ``on_epoch_start`` and
  accumulates only within the current epoch.
- Each entry stores a compact summary of the rejected patch **and** the
  observed failure patterns from that step, so the formatted context
  stays within the token budget while providing rich negative feedback.
- The formatted text is injected into the LLM user prompt alongside
  ``meta_skill_context`` and ``step_buffer_context``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RejectedEntry:
    """One rejected patch record."""

    step: int
    score_before: float
    score_after: float
    edits_summary: list[str] = field(default_factory=list)
    failure_patterns: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "edits_summary": list(self.edits_summary),
            "failure_patterns": list(self.failure_patterns),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RejectedEntry:
        return cls(
            step=d.get("step", 0),
            score_before=d.get("score_before", 0.0),
            score_after=d.get("score_after", 0.0),
            edits_summary=d.get("edits_summary", []),
            failure_patterns=d.get("failure_patterns", []),
        )


class RejectedBuffer:
    """Circular buffer of rejected patch summaries.

    Parameters
    ----------
    max_size : int
        Maximum number of rejected entries to retain.  When the buffer
        is full, the oldest entry is evicted (FIFO).
    max_summary_chars : int
        Maximum character length for each edit summary string.  Longer
        summaries are truncated with an ellipsis.
    """

    def __init__(
        self,
        max_size: int = 10,
        max_summary_chars: int = 200,
    ) -> None:
        self.max_size = max_size
        self.max_summary_chars = max_summary_chars
        self._entries: list[RejectedEntry] = []

    # ── Mutators ──────────────────────────────────────────────────────

    def add(
        self,
        step: int,
        edits: list[Any],
        score_before: float,
        score_after: float,
        failure_patterns: list[dict] | None = None,
    ) -> None:
        """Record a rejected patch.

        Parameters
        ----------
        step : int
            Global step at which the rejection occurred.
        edits : list
            The list of ``Edit`` objects (or dicts) from the rejected patch.
        score_before : float
            Current skill score before the update attempt.
        score_after : float
            Candidate skill score that was rejected.
        failure_patterns : list[dict] | None
            Extracted failure patterns from the rollout that produced
            the rejected candidate.  Stored in the buffer alongside the
            rejected edits for richer negative feedback (aligned with
            official SkillOpt paper: "observed failure patterns").
        """
        summaries: list[str] = []
        for edit in edits:
            if hasattr(edit, "op") and hasattr(edit, "content"):
                # Edit dataclass
                text = f"{edit.op}: {edit.content[:self.max_summary_chars]}"
                if hasattr(edit, "target") and edit.target:
                    text += f" (target: {edit.target[:60]})"
            elif isinstance(edit, dict):
                op = edit.get("op", "?")
                content = edit.get("content", "")[:self.max_summary_chars]
                target = edit.get("target", "")
                text = f"{op}: {content}"
                if target:
                    text += f" (target: {target[:60]})"
            else:
                text = str(edit)[:self.max_summary_chars]
            summaries.append(text)

        entry = RejectedEntry(
            step=step,
            score_before=score_before,
            score_after=score_after,
            edits_summary=summaries,
            failure_patterns=list(failure_patterns or []),
        )
        self._entries.append(entry)

        # FIFO eviction
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size:]

        logger.info(
            "[REJECTED_BUFFER] added step={} (score {:.3f}→{:.3f}, {} edits, buffer={}/{})",
            step, score_before, score_after,
            len(summaries), len(self._entries), self.max_size,
        )

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def entries(self) -> list[RejectedEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    # ── Formatting ────────────────────────────────────────────────────

    def format_context(self) -> str:
        """Format buffer contents as prompt text for LLM injection.

        Returns an empty string when the buffer is empty.
        """
        if not self._entries:
            return ""

        lines: list[str] = []
        for entry in self._entries:
            delta = entry.score_after - entry.score_before
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"[Step {entry.step}] "
                f"score={entry.score_before:.3f}→{entry.score_after:.3f} "
                f"(delta={sign}{delta:.3f})"
            )
            # Failure patterns (observed issues from rollout)
            if entry.failure_patterns:
                for fp in entry.failure_patterns[:3]:  # cap at 3 per entry
                    pattern = fp.get("pattern", str(fp))
                    lines.append(f"  [failure] {pattern[:120]}")
            # Rejected edits
            for summary in entry.edits_summary:
                lines.append(f"  - {summary}")

        header = (
            "## Previously Rejected Edits (this epoch)\n"
            "The following edits were proposed earlier in this epoch but **rejected** "
            "by the validation gate (candidate score did not improve).  "
            "Avoid generating similar or identical edits.\n"
        )
        return header + "\n".join(lines)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "max_summary_chars": self.max_summary_chars,
            "entries": [e.to_dict() for e in self._entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RejectedBuffer:
        buf = cls(
            max_size=d.get("max_size", 10),
            max_summary_chars=d.get("max_summary_chars", 200),
        )
        buf._entries = [
            RejectedEntry.from_dict(e) for e in d.get("entries", [])
        ]
        return buf
