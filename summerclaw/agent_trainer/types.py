"""Agent Trainer — common types for the pluggable training pipeline.

Shared dataclass definitions for the 6-stage per-step pipeline:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject

All types support round-trip conversion to/from plain dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Literal


# ── Atomic types ─────────────────────────────────────────────────────────

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
    """A single edit operation on a skill document."""

    op: EditOp
    content: str = ""
    target: str = ""
    support_count: int | None = None
    source_type: Literal["failure", "success"] | None = None
    merge_level: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        return cls(
            op=d.get("op", "append"),
            content=d.get("content", ""),
            target=d.get("target", ""),
            support_count=d.get("support_count"),
            source_type=d.get("source_type"),
            merge_level=d.get("merge_level"),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target:
            d["target"] = self.target
        if self.support_count is not None:
            d["support_count"] = self.support_count
        if self.source_type is not None:
            d["source_type"] = self.source_type
        if self.merge_level is not None:
            d["merge_level"] = self.merge_level
        return d


@dataclass
class Patch:
    """A set of edits with reasoning.

    Output of Aggregate, Select; input to Update.
    """

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        edits_raw = d.get("edits", [])
        return cls(
            edits=[Edit.from_dict(e) if isinstance(e, dict) else e for e in edits_raw],
            reasoning=d.get("reasoning", ""),
            ranking_details=d.get("ranking_details"),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "reasoning": self.reasoning,
            "edits": [e.to_dict() if isinstance(e, Edit) else e for e in self.edits],
        }
        if self.ranking_details is not None:
            d["ranking_details"] = self.ranking_details
        return d


# ── Stage 1: ROLLOUT ─────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Result of a single episode/task rollout."""

    id: str
    hard: int
    soft: float
    n_turns: int = 0
    fail_reason: str = ""
    task_type: str = ""
    task_description: str = ""
    predicted_answer: str = ""
    question: str = ""
    reference_text: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    @classmethod
    def _get_known_fields(cls) -> frozenset[str]:
        if cls._KNOWN_FIELDS is None:
            cls._KNOWN_FIELDS = frozenset(
                f.name for f in dc_fields(cls)
                if f.name != "_KNOWN_FIELDS"
            )
        return cls._KNOWN_FIELDS

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        known = cls._get_known_fields()
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")),
            hard=int(d.get("hard", 0)),
            soft=float(d.get("soft", 0.0)),
            n_turns=int(d.get("n_turns", 0)),
            fail_reason=str(d.get("fail_reason", "")),
            task_type=str(d.get("task_type", "")),
            task_description=str(d.get("task_description", "")),
            predicted_answer=str(d.get("predicted_answer", "")),
            question=str(d.get("question", "")),
            reference_text=str(d.get("reference_text", "")),
            trajectory=d.get("trajectory", []),
            extras=extras,
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "hard": self.hard,
            "soft": self.soft,
        }
        for attr in (
            "n_turns", "fail_reason", "task_type", "task_description",
            "predicted_answer", "question", "reference_text",
        ):
            val = getattr(self, attr)
            if val:
                d[attr] = val
        if self.trajectory:
            d["trajectory"] = self.trajectory
        d.update(self.extras)
        return d


# ── Stage 2: REFLECT ─────────────────────────────────────────────────────

@dataclass
class FailureSummaryEntry:
    """One entry in the failure summary produced by error analysts."""

    failure_type: str
    count: int = 0
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> FailureSummaryEntry:
        return cls(
            failure_type=d.get("failure_type", ""),
            count=int(d.get("count", 0)),
            description=d.get("description", ""),
        )

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "description": self.description,
        }


@dataclass
class RawPatch:
    """Analyst output from the Reflect stage — a patch with provenance."""

    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
    batch_size: int = 0
    failure_summary: list[FailureSummaryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> RawPatch | None:
        if d is None:
            return None
        inner = d.get("patch", d)
        if not isinstance(inner, dict):
            return None
        patch = Patch.from_dict(inner)
        return cls(
            patch=patch,
            source_type=d.get("source_type", "failure"),
            batch_size=int(d.get("batch_size", 0)),
            failure_summary=[
                FailureSummaryEntry.from_dict(fs)
                for fs in d.get("failure_summary", [])
            ],
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "patch": self.patch.to_dict(),
            "source_type": self.source_type,
            "batch_size": self.batch_size,
        }
        if self.failure_summary:
            d["failure_summary"] = [fs.to_dict() for fs in self.failure_summary]
        return d


# ── Stage 6: EVALUATE (Gate) ─────────────────────────────────────────────

GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass(frozen=True)
class GateResult:
    """Immutable outcome of the validation gate."""

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


# ── Training metadata ────────────────────────────────────────────────────

@dataclass
class TrainingStep:
    """Single-step training snapshot.

    Fields marked with ``step_rec`` are aligned with the official SkillOpt
    ``step_record.json`` schema for parity analysis.
    """

    step: int
    epoch: int
    score: float
    action: str  # accept_new_best / accept / reject
    skill_hash: str = ""
    n_edits_applied: int = 0
    n_edits_rejected: int = 0
    # --- step_rec alignment fields ---
    step_in_epoch: int = 0
    timing: dict = field(default_factory=dict)
    rollout_hard: float = 0.0
    rollout_soft: float = 0.0
    rollout_n: int = 0
    n_patches: int = 0
    n_failure_patches: int = 0
    n_success_patches: int = 0
    n_edits_merged: int = 0
    edit_budget: int = 0
    lr_control_mode: str = ""
    selection_hard: float = 0.0
    selection_soft: float = 0.0
    candidate_skill_len: int = 0
    current_score: float = 0.0
    best_score: float = 0.0
    best_step: int = 0
    current_origin: str = ""
    best_origin: str = ""
    skill_len: int = 0
    wall_time_s: float = 0.0
    edit_apply_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "score": self.score,
            "action": self.action,
            "skill_hash": self.skill_hash,
            "n_edits_applied": self.n_edits_applied,
            "n_edits_rejected": self.n_edits_rejected,
            "step_in_epoch": self.step_in_epoch,
            "timing": self.timing,
            "rollout_hard": self.rollout_hard,
            "rollout_soft": self.rollout_soft,
            "rollout_n": self.rollout_n,
            "n_patches": self.n_patches,
            "n_failure_patches": self.n_failure_patches,
            "n_success_patches": self.n_success_patches,
            "n_edits_merged": self.n_edits_merged,
            "edit_budget": self.edit_budget,
            "lr_control_mode": self.lr_control_mode,
            "selection_hard": self.selection_hard,
            "selection_soft": self.selection_soft,
            "candidate_skill_len": self.candidate_skill_len,
            "current_score": self.current_score,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "current_origin": self.current_origin,
            "best_origin": self.best_origin,
            "skill_len": self.skill_len,
            "wall_time_s": self.wall_time_s,
            "edit_apply_summary": self.edit_apply_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingStep:
        return cls(
            step=d.get("step", 0),
            epoch=d.get("epoch", 0),
            score=d.get("score", 0.0),
            action=d.get("action", ""),
            skill_hash=d.get("skill_hash", ""),
            n_edits_applied=d.get("n_edits_applied", 0),
            n_edits_rejected=d.get("n_edits_rejected", 0),
            step_in_epoch=d.get("step_in_epoch", 0),
            timing=d.get("timing", {}),
            rollout_hard=d.get("rollout_hard", 0.0),
            rollout_soft=d.get("rollout_soft", 0.0),
            rollout_n=d.get("rollout_n", 0),
            n_patches=d.get("n_patches", 0),
            n_failure_patches=d.get("n_failure_patches", 0),
            n_success_patches=d.get("n_success_patches", 0),
            n_edits_merged=d.get("n_edits_merged", 0),
            edit_budget=d.get("edit_budget", 0),
            lr_control_mode=d.get("lr_control_mode", ""),
            selection_hard=d.get("selection_hard", 0.0),
            selection_soft=d.get("selection_soft", 0.0),
            candidate_skill_len=d.get("candidate_skill_len", 0),
            current_score=d.get("current_score", 0.0),
            best_score=d.get("best_score", 0.0),
            best_step=d.get("best_step", 0),
            current_origin=d.get("current_origin", ""),
            best_origin=d.get("best_origin", ""),
            skill_len=d.get("skill_len", 0),
            wall_time_s=d.get("wall_time_s", 0.0),
            edit_apply_summary=d.get("edit_apply_summary", {}),
        )


@dataclass
class TrainingHistory:
    """Training history — a list of step snapshots."""

    steps: list[TrainingStep] = field(default_factory=list)
    best_score: float = 0.0
    best_step: int = 0
    total_epochs: int = 0
    total_steps: int = 0

    def add_step(self, step: TrainingStep) -> None:
        self.steps.append(step)
        self.total_steps = len(self.steps)
        if step.score > self.best_score:
            self.best_score = step.score
            self.best_step = step.step

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "best_score": self.best_score,
            "best_step": self.best_step,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingHistory:
        return cls(
            steps=[TrainingStep.from_dict(s) for s in d.get("steps", [])],
            best_score=d.get("best_score", 0.0),
            best_step=d.get("best_step", 0),
            total_epochs=d.get("total_epochs", 0),
            total_steps=d.get("total_steps", 0),
        )
