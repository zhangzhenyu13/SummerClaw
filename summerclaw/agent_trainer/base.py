"""Abstract base class for pluggable training algorithms.

Every training algorithm (SkillOpt, DSPy, TextGrad, etc.) must subclass
:class:`BaseAlgorithm` and implement all abstract methods.  The
:class:`~summerclaw.agent_trainer.engine.trainer.TrainerEngine` calls
these methods at the appropriate pipeline stages.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from summerclaw.agent_trainer.types import (
        GateResult,
        Patch,
        RawPatch,
        RolloutResult,
    )


class BaseAlgorithm(ABC):
    """Abstract base for all pluggable training algorithms.

    Subclasses must set a unique ``name`` attribute and implement all
    abstract methods.  The trainer engine calls these at the appropriate
    stages of the 6-phase per-step pipeline.
    """

    name: str = ""

    # ── Per-step pipeline (6 stages) ────────────────────────────────────

    @abstractmethod
    async def rollout(
        self,
        env: Any,
        skill: str,
        items: list[dict],
        out_dir: str,
    ) -> list[RolloutResult]:
        """Stage 1: Execute episodes with the current skill.

        Parameters
        ----------
        env : SummerClawEnvAdapter
            The environment adapter.
        skill : str
            Current skill document content.
        items : list[dict]
            Training items for this batch.
        out_dir : str
            Output directory for artifacts.

        Returns
        -------
        list[RolloutResult]
            Per-item rollout results.
        """

    @abstractmethod
    async def reflect(
        self,
        results: list[RolloutResult],
        skill: str,
        out_dir: str,
    ) -> list[RawPatch]:
        """Stage 2: Analyze rollout trajectories and produce patches.

        Parameters
        ----------
        results : list[RolloutResult]
            Rollout results from stage 1.
        skill : str
            Current skill document content.
        out_dir : str
            Output directory for patch artifacts.

        Returns
        -------
        list[RawPatch]
            Analyst outputs (patches with provenance).
        """

    @abstractmethod
    async def aggregate(
        self,
        patches: list[RawPatch],
        skill: str,
    ) -> Patch:
        """Stage 3: Hierarchical merge of independently-generated patches.

        Parameters
        ----------
        patches : list[RawPatch]
            Raw patches from reflect stage.
        skill : str
            Current skill document content.

        Returns
        -------
        Patch
            Merged patch with edits and reasoning.
        """

    @abstractmethod
    async def select(
        self,
        patch: Patch,
        budget: int,
        skill: str,
    ) -> Patch:
        """Stage 4: Rank edits by importance and select top-L (gradient clipping).

        Parameters
        ----------
        patch : Patch
            Aggregated patch.
        budget : int
            Maximum number of edits to keep.
        skill : str
            Current skill document content.

        Returns
        -------
        Patch
            Selected patch with ranking details.
        """

    @abstractmethod
    async def update(
        self,
        skill: str,
        patch: Patch,
    ) -> tuple[str, list[dict]]:
        """Stage 5: Apply selected edits to the skill document (optimizer step).

        Parameters
        ----------
        skill : str
            Current skill document content.
        patch : Patch
            Selected patch from stage 4.

        Returns
        -------
        tuple[str, list[dict]]
            (updated_skill, per-edit_report)
        """

    @abstractmethod
    async def evaluate(
        self,
        env: Any,
        skill: str,
        items: list[dict],
        out_dir: str,
    ) -> float:
        """Stage 6: Evaluate the candidate skill on the validation split.

        Parameters
        ----------
        env : SummerClawEnvAdapter
            The environment adapter.
        skill : str
            Candidate skill document content.
        items : list[dict]
            Validation items.
        out_dir : str
            Output directory for evaluation artifacts.

        Returns
        -------
        float
            Hard accuracy score on validation set.
        """

    # ── Per-step budget ─────────────────────────────────────────────────

    def get_edit_budget(self, step: int, total_steps: int) -> int:
        """Return the per-step edit budget.

        Override in subclasses to implement learning-rate scheduling.
        Default: ``self.edit_budget`` (constant).

        Parameters
        ----------
        step : int
            Current global step (1-indexed).
        total_steps : int
            Total number of steps in training.
        """
        return getattr(self, "edit_budget", 4)

    # ── Rejected buffer hook ────────────────────────────────────────────

    def record_rejection(
        self,
        step: int,
        patch: Any,
        score_before: float,
        score_after: float,
        failure_patterns: list[dict] | None = None,
    ) -> None:
        """Optional hook called when the gate rejects a candidate.

        Override in subclasses to track rejected edits for negative
        feedback (e.g. :class:`RejectedBuffer`).  Default: no-op.

        Parameters
        ----------
        step : int
            Global step at which the rejection occurred.
        patch : Patch
            The selected patch whose candidate was rejected.
        score_before : float
            Current skill score before the update attempt.
        score_after : float
            Candidate score that was rejected.
        failure_patterns : list[dict] | None
            Extracted failure patterns from the rollout that produced
            the rejected candidate.  Stored in the buffer alongside the
            rejected edits for richer negative feedback.
        """
        pass

    # ── Epoch-level hooks ───────────────────────────────────────────────

    def on_epoch_start(self, epoch: int) -> None:
        """Optional hook called at the start of each epoch.

        Override in subclasses to reset per-epoch state (e.g. clear
        epoch-local buffers).  Default: no-op.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based).
        """
        pass

    async def on_epoch_end(
        self,
        epoch: int,
        history: Any,
        skill: str,
        *,
        prev_results: Any = None,
        curr_results: Any = None,
        items: list[dict] | None = None,
        out_dir: str = "",
    ) -> str:
        """Optional epoch-level hook (slow update, meta skill, etc.).

        Called after all steps in an epoch are complete.  May return a
        modified skill document.  Default: return skill unchanged.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based).
        history : Any
            Training history object.
        skill : str
            Current skill document content.
        prev_results : Any
            Rollout results from the previous epoch's last step (if available).
        curr_results : Any
            Rollout results from the current epoch's last step (if available).
        items : list[dict] | None
            The task items used in the last step.
        out_dir : str
            Output directory for epoch artifacts.
        """
        return skill
