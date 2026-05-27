"""SkillOpt algorithm — the main SkillOptAlgorithm entry point.

Implements all abstract methods of :class:`BaseAlgorithm` by delegating
to the individual stage modules (reflect, aggregate, select, update,
slow_update, meta_skill).
"""
from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.base import BaseAlgorithm
from summerclaw.agent_trainer.registry import algorithm
from summerclaw.agent_trainer.types import (
    Patch,
    RawPatch,
    RolloutResult,
)

from .aggregate import merge_patches
from .lr_autonomous import decide_autonomous_learning_rate
from .meta_skill import format_meta_skill_context, run_meta_skill
from .reflect import run_minibatch_reflect
from .rewrite import rewrite_skill_from_suggestions
from .scheduler import AutonomousScheduler, build_scheduler
from .select import rank_and_select
from .slow_update import (
    SlowUpdateResult,
    build_comparison_pairs,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from .update import apply_patch_with_report


@algorithm("skillopt")
class SkillOptAlgorithm(BaseAlgorithm):
    """SkillOpt — structured skill optimization via reflection.

    6-stage per-step pipeline:
      1. Rollout   — execute episodes with current skill
      2. Reflect   — analyze trajectories, generate patches (minibatch)
      3. Aggregate — hierarchical merge of patches
      4. Select    — rank and select top edits (gradient clipping)
      5. Update    — apply edits to skill document (optimizer step)
      6. Evaluate  — validate candidate skill, accept/reject

    Epoch-level hooks:
      - Slow Update: LLM-driven longitudinal analysis → protected skill region
      - Meta Skill: cross-epoch optimizer memory → injected into all LLM calls
    """

    name: str = "skillopt"

    def __init__(
        self,
        provider: Any = None,
        model: str = "",
        minibatch_size: int = 5,
        edit_budget: int = 4,
        workers: int = 4,
        optimizer_model: str | None = None,
        update_mode: str = "patch",
        lr_mode: str = "constant",
        min_lr: int = 2,
        reasoning_effort: str = "high",
        env: str | None = None,
        *,
        merge_batch_size: int = 8,
        max_analyst_rounds: int = 3,
        use_slow_update: bool = True,
        use_meta_skill: bool = True,
        longitudinal_pair_policy: str = "mixed",
        rewrite_reasoning_effort: str | None = None,
        rewrite_max_completion_tokens: int = 64000,
    ):
        """Initialize SkillOpt algorithm.

        Parameters
        ----------
        provider : LLMProvider
            SummerClaw LLM provider (used for optimizer calls).
        model : str
            Default model name.
        minibatch_size : int
            Trajectories per reflect minibatch (M).
        edit_budget : int
            Maximum edits per step (L, the "learning rate").
        workers : int
            Max concurrent LLM calls.
        optimizer_model : str | None
            Optional separate model for optimizer (reflect/aggregate/select).
            Falls back to *model* if not set.
        update_mode : str
            One of "patch", "rewrite_from_suggestions",
            "full_rewrite_minibatch".
        lr_mode : str
            Learning-rate scheduler mode: "constant", "linear",
            "cosine", or "autonomous".
        min_lr : int
            Minimum edit budget (for linear/cosine decay).
        reasoning_effort : str
            Reasoning effort hint for rewrite LLM calls.
        env : str | None
            Optional environment label for prompt variant selection
            (e.g. ``"swe_bench"``, ``"tau_bench"``).
        merge_batch_size : int
            Batch size for hierarchical merge in aggregate stage.
        max_analyst_rounds : int
            Maximum reflect analyst rounds (mirrors official ``gradient.max_analyst_rounds``).
        use_slow_update : bool
            Whether to run slow update at epoch end.
        use_meta_skill : bool
            Whether to run meta skill at epoch end.
        longitudinal_pair_policy : str
            Policy for building comparison pairs: "mixed", "changed", or "unchanged".
        rewrite_reasoning_effort : str | None
            Separate reasoning effort for rewrite calls (None = use ``reasoning_effort``).
        rewrite_max_completion_tokens : int
            Max completion tokens for rewrite calls.
        """
        self.provider = provider
        self.model = model
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.workers = workers
        self.optimizer_model = optimizer_model or model
        self.update_mode = update_mode
        self.lr_mode = lr_mode
        self.min_lr = min_lr
        self.reasoning_effort = reasoning_effort
        self.env = env
        self.merge_batch_size = merge_batch_size
        self.max_analyst_rounds = max_analyst_rounds
        self.use_slow_update = use_slow_update
        self.use_meta_skill = use_meta_skill
        self.longitudinal_pair_policy = longitudinal_pair_policy
        self.rewrite_reasoning_effort = rewrite_reasoning_effort or reasoning_effort
        self.rewrite_max_completion_tokens = rewrite_max_completion_tokens
        self._scheduler = None  # built in init_training_run()

        # Cross-epoch runtime state
        self._meta_skill_content: str = ""
        self._prev_epoch_skill: str = ""
        self._prev_epoch_results: list[RolloutResult] = []
        self._prev_epoch_items: list[dict] = []
        self._curr_epoch_last_results: list[RolloutResult] = []
        self._curr_epoch_last_items: list[dict] = []

        # Step buffer context (accumulated within epoch)
        self._step_buffer_context: str = ""

    # ── Training run init ─────────────────────────────────────────────

    def init_training_run(self, total_steps: int) -> None:
        """Called by the trainer after computing total_steps.

        Builds the LR scheduler with the correct total step count.
        Skips if scheduler already exists with matching total_steps
        (e.g. after resume via load_state_dict).
        """
        if self._scheduler and self._scheduler.total_steps == total_steps:
            logger.info(
                "[SkillOpt] LR scheduler already initialized (mode={} step={}/{}); skipping rebuild",
                self.lr_mode, self._scheduler._current_step, total_steps,
            )
            return
        self._scheduler = build_scheduler(
            mode=self.lr_mode,
            max_lr=self.edit_budget,
            min_lr=self.min_lr,
            total_steps=total_steps,
        )
        logger.info(
            "[SkillOpt] LR scheduler: mode={} max_lr={} min_lr={} total_steps={}",
            self.lr_mode, self.edit_budget, self.min_lr, total_steps,
        )

    # ── Per-step budget ───────────────────────────────────────────────

    def get_edit_budget(self, step: int, total_steps: int) -> int:
        """Return the per-step edit budget from the scheduler."""
        if self._scheduler is None:
            return self.edit_budget
        return self._scheduler.step()

    # ── State persistence ─────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize scheduler state for resume support."""
        return {
            "scheduler": self._scheduler.state_dict() if self._scheduler else {},
            "lr_mode": self.lr_mode,
            "meta_skill_content": self._meta_skill_content,
            "step_buffer_context": self._step_buffer_context,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state on resume."""
        if self._scheduler and "scheduler" in state:
            self._scheduler.load_state_dict(state["scheduler"])
        self._meta_skill_content = state.get("meta_skill_content", "")
        self._step_buffer_context = state.get("step_buffer_context", "")

    # ── Stage 1: Rollout ────────────────────────────────────────────────

    async def rollout(
        self,
        env: Any,
        skill: str,
        items: list[dict],
        out_dir: str,
    ) -> list[RolloutResult]:
        """Execute rollout batch via the environment adapter."""
        logger.info(
            "[1/6 ROLLOUT] {} items with skill ({} chars, mode={})",
            len(items), len(skill), self.update_mode,
        )
        results = await env.rollout_batch(items, skill, phase_label="1/6 ROLLOUT")
        hard_sum = sum(r.hard for r in results)
        soft_mean = sum(r.soft for r in results) / max(len(results), 1)
        logger.info(
            "[1/6 ROLLOUT] done: hard_acc={:.3f} soft_mean={:.3f}",
            hard_sum / max(len(results), 1), soft_mean,
        )
        # Track last batch results for epoch-end comparison
        self._curr_epoch_last_results = list(results)
        self._curr_epoch_last_items = list(items)
        return results

    # ── Stage 2: Reflect ────────────────────────────────────────────────

    async def reflect(
        self,
        results: list[RolloutResult],
        skill: str,
        out_dir: str,
    ) -> list[RawPatch]:
        """Minibatch trajectory analysis → patches."""
        patches_dir = os.path.join(out_dir, "patches")
        meta_ctx = format_meta_skill_context(self._meta_skill_content)
        return await run_minibatch_reflect(
            provider=self.provider,
            model=self.optimizer_model,
            results=results,
            skill_content=skill,
            patches_dir=patches_dir,
            workers=self.workers,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            update_mode=self.update_mode,
            meta_skill_context=meta_ctx,
        )

    # ── Stage 3: Aggregate ──────────────────────────────────────────────

    async def aggregate(
        self,
        patches: list[RawPatch],
        skill: str,
    ) -> Patch:
        """Hierarchical merge of patches."""
        failure_patches: list[dict] = []
        success_patches: list[dict] = []
        for p in patches:
            d = p.patch.to_dict()
            if not d.get("edits"):
                continue
            if p.source_type == "success":
                success_patches.append(d)
            else:
                failure_patches.append(d)

        meta_ctx = format_meta_skill_context(self._meta_skill_content)
        return await merge_patches(
            provider=self.provider,
            model=self.optimizer_model,
            skill_content=skill,
            failure_patches=failure_patches,
            success_patches=success_patches,
            update_mode=self.update_mode,
            meta_skill_context=meta_ctx,
        )

    # ── Stage 4: Select ─────────────────────────────────────────────────

    async def select(
        self,
        patch: Patch,
        budget: int,
        skill: str,
        *,
        rollout_hard: float = 0.0,
        rollout_soft: float = 0.0,
        rollout_n: int = 0,
    ) -> Patch:
        """Rank edits and select top-L (gradient clipping).

        When ``lr_mode == "autonomous"``, the LLM decides the actual
        number of edits to apply.
        """
        meta_ctx = format_meta_skill_context(self._meta_skill_content)

        actual_budget = budget
        if self.lr_mode == "autonomous" and self.provider:
            try:
                lr_record = await decide_autonomous_learning_rate(
                    provider=self.provider,
                    model=self.optimizer_model,
                    skill_content=skill,
                    merged_patch=patch.to_dict(),
                    update_mode=self.update_mode,
                    rollout_hard=rollout_hard,
                    rollout_soft=rollout_soft,
                    rollout_n=rollout_n,
                    step_buffer_context=self._step_buffer_context,
                    meta_skill_context=meta_ctx,
                )
                actual_budget = lr_record.get("learning_rate", budget)
                logger.info(
                    "[SELECT] autonomous LR={} (budget={}, fallback={})",
                    actual_budget, budget, lr_record.get("fallback", False),
                )
            except Exception as exc:
                logger.error("[SELECT] autonomous LR failed, using budget={}: {}", budget, exc)
                actual_budget = budget

        return await rank_and_select(
            provider=self.provider,
            model=self.optimizer_model,
            skill_content=skill,
            patch=patch,
            max_edits=actual_budget,
            update_mode=self.update_mode,
            meta_skill_context=meta_ctx,
        )

    # ── Stage 5: Update ─────────────────────────────────────────────────

    async def update(
        self,
        skill: str,
        patch: Patch,
    ) -> tuple[str, list[dict]]:
        """Apply selected edits to skill document (optimizer step).

        In ``rewrite_from_suggestions`` mode, the LLM generates a full
        rewrite of the skill integrating the selected suggestions.
        Falls back to standard patch apply if rewrite fails.
        """
        # Rewrite path: rewrite_from_suggestions mode
        if self.update_mode == "rewrite_from_suggestions" and self.provider:
            meta_ctx = format_meta_skill_context(self._meta_skill_content)
            rewrite_result = await rewrite_skill_from_suggestions(
                provider=self.provider,
                model=self.optimizer_model,
                skill_content=skill,
                patch=patch.to_dict(),
                step_buffer_context=self._step_buffer_context,
                env=self.env,
                reasoning_effort=self.rewrite_reasoning_effort,
                max_completion_tokens=self.rewrite_max_completion_tokens,
            )
            if rewrite_result and rewrite_result.get("new_skill"):
                new_skill = rewrite_result["new_skill"]
                report = [{
                    "action": "rewrite_from_suggestions",
                    "reasoning": rewrite_result.get("reasoning", ""),
                    "change_summary": rewrite_result.get("change_summary", []),
                }]
                logger.info(
                    "[UPDATE] rewrite success: {} chars (was {})",
                    len(new_skill), len(skill),
                )
                return new_skill, report
            # Fallback to standard apply
            logger.warning("[UPDATE] rewrite failed; falling back to apply_patch_with_report")

        # Standard path: apply edits
        return apply_patch_with_report(skill, patch)

    # ── Stage 6: Evaluate ───────────────────────────────────────────────

    async def evaluate(
        self,
        env: Any,
        skill: str,
        items: list[dict],
        out_dir: str,
    ) -> float:
        """Evaluate candidate skill on validation items.

        Returns hard accuracy.
        """
        logger.info("[6/6 EVALUATE] {} val items", len(items))
        results = await env.rollout_batch(items, skill, phase_label="6/6 EVALUATE")
        if not results:
            return 0.0
        hard_acc = sum(r.hard for r in results) / len(results)
        logger.info("[6/6 EVALUATE] hard_acc={:.3f}", hard_acc)
        return hard_acc

    # ── Epoch hook: Slow Update + Meta Skill ────────────────────────────

    async def on_epoch_end(
        self,
        epoch: int,
        history: Any,
        skill: str,
        *,
        prev_results: list[RolloutResult] | None = None,
        curr_results: list[RolloutResult] | None = None,
        items: list[dict] | None = None,
        out_dir: str = "",
    ) -> str:
        """Epoch-level slow update + meta skill update.

        1. Build comparison pairs from prev/curr epoch results.
        2. Run LLM-driven slow update → write guidance to protected region.
        3. Run meta skill update → update optimizer memory.
        4. Persist artifacts to ``out_dir/slow_update/`` and ``out_dir/meta_skill/``.
        """
        if not self.provider:
            logger.warning("[EPOCH END] no provider; skipping slow/meta update")
            return skill

        # Use passed-in results or fall back to tracked state
        prev_res = prev_results or self._prev_epoch_results
        curr_res = curr_results or self._curr_epoch_last_results
        epoch_items = items or self._curr_epoch_last_items

        logger.info(
            "[EPOCH END] epoch={} prev_results={} curr_results={} items={}",
            epoch, len(prev_res), len(curr_res), len(epoch_items),
        )

        # Ensure SLOW_UPDATE region exists in skill
        skill = inject_empty_slow_update_field(skill)

        # ── Slow Update ─────────────────────────────────────────────
        su_result: SlowUpdateResult | None = None
        if self.use_slow_update and prev_res and curr_res and epoch_items:
            comparison_pairs = build_comparison_pairs(prev_res, curr_res, epoch_items)

            # Persist comparison pairs
            if out_dir:
                su_dir = os.path.join(out_dir, "slow_update", f"epoch_{epoch:02d}")
                os.makedirs(su_dir, exist_ok=True)
                save_comparison_pairs(
                    comparison_pairs,
                    os.path.join(su_dir, "comparison_pairs.json"),
                )

            prev_skill = self._prev_epoch_skill or skill
            su_result = await run_slow_update(
                provider=self.provider,
                model=self.optimizer_model,
                prev_skill=prev_skill,
                curr_skill=skill,
                comparison_pairs=comparison_pairs,
            )

            if su_result.guidance:
                skill = replace_slow_update_field(skill, su_result.guidance)
                logger.info(
                    "[SLOW UPDATE] epoch={} action={} guidance={} chars",
                    epoch, su_result.action, len(su_result.guidance),
                )

                if out_dir:
                    su_path = os.path.join(su_dir, "guidance.txt")
                    with open(su_path, "w") as f:
                        f.write(su_result.guidance)
                    reasoning_path = os.path.join(su_dir, "reasoning.txt")
                    with open(reasoning_path, "w") as f:
                        f.write(su_result.reasoning)
                    # Persist candidate skill snapshot (aligns with official SkillOpt)
                    cand_path = os.path.join(su_dir, "candidate_skill.md")
                    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
                    with open(cand_path, "w") as f:
                        f.write(skill)
            else:
                logger.info(
                    "[SLOW UPDATE] epoch={} action={} (no guidance produced)",
                    epoch, su_result.action,
                )
        elif not self.use_slow_update:
            logger.info("[SLOW UPDATE] epoch={} disabled (use_slow_update=false)", epoch)
        else:
            logger.info(
                "[SLOW UPDATE] epoch={} skipped (insufficient data for comparison)",
                epoch,
            )

        # Write slow_result.json as done marker (aligns with official SkillOpt)
        if out_dir and self.use_slow_update:
            su_done_dir = os.path.join(out_dir, "slow_update", f"epoch_{epoch:02d}")
            os.makedirs(su_done_dir, exist_ok=True)
            su_done = {
                "epoch": epoch,
                "action": su_result.action if su_result else "skipped",
            }
            if su_result and su_result.guidance:
                su_done["slow_update_content"] = su_result.guidance
            su_done_path = os.path.join(su_done_dir, "slow_result.json")
            with open(su_done_path, "w") as f:
                json.dump(su_done, f, indent=2, ensure_ascii=False)

        # ── Meta Skill ──────────────────────────────────────────────
        if self.use_meta_skill and prev_res and curr_res and epoch_items:
            comparison_pairs_meta = build_comparison_pairs(
                prev_res, curr_res, epoch_items,
            )
            prev_skill_for_meta = self._prev_epoch_skill or skill

            meta_result = await run_meta_skill(
                provider=self.provider,
                model=self.optimizer_model,
                prev_skill=prev_skill_for_meta,
                curr_skill=skill,
                comparison_pairs=comparison_pairs_meta,
                prev_meta_skill_content=self._meta_skill_content,
            )

            if meta_result:
                self._meta_skill_content = meta_result["meta_skill_content"]
                logger.info(
                    "[META SKILL] epoch={} updated ({} chars)",
                    epoch, len(self._meta_skill_content),
                )

                if out_dir:
                    ms_dir = os.path.join(out_dir, "meta_skill", f"epoch_{epoch:02d}")
                    os.makedirs(ms_dir, exist_ok=True)
                    ms_path = os.path.join(ms_dir, "meta_skill.json")
                    with open(ms_path, "w") as f:
                        json.dump(meta_result, f, ensure_ascii=False, indent=2)
                    # Also write meta_skill_result.json (official done marker)
                    ms_result_path = os.path.join(ms_dir, "meta_skill_result.json")
                    with open(ms_result_path, "w") as f:
                        json.dump(
                            {
                                "meta_skill_content": self._meta_skill_content,
                                "action": "write_meta_skill",
                            },
                            f, ensure_ascii=False, indent=2,
                        )
            else:
                logger.info("[META SKILL] epoch={} no update produced", epoch)
        elif not self.use_meta_skill:
            logger.info("[META SKILL] epoch={} disabled (use_meta_skill=false)", epoch)

        # ── Update epoch tracking state ─────────────────────────────
        self._prev_epoch_skill = skill
        self._prev_epoch_results = list(curr_res) if curr_res else []
        self._prev_epoch_items = list(epoch_items) if epoch_items else []
        self._curr_epoch_last_results = []
        self._curr_epoch_last_items = []

        return skill
