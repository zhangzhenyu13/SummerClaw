"""MOSCOPT algorithm — Mixture-of-Skill Collective Optimization.

Implements :class:`BaseAlgorithm` with multi-skill pool management,
text-based gating, and three-phase interleaved updates:

  Phase 1: Skill editing (gate fixed) — SkillOpt 6-stage per skill
  Phase 2: Gate editing (skills fixed) — bounded edits to gate text
  Phase 3: Collective evolution (every E epochs) — cull/breed/merge

Backward compatible with SkillOpt: when pool_size=1, activate_count=1
the algorithm degrades to single-skill SkillOpt with no gate LLM calls.

6-stage per-step pipeline (inherited from SkillOpt):
  1. Rollout   — gate selects K skills → execute with activated skills
  2. Reflect   — analyze trajectories, generate skill + gate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits (gradient clipping)
  5. Update    — apply edits to skill pool or gate document
  6. Evaluate  — validate candidate skill, accept/reject

Epoch-level hooks:
  - Slow Update: LLM-driven longitudinal analysis → protected skill region
  - Meta Skill: cross-epoch optimizer memory → injected into all LLM calls
  - Collective Evolution: cull + breed + merge (every E epochs)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.base import BaseAlgorithm
from summerclaw.agent_trainer.registry import algorithm
from summerclaw.agent_trainer.types import (
    Edit,
    FailureSummaryEntry,
    Patch,
    RawPatch,
    RolloutResult,
)

from .aggregate import merge_patches
from .lr_autonomous import decide_autonomous_learning_rate
from .meta_skill import format_meta_skill_context, run_meta_skill
from .pool import (
    DEFAULT_GATE_PROMPT,
    SkillPool,
    _MERGE_DISTILL_SYSTEM,
    build_agent_prompt,
    call_gate_llm,
    compute_diversity,
    fallback_top_k,
    format_summary_table,
    generate_diverse_pool,
    generate_gate_prompt,
    get_activated_skill_ids,
    get_top_cooccurrence_pair,
    inject_foreign_gene,
    mutate_skill,
    parse_pool,
    rank_skills_by_failure_contribution,
    reassign_skill_ids,
    select_lowest_scored,
    select_top_parents,
    serialize_pool,
    update_cooccurrence,
    update_q_scores,
    update_summaries,
)
from .reflect import run_minibatch_reflect
from .rejected_buffer import RejectedBuffer
from .rewrite import rewrite_skill_from_suggestions
from .scheduler import build_scheduler
from .select import rank_and_select
from .slow_update import (
    SlowUpdateResult,
    build_comparison_pairs,
    extract_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from .update import apply_patch_with_report


@algorithm("moscopt")
class MOSCOPTAlgorithm(BaseAlgorithm):
    """MOSCOPT — Mixture-of-Skill Collective Optimization.

    Extends SkillOpt with a multi-skill pool, text-based gating, and
    collective evolution mechanisms.  When pool_size=1 and
    activate_count=1, degenerates to standard SkillOpt.

    Three-phase per-opt_step update:
      Phase 1: Skill editing (gate fixed)
      Phase 2: Gate editing (skills fixed)
      Phase 3: Collective evolution (every E epochs)
    """

    name: str = "moscopt"

    def __init__(
        self,
        provider: Any = None,
        model: str = "",
        minibatch_size: int = 4,
        edit_budget: int = 8,
        workers: int = 4,
        optimizer_model: str | None = None,
        update_mode: str = "patch",
        lr_mode: str = "cosine",
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
        use_rejected_buffer: bool = True,
        rejected_buffer_max_size: int = 10,
        rejected_buffer_max_summary_chars: int = 200,
        # ── MOSCOPT-specific ──────────────────────────────────────
        pool_size: int = 5,
        activate_count: int = 2,
        evolution_interval: int = 5,
        evolution_count: int = 1,
        gating_granularity: str = "task",
        ema_beta: float = 0.3,
        min_activations: int = 5,
        diversity_threshold: float = 0.85,
        val_sample_ratio: float = 1.0,
        summary_enrichment_epochs: tuple[int, int] = (2, 4),
    ):
        """Initialize MOSCOPT algorithm.

        Parameters
        ----------
        pool_size : int
            Number of skills in the pool (N).
        activate_count : int
            Number of skills to activate per step (K).  1 <= K <= N.
        evolution_interval : int
            Trigger collective evolution every E epochs.
        evolution_count : int
            Number of skills to cull/breed per evolution (M).
        gating_granularity : str
            "task" for per-task gate selection, "step" for per-exec-step.
        ema_beta : float
            EMA smoothing coefficient for Q-score updates.
        min_activations : int
            Minimum activation count before Q-score is trusted (c_min).
        diversity_threshold : float
            Trigger forced mutation when avg similarity exceeds this.
        val_sample_ratio : float
            Fraction of validation set to use for evaluation (1.0 = full).
        """
        # ── SkillOpt base parameters ──────────────────────────────
        self.provider = provider
        self.model = model
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget

        # workers: 0 = auto-derive 80% of provider.max_concurrency
        if workers <= 0:
            provider_max = getattr(provider, "max_concurrency", 0) or 0
            self.workers = max(1, int(provider_max * 0.8)) if provider_max > 0 else 4
        else:
            self.workers = workers

        self.analyst_workers = self.workers
        self.aggregate_workers = self.workers
        self.evaluate_workers = self.workers
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

        # ── MOSCOPT-specific parameters ───────────────────────────
        self.pool_size = pool_size
        self.activate_count = activate_count
        self.evolution_interval = evolution_interval
        self.evolution_count = evolution_count
        self.gating_granularity = gating_granularity
        self.ema_beta = ema_beta
        self.min_activations = min_activations
        self.diversity_threshold = diversity_threshold
        self.val_sample_ratio = val_sample_ratio
        self.summary_enrichment_epochs = summary_enrichment_epochs

        # Validate K <= N
        if self.activate_count > self.pool_size:
            logger.warning(
                "[MOSCOPT] K={} > N={}, clamping K to N",
                self.activate_count, self.pool_size,
            )
            self.activate_count = self.pool_size

        # K/N ratio warning (Section 5.4)
        if self.pool_size > 1 and self.activate_count / self.pool_size > 0.8:
            logger.warning(
                "[MOSCOPT] K/N ratio {:.2f} > 0.8: gate selection has limited "
                "discriminative power. Consider increasing pool_size or "
                "decreasing activate_count.",
                self.activate_count / self.pool_size,
            )
        # N >= K + M constraint warning
        if self.pool_size < self.activate_count + self.evolution_count:
            logger.warning(
                "[MOSCOPT] N={} < K+M={} (K={}, M={}): collective evolution "
                "may shrink pool below K.",
                self.pool_size,
                self.activate_count + self.evolution_count,
                self.activate_count,
                self.evolution_count,
            )

        # ── Pool state ────────────────────────────────────────────
        self._pool: SkillPool = SkillPool(n=pool_size, k=self.activate_count)

        # ── Rejected buffers (per-skill + gate) ───────────────────
        self.use_rejected_buffer = use_rejected_buffer
        self._rb_max_size = rejected_buffer_max_size
        self._rb_max_chars = rejected_buffer_max_summary_chars
        self._skill_reject_buffers: dict[str, RejectedBuffer] = {}
        self._gate_reject_buffer = RejectedBuffer(
            max_size=rejected_buffer_max_size,
            max_summary_chars=rejected_buffer_max_summary_chars,
        )

        # ── Step buffers (per-skill + gate) ───────────────────────
        self._skill_step_buffers: dict[str, list[str]] = {}
        self._gate_step_buffer: list[str] = []

        # ── Scoring and read-state ────────────────────────────────
        self._last_evaluate_soft_score: float = 0.0
        self._last_rollout_results: list[RolloutResult] = []
        self._last_analysis_failures: int = 0

        # ── Gate parse failure monitor (Section 5.1) ───────────────
        self._gate_parse_failures: int = 0
        self._gate_parse_total: int = 0

        # ── Phase 1 rotation index (multi-skill round-robin) ────────
        self._edit_rotation_index: int = 0

        # ── Gate parse failure events (for Phase 2 reflect signal) ─────
        self._gate_parse_failure_events: list[str] = []

        # ── Gate validation state (GAP-3: Section 3.8) ───────────
        self._pre_edit_gate: str | None = None

        # ── Gate positive feedback (Section 3.8) ───────────────────
        self._last_gate_successes: list[RolloutResult] = []

        # ── Convergence detection (Section 3.10) ────────────────────
        self._convergence_window: list[float] = []
        self._convergence_threshold: float = 0.01
        self._convergence_window_size: int = 5
        self.converged: bool = False
        self._current_score: float = 0.0
        self._prev_pool_size: int = 0
        self._gate_selection_counts: dict[str, int] = {}

        # ── Pool history for Dashboard API (GAP-C) ─────────────────
        self._pool_history: list[dict] = []

        # ── Pending validation candidates (GAP-G) ──────────────────
        # Each entry: (cand_id, old_text, new_text, patch_or_None)
        self._pending_val_candidates: list[tuple] = []

        # ── Cross-epoch state ─────────────────────────────────────
        self._meta_skill_content: str = ""
        self._prev_epoch_pool: dict[str, str] = {}
        self._prev_epoch_results: list[RolloutResult] = []
        self._prev_epoch_items: list[dict] = []
        self._curr_epoch_last_results: list[RolloutResult] = []
        self._curr_epoch_last_items: list[dict] = []

        # ── Step buffer context (global, for compat) ──────────────
        self._step_buffer_context: str = ""
        self._step_buffer_entries: list[str] = []
        self._analysis_failure_count: int = 0

        # ── Current target skill (set by reflect, used by update) ─
        self._current_target_skill: str | None = None
        self._last_rollout_for_attribution: list[RolloutResult] = []

    # ── Backward compat check ──────────────────────────────────────

    @property
    def _is_single_skill(self) -> bool:
        return self.pool_size <= 1 and self.activate_count <= 1

    # ── Training run init ──────────────────────────────────────────

    def init_training_run(self, total_steps: int) -> None:
        """Build the LR scheduler. Skips if already built with matching total."""
        if self._scheduler and self._scheduler.total_steps == total_steps:
            logger.info(
                "[MOSCOPT] LR scheduler already initialized (mode={} step={}/{}); skipping",
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
            "[MOSCOPT] LR scheduler: mode={} max_lr={} min_lr={} total_steps={}",
            self.lr_mode, self.edit_budget, self.min_lr, total_steps,
        )

    # ── Epoch hooks ────────────────────────────────────────────────

    def on_epoch_start(self, epoch: int) -> None:
        """Clear all epoch-local buffers (per-skill + gate)."""
        for sid, buf in self._skill_reject_buffers.items():
            if not buf.is_empty():
                logger.info("[REJECTED_BUFFER] clearing skill {} ({} entries)", sid, len(buf))
            buf.clear()
        self._gate_reject_buffer.clear()

        for sid in list(self._skill_step_buffers.keys()):
            self._skill_step_buffers[sid] = []
        self._gate_step_buffer = []

        self._step_buffer_entries.clear()
        self._step_buffer_context = ""
        self._analysis_failure_count = 0

        # Reset gate failure monitor for this epoch
        self._gate_parse_failures = 0
        self._gate_parse_total = 0

        # Reset Phase 1 rotation index
        self._edit_rotation_index = 0

        # Reset gate parse failure events
        self._gate_parse_failure_events = []

        logger.info(
            "[MOSCOPT] epoch {} started (pool_size={}, K={})",
            epoch, self._pool.size, self.activate_count,
        )

    # ── Per-step budget ────────────────────────────────────────────

    def get_edit_budget(self, step: int, total_steps: int) -> int:
        """Return the per-step edit budget from the scheduler."""
        if self._scheduler is None:
            return self.edit_budget
        return self._scheduler.step()

    # ── State persistence ──────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize full MOSCOPT state for resume support."""
        skill_rb = {
            sid: buf.to_dict()
            for sid, buf in self._skill_reject_buffers.items()
        }
        return {
            "scheduler": self._scheduler.state_dict() if self._scheduler else {},
            "lr_mode": self.lr_mode,
            "meta_skill_content": self._meta_skill_content,
            "step_buffer_context": self._step_buffer_context,
            "step_buffer_entries": self._step_buffer_entries,
            "analysis_failure_count": self._analysis_failure_count,
            "pool": self._pool.__dict__,
            "gate_reject_buffer": self._gate_reject_buffer.to_dict(),
            "skill_reject_buffers": skill_rb,
            "skill_step_buffers": dict(self._skill_step_buffers),
            "gate_step_buffer": list(self._gate_step_buffer),
            "prev_epoch_pool": dict(self._prev_epoch_pool),
            # Spec-alignment state
            "gate_parse_failures": self._gate_parse_failures,
            "gate_parse_total": self._gate_parse_total,
            "edit_rotation_index": self._edit_rotation_index,
            "convergence_window": list(self._convergence_window),
            "converged": self.converged,
            "gate_parse_failure_events": list(self._gate_parse_failure_events),
            "pre_edit_gate": self._pre_edit_gate,
            "prev_pool_size": self._prev_pool_size,
            "gate_selection_counts": dict(self._gate_selection_counts),
            "pool_history": list(self._pool_history),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore MOSCOPT state on resume."""
        if self._scheduler and "scheduler" in state:
            self._scheduler.load_state_dict(state["scheduler"])
        self._meta_skill_content = state.get("meta_skill_content", "")
        self._step_buffer_context = state.get("step_buffer_context", "")
        self._step_buffer_entries = state.get("step_buffer_entries", [])
        self._analysis_failure_count = state.get("analysis_failure_count", 0)

        # Restore pool
        pool_data = state.get("pool")
        if pool_data:
            self._pool = SkillPool()
            self._pool.__dict__.update(pool_data)

        # Restore gate reject buffer
        grb = state.get("gate_reject_buffer")
        if grb:
            self._gate_reject_buffer = RejectedBuffer.from_dict(grb)

        # Restore per-skill reject buffers
        for sid, rb_data in state.get("skill_reject_buffers", {}).items():
            self._skill_reject_buffers[sid] = RejectedBuffer.from_dict(rb_data)

        # Restore step buffers
        self._skill_step_buffers = state.get("skill_step_buffers", {})
        self._gate_step_buffer = state.get("gate_step_buffer", [])
        self._prev_epoch_pool = state.get("prev_epoch_pool", {})

        # Restore spec-alignment state
        self._gate_parse_failures = state.get("gate_parse_failures", 0)
        self._gate_parse_total = state.get("gate_parse_total", 0)
        self._edit_rotation_index = state.get("edit_rotation_index", 0)
        self._convergence_window = state.get("convergence_window", [])
        self.converged = state.get("converged", False)
        self._gate_parse_failure_events = state.get("gate_parse_failure_events", [])
        self._pre_edit_gate = state.get("pre_edit_gate", None)
        self._prev_pool_size = state.get("prev_pool_size", 0)
        self._gate_selection_counts = state.get("gate_selection_counts", {})
        self._pool_history = state.get("pool_history", [])

    # ── Rejection recording ────────────────────────────────────────

    def record_rejection(
        self,
        step: int,
        patch: Patch,
        score_before: float,
        score_after: float,
        failure_patterns: list[dict] | None = None,
    ) -> None:
        """Record a rejected patch into the appropriate buffer.

        Uses ``patch.reasoning`` metadata to route to gate or skill buffer.
        """
        if not self.use_rejected_buffer:
            return
        if "[MOSCOPT:GATE]" in (patch.reasoning or ""):
            self._gate_reject_buffer.add(
                step=step, edits=patch.edits,
                score_before=score_before, score_after=score_after,
                failure_patterns=failure_patterns,
            )
        else:
            target = self._current_target_skill
            if target and target in self._skill_reject_buffers:
                self._skill_reject_buffers[target].add(
                    step=step, edits=patch.edits,
                    score_before=score_before, score_after=score_after,
                    failure_patterns=failure_patterns,
                )
            elif self._skill_reject_buffers:
                # Fallback: record to first available buffer
                first_buf = next(iter(self._skill_reject_buffers.values()))
                first_buf.add(
                    step=step, edits=patch.edits,
                    score_before=score_before, score_after=score_after,
                    failure_patterns=failure_patterns,
                )

    # ── Step buffer accumulation ───────────────────────────────────

    def update_step_buffer(
        self,
        step: int,
        *,
        rollout_hard: float = 0.0,
        rollout_soft: float = 0.0,
        n_patches: int = 0,
        n_analysis_failures: int = 0,
        gate_action: str = "",
        selected_edits: list[Edit] | None = None,
        failure_summaries: list[FailureSummaryEntry] | None = None,
        score_before: float = 0.0,
        score_after: float = 0.0,
    ) -> None:
        """Accumulate one step's reflect outcome into epoch-local buffers."""
        parts: list[str] = []
        parts.append(
            f"[Step {step}] rollout_hard={rollout_hard:.4f} "
            f"rollout_soft={rollout_soft:.4f} "
            f"patches={n_patches} "
            f"analysis_failures={n_analysis_failures} "
            f"gate={gate_action} "
            f"score={score_before:.4f}->{score_after:.4f}"
        )

        if failure_summaries:
            for fs in failure_summaries[:5]:
                parts.append(
                    f"  [failure_type={fs.failure_type}] "
                    f"count={fs.count}: {fs.description[:120]}"
                )

        if selected_edits:
            for edit in selected_edits[:6]:
                content_preview = edit.content[:80] if edit.content else ""
                parts.append(f"  [edit] {edit.op}: {content_preview}")

        entry = "\n".join(parts)
        self._step_buffer_entries.append(entry)

        header = "## Previous Steps in This Epoch\n"
        self._step_buffer_context = header + "\n\n".join(self._step_buffer_entries)

        # Route to per-skill step buffer
        target = self._current_target_skill
        if target:
            self._skill_step_buffers.setdefault(target, []).append(entry)

        self._analysis_failure_count += n_analysis_failures

        logger.info(
            "[STEP_BUFFER] step={} added (patches={} failures={} gate={} buffer_size={})",
            step, n_patches, n_analysis_failures, gate_action,
            len(self._step_buffer_entries),
        )

    # ── Helpers: pool initialization ────────────────────────────────

    async def _ensure_pool_initialized(self, skill: str) -> SkillPool:
        """Parse compound doc into pool, or bootstrap from a single skill."""
        pool = parse_pool(skill)

        if pool.size == 0:
            if self._is_single_skill:
                # Single-skill backward compat: simple bootstrap
                pool.skills["1"] = skill
                pool.n = 1
                pool.k = 1
                pool.gate = DEFAULT_GATE_PROMPT
                pool.ensure_state()
            else:
                # Multi-skill: generate diverse pool via LLM (Section 3.2)
                logger.info(
                    "[POOL INIT] generating diverse pool of {} skills via LLM",
                    self.pool_size,
                )
                try:
                    candidates = await generate_diverse_pool(
                        provider=self.provider,
                        model=self.optimizer_model,
                        seed_skill=skill,
                        n=self.pool_size,
                    )
                    for idx, (label, text) in enumerate(candidates, 1):
                        pool.skills[str(idx)] = text
                        pool.summaries[str(idx)] = {"id": str(idx), "label": label}
                except Exception as exc:
                    logger.error("[POOL INIT] diverse generation failed: {}; bootstrapping from seed", exc)
                    pool.skills["1"] = skill
                    for idx in range(2, self.pool_size + 1):
                        pool.skills[str(idx)] = f"# Variant {idx}\n\n{skill[:3000]}"
                        pool.summaries[str(idx)] = {"id": str(idx), "label": f"Variant {idx}"}

                pool.n = self.pool_size
                pool.k = self.activate_count

                # Generate customized gate prompt via LLM (Section 3.3)
                pool_summaries = [
                    (sid, pool.summaries.get(sid, {}).get("label", f"Skill {sid}"))
                    for sid in pool.skills
                ]
                try:
                    pool.gate = await generate_gate_prompt(
                        provider=self.provider,
                        model=self.optimizer_model,
                        task_description=skill[:500],
                        pool_summaries=pool_summaries,
                        k=self.activate_count,
                    )
                except Exception as exc:
                    logger.warning("[POOL INIT] gate generation failed: {}; using default", exc)
                    pool.gate = DEFAULT_GATE_PROMPT

                pool.ensure_state()

        # Inject Slow Update protected region into each skill and gate
        for sid in list(pool.skills.keys()):
            pool.skills[sid] = inject_empty_slow_update_field(pool.skills[sid])
        pool.gate = inject_empty_slow_update_field(pool.gate)

        # Ensure per-skill buffers exist
        for sid in pool.skills:
            if sid not in self._skill_reject_buffers:
                self._skill_reject_buffers[sid] = RejectedBuffer(
                    max_size=self._rb_max_size,
                    max_summary_chars=self._rb_max_chars,
                )
            if sid not in self._skill_step_buffers:
                self._skill_step_buffers[sid] = []

        return pool

    def _select_activated_skills(
        self,
        pool: SkillPool,
        state: str = "",
        history: str = "",
    ) -> list[str]:
        """Select K skills via gate LLM or fallback."""
        if self._is_single_skill:
            return list(pool.skills.keys())[:1]

        summaries = format_summary_table(pool, pool.epoch, self.summary_enrichment_epochs)
        valid_ids = set(pool.skills.keys())

        # Synchronous gate selection (we'll call from async context)
        # Use fallback directly for now; the async gate call happens in rollout
        activated = fallback_top_k(
            pool.q_scores,
            self.activate_count,
            pool.activation_counts,
            c_min=self.min_activations,
        )
        return activated

    # ══════════════════════════════════════════════════════════════
    # Stage 1: Rollout
    # ══════════════════════════════════════════════════════════════

    async def rollout(
        self,
        env: Any,
        skill: str,
        items: list[dict],
        out_dir: str,
    ) -> list[RolloutResult]:
        """Execute rollout with gate-based skill activation."""
        pool = await self._ensure_pool_initialized(skill)
        self._pool = pool
        pool.epoch = getattr(self, "_current_epoch", 0)

        # Backward compat: N=1, K=1 → plain SkillOpt
        if self._is_single_skill:
            sid = pool.skill_ids()[0]
            effective_skill = pool.skills[sid]
            logger.info(
                "[1/6 ROLLOUT] {} items, single skill (backward compat, {} chars)",
                len(items), len(effective_skill),
            )
            results = await env.rollout_batch(items, effective_skill, phase_label="1/6 ROLLOUT")
            # Tag results with activated skill
            for r in results:
                r.extras["moscopt_activated_skills"] = [sid]
            self._post_rollout(results, items)
            return results

        # ── Step-level gating: per-item gate selection (Section 3.4) ─
        if self.gating_granularity == "step":
            results = await self._rollout_per_item_gating(env, pool, items)
            self._post_rollout(results, items)
            return results

        # ── Task-level gating: select K skills once for the batch ─
        summary_table = format_summary_table(pool, pool.epoch, self.summary_enrichment_epochs)
        valid_ids = set(pool.skills.keys())
        state = items[0].get("question", "") if items else ""

        activated_ids, parse_failed = await call_gate_llm(
            provider=self.provider,
            model=self.optimizer_model,
            gate_text=pool.gate,
            summary_table=summary_table,
            state=state,
            history="",
            k=self.activate_count,
            valid_ids=valid_ids,
        )

        # Gate parse failure monitoring (Section 5.1)
        self._gate_parse_total += 1
        if parse_failed:
            self._gate_parse_failures += 1
            self._gate_parse_failure_events.append(
                f"step@epoch{getattr(self, '_current_epoch', '?')}: "
                f"gate output did not contain exactly K={self.activate_count} valid IDs"
            )

        if activated_ids is None:
            activated_ids = fallback_top_k(
                pool.q_scores, self.activate_count,
                pool.activation_counts, c_min=self.min_activations,
            )
            logger.info("[ROLLOUT] gate fallback: activated={}", activated_ids)
        else:
            logger.info("[ROLLOUT] gate selected: activated={}", activated_ids)

        # Gate failure rate monitor: rebuild G if > 30% failures after 5+ attempts
        if (
            self._gate_parse_total >= 5
            and self._gate_parse_failures / self._gate_parse_total > 0.3
        ):
            logger.warning(
                "[GATE MONITOR] parse failure rate {}/{} ({:.0%}) > 30%, rebuilding gate",
                self._gate_parse_failures, self._gate_parse_total,
                self._gate_parse_failures / self._gate_parse_total,
            )
            pool.gate = inject_empty_slow_update_field(DEFAULT_GATE_PROMPT)
            self._gate_parse_failures = 0
            self._gate_parse_total = 0

        # Build effective skill: concatenate K activated skills
        activated_texts = {sid: pool.skills[sid] for sid in activated_ids if sid in pool.skills}
        effective_skill = build_agent_prompt(activated_texts)

        logger.info(
            "[1/6 ROLLOUT] {} items, {} activated skills ({} chars effective)",
            len(items), len(activated_ids), len(effective_skill),
        )

        results = await env.rollout_batch(items, effective_skill, phase_label="1/6 ROLLOUT")

        # Tag results with activated skill IDs for attribution
        for r in results:
            r.extras["moscopt_activated_skills"] = list(activated_ids)

        self._post_rollout(results, items)
        return results

    async def _rollout_per_item_gating(
        self,
        env: Any,
        pool: SkillPool,
        items: list[dict],
    ) -> list[RolloutResult]:
        """Per-item gate selection: each item gets its own gate LLM call (Section 3.4).

        Used when ``gating_granularity == "step"``.  Each item is independently
        gated, so different items can activate different skill subsets.
        """
        summary_table = format_summary_table(pool, pool.epoch, self.summary_enrichment_epochs)
        valid_ids = set(pool.skills.keys())
        gate_sem = asyncio.Semaphore(self.workers)
        rollout_sem = asyncio.Semaphore(self.workers)
        timeout_s = getattr(env, "rollout_timeout_s", 300) or 300

        import os as _os
        debug = _os.environ.get("SUMMERCLAW_DEBUG_LLM")
        if debug:
            timeout_s = 0

        async def _process_item(item: dict) -> RolloutResult:
            item_id = str(item.get("id", "?"))
            # Gate selection for this item
            async with gate_sem:
                state = item.get("question", "")
                activated_ids, parse_failed = await call_gate_llm(
                    provider=self.provider,
                    model=self.optimizer_model,
                    gate_text=pool.gate,
                    summary_table=summary_table,
                    state=state,
                    history="",
                    k=self.activate_count,
                    valid_ids=valid_ids,
                )

            # Track parse failures (Section 5.1)
            self._gate_parse_total += 1
            if parse_failed:
                self._gate_parse_failures += 1
                self._gate_parse_failure_events.append(
                    f"item@epoch{getattr(self, '_current_epoch', '?')}:{item_id}: "
                    f"gate output did not contain exactly K={self.activate_count} valid IDs"
                )

            if activated_ids is None:
                activated_ids = fallback_top_k(
                    pool.q_scores, self.activate_count,
                    pool.activation_counts, c_min=self.min_activations,
                )

            activated_texts = {
                sid: pool.skills[sid] for sid in activated_ids if sid in pool.skills
            }
            effective_skill = build_agent_prompt(activated_texts)

            # Rollout with item-specific skill
            async with rollout_sem:
                try:
                    if timeout_s > 0:
                        result = await asyncio.wait_for(
                            env.rollout_one(item, effective_skill),
                            timeout=timeout_s,
                        )
                    else:
                        result = await env.rollout_one(item, effective_skill)
                except asyncio.TimeoutError:
                    logger.error("[STEP ROLLOUT] item={} timed out after {}s", item_id, timeout_s)
                    result = RolloutResult(
                        id=item_id,
                        hard=0,
                        soft=0.0,
                        fail_reason=f"rollout_timeout_{timeout_s}s",
                        question=item.get("question", ""),
                    )

            result.extras["moscopt_activated_skills"] = list(activated_ids)
            return result

        logger.info(
            "[1/6 ROLLOUT] step-level gating: {} items (per-item gate)", len(items),
        )
        results = await asyncio.gather(*[_process_item(item) for item in items])
        return list(results)

    def _post_rollout(
        self,
        results: list[RolloutResult],
        items: list[dict],
    ) -> None:
        """Common post-rollout bookkeeping."""
        hard_sum = sum(r.hard for r in results)
        soft_mean = sum(r.soft for r in results) / max(len(results), 1)
        logger.info(
            "[1/6 ROLLOUT] done: hard_acc={:.3f} soft_mean={:.3f}",
            hard_sum / max(len(results), 1), soft_mean,
        )
        self._curr_epoch_last_results = list(results)
        self._curr_epoch_last_items = list(items)
        self._last_rollout_for_attribution = list(results)
        # Update pool scoring
        if self._pool.size > 0:
            update_q_scores(self._pool, results, self.ema_beta)
            update_cooccurrence(self._pool, results)
        # Track gate selection counts for convergence detection (Section 3.10)
        for r in results:
            for sid in get_activated_skill_ids(r):
                self._gate_selection_counts[sid] = (
                    self._gate_selection_counts.get(sid, 0) + 1
                )

    # ══════════════════════════════════════════════════════════════
    # Stage 2: Reflect
    # ══════════════════════════════════════════════════════════════

    async def reflect(
        self,
        results: list[RolloutResult],
        skill: str,
        out_dir: str,
    ) -> list[RawPatch]:
        """Analyze trajectories → produce skill patches + gate patches.

        Phase 1: Skill editing — reflect on the lowest-Q-score skill.
        Phase 2: Gate editing — reflect on gate selection errors.
        """
        pool = await self._ensure_pool_initialized(skill)
        self._pool = pool
        patches_dir = os.path.join(out_dir, "patches")
        meta_ctx = format_meta_skill_context(self._meta_skill_content)

        all_patches: list[RawPatch] = []
        total_analysis_failures = 0

        # ── Phase 1: Skill reflect (loop over up to 3 candidates, Section 3.9) ──
        if not self._is_single_skill:
            ranked = rank_skills_by_failure_contribution(pool, results)
            max_skills_per_step = min(3, pool.size)
            edit_candidates: list[str] = []

            if ranked:
                n_candidates = len(ranked)
                for offset in range(n_candidates):
                    if len(edit_candidates) >= max_skills_per_step:
                        break
                    candidate_sid = ranked[(self._edit_rotation_index + offset) % n_candidates]
                    # Skip stable skills (Section 5.5 priority validation)
                    q = pool.q_scores.get(candidate_sid, 0.0)
                    act = pool.activation_counts.get(candidate_sid, 0)
                    if q >= 0.8 and act >= self.min_activations * 2:
                        logger.info(
                            "[REFLECT] skip stable skill {} (Q={:.2f}, act={})",
                            candidate_sid, q, act,
                        )
                        continue
                    edit_candidates.append(candidate_sid)
                # Advance rotation by the number of candidates examined
                self._edit_rotation_index = (
                    self._edit_rotation_index + min(n_candidates, max_skills_per_step + 2)
                ) % n_candidates
                if not edit_candidates:
                    edit_candidates = [ranked[0]]
            else:
                edit_candidates = [pool.skill_ids()[0]]

            # Set primary target (first candidate) for aggregate/select/update routing
            primary_sid = edit_candidates[0]
            self._current_target_skill = primary_sid

            # Reflect on each candidate skill
            _skill_semaphore = asyncio.Semaphore(2)  # limit concurrent skill reflects

            async def _reflect_one_skill(target_sid: str) -> list[RawPatch]:
                nonlocal total_analysis_failures
                async with _skill_semaphore:
                    target_text = pool.get_skill(target_sid) or skill
                    rb_ctx = ""
                    if self.use_rejected_buffer and target_sid in self._skill_reject_buffers:
                        rb_ctx = self._skill_reject_buffers[target_sid].format_context()
                    sb_ctx = "\n".join(self._skill_step_buffers.get(target_sid, []))
                    if sb_ctx:
                        sb_ctx = "## Previous Steps for This Skill\n" + sb_ctx

                    skill_failures = [
                        r for r in results
                        if r.hard == 0 and target_sid in get_activated_skill_ids(r)
                    ]
                    reflect_results = skill_failures if skill_failures else results

                    logger.info(
                        "[2/6 REFLECT] Phase 1: skill {} ({} failures / {} total)",
                        target_sid, len(skill_failures), len(results),
                    )
                    skill_patches, n_af = await run_minibatch_reflect(
                        provider=self.provider,
                        model=self.optimizer_model,
                        results=reflect_results,
                        skill_content=target_text,
                        patches_dir=os.path.join(patches_dir, f"skill_{target_sid}"),
                        workers=self.analyst_workers,
                        minibatch_size=self.minibatch_size,
                        edit_budget=self.edit_budget,
                        update_mode=self.update_mode,
                        step_buffer_context=sb_ctx,
                        meta_skill_context=meta_ctx,
                        rejected_buffer_context=rb_ctx,
                    )
                    total_analysis_failures += n_af
                    for p in skill_patches:
                        p.patch.reasoning = (
                            f"[MOSCOPT:SKILL:{target_sid}] " + (p.patch.reasoning or "")
                        )
                    return skill_patches

            # Run reflects concurrently
            tasks = [_reflect_one_skill(sid) for sid in edit_candidates]
            patch_lists = await asyncio.gather(*tasks)
            for pl in patch_lists:
                all_patches.extend(pl)

            logger.info(
                "[2/6 REFLECT] Phase 1: {} skills reflected, {} total patches",
                len(edit_candidates), len(all_patches),
            )
            self._last_analysis_failures = total_analysis_failures

        else:
            # Single-skill backward compat
            target_sid = pool.skill_ids()[0] if pool.skills else "1"
            self._current_target_skill = target_sid
            target_text = pool.get_skill(target_sid) or skill

            rb_ctx = ""
            if self.use_rejected_buffer and target_sid in self._skill_reject_buffers:
                rb_ctx = self._skill_reject_buffers[target_sid].format_context()
            sb_ctx = "\n".join(self._skill_step_buffers.get(target_sid, []))
            if sb_ctx:
                sb_ctx = "## Previous Steps for This Skill\n" + sb_ctx

            skill_patches, n_af = await run_minibatch_reflect(
                provider=self.provider,
                model=self.optimizer_model,
                results=results,
                skill_content=target_text,
                patches_dir=patches_dir,
                workers=self.analyst_workers,
                minibatch_size=self.minibatch_size,
                edit_budget=self.edit_budget,
                update_mode=self.update_mode,
                step_buffer_context=sb_ctx,
                meta_skill_context=meta_ctx,
                rejected_buffer_context=rb_ctx,
            )
            for p in skill_patches:
                p.patch.reasoning = (
                    f"[MOSCOPT:SKILL:{target_sid}] " + (p.patch.reasoning or "")
                )
            all_patches.extend(skill_patches)
            self._last_analysis_failures = n_af

        # ── Phase 2: Gate reflect (only in multi-skill mode) ──────
        if not self._is_single_skill:
            # Extract gate successes unconditionally (Section 3.8 positive feedback)
            self._last_gate_successes = self._extract_gate_successes(results, pool)

            gate_failures = self._extract_gate_failures(results, pool)
            if gate_failures or self._gate_parse_failure_events:
                gate_rb_ctx = (
                    self._gate_reject_buffer.format_context()
                    if self.use_rejected_buffer else ""
                )
                gate_sb_ctx = "\n".join(self._skill_step_buffers.get("_gate", []) or self._gate_step_buffer)
                if gate_sb_ctx:
                    gate_sb_ctx = "## Previous Gate Steps\n" + gate_sb_ctx

                # Inject parse failure events as additional gate signal (Section 5.1)
                if self._gate_parse_failure_events:
                    pf_summary = (
                        f"\n## Gate Parse Failures (this epoch)\n"
                        f"Gate output failed to parse {len(self._gate_parse_failure_events)} times. "
                        f"The gate must output exactly K={self.activate_count} valid skill IDs.\n"
                        f"Events:\n" +
                        "\n".join(f"- {e}" for e in self._gate_parse_failure_events[:10])
                    )
                    gate_sb_ctx = (gate_sb_ctx or "") + pf_summary

                # Inject positive feedback: successful gate selections (Section 3.8)
                if self._last_gate_successes:
                    success_lines = []
                    for gs in self._last_gate_successes[:10]:
                        activated = get_activated_skill_ids(gs)
                        stype = gs.extras.get("gate_success_type", "unknown")
                        success_lines.append(
                            f"- activated={activated} type={stype} reward=success"
                        )
                    pos_feedback = (
                        f"\n## Gate Positive Feedback (Section 3.8)\n"
                        f"The following selections succeeded. Consider solidifying these patterns:\n"
                        + "\n".join(success_lines)
                    )
                    gate_sb_ctx = (gate_sb_ctx or "") + pos_feedback

                logger.info(
                    "[2/6 REFLECT] Phase 2: gate ({} failures, {} successes)",
                    len(gate_failures), len(self._last_gate_successes),
                )
                gate_patches, gate_n_failures = await run_minibatch_reflect(
                    provider=self.provider,
                    model=self.optimizer_model,
                    results=gate_failures,
                    skill_content=pool.gate,
                    patches_dir=os.path.join(patches_dir, "gate"),
                    workers=self.analyst_workers,
                    minibatch_size=self.minibatch_size,
                    edit_budget=self.edit_budget,
                    update_mode=self.update_mode,
                    step_buffer_context=gate_sb_ctx,
                    meta_skill_context=meta_ctx,
                    rejected_buffer_context=gate_rb_ctx,
                )
                for p in gate_patches:
                    p.patch.reasoning = (
                        f"[MOSCOPT:GATE] " + (p.patch.reasoning or "")
                    )
                all_patches.extend(gate_patches)
                self._last_analysis_failures += gate_n_failures

        return all_patches

    def _extract_gate_failures(
        self,
        results: list[RolloutResult],
        pool: SkillPool,
    ) -> list[RolloutResult]:
        """Identify rollout results likely caused by poor gate selections.

        Returns failures annotated with attribution type in ``extras``:
        - ``missed_high_q``: a high-Q skill was not activated while activated
          skills had low average Q.
        - ``bad_combo``: individually high-Q skills were activated but the
          combination still failed.
        - (no tag): generic gate failure based on avg-Q heuristic.
        """
        gate_failures: list[RolloutResult] = []
        all_sids = set(pool.skills.keys())
        for r in results:
            if r.hard != 0:
                continue
            activated = get_activated_skill_ids(r)
            if not activated:
                continue
            activated_set = set(activated)
            activated_qs = [pool.q_scores.get(sid, 0.0) for sid in activated]
            avg_q = sum(activated_qs) / max(len(activated_qs), 1)

            # Missed high-Q skill detection (Section 3.5)
            inactive_sids = all_sids - activated_set
            high_q_inactive = [
                sid for sid in inactive_sids
                if pool.q_scores.get(sid, 0.0) > 0.7
                and pool.activation_counts.get(sid, 0) >= self.min_activations
            ]
            if high_q_inactive and avg_q < 0.5:
                r.extras["gate_failure_type"] = "missed_high_q"
                r.extras["missed_skills"] = high_q_inactive
                gate_failures.append(r)
                continue

            # Bad combo detection: individually good skills but combined failure
            if all(q > 0.5 for q in activated_qs) and len(activated_qs) > 1:
                r.extras["gate_failure_type"] = "bad_combo"
                gate_failures.append(r)
                continue

            # Generic gate failure
            if avg_q > 0.5:
                gate_failures.append(r)
        return gate_failures

    def _extract_gate_successes(
        self,
        results: list[RolloutResult],
        pool: SkillPool,
    ) -> list[RolloutResult]:
        """Identify rollout results with good gate selections (positive feedback).

        Returns successes annotated with ``gate_success_type`` in extras:
        - ``good_combo``: multiple high-Q skills activated and task succeeded.
        - ``high_activation``: single high-Q skill activated and task succeeded.
        """
        gate_successes: list[RolloutResult] = []
        for r in results:
            if r.hard != 1:
                continue
            activated = get_activated_skill_ids(r)
            if not activated:
                continue
            activated_qs = [pool.q_scores.get(sid, 0.0) for sid in activated]
            avg_q = sum(activated_qs) / max(len(activated_qs), 1)

            if avg_q > 0.7 and len(activated_qs) > 1:
                r.extras["gate_success_type"] = "good_combo"
                gate_successes.append(r)
            elif avg_q > 0.7:
                r.extras["gate_success_type"] = "high_activation"
                gate_successes.append(r)
        return gate_successes

    # ══════════════════════════════════════════════════════════════
    # Stage 3: Aggregate
    # ══════════════════════════════════════════════════════════════

    async def aggregate(
        self,
        patches: list[RawPatch],
        skill: str,
    ) -> Patch:
        """Hierarchical merge of patches (skill + gate combined).

        When multiple skills' patches are present (from multi-skill reflect),
        filters to only the primary target skill + gate patches to ensure
        clean merge context.
        """
        pool = await self._ensure_pool_initialized(skill)

        # Filter patches for primary skill + gate (GAP-2: multi-skill reflect compat)
        target = self._current_target_skill
        primary_tag = f"[MOSCOPT:SKILL:{target}]" if target else None
        gate_tag = "[MOSCOPT:GATE]"

        filtered_patches = []
        for p in patches:
            reasoning = p.patch.reasoning or ""
            if primary_tag and primary_tag in reasoning:
                filtered_patches.append(p)
            elif gate_tag in reasoning:
                filtered_patches.append(p)
            elif not primary_tag:
                # Single-skill mode: accept all
                filtered_patches.append(p)

        if len(filtered_patches) < len(patches):
            logger.info(
                "[AGGREGATE] filtered {}/{} patches for primary skill {}",
                len(filtered_patches), len(patches), target,
            )

        failure_patches: list[dict] = []
        success_patches: list[dict] = []
        for p in filtered_patches:
            d = p.patch.to_dict()
            if not d.get("edits"):
                continue
            if p.source_type == "success":
                success_patches.append(d)
            else:
                failure_patches.append(d)

        meta_ctx = format_meta_skill_context(self._meta_skill_content)
        rb_ctx = ""
        if self.use_rejected_buffer:
            target = self._current_target_skill
            if target and target in self._skill_reject_buffers:
                rb_ctx = self._skill_reject_buffers[target].format_context()

        # Get the correct skill text for merge context
        merge_skill = skill
        if target and pool.get_skill(target):
            merge_skill = pool.get_skill(target)

        return await merge_patches(
            provider=self.provider,
            model=self.optimizer_model,
            skill_content=merge_skill,
            failure_patches=failure_patches,
            success_patches=success_patches,
            update_mode=self.update_mode,
            meta_skill_context=meta_ctx,
            workers=self.aggregate_workers,
            rejected_buffer_context=rb_ctx,
        )

    # ══════════════════════════════════════════════════════════════
    # Stage 4: Select
    # ══════════════════════════════════════════════════════════════

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
        """Rank edits and select top-L (gradient clipping)."""
        pool = await self._ensure_pool_initialized(skill)
        meta_ctx = format_meta_skill_context(self._meta_skill_content)

        # Determine target skill text
        target = self._current_target_skill
        select_skill = skill
        if target and pool.get_skill(target):
            select_skill = pool.get_skill(target)
        elif "[MOSCOPT:GATE]" in (patch.reasoning or ""):
            select_skill = pool.gate

        actual_budget = budget
        if self.lr_mode == "autonomous" and self.provider:
            try:
                lr_record = await decide_autonomous_learning_rate(
                    provider=self.provider,
                    model=self.optimizer_model,
                    skill_content=select_skill,
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
                logger.error("[SELECT] autonomous LR failed: {}", exc)

        return await rank_and_select(
            provider=self.provider,
            model=self.optimizer_model,
            skill_content=select_skill,
            patch=patch,
            max_edits=actual_budget,
            update_mode=self.update_mode,
            meta_skill_context=meta_ctx,
        )

    # ══════════════════════════════════════════════════════════════
    # Stage 5: Update
    # ══════════════════════════════════════════════════════════════

    async def update(
        self,
        skill: str,
        patch: Patch,
    ) -> tuple[str, list[dict]]:
        """Apply edits to the appropriate skill or gate.

        Routes based on ``[MOSCOPT:SKILL:ID]`` / ``[MOSCOPT:GATE]``
        markers in the patch reasoning.
        """
        pool = await self._ensure_pool_initialized(skill)
        reasoning = patch.reasoning or ""
        report: list[dict] = []

        # Determine target
        is_gate = "[MOSCOPT:GATE]" in reasoning
        target_sid: str | None = None
        if not is_gate:
            import re
            m = re.search(r"\[MOSCOPT:SKILL:(\w+)\]", reasoning)
            if m:
                target_sid = m.group(1)

        if is_gate:
            # Save pre-edit gate for validation in evaluate() (Section 3.8)
            self._pre_edit_gate = pool.gate
            # Apply to gate text
            new_gate, gate_report = apply_patch_with_report(pool.gate, patch)
            pool.gate = new_gate
            report.extend(gate_report)
            # Track for delayed batch validation (Section 3.11)
            self._pending_val_candidates.append(("gate", pool.gate, new_gate, None))
            logger.info("[UPDATE] applied {} edits to gate (pre-edit saved)", len(patch.edits))
        elif target_sid and target_sid in pool.skills:
            # Apply to specific skill
            old_text = pool.skills[target_sid]
            new_skill_text, skill_report = apply_patch_with_report(
                pool.skills[target_sid], patch,
            )
            pool.skills[target_sid] = new_skill_text
            report.extend(skill_report)
            # Track for delayed batch validation (Section 3.11)
            self._pending_val_candidates.append(
                (f"skill:{target_sid}", old_text, new_skill_text, patch)
            )
            logger.info(
                "[UPDATE] applied {} edits to skill {}",
                len(patch.edits), target_sid,
            )
        else:
            # Fallback: apply to full compound doc
            new_text, fallback_report = apply_patch_with_report(skill, patch)
            report.extend(fallback_report)
            return new_text, report

        # Re-serialize the pool
        return serialize_pool(pool), report

    # ══════════════════════════════════════════════════════════════
    # Stage 6: Evaluate
    # ══════════════════════════════════════════════════════════════

    async def evaluate(
        self,
        env: Any,
        skill: str,
        items: list[dict],
        out_dir: str,
    ) -> float:
        """Evaluate current pool+gate on validation items."""
        self._last_evaluate_soft_score = 0.0
        self._last_rollout_results = []

        # Validation sampling (Section 5.5, p_val)
        if self.val_sample_ratio < 1.0 and items:
            sample_size = max(1, int(len(items) * self.val_sample_ratio))
            items = random.sample(items, sample_size)

        pool = await self._ensure_pool_initialized(skill)

        # ── Gate validation (Section 3.8): accept/reject gate edit ──────
        if self._pre_edit_gate and not self._is_single_skill:
            new_gate_score = await self._evaluate_gate_candidate(env, pool, items)
            # Temporarily restore old gate to measure its quality
            old_gate = pool.gate
            pool.gate = self._pre_edit_gate
            old_gate_score = await self._evaluate_gate_candidate(env, pool, items)
            pool.gate = old_gate  # restore new gate for now

            if new_gate_score > old_gate_score:
                logger.info(
                    "[EVALUATE] gate edit accepted: new={:.3f} > old={:.3f}",
                    new_gate_score, old_gate_score,
                )
            else:
                # Rollback gate to pre-edit version
                pool.gate = self._pre_edit_gate
                logger.info(
                    "[EVALUATE] gate edit rejected: new={:.3f} <= old={:.3f}; rolling back",
                    new_gate_score, old_gate_score,
                )
                if self.use_rejected_buffer:
                    self._gate_reject_buffer.add(
                        step=getattr(self, "_current_step", 0),
                        edits=[],
                        score_before=old_gate_score,
                        score_after=new_gate_score,
                    )
            self._pre_edit_gate = None

        # ── Delayed batch validation of skill candidates (Section 3.11) ──
        if self._pending_val_candidates and not self._is_single_skill:
            for cand_id, old_text, new_text, patch_info in self._pending_val_candidates:
                if cand_id == "gate":
                    continue  # gate already validated above
                if not cand_id.startswith("skill:"):
                    continue
                sid = cand_id.split(":", 1)[1]
                if sid not in pool.skills:
                    continue
                # Mini validation: sample a few items, compare old vs new
                sample_n = min(5, len(items))
                val_sample = random.sample(items, sample_n) if sample_n < len(items) else items
                # Test with new text
                pool.skills[sid] = new_text
                new_ids = fallback_top_k(
                    pool.q_scores, self.activate_count,
                    pool.activation_counts, c_min=self.min_activations,
                )
                new_texts = {s: pool.skills[s] for s in new_ids if s in pool.skills}
                new_eff = build_agent_prompt(new_texts)
                new_results = await env.rollout_batch(
                    val_sample, new_eff, phase_label="EVAL(skill-val)",
                )
                new_score = sum(r.hard for r in new_results) / max(len(new_results), 1)
                # Test with old text
                pool.skills[sid] = old_text
                old_texts = {s: pool.skills[s] for s in new_ids if s in pool.skills}
                old_eff = build_agent_prompt(old_texts)
                old_results = await env.rollout_batch(
                    val_sample, old_eff, phase_label="EVAL(skill-val)",
                )
                old_score = sum(r.hard for r in old_results) / max(len(old_results), 1)
                # Accept or rollback
                if new_score > old_score:
                    pool.skills[sid] = new_text
                    logger.info(
                        "[EVALUATE] skill {} edit accepted (mini-val): new={:.3f} > old={:.3f}",
                        sid, new_score, old_score,
                    )
                else:
                    pool.skills[sid] = old_text
                    logger.info(
                        "[EVALUATE] skill {} edit rejected (mini-val): new={:.3f} <= old={:.3f}",
                        sid, new_score, old_score,
                    )
                    # Record to Rejected Buffer (Section 3.6, pseudocode L598-603)
                    if self.use_rejected_buffer and patch_info and sid in self._skill_reject_buffers:
                        self._skill_reject_buffers[sid].add(
                            step=getattr(self, "_current_step", 0),
                            edits=patch_info.edits if hasattr(patch_info, "edits") else [],
                            score_before=old_score,
                            score_after=new_score,
                        )
            logger.info(
                "[EVALUATE] batch validated {} pending candidates",
                len(self._pending_val_candidates),
            )
            self._pending_val_candidates.clear()

        # Build effective skill for evaluation
        if self._is_single_skill:
            sid = pool.skill_ids()[0]
            effective_skill = pool.skills[sid]
        else:
            activated_ids = fallback_top_k(
                pool.q_scores, self.activate_count,
                pool.activation_counts, c_min=self.min_activations,
            )
            activated_texts = {
                sid: pool.skills[sid]
                for sid in activated_ids if sid in pool.skills
            }
            effective_skill = build_agent_prompt(activated_texts)

        logger.info("[6/6 EVALUATE] {} val items", len(items))
        results = await env.rollout_batch(items, effective_skill, phase_label="6/6 EVALUATE")

        if not results:
            return 0.0

        self._last_rollout_results = list(results)
        hard_acc = sum(r.hard for r in results) / len(results)
        soft_mean = sum(r.soft for r in results) / len(results)
        self._last_evaluate_soft_score = soft_mean
        self._current_score = hard_acc

        n_timeout = sum(
            1 for r in results
            if r.fail_reason.startswith("rollout_timeout")
        )
        n_error = sum(
            1 for r in results
            if r.fail_reason
            and not r.fail_reason.startswith("rollout_timeout")
            and r.hard == 0
        )
        n_correct = sum(1 for r in results if r.hard == 1)

        logger.info(
            "[6/6 EVALUATE] hard_acc={:.3f} soft={:.3f} correct={}/{} timeout={} error={}",
            hard_acc, soft_mean, n_correct, len(results), n_timeout, n_error,
        )
        return hard_acc

    # ══════════════════════════════════════════════════════════════
    # Epoch end: Slow Update + Meta Skill + Collective Evolution
    # ══════════════════════════════════════════════════════════════

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
        """Epoch-level slow update + meta skill + collective evolution.

        1. Per-skill Slow Update (longitudinal comparison → protected region)
        2. Gate Slow Update
        3. Global Meta Skill update (optimizer memory)
        4. Phase 3 Collective Evolution (every E epochs):
           cull → breed → merge → diversity check → summary update
        """
        if not self.provider:
            logger.warning("[EPOCH END] no provider; skipping")
            return skill

        prev_res = prev_results or self._prev_epoch_results
        curr_res = curr_results or self._curr_epoch_last_results
        epoch_items = items or self._curr_epoch_last_items

        logger.info(
            "[EPOCH END] epoch={} prev={} curr={} items={}",
            epoch, len(prev_res), len(curr_res), len(epoch_items),
        )

        pool = await self._ensure_pool_initialized(skill)

        # ═══ Slow Update (per-skill + gate) ═══════════════════════
        if self.use_slow_update and prev_res and curr_res and epoch_items:
            for sid in list(pool.skills.keys()):
                skill_text = pool.skills[sid]
                skill_text = inject_empty_slow_update_field(skill_text)
                comparison_pairs = build_comparison_pairs(
                    prev_res, curr_res, epoch_items,
                )

                if out_dir:
                    su_dir = os.path.join(
                        out_dir, "slow_update", f"epoch_{epoch:02d}", f"skill_{sid}",
                    )
                    os.makedirs(su_dir, exist_ok=True)
                    save_comparison_pairs(
                        comparison_pairs,
                        os.path.join(su_dir, "comparison_pairs.json"),
                    )

                prev_skill_text = self._prev_epoch_pool.get(sid, skill_text)
                su_result = await run_slow_update(
                    provider=self.provider,
                    model=self.optimizer_model,
                    prev_skill=prev_skill_text,
                    curr_skill=skill_text,
                    comparison_pairs=comparison_pairs,
                )

                if su_result.guidance:
                    pool.skills[sid] = replace_slow_update_field(
                        skill_text, su_result.guidance,
                    )
                    logger.info(
                        "[SLOW UPDATE] skill {} epoch={} action={} guidance={} chars",
                        sid, epoch, su_result.action, len(su_result.guidance),
                    )

                    if out_dir:
                        su_dir = os.path.join(
                            out_dir, "slow_update", f"epoch_{epoch:02d}", f"skill_{sid}",
                        )
                        os.makedirs(su_dir, exist_ok=True)
                        with open(os.path.join(su_dir, "guidance.txt"), "w") as f:
                            f.write(su_result.guidance)
                        with open(os.path.join(su_dir, "reasoning.txt"), "w") as f:
                            f.write(su_result.reasoning)
                else:
                    logger.info(
                        "[SLOW UPDATE] skill {} epoch={} action={} (no guidance)",
                        sid, epoch, su_result.action,
                    )

            # Gate slow update
            pool.gate = inject_empty_slow_update_field(pool.gate)
            gate_comparison = build_comparison_pairs(prev_res, curr_res, epoch_items)

            # Inject gate positive feedback into slow update (Section 3.8)
            if self._last_gate_successes:
                success_lines = []
                for gs in self._last_gate_successes[:10]:
                    activated = get_activated_skill_ids(gs)
                    stype = gs.extras.get("gate_success_type", "unknown")
                    success_lines.append(f"activated={activated} type={stype}")
                gate_pos_ctx = (
                    "## Gate Successful Patterns (this epoch)\n"
                    + "\n".join(success_lines)
                )
                for pair in gate_comparison:
                    pair["context"] = pair.get("context", "") + "\n" + gate_pos_ctx

            gate_su = await run_slow_update(
                provider=self.provider,
                model=self.optimizer_model,
                prev_skill=self._prev_epoch_pool.get("_gate", pool.gate),
                curr_skill=pool.gate,
                comparison_pairs=gate_comparison,
            )
            if gate_su.guidance:
                pool.gate = replace_slow_update_field(pool.gate, gate_su.guidance)
                logger.info("[SLOW UPDATE] gate epoch={} guidance={} chars", epoch, len(gate_su.guidance))

        elif not self.use_slow_update:
            logger.info("[SLOW UPDATE] epoch={} disabled", epoch)
        else:
            logger.info("[SLOW UPDATE] epoch={} skipped (insufficient data)", epoch)

        # ═══ Meta Skill (global optimizer memory) ═════════════════
        if self.use_meta_skill and prev_res and curr_res and epoch_items:
            comparison_pairs_meta = build_comparison_pairs(prev_res, curr_res, epoch_items)
            # Aggregate top-3 skills as global representatives (Section 3.7.4)
            top_sids = sorted(
                pool.skills,
                key=lambda s: pool.q_scores.get(s, 0.0),
                reverse=True,
            )[:3]
            agg_prev = "\n\n---\n\n".join(
                f"### Skill {sid}\n{self._prev_epoch_pool.get(sid, pool.skills[sid])}"
                for sid in top_sids
            )
            agg_curr = "\n\n---\n\n".join(
                f"### Skill {sid}\n{pool.skills[sid]}"
                for sid in top_sids
            )

            meta_result = await run_meta_skill(
                provider=self.provider,
                model=self.optimizer_model,
                prev_skill=agg_prev,
                curr_skill=agg_curr,
                comparison_pairs=comparison_pairs_meta,
                prev_meta_skill_content=self._meta_skill_content,
            )

            if meta_result:
                self._meta_skill_content = meta_result["meta_skill_content"]
                logger.info("[META SKILL] epoch={} updated ({} chars)", epoch, len(self._meta_skill_content))

                if out_dir:
                    ms_dir = os.path.join(out_dir, "meta_skill", f"epoch_{epoch:02d}")
                    os.makedirs(ms_dir, exist_ok=True)
                    with open(os.path.join(ms_dir, "meta_skill.json"), "w") as f:
                        json.dump(meta_result, f, ensure_ascii=False, indent=2)
            else:
                logger.info("[META SKILL] epoch={} no update", epoch)

        # ═══ Update epoch tracking ════════════════════════════════
        self._prev_epoch_pool = {sid: pool.skills[sid] for sid in pool.skills}
        self._prev_epoch_pool["_gate"] = pool.gate
        self._prev_epoch_results = list(curr_res) if curr_res else []
        self._prev_epoch_items = list(epoch_items) if epoch_items else []
        self._curr_epoch_last_results = []
        self._curr_epoch_last_items = []

        # ═══ Phase 3: Collective Evolution (every E epochs) ══════
        if epoch > 0 and epoch % self.evolution_interval == 0:
            await self._collective_evolution(pool, epoch, out_dir)

        # ═══ Convergence detection (Section 3.10) ═══════════════════
        # Three convergence signals; 2/3 required to declare convergence
        convergence_signals = 0
        convergence_reasons: list[str] = []

        # Signal 1: Score stability
        curr_score = self._current_score
        if curr_score == 0.0 and pool.q_scores:
            curr_score = sum(pool.q_scores.values()) / len(pool.q_scores)
        self._convergence_window.append(curr_score)
        if len(self._convergence_window) > self._convergence_window_size:
            self._convergence_window.pop(0)
        if len(self._convergence_window) >= self._convergence_window_size:
            score_range = max(self._convergence_window) - min(self._convergence_window)
            if score_range < self._convergence_threshold:
                convergence_signals += 1
                convergence_reasons.append(
                    f"score_stable (range={score_range:.4f}<{self._convergence_threshold})"
                )

        # Signal 2: Pool size stability
        if self._prev_pool_size > 0 and abs(pool.size - self._prev_pool_size) == 0:
            convergence_signals += 1
            convergence_reasons.append(f"pool_stable (size={pool.size})")

        # Signal 3: Gate distribution concentration (entropy)
        total_sel = sum(self._gate_selection_counts.values())
        if total_sel > 0 and pool.size > 1:
            max_entropy = math.log(pool.size)
            entropy = 0.0
            for cnt in self._gate_selection_counts.values():
                if cnt > 0:
                    p = cnt / total_sel
                    entropy -= p * math.log(p)
            # Concentrated when entropy < 30% of maximum
            if max_entropy > 0 and entropy < max_entropy * 0.3:
                convergence_signals += 1
                convergence_reasons.append(
                    f"gate_concentrated (entropy={entropy:.3f}<{max_entropy * 0.3:.3f})"
                )

        self._prev_pool_size = pool.size

        if convergence_signals >= 2 and not self.converged:
            self.converged = True
            logger.info(
                "[CONVERGENCE] detected at epoch {} ({} signals: {})",
                epoch, convergence_signals, ", ".join(convergence_reasons),
            )

        # ═══ Save pool history snapshot (Dashboard API) ════════════
        self._pool_history.append({
            "epoch": epoch,
            "pool_size": pool.size,
            "q_scores": dict(pool.q_scores),
            "activation_counts": dict(pool.activation_counts),
            "skill_ids": list(pool.skills.keys()),
            "converged": self.converged,
        })

        # Re-serialize pool
        pool.epoch = epoch
        return serialize_pool(pool)

    # ── Collective evolution ───────────────────────────────────────

    async def _collective_evolution(
        self,
        pool: SkillPool,
        epoch: int,
        out_dir: str,
    ) -> None:
        """Phase 3: cull → breed → merge → diversity → summary update."""
        logger.info(
            "[EVOLUTION] epoch={} pool_size={} M={} K={}",
            epoch, pool.size, self.evolution_count, self.activate_count,
        )

        M = self.evolution_count

        # ── 3a. Cull lowest-scored skills ─────────────────────────
        # Protect high co-occurrence pairs
        protected: set[str] = set()
        top_pair = get_top_cooccurrence_pair(pool)
        if top_pair:
            max_cooc = top_pair[1]
            for si, partners in pool.cooccurrence.items():
                for sj, count in partners.items():
                    if count >= max_cooc * 0.5:
                        protected.add(si)
                        protected.add(sj)

        culled = select_lowest_scored(
            pool, M,
            min_activations=self.min_activations,
            protected=protected,
        )
        for sid in culled:
            logger.info("[EVOLUTION] culling skill {} (Q={:.3f})", sid, pool.q_scores.get(sid, 0.0))
            del pool.skills[sid]
            pool.q_scores.pop(sid, None)
            pool.activation_counts.pop(sid, None)
            pool.summaries.pop(sid, None)
            pool.cooccurrence.pop(sid, None)
            self._skill_reject_buffers.pop(sid, None)
            self._skill_step_buffers.pop(sid, None)
            # Clean co-occurrence references
            for partners in pool.cooccurrence.values():
                partners.pop(sid, None)

        # ── 3b. Breed from top parents ────────────────────────────
        parents = select_top_parents(pool, M)
        new_skills: list[tuple[str, str]] = []  # (label, text)
        for parent_sid in parents:
            parent_text = pool.skills.get(parent_sid, "")
            inherited_rules = extract_slow_update_field(parent_text) or ""
            mutated = await mutate_skill(
                self.provider, self.optimizer_model,
                parent_text, inherited_rules,
            )
            label = f"Mutant of {pool.summaries.get(parent_sid, {}).get('label', parent_sid)}"
            new_skills.append((label, mutated))
            logger.info("[EVOLUTION] bred from skill {} -> new variant", parent_sid)

        # Add new skills to pool
        next_id = max((int(s) for s in pool.skills), default=0) + 1
        for label, text in new_skills[:M]:
            sid = str(next_id)
            text = inject_empty_slow_update_field(text)
            pool.skills[sid] = text
            pool.q_scores[sid] = 0.0
            pool.activation_counts[sid] = 0
            pool.summaries[sid] = {"id": sid, "label": label}
            pool.cooccurrence[sid] = {}
            self._skill_reject_buffers[sid] = RejectedBuffer(
                max_size=self._rb_max_size,
                max_summary_chars=self._rb_max_chars,
            )
            self._skill_step_buffers[sid] = []
            next_id += 1

        # ── 3c. Co-occurrence merge (optional) ──────────────────
        if top_pair and top_pair[1] > 5:
            (si, sj), _ = top_pair
            if si in pool.skills and sj in pool.skills and si not in protected:
                # Use specialized merge prompt (Section 3.10) instead of generic mutate
                from .reflect import _call_llm
                merge_user = (
                    f"## Skill {si}: {pool.summaries.get(si, {}).get('label', si)} "
                    f"(Q={pool.q_scores.get(si, 0.0):.2f})\n{pool.skills[si]}\n\n"
                    f"---\n\n"
                    f"## Skill {sj}: {pool.summaries.get(sj, {}).get('label', sj)} "
                    f"(Q={pool.q_scores.get(sj, 0.0):.2f})\n{pool.skills[sj]}\n\n"
                    "Merge the above two co-occurring skills into a single unified skill "
                    "document that combines their complementary strengths."
                )
                try:
                    merge_result = await _call_llm(
                        provider=self.provider,
                        model=self.optimizer_model,
                        system=_MERGE_DISTILL_SYSTEM,
                        user=merge_user,
                        max_tokens=8192,
                        retries=2,
                        stage="cooccurrence_merge",
                    )
                    merged_text = (
                        merge_result.strip()
                        if merge_result and len(merge_result.strip()) > 100
                        else pool.skills[si] + "\n\n---\n\n" + pool.skills[sj]
                    )
                except Exception as exc:
                    logger.warning("[EVOLUTION] co-occurrence merge LLM failed: {}; concatenating", exc)
                    merged_text = pool.skills[si] + "\n\n---\n\n" + pool.skills[sj]
                merged_id = str(next_id)
                merged_text = inject_empty_slow_update_field(merged_text)
                pool.skills[merged_id] = merged_text
                pool.q_scores[merged_id] = max(
                    pool.q_scores.get(si, 0.0), pool.q_scores.get(sj, 0.0),
                )
                pool.activation_counts[merged_id] = 0
                pool.summaries[merged_id] = {
                    "id": merged_id,
                    "label": f"Merged {pool.summaries.get(si, {}).get('label', si)}+{pool.summaries.get(sj, {}).get('label', sj)}",
                }
                pool.cooccurrence[merged_id] = {}
                self._skill_reject_buffers[merged_id] = RejectedBuffer(
                    max_size=self._rb_max_size, max_summary_chars=self._rb_max_chars,
                )
                self._skill_step_buffers[merged_id] = []
                next_id += 1

                # Remove lowest remaining to keep pool at N
                remaining = [
                    s for s in pool.skills
                    if s not in {si, sj, merged_id}
                ]
                if remaining:
                    lowest = min(remaining, key=lambda s: pool.q_scores.get(s, 0.0))
                    if pool.q_scores.get(lowest, 0.0) < pool.q_scores.get(merged_id, 0.0):
                        del pool.skills[lowest]
                        pool.q_scores.pop(lowest, None)
                        pool.activation_counts.pop(lowest, None)
                        pool.summaries.pop(lowest, None)
                        pool.cooccurrence.pop(lowest, None)
                        self._skill_reject_buffers.pop(lowest, None)
                        self._skill_step_buffers.pop(lowest, None)

                logger.info("[EVOLUTION] merged skills {} + {} -> {}", si, sj, merged_id)

        # ── 3d. Diversity check + foreign gene injection (Section 5.2) ──
        diversity = compute_diversity(pool)
        if diversity > self.diversity_threshold:
            logger.warning(
                "[EVOLUTION] low diversity ({:.2f} > {:.2f}), injecting foreign gene",
                diversity, self.diversity_threshold,
            )
            try:
                fg_label, fg_text = await inject_foreign_gene(
                    self.provider, self.optimizer_model, pool,
                )
            except Exception as exc:
                logger.warning("[EVOLUTION] foreign gene injection failed: {}", exc)
                fg_label, fg_text = "Diverse Explorer", (
                    "# Diverse Explorer\n\n"
                    "You are an exploratory problem-solver. Try unconventional approaches "
                    "and consider multiple alternative strategies before committing.\n"
                )
            new_sid = str(max((int(s) for s in pool.skills), default=0) + 1)
            new_skill = inject_empty_slow_update_field(fg_text)
            pool.skills[new_sid] = new_skill
            pool.q_scores[new_sid] = 0.0
            pool.activation_counts[new_sid] = 0
            pool.summaries[new_sid] = {"id": new_sid, "label": fg_label}
            pool.cooccurrence[new_sid] = {}
            self._skill_reject_buffers[new_sid] = RejectedBuffer(
                max_size=self._rb_max_size, max_summary_chars=self._rb_max_chars,
            )
            self._skill_step_buffers[new_sid] = []

            # Additional forced mutation of top parent (Section 5.2)
            top_parents = select_top_parents(pool, 1)
            if top_parents:
                parent_sid = top_parents[0]
                parent_text = pool.skills.get(parent_sid, "")
                inherited_rules = extract_slow_update_field(parent_text) or ""
                try:
                    forced_mutant = await mutate_skill(
                        self.provider, self.optimizer_model,
                        parent_text, inherited_rules, force=True,
                    )
                    fm_sid = str(max((int(s) for s in pool.skills), default=0) + 1)
                    fm_text = inject_empty_slow_update_field(forced_mutant)
                    pool.skills[fm_sid] = fm_text
                    pool.q_scores[fm_sid] = 0.0
                    pool.activation_counts[fm_sid] = 0
                    pool.summaries[fm_sid] = {
                        "id": fm_sid,
                        "label": f"Forced mutant of {pool.summaries.get(parent_sid, {}).get('label', parent_sid)}",
                    }
                    pool.cooccurrence[fm_sid] = {}
                    self._skill_reject_buffers[fm_sid] = RejectedBuffer(
                        max_size=self._rb_max_size, max_summary_chars=self._rb_max_chars,
                    )
                    self._skill_step_buffers[fm_sid] = []
                    logger.info("[EVOLUTION] forced mutation of skill {} -> {}", parent_sid, fm_sid)
                except Exception as exc:
                    logger.warning("[EVOLUTION] forced mutation failed: {}", exc)

        # ── 3e. Re-index and update summaries ─────────────────────
        reassign_skill_ids(pool)
        update_summaries(pool, epoch, self.summary_enrichment_epochs)

        # Rebuild per-skill buffers after re-indexing
        new_rb: dict[str, RejectedBuffer] = {}
        new_sb: dict[str, list[str]] = {}
        for sid in pool.skills:
            new_rb[sid] = self._skill_reject_buffers.get(
                sid,
                RejectedBuffer(max_size=self._rb_max_size, max_summary_chars=self._rb_max_chars),
            )
            new_sb[sid] = self._skill_step_buffers.get(sid, [])
        self._skill_reject_buffers = new_rb
        self._skill_step_buffers = new_sb

        logger.info("[EVOLUTION] done: pool_size={}", pool.size)

    # ── Gate independent validation (Section 3.8) ─────────────────────

    async def _evaluate_gate_candidate(
        self,
        env: Any,
        pool: SkillPool,
        items: list[dict],
    ) -> float:
        """Evaluate gate selection quality on a sample of validation items.

        Returns the average Q-score of skills selected by the current gate
        across a sample of tasks.  Higher is better.
        """
        if not items or pool.size == 0:
            return 0.0

        sample_size = min(len(items), max(1, int(len(items) * self.val_sample_ratio)))
        sample = random.sample(items, sample_size) if sample_size < len(items) else items
        summary_table = format_summary_table(pool, pool.epoch, self.summary_enrichment_epochs)
        valid_ids = set(pool.skills.keys())

        total_q = 0.0
        n_calls = 0
        for item in sample[:10]:  # cap at 10 to limit LLM cost
            state = item.get("question", "")
            activated_ids, _ = await call_gate_llm(
                provider=self.provider,
                model=self.optimizer_model,
                gate_text=pool.gate,
                summary_table=summary_table,
                state=state,
                history="",
                k=self.activate_count,
                valid_ids=valid_ids,
            )
            if activated_ids is None:
                activated_ids = fallback_top_k(
                    pool.q_scores, self.activate_count,
                    pool.activation_counts, c_min=self.min_activations,
                )
            avg_q = sum(pool.q_scores.get(sid, 0.0) for sid in activated_ids) / max(len(activated_ids), 1)
            total_q += avg_q
            n_calls += 1

        return total_q / max(n_calls, 1)
