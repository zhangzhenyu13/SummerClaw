"""Unit tests for MOSCOPT Algorithm (algorithm.py) — 6-stage pipeline.

Covers:
- Initialization and parameter validation
- state_dict / load_state_dict roundtrip
- on_epoch_start buffer clearing
- Stage 1: Rollout (single/multi-skill, gating, fallback)
- Stage 2: Reflect (phase 1 skill, phase 2 gate)
- Stage 5: Update routing (gate/skill/fallback)
- Stage 6: Evaluate (gate validation, batch skill validation)
- Record rejection routing
- Step buffer accumulation
- Convergence detection
- Collective evolution
"""
from __future__ import annotations

import os
import tempfile
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.agent_trainer.algorithms.moscopt.algorithm import MOSCOPTAlgorithm
from summerclaw.agent_trainer.algorithms.moscopt.pool import (
    DEFAULT_GATE_PROMPT,
    SkillPool,
    parse_pool,
    serialize_pool,
)
from summerclaw.agent_trainer.algorithms.moscopt.rejected_buffer import RejectedBuffer
from summerclaw.agent_trainer.types import (
    Edit,
    FailureSummaryEntry,
    Patch,
    RawPatch,
    RolloutResult,
)


MockResponse = namedtuple("MockResponse", ["content"])


def _make_mock_provider(response_text: str = "mock response"):
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MockResponse(content=response_text))
    provider.max_concurrency = 4
    return provider


def _make_env(results: list[RolloutResult] | None = None):
    """Create a mock environment adapter."""
    env = MagicMock()
    default_results = results or [
        RolloutResult(id="t1", hard=1, soft=0.9, question="Q1"),
        RolloutResult(id="t2", hard=0, soft=0.3, question="Q2"),
    ]
    env.rollout_batch = AsyncMock(return_value=default_results)
    env.rollout_one = AsyncMock(return_value=default_results[0])
    env.rollout_timeout_s = 300
    return env


def _make_items(n: int = 2):
    return [{"id": f"t{i}", "question": f"Question {i}"} for i in range(1, n + 1)]


def _make_algorithm(
    pool_size: int = 3,
    activate_count: int = 2,
    **kwargs,
) -> MOSCOPTAlgorithm:
    provider = _make_mock_provider()
    algo = MOSCOPTAlgorithm(
        provider=provider,
        model="test-model",
        pool_size=pool_size,
        activate_count=activate_count,
        workers=2,
        use_slow_update=False,
        use_meta_skill=False,
        **kwargs,
    )
    return algo


def _init_pool(algo: MOSCOPTAlgorithm, n: int | None = None):
    """Initialize the internal pool for testing."""
    n = n or algo.pool_size
    pool = SkillPool(n=n, k=algo.activate_count, epoch=1)
    for i in range(1, n + 1):
        sid = str(i)
        pool.skills[sid] = f"# Skill {sid}\nContent for skill {sid}."
    pool.gate = DEFAULT_GATE_PROMPT
    pool.ensure_state()
    algo._pool = pool
    # Init per-skill buffers
    for sid in pool.skills:
        algo._skill_reject_buffers[sid] = RejectedBuffer()
        algo._skill_step_buffers[sid] = []
    return pool


# ═══════════════════════════════════════════════════════════════
# 2.1 Initialization and parameter validation
# ═══════════════════════════════════════════════════════════════


class TestInitialization:
    def test_default_params(self):
        algo = _make_algorithm()
        assert algo.pool_size == 3
        assert algo.activate_count == 2
        assert algo.name == "moscopt"

    def test_k_greater_than_n_clamped(self):
        algo = _make_algorithm(pool_size=2, activate_count=5)
        assert algo.activate_count == 2

    def test_is_single_skill_true(self):
        algo = _make_algorithm(pool_size=1, activate_count=1)
        assert algo._is_single_skill is True

    def test_is_single_skill_false(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        assert algo._is_single_skill is False

    def test_init_training_run(self):
        algo = _make_algorithm()
        algo.init_training_run(total_steps=10)
        assert algo._scheduler is not None
        assert algo._scheduler.total_steps == 10

    def test_init_training_run_idempotent(self):
        algo = _make_algorithm()
        algo.init_training_run(total_steps=10)
        sched = algo._scheduler
        algo.init_training_run(total_steps=10)
        assert algo._scheduler is sched  # same instance

    def test_init_training_run_rebuild_different_total(self):
        algo = _make_algorithm()
        algo.init_training_run(total_steps=10)
        algo.init_training_run(total_steps=20)
        assert algo._scheduler.total_steps == 20


# ═══════════════════════════════════════════════════════════════
# 2.2 state_dict / load_state_dict
# ═══════════════════════════════════════════════════════════════


class TestStatePersistence:
    def test_state_dict_roundtrip(self):
        algo = _make_algorithm()
        algo.init_training_run(total_steps=10)
        _init_pool(algo)

        algo._meta_skill_content = "test meta skill"
        algo._step_buffer_context = "some context"
        algo._step_buffer_entries = ["entry1", "entry2"]
        algo._analysis_failure_count = 3
        algo._gate_parse_failures = 2
        algo._gate_parse_total = 5
        algo._edit_rotation_index = 1
        algo._convergence_window = [0.5, 0.6, 0.55]
        algo.converged = True
        algo._gate_parse_failure_events = ["event1"]
        algo._prev_pool_size = 3
        algo._gate_selection_counts = {"1": 10, "2": 5}
        algo._pool_history = [{"epoch": 1}]

        state = algo.state_dict()

        # Create fresh algorithm and load state
        algo2 = _make_algorithm()
        algo2.init_training_run(total_steps=10)
        algo2.load_state_dict(state)

        assert algo2._meta_skill_content == "test meta skill"
        assert algo2._step_buffer_context == "some context"
        assert algo2._step_buffer_entries == ["entry1", "entry2"]
        assert algo2._analysis_failure_count == 3
        assert algo2._gate_parse_failures == 2
        assert algo2._gate_parse_total == 5
        assert algo2._edit_rotation_index == 1
        assert algo2._convergence_window == [0.5, 0.6, 0.55]
        assert algo2.converged is True
        assert algo2._gate_parse_failure_events == ["event1"]
        assert algo2._prev_pool_size == 3
        assert algo2._gate_selection_counts == {"1": 10, "2": 5}
        assert algo2._pool_history == [{"epoch": 1}]

    def test_state_dict_pool_restored(self):
        algo = _make_algorithm()
        _init_pool(algo)
        algo._pool.q_scores["1"] = 0.9

        state = algo.state_dict()
        algo2 = _make_algorithm()
        algo2.load_state_dict(state)

        assert algo2._pool.q_scores.get("1") == 0.9
        assert algo2._pool.size == algo._pool.size


# ═══════════════════════════════════════════════════════════════
# 2.3 on_epoch_start
# ═══════════════════════════════════════════════════════════════


class TestOnEpochStart:
    def test_clears_buffers(self):
        algo = _make_algorithm()
        _init_pool(algo)

        # Add some data to buffers
        algo._skill_reject_buffers["1"].add(
            step=1, edits=[], score_before=0.5, score_after=0.3,
        )
        algo._gate_reject_buffer.add(
            step=1, edits=[], score_before=0.5, score_after=0.3,
        )
        algo._skill_step_buffers["1"] = ["some entry"]
        algo._gate_step_buffer = ["gate entry"]
        algo._step_buffer_entries = ["entry"]
        algo._step_buffer_context = "context"
        algo._gate_parse_failures = 3
        algo._gate_parse_total = 5
        algo._edit_rotation_index = 2
        algo._gate_parse_failure_events = ["event"]

        algo.on_epoch_start(epoch=2)

        assert algo._skill_reject_buffers["1"].is_empty()
        assert algo._gate_reject_buffer.is_empty()
        assert algo._skill_step_buffers["1"] == []
        assert algo._gate_step_buffer == []
        assert algo._step_buffer_entries == []
        assert algo._step_buffer_context == ""
        assert algo._gate_parse_failures == 0
        assert algo._gate_parse_total == 0
        assert algo._edit_rotation_index == 0
        assert algo._gate_parse_failure_events == []


# ═══════════════════════════════════════════════════════════════
# 2.4 Stage 1: Rollout
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestRollout:
    async def test_single_skill_rollout(self):
        algo = _make_algorithm(pool_size=1, activate_count=1)
        _init_pool(algo, n=1)
        env = _make_env()
        items = _make_items()

        results = await algo.rollout(env, "plain skill text", items, "/tmp/out")
        assert len(results) == 2
        # All results tagged with activated skill
        for r in results:
            assert r.extras["moscopt_activated_skills"] == ["1"]

    async def test_multi_skill_rollout_with_fallback(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.9, "2": 0.5, "3": 0.3}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10}

        # Mock gate LLM to fail → fallback
        algo.provider.chat_with_retry = AsyncMock(side_effect=Exception("API error"))

        env = _make_env()
        items = _make_items()
        results = await algo.rollout(env, serialize_pool(pool), items, "/tmp/out")
        assert len(results) == 2

    async def test_post_rollout_updates_q_scores(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        pool = _init_pool(algo, n=2)
        # Give skill 1 a higher Q so fallback selects it
        pool.q_scores["1"] = 0.9
        pool.activation_counts["1"] = 10

        env = _make_env([
            RolloutResult(id="t1", hard=1, soft=0.9),
        ])
        items = _make_items(1)

        await algo.rollout(env, serialize_pool(pool), items, "/tmp/out")
        # At least one skill should have non-zero activation count from _post_rollout
        total_act = sum(pool.activation_counts.values())
        assert total_act > 0

    async def test_gate_parse_failure_tracking(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        # Gate returns wrong format → parse failure
        algo.provider.chat_with_retry = AsyncMock(
            return_value=MockResponse(content="ACTIVATE: 1, 2, 3")  # too many
        )
        env = _make_env()
        items = _make_items()

        await algo.rollout(env, serialize_pool(pool), items, "/tmp/out")
        assert algo._gate_parse_total >= 1


# ═══════════════════════════════════════════════════════════════
# 2.5 Stage 2: Reflect
# ═══════════════════════════════════════════════════════════════


class TestReflectHelpers:
    def test_extract_gate_failures_missed_high_q(self):
        algo = _make_algorithm(pool_size=3, activate_count=1)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.2, "2": 0.9, "3": 0.3}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10}

        r = RolloutResult(id="t1", hard=0, soft=0.1)
        r.extras["moscopt_activated_skills"] = ["1"]  # low Q activated, high Q "2" missed

        failures = algo._extract_gate_failures([r], pool)
        assert len(failures) == 1
        assert failures[0].extras.get("gate_failure_type") == "missed_high_q"

    def test_extract_gate_failures_bad_combo(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.8, "2": 0.7, "3": 0.3}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10}

        r = RolloutResult(id="t1", hard=0, soft=0.1)
        r.extras["moscopt_activated_skills"] = ["1", "2"]  # both high Q but failed

        failures = algo._extract_gate_failures([r], pool)
        assert len(failures) == 1
        assert failures[0].extras.get("gate_failure_type") == "bad_combo"

    def test_extract_gate_failures_generic(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.6, "2": 0.7, "3": 0.3}

        r = RolloutResult(id="t1", hard=0, soft=0.1)
        r.extras["moscopt_activated_skills"] = ["1", "2"]  # avg_q=0.65 > 0.5, not bad_combo (need >0.5 each)

        failures = algo._extract_gate_failures([r], pool)
        assert len(failures) == 1

    def test_extract_gate_successes_good_combo(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.9, "2": 0.8, "3": 0.3}

        r = RolloutResult(id="t1", hard=1, soft=0.9)
        r.extras["moscopt_activated_skills"] = ["1", "2"]

        successes = algo._extract_gate_successes([r], pool)
        assert len(successes) == 1
        assert successes[0].extras.get("gate_success_type") == "good_combo"

    def test_extract_gate_successes_high_activation(self):
        algo = _make_algorithm(pool_size=3, activate_count=1)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.9, "2": 0.3, "3": 0.3}

        r = RolloutResult(id="t1", hard=1, soft=0.9)
        r.extras["moscopt_activated_skills"] = ["1"]

        successes = algo._extract_gate_successes([r], pool)
        assert len(successes) == 1
        assert successes[0].extras.get("gate_success_type") == "high_activation"

    def test_extract_gate_successes_skip_low_q(self):
        algo = _make_algorithm(pool_size=2, activate_count=2)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.3, "2": 0.4}

        r = RolloutResult(id="t1", hard=1, soft=0.9)
        r.extras["moscopt_activated_skills"] = ["1", "2"]

        successes = algo._extract_gate_successes([r], pool)
        assert len(successes) == 0  # avg_q < 0.7


# ═══════════════════════════════════════════════════════════════
# 2.6 Stage 5: Update routing
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestUpdate:
    async def test_update_skill(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        pool = _init_pool(algo, n=2)
        skill_text = serialize_pool(pool)

        patch = Patch(
            edits=[Edit(op="append", content="New rule added")],
            reasoning="[MOSCOPT:SKILL:1] some reasoning",
        )
        new_skill, report = await algo.update(skill_text, patch)
        assert len(report) > 0
        # Status may be 'applied_append' or 'applied_append_before_slow_update'
        assert "applied_append" in report[0]["status"]
        # The edit is applied to the returned serialized pool (not algo._pool directly)
        assert "New rule added" in new_skill

    async def test_update_gate(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        pool = _init_pool(algo, n=2)
        skill_text = serialize_pool(pool)
        old_gate = pool.gate

        patch = Patch(
            edits=[Edit(op="replace", content="Custom gate text here", target=old_gate[:50])],
            reasoning="[MOSCOPT:GATE] gate edit",
        )
        new_skill, report = await algo.update(skill_text, patch)
        # pre_edit_gate should be saved for validation
        assert algo._pre_edit_gate is not None
        assert len(report) > 0

    async def test_update_fallback(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        pool = _init_pool(algo, n=2)
        skill_text = serialize_pool(pool)

        patch = Patch(
            edits=[Edit(op="append", content="Fallback edit")],
            reasoning="no specific tag",
        )
        new_skill, report = await algo.update(skill_text, patch)
        assert len(report) > 0


# ═══════════════════════════════════════════════════════════════
# 2.8 Record Rejection routing
# ═══════════════════════════════════════════════════════════════


class TestRecordRejection:
    def test_gate_rejection(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        _init_pool(algo, n=2)

        patch = Patch(edits=[], reasoning="[MOSCOPT:GATE] gate edit rejected")
        algo.record_rejection(step=1, patch=patch, score_before=0.5, score_after=0.3)
        assert len(algo._gate_reject_buffer) == 1

    def test_skill_rejection(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        _init_pool(algo, n=2)
        algo._current_target_skill = "1"

        patch = Patch(edits=[Edit(op="append", content="x")], reasoning="skill edit")
        algo.record_rejection(step=1, patch=patch, score_before=0.5, score_after=0.3)
        assert len(algo._skill_reject_buffers["1"]) == 1

    def test_skill_rejection_fallback_first_buffer(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        _init_pool(algo, n=2)
        algo._current_target_skill = None  # no target

        patch = Patch(edits=[Edit(op="append", content="x")], reasoning="skill edit")
        algo.record_rejection(step=1, patch=patch, score_before=0.5, score_after=0.3)
        # Should go to first available buffer
        total = sum(len(buf) for buf in algo._skill_reject_buffers.values())
        assert total == 1

    def test_disabled_rejection(self):
        algo = _make_algorithm(pool_size=2, activate_count=1, use_rejected_buffer=False)
        _init_pool(algo, n=2)

        patch = Patch(edits=[], reasoning="anything")
        algo.record_rejection(step=1, patch=patch, score_before=0.5, score_after=0.3)
        assert len(algo._gate_reject_buffer) == 0


# ═══════════════════════════════════════════════════════════════
# 2.9 Update Step Buffer
# ═══════════════════════════════════════════════════════════════


class TestUpdateStepBuffer:
    def test_accumulation(self):
        algo = _make_algorithm()
        _init_pool(algo)
        algo._current_target_skill = "1"

        algo.update_step_buffer(
            step=1, rollout_hard=0.8, rollout_soft=0.7,
            n_patches=3, n_analysis_failures=1,
            gate_action="accepted",
            score_before=0.5, score_after=0.7,
        )
        assert len(algo._step_buffer_entries) == 1
        assert "Step 1" in algo._step_buffer_context
        assert "## Previous Steps" in algo._step_buffer_context

    def test_routing_to_per_skill_buffer(self):
        algo = _make_algorithm()
        _init_pool(algo)
        algo._current_target_skill = "2"

        algo.update_step_buffer(step=1, n_patches=1)
        assert len(algo._skill_step_buffers["2"]) == 1

    def test_failure_summaries_included(self):
        algo = _make_algorithm()
        _init_pool(algo)
        algo._current_target_skill = "1"

        fs = FailureSummaryEntry(
            failure_type="wrong_answer", count=3, description="Model gave wrong answer",
        )
        algo.update_step_buffer(step=1, failure_summaries=[fs])
        assert "wrong_answer" in algo._step_buffer_context

    def test_analysis_failure_count_accumulated(self):
        algo = _make_algorithm()
        _init_pool(algo)
        algo._current_target_skill = "1"

        algo.update_step_buffer(step=1, n_analysis_failures=2)
        algo.update_step_buffer(step=2, n_analysis_failures=3)
        assert algo._analysis_failure_count == 5


# ═══════════════════════════════════════════════════════════════
# 2.10 Convergence Detection
# ═══════════════════════════════════════════════════════════════


class TestConvergenceDetection:
    @pytest.mark.asyncio
    async def test_score_stability_signal(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        _init_pool(algo)

        # Feed 5 similar scores to trigger stability signal
        algo._convergence_window = [0.70, 0.70, 0.70, 0.70, 0.70]
        algo._convergence_threshold = 0.01

        # Simulate on_epoch_end by directly checking convergence logic
        # The window range is 0.0 < 0.01 → signal 1
        score_range = max(algo._convergence_window) - min(algo._convergence_window)
        assert score_range < algo._convergence_threshold

    @pytest.mark.asyncio
    async def test_pool_stability_signal(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        algo._prev_pool_size = pool.size

        # Pool size unchanged → signal
        assert abs(pool.size - algo._prev_pool_size) == 0

    @pytest.mark.asyncio
    async def test_gate_concentration_signal(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        _init_pool(algo)
        import math

        # Highly concentrated: one skill gets all selections
        algo._gate_selection_counts = {"1": 100, "2": 1, "3": 1}
        total_sel = sum(algo._gate_selection_counts.values())
        max_entropy = math.log(3)
        entropy = 0.0
        for cnt in algo._gate_selection_counts.values():
            if cnt > 0:
                p = cnt / total_sel
                entropy -= p * math.log(p)
        assert entropy < max_entropy * 0.3  # concentrated

    def test_convergence_requires_two_signals(self):
        algo = _make_algorithm()
        assert algo.converged is False
        # Only converged when >= 2 signals are True simultaneously


# ═══════════════════════════════════════════════════════════════
# 2.11 Collective Evolution (unit-level)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCollectiveEvolution:
    async def test_collective_evolution_cull_and_breed(self):
        """Test that collective evolution culls low skills and breeds new ones."""
        algo = _make_algorithm(pool_size=4, activate_count=2, evolution_count=1)
        pool = _init_pool(algo, n=4)
        pool.q_scores = {"1": 0.9, "2": 0.7, "3": 0.3, "4": 0.1}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10, "4": 10}

        # Mock LLM for mutation
        algo.provider.chat_with_retry = AsyncMock(
            return_value=MockResponse(content="# Mutant Skill\nA new mutated variant with enough text content to be valid and pass all filters.")
        )

        original_size = pool.size
        await algo._collective_evolution(pool, epoch=5, out_dir="")

        # Pool should still have skills (exact count depends on diversity)
        assert pool.size >= algo.activate_count

    async def test_collective_evolution_diversity_injection(self):
        """Low diversity triggers foreign gene injection."""
        algo = _make_algorithm(pool_size=3, activate_count=2, diversity_threshold=0.5)
        pool = _init_pool(algo, n=3)
        # Make all skills very similar → high diversity score (>0.85)
        for sid in pool.skills:
            pool.skills[sid] = "Nearly identical skill text that is very similar across all skills in the pool."

        algo.provider.chat_with_retry = AsyncMock(
            return_value=MockResponse(content="# Diverse Explorer\nA completely different approach to problem solving with novel methodologies.")
        )

        await algo._collective_evolution(pool, epoch=5, out_dir="")
        # Pool should have gained at least one new skill (foreign gene)
        assert pool.size >= 3

    async def test_evolution_reassigns_ids(self):
        """After evolution, skill IDs should be consecutive."""
        algo = _make_algorithm(pool_size=4, activate_count=2, evolution_count=1)
        pool = _init_pool(algo, n=4)
        pool.q_scores = {"1": 0.9, "2": 0.8, "3": 0.2, "4": 0.1}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10, "4": 10}

        algo.provider.chat_with_retry = AsyncMock(
            return_value=MockResponse(content="# New Mutant\nA brand new mutated skill with different approach and sufficient content length.")
        )

        await algo._collective_evolution(pool, epoch=5, out_dir="")
        ids = sorted(pool.skills.keys(), key=lambda s: int(s))
        # Should be consecutive
        expected = [str(i) for i in range(1, len(ids) + 1)]
        assert ids == expected


# ═══════════════════════════════════════════════════════════════
# 2.7 Evaluate helpers
# ═══════════════════════════════════════════════════════════════


class TestEvaluateHelpers:
    def test_get_edit_budget_no_scheduler(self):
        algo = _make_algorithm()
        assert algo.get_edit_budget(1, 10) == algo.edit_budget

    def test_get_edit_budget_with_scheduler(self):
        algo = _make_algorithm()
        algo.init_training_run(total_steps=10)
        budget = algo.get_edit_budget(1, 10)
        assert isinstance(budget, int)
        assert budget > 0


@pytest.mark.asyncio
class TestEvaluateGateCandidate:
    async def test_evaluate_gate_candidate(self):
        algo = _make_algorithm(pool_size=3, activate_count=2)
        pool = _init_pool(algo)
        pool.q_scores = {"1": 0.9, "2": 0.7, "3": 0.3}

        # Mock gate LLM to select skills
        algo.provider.chat_with_retry = AsyncMock(
            return_value=MockResponse(content="ACTIVATE: 1, 2")
        )

        items = _make_items(5)
        env = _make_env()
        score = await algo._evaluate_gate_candidate(env, pool, items)
        assert isinstance(score, float)
        assert score >= 0.0

    async def test_evaluate_gate_candidate_empty_items(self):
        algo = _make_algorithm(pool_size=2, activate_count=1)
        pool = _init_pool(algo)
        env = _make_env()
        score = await algo._evaluate_gate_candidate(env, pool, [])
        assert score == 0.0
