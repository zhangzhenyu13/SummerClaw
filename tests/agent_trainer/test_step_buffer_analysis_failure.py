"""Unit tests for SkillOpt step buffer accumulation and analysis failure handling.

Covers:
- SkillOptAlgorithm.update_step_buffer: epoch-local accumulation
- SkillOptAlgorithm.on_epoch_start: step buffer clearing
- SkillOptAlgorithm._last_analysis_failures: tracking
- reflect.run_error_analyst_minibatch: failure_summary extraction
- reflect.run_minibatch_reflect: analysis failure counting
- state_dict / load_state_dict: step buffer persistence
"""
from __future__ import annotations

import json
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.algorithm import SkillOptAlgorithm
from summerclaw.agent_trainer.algorithms.skillopt.reflect import (
    run_error_analyst_minibatch,
    run_minibatch_reflect,
)
from summerclaw.agent_trainer.types import (
    Edit,
    FailureSummaryEntry,
    Patch,
    RawPatch,
    RolloutResult,
)

MockResponse = namedtuple("MockResponse", ["content"])


def _make_mock_provider(response_text: str):
    """Create a mock provider that returns the given text."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MockResponse(content=response_text))
    provider.max_concurrency = 4
    return provider


def _make_rollout_result(hard: int = 0, soft: float = 0.0, id: str = "test_001"):
    """Create a minimal RolloutResult."""
    return RolloutResult(
        id=id,
        hard=hard,
        soft=soft,
        trajectory=[{"role": "user", "content": "test"}],
        question="test question",
        task_type="test",
    )


def _make_edit(op: str = "append", content: str = "new rule", target: str = ""):
    return Edit(op=op, content=content, target=target)


# ── Step buffer accumulation ────────────────────────────────────────────

class TestStepBufferAccumulation:
    def test_update_step_buffer_appends_entry(self):
        """update_step_buffer should add an entry and rebuild context."""
        algo = SkillOptAlgorithm(provider=None, model="test")

        algo.update_step_buffer(
            step=1,
            rollout_hard=0.5,
            rollout_soft=0.6,
            n_patches=3,
            n_analysis_failures=0,
            gate_action="accept",
            score_before=0.4,
            score_after=0.5,
        )

        assert len(algo._step_buffer_entries) == 1
        assert "Step 1" in algo._step_buffer_context
        assert "rollout_hard=0.5" in algo._step_buffer_context
        assert "gate=accept" in algo._step_buffer_context

    def test_update_step_buffer_accumulates_multiple_steps(self):
        """Multiple calls should accumulate entries."""
        algo = SkillOptAlgorithm(provider=None, model="test")

        for i in range(1, 4):
            algo.update_step_buffer(
                step=i,
                rollout_hard=0.5 + i * 0.1,
                rollout_soft=0.6,
                n_patches=2,
                gate_action="accept",
                score_before=0.4,
                score_after=0.5,
            )

        assert len(algo._step_buffer_entries) == 3
        assert "Step 1" in algo._step_buffer_context
        assert "Step 2" in algo._step_buffer_context
        assert "Step 3" in algo._step_buffer_context

    def test_update_step_buffer_includes_failure_summaries(self):
        """failure_summaries should be rendered in the buffer entry."""
        algo = SkillOptAlgorithm(provider=None, model="test")

        fs = [
            FailureSummaryEntry(failure_type="wrong_answer", count=3, description="Model gave incorrect response"),
            FailureSummaryEntry(failure_type="tool_error", count=1, description="Tool call failed"),
        ]
        algo.update_step_buffer(
            step=1,
            n_patches=2,
            gate_action="reject",
            failure_summaries=fs,
            score_before=0.5,
            score_after=0.4,
        )

        assert "wrong_answer" in algo._step_buffer_context
        assert "tool_error" in algo._step_buffer_context
        assert "incorrect response" in algo._step_buffer_context

    def test_update_step_buffer_includes_edit_summaries(self):
        """selected_edits should be rendered in the buffer entry."""
        algo = SkillOptAlgorithm(provider=None, model="test")

        edits = [
            _make_edit("append", "Add error handling rule"),
            _make_edit("replace", "Update timeout value"),
        ]
        algo.update_step_buffer(
            step=1,
            n_patches=2,
            gate_action="accept",
            selected_edits=edits,
            score_before=0.4,
            score_after=0.5,
        )

        assert "Add error handling rule" in algo._step_buffer_context
        assert "Update timeout value" in algo._step_buffer_context

    def test_update_step_buffer_tracks_analysis_failures(self):
        """_analysis_failure_count should accumulate."""
        algo = SkillOptAlgorithm(provider=None, model="test")

        algo.update_step_buffer(step=1, n_analysis_failures=2, gate_action="skip_no_patches")
        algo.update_step_buffer(step=2, n_analysis_failures=1, gate_action="accept")

        assert algo._analysis_failure_count == 3


class TestStepBufferEpochClearing:
    def test_on_epoch_start_clears_step_buffer(self):
        """on_epoch_start should clear step buffer entries."""
        algo = SkillOptAlgorithm(provider=None, model="test")

        algo.update_step_buffer(step=1, n_patches=2, gate_action="accept")
        assert len(algo._step_buffer_entries) == 1

        algo.on_epoch_start(epoch=2)

        assert len(algo._step_buffer_entries) == 0
        assert algo._step_buffer_context == ""
        assert algo._analysis_failure_count == 0

    def test_on_epoch_start_clears_rejected_buffer(self):
        """on_epoch_start should also clear the rejected buffer."""
        algo = SkillOptAlgorithm(provider=None, model="test", use_rejected_buffer=True)

        algo._rejected_buffer.add(
            step=1,
            edits=[{"op": "append", "content": "bad rule"}],
            score_before=0.5,
            score_after=0.4,
        )
        assert len(algo._rejected_buffer) == 1

        algo.on_epoch_start(epoch=2)

        assert algo._rejected_buffer.is_empty()


# ── State persistence ────────────────────────────────────────────────────

class TestStepBufferPersistence:
    def test_state_dict_includes_step_buffer(self):
        """state_dict should serialize step buffer state."""
        algo = SkillOptAlgorithm(provider=None, model="test")
        algo.update_step_buffer(step=1, n_patches=2, gate_action="accept", n_analysis_failures=1)

        state = algo.state_dict()

        assert "step_buffer_entries" in state
        assert "analysis_failure_count" in state
        assert len(state["step_buffer_entries"]) == 1
        assert state["analysis_failure_count"] == 1

    def test_load_state_dict_restores_step_buffer(self):
        """load_state_dict should restore step buffer state."""
        algo1 = SkillOptAlgorithm(provider=None, model="test")
        algo1.update_step_buffer(step=1, n_patches=2, gate_action="accept", n_analysis_failures=1)
        state = algo1.state_dict()

        algo2 = SkillOptAlgorithm(provider=None, model="test")
        algo2.load_state_dict(state)

        assert len(algo2._step_buffer_entries) == 1
        assert algo2._analysis_failure_count == 1
        assert "Step 1" in algo2._step_buffer_context


# ── Analysis failure handling ───────────────────────────────────────────

class TestAnalysisFailureHandling:
    @pytest.mark.asyncio
    async def test_error_analyst_returns_none_on_llm_failure(self):
        """LLM call failure should return None (analysis failure)."""
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API error"))

        results = [_make_rollout_result(hard=0)]
        patch = await run_error_analyst_minibatch(
            provider, "test-model",
            skill_content="# Skill",
            results=results,
            system_prompt="Test prompt",
        )
        assert patch is None

    @pytest.mark.asyncio
    async def test_error_analyst_returns_none_on_bad_json(self):
        """Unparseable JSON should return None (analysis failure)."""
        provider = _make_mock_provider("this is not json at all {{{")

        results = [_make_rollout_result(hard=0)]
        patch = await run_error_analyst_minibatch(
            provider, "test-model",
            skill_content="# Skill",
            results=results,
            system_prompt="Test prompt",
        )
        assert patch is None

    @pytest.mark.asyncio
    async def test_error_analyst_extracts_failure_summary(self):
        """LLM response with failure_summary should be parsed."""
        response = json.dumps({
            "patch": {
                "reasoning": "Found issues",
                "edits": [{"op": "append", "content": "fix this"}],
            },
            "failure_summary": [
                {"failure_type": "wrong_answer", "count": 3, "description": "Model was wrong"},
                {"failure_type": "tool_error", "count": 1, "description": "Tool failed"},
            ],
            "batch_size": 2,
        })
        provider = _make_mock_provider(response)

        results = [_make_rollout_result(hard=0), _make_rollout_result(hard=0, id="test_002")]
        patch = await run_error_analyst_minibatch(
            provider, "test-model",
            skill_content="# Skill",
            results=results,
            system_prompt="Test prompt",
        )

        assert patch is not None
        assert patch.source_type == "failure"
        assert len(patch.failure_summary) == 2
        assert patch.failure_summary[0].failure_type == "wrong_answer"
        assert patch.failure_summary[0].count == 3
        assert patch.failure_summary[1].failure_type == "tool_error"

    @pytest.mark.asyncio
    async def test_error_analyst_empty_failure_summary(self):
        """LLM response without failure_summary should have empty list."""
        response = json.dumps({
            "patch": {
                "reasoning": "Found issues",
                "edits": [{"op": "append", "content": "fix"}],
            },
        })
        provider = _make_mock_provider(response)

        results = [_make_rollout_result(hard=0)]
        patch = await run_error_analyst_minibatch(
            provider, "test-model",
            skill_content="# Skill",
            results=results,
            system_prompt="Test prompt",
        )

        assert patch is not None
        assert patch.failure_summary == []

    @pytest.mark.asyncio
    async def test_run_minibatch_reflect_counts_failures(self, tmp_path):
        """run_minibatch_reflect should return (patches, n_analysis_failures)."""
        # Provider always fails — all minibatches should fail
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API down"))
        provider.max_concurrency = 4

        # One failure and one success → two minibatches
        results = [
            _make_rollout_result(hard=0, id="fail_1"),
            _make_rollout_result(hard=1, id="succ_1"),
        ]

        patches, n_failures = await run_minibatch_reflect(
            provider=provider,
            model="test-model",
            results=results,
            skill_content="# Skill",
            patches_dir=str(tmp_path / "patches"),
            workers=2,
            minibatch_size=5,
            edit_budget=4,
            error_system="Test error prompt",
            success_system="Test success prompt",
        )

        assert isinstance(patches, list)
        assert isinstance(n_failures, int)
        # Both minibatches failed (error + success analyst)
        assert n_failures == 2
        assert patches == []

    @pytest.mark.asyncio
    async def test_run_minibatch_reflect_all_fail(self, tmp_path):
        """When ALL minibatches fail, should return empty list + full count."""
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API down"))
        provider.max_concurrency = 4

        results = [
            _make_rollout_result(hard=0, id="fail_1"),
            _make_rollout_result(hard=0, id="fail_2"),
        ]

        patches, n_failures = await run_minibatch_reflect(
            provider=provider,
            model="test-model",
            results=results,
            skill_content="# Skill",
            patches_dir=str(tmp_path / "patches"),
            workers=2,
            minibatch_size=5,
            edit_budget=4,
            error_system="Test error prompt",
            success_system="Test success prompt",
        )

        assert patches == []
        assert n_failures == 1  # 1 failure minibatch, all failed


# ── Reflect integration with algorithm ──────────────────────────────────

class TestReflectIntegration:
    @pytest.mark.asyncio
    async def test_reflect_stores_analysis_failures(self, tmp_path):
        """algorithm.reflect should store _last_analysis_failures."""
        response = json.dumps({
            "patch": {
                "reasoning": "ok",
                "edits": [{"op": "append", "content": "fix"}],
            },
        })
        provider = _make_mock_provider(response)
        provider.max_concurrency = 4

        algo = SkillOptAlgorithm(
            provider=provider,
            model="test-model",
            minibatch_size=5,
            workers=2,
        )

        results = [_make_rollout_result(hard=0)]
        patches = await algo.reflect(
            results=results,
            skill="# Skill",
            out_dir=str(tmp_path),
        )

        assert isinstance(patches, list)
        assert hasattr(algo, "_last_analysis_failures")
        assert isinstance(algo._last_analysis_failures, int)
