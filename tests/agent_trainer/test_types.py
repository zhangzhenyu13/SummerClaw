"""Unit tests for agent_trainer types module."""
from __future__ import annotations

import pytest

from summerclaw.agent_trainer.types import (
    Edit,
    EditOp,
    FailureSummaryEntry,
    GateAction,
    GateResult,
    Patch,
    RawPatch,
    RolloutResult,
    TrainingHistory,
    TrainingStep,
)


class TestEdit:
    def test_from_dict_minimal(self):
        e = Edit.from_dict({"op": "append", "content": "new rule"})
        assert e.op == "append"
        assert e.content == "new rule"
        assert e.target == ""
        assert e.support_count is None

    def test_from_dict_full(self):
        e = Edit.from_dict({
            "op": "replace",
            "content": "new text",
            "target": "old text",
            "support_count": 3,
            "source_type": "failure",
            "merge_level": 2,
        })
        assert e.op == "replace"
        assert e.target == "old text"
        assert e.support_count == 3
        assert e.source_type == "failure"
        assert e.merge_level == 2

    def test_to_dict_roundtrip(self):
        original = {
            "op": "insert_after",
            "content": "inserted",
            "target": "anchor",
            "support_count": 1,
            "source_type": "success",
        }
        e = Edit.from_dict(original)
        result = e.to_dict()
        assert result["op"] == "insert_after"
        assert result["content"] == "inserted"
        assert result["target"] == "anchor"
        assert result["support_count"] == 1
        assert result["source_type"] == "success"

    def test_to_dict_omits_empty(self):
        e = Edit(op="delete")
        d = e.to_dict()
        assert "target" not in d
        assert "support_count" not in d


class TestPatch:
    def test_from_dict_with_edits(self):
        p = Patch.from_dict({
            "edits": [
                {"op": "append", "content": "rule1"},
                {"op": "delete", "target": "old"},
            ],
            "reasoning": "merge result",
        })
        assert len(p.edits) == 2
        assert isinstance(p.edits[0], Edit)
        assert p.reasoning == "merge result"

    def test_to_dict_roundtrip(self):
        p = Patch(
            edits=[Edit(op="append", content="x")],
            reasoning="test",
        )
        d = p.to_dict()
        assert d["reasoning"] == "test"
        assert len(d["edits"]) == 1
        assert d["edits"][0]["op"] == "append"


class TestRolloutResult:
    def test_from_dict_basic(self):
        r = RolloutResult.from_dict({
            "id": "task_001",
            "hard": 1,
            "soft": 0.9,
            "n_turns": 5,
            "task_type": "qa",
        })
        assert r.id == "task_001"
        assert r.hard == 1
        assert r.soft == 0.9
        assert r.n_turns == 5
        assert r.task_type == "qa"

    def test_from_dict_extras(self):
        r = RolloutResult.from_dict({
            "id": "task_002",
            "hard": 0,
            "soft": 0.3,
            "custom_field": "extra_value",
        })
        assert r.extras == {"custom_field": "extra_value"}

    def test_to_dict_includes_extras(self):
        r = RolloutResult(id="t1", hard=1, soft=1.0, extras={"foo": "bar"})
        d = r.to_dict()
        assert d["foo"] == "bar"
        assert d["id"] == "t1"


class TestRawPatch:
    def test_from_dict(self):
        rp = RawPatch.from_dict({
            "patch": {
                "edits": [{"op": "append", "content": "x"}],
                "reasoning": "test",
            },
            "source_type": "failure",
            "batch_size": 5,
            "failure_summary": [
                {"failure_type": "wrong_tool", "count": 3, "description": "used wrong tool"},
            ],
        })
        assert rp is not None
        assert rp.source_type == "failure"
        assert rp.batch_size == 5
        assert len(rp.failure_summary) == 1
        assert rp.failure_summary[0].failure_type == "wrong_tool"

    def test_from_dict_none(self):
        assert RawPatch.from_dict(None) is None


class TestGateResult:
    def test_frozen(self):
        gr = GateResult(
            action="accept_new_best",
            current_skill="skill",
            current_score=0.8,
            best_skill="skill",
            best_score=0.8,
            best_step=5,
        )
        with pytest.raises(AttributeError):
            gr.action = "reject"


class TestTrainingStep:
    def test_roundtrip(self):
        s = TrainingStep(
            step=1, epoch=1, score=0.75, action="accept",
            skill_hash="abc123", n_edits_applied=3, n_edits_rejected=1,
        )
        d = s.to_dict()
        s2 = TrainingStep.from_dict(d)
        assert s2.step == 1
        assert s2.score == 0.75
        assert s2.action == "accept"
        assert s2.n_edits_applied == 3

    def test_step_rec_alignment_fields(self):
        """Verify all step_rec alignment fields survive round-trip."""
        s = TrainingStep(
            step=2, epoch=1, score=0.61, action="accept_new_best",
            step_in_epoch=0,
            timing={"rollout_s": 101.9, "reflect_s": 13.0},
            rollout_hard=0.14, rollout_soft=0.14, rollout_n=35,
            n_patches=5, n_failure_patches=4, n_success_patches=1,
            n_edits_merged=3, edit_budget=4, lr_control_mode="fixed",
            selection_hard=0.61, selection_soft=0.61,
            candidate_skill_len=3584,
            current_score=0.61, best_score=0.61, best_step=2,
            current_origin="step_0002", best_origin="step_0002",
            skill_len=3584, wall_time_s=223.6,
            edit_apply_summary={"total": 3, "applied": 3, "skipped": 0},
        )
        d = s.to_dict()
        s2 = TrainingStep.from_dict(d)
        assert s2.step_in_epoch == 0
        assert s2.timing["rollout_s"] == 101.9
        assert s2.rollout_hard == 0.14
        assert s2.n_patches == 5
        assert s2.n_failure_patches == 4
        assert s2.edit_budget == 4
        assert s2.selection_hard == 0.61
        assert s2.current_origin == "step_0002"
        assert s2.best_origin == "step_0002"
        assert s2.edit_apply_summary["applied"] == 3

    def test_defaults(self):
        """New fields have safe defaults for old code."""
        s = TrainingStep(step=1, epoch=1, score=0.0, action="skip_no_patches")
        assert s.timing == {}
        assert s.rollout_n == 0
        assert s.current_origin == ""
        assert s.edit_apply_summary == {}


class TestTrainingHistory:
    def test_add_step(self):
        h = TrainingHistory()
        h.add_step(TrainingStep(step=1, epoch=1, score=0.5, action="accept"))
        h.add_step(TrainingStep(step=2, epoch=1, score=0.7, action="accept_new_best"))
        h.add_step(TrainingStep(step=3, epoch=1, score=0.6, action="reject"))

        assert h.total_steps == 3
        assert h.best_score == 0.7
        assert h.best_step == 2

    def test_roundtrip(self):
        h = TrainingHistory()
        h.add_step(TrainingStep(step=1, epoch=1, score=0.5, action="accept"))
        d = h.to_dict()
        h2 = TrainingHistory.from_dict(d)
        assert h2.total_steps == 1
        assert h2.best_score == 0.5
