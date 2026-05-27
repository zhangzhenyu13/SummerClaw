"""Unit tests for agent_trainer evaluation gate."""
from __future__ import annotations

from summerclaw.agent_trainer.evaluation.gate import evaluate_gate


class TestEvaluateGate:
    def test_accept_new_best(self):
        result = evaluate_gate(
            candidate_skill="new_skill",
            cand_hard=0.9,
            current_skill="old_skill",
            current_score=0.7,
            best_skill="old_skill",
            best_score=0.8,
            best_step=3,
            global_step=5,
        )
        assert result.action == "accept_new_best"
        assert result.current_skill == "new_skill"
        assert result.current_score == 0.9
        assert result.best_skill == "new_skill"
        assert result.best_score == 0.9
        assert result.best_step == 5

    def test_accept_not_best(self):
        result = evaluate_gate(
            candidate_skill="better_skill",
            cand_hard=0.85,
            current_skill="old_skill",
            current_score=0.7,
            best_skill="best_skill",
            best_score=0.9,
            best_step=3,
            global_step=5,
        )
        assert result.action == "accept"
        assert result.current_skill == "better_skill"
        assert result.current_score == 0.85
        assert result.best_skill == "best_skill"
        assert result.best_score == 0.9
        assert result.best_step == 3

    def test_reject(self):
        result = evaluate_gate(
            candidate_skill="worse_skill",
            cand_hard=0.5,
            current_skill="current_skill",
            current_score=0.7,
            best_skill="best_skill",
            best_score=0.9,
            best_step=3,
            global_step=5,
        )
        assert result.action == "reject"
        assert result.current_skill == "current_skill"
        assert result.current_score == 0.7
        assert result.best_skill == "best_skill"
        assert result.best_score == 0.9

    def test_equal_score_reject(self):
        """Equal score should be rejected (no improvement)."""
        result = evaluate_gate(
            candidate_skill="same_skill",
            cand_hard=0.7,
            current_skill="current_skill",
            current_score=0.7,
            best_skill="best_skill",
            best_score=0.9,
            best_step=3,
            global_step=5,
        )
        assert result.action == "reject"

    def test_accept_new_best_from_scratch(self):
        """First evaluation should always be accept_new_best."""
        result = evaluate_gate(
            candidate_skill="initial_skill",
            cand_hard=0.3,
            current_skill="initial_skill",
            current_score=-1.0,
            best_skill="initial_skill",
            best_score=-1.0,
            best_step=0,
            global_step=1,
        )
        assert result.action == "accept_new_best"
        assert result.best_score == 0.3
