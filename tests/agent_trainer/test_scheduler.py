"""Unit tests for SkillOpt LR scheduler module."""
from __future__ import annotations

import math

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.scheduler import (
    AutonomousScheduler,
    ConstantScheduler,
    CosineScheduler,
    LinearScheduler,
    build_scheduler,
)


class TestConstantScheduler:
    def test_always_returns_max_lr(self):
        s = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        for _ in range(10):
            assert s.step() == 8

    def test_beyond_total_steps(self):
        s = ConstantScheduler(max_lr=4, min_lr=1, total_steps=3)
        for _ in range(10):
            assert s.step() == 4

    def test_get_lr(self):
        s = ConstantScheduler(max_lr=6, min_lr=2, total_steps=10)
        assert s.get_lr(1) == 6
        assert s.get_lr(5) == 6
        assert s.get_lr(10) == 6


class TestLinearScheduler:
    def test_start_and_end(self):
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        assert s.get_lr(0) == 10
        assert s.get_lr(10) == 2

    def test_midpoint(self):
        s = LinearScheduler(max_lr=10, min_lr=0, total_steps=10)
        mid = s.get_lr(5)
        assert mid == 5

    def test_step_advances(self):
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=8)
        budgets = [s.step() for _ in range(8)]
        # Should be monotonically decreasing (or equal)
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1]
        assert budgets[0] >= budgets[-1]

    def test_single_step(self):
        s = LinearScheduler(max_lr=8, min_lr=2, total_steps=1)
        assert s.step() == 8  # edge case: total_steps <= 1

    def test_clamped_to_min(self):
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=5)
        for _ in range(20):
            assert s.step() >= 2


class TestCosineScheduler:
    def test_start_is_max(self):
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=10)
        assert s.get_lr(0) == 10

    def test_end_is_min(self):
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=10)
        assert s.get_lr(10) == 2

    def test_midpoint(self):
        s = CosineScheduler(max_lr=10, min_lr=0, total_steps=10)
        mid = s.get_lr(5)
        # cos(pi * 0.5) = 0 => mid = 0 + 0.5 * 10 * 1 = 5
        assert mid == 5

    def test_step_monotonic(self):
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=8)
        budgets = [s.step() for _ in range(8)]
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1]

    def test_single_step(self):
        s = CosineScheduler(max_lr=8, min_lr=2, total_steps=1)
        assert s.step() == 8


class TestAutonomousScheduler:
    def test_returns_no_limit(self):
        s = AutonomousScheduler(max_lr=10, min_lr=2, total_steps=5)
        for _ in range(5):
            assert s.step() == 999

    def test_get_lr(self):
        s = AutonomousScheduler(max_lr=10, min_lr=2, total_steps=5)
        assert s.get_lr(1) == AutonomousScheduler.NO_LIMIT
        assert s.get_lr(5) == AutonomousScheduler.NO_LIMIT


class TestBuildScheduler:
    def test_constant(self):
        s = build_scheduler(mode="constant", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, ConstantScheduler)
        assert s.step() == 8

    def test_linear(self):
        s = build_scheduler(mode="linear", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, LinearScheduler)

    def test_cosine(self):
        s = build_scheduler(mode="cosine", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, CosineScheduler)

    def test_autonomous(self):
        s = build_scheduler(mode="autonomous", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, AutonomousScheduler)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            build_scheduler(mode="invalid")


class TestStateDict:
    def test_save_and_restore_constant(self):
        s1 = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        for _ in range(3):
            s1.step()
        state = s1.state_dict()
        assert state["current_step"] == 3

        s2 = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        s2.load_state_dict(state)
        assert s2._current_step == 3
        # Next step should be 4
        s2.step()
        assert s2._current_step == 4

    def test_save_and_restore_linear(self):
        s1 = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(5):
            s1.step()
        state = s1.state_dict()

        s2 = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        s2.load_state_dict(state)
        assert s1._compute_lr(6) == s2._compute_lr(6)

    def test_load_empty_state(self):
        s = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        s.load_state_dict({})
        assert s._current_step == 0
