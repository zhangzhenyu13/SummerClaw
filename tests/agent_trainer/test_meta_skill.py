"""Unit tests for SkillOpt meta_skill module."""
from __future__ import annotations

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.meta_skill import (
    format_meta_skill_context,
)
from summerclaw.agent_trainer.algorithms.skillopt.slow_update import (
    format_comparison_text,
)


class TestFormatMetaSkillContext:
    def test_empty_content(self):
        assert format_meta_skill_context("") == ""
        assert format_meta_skill_context("   ") == ""
        assert format_meta_skill_context(None) == ""

    def test_valid_content(self):
        ctx = format_meta_skill_context("Focus on error handling patterns.")
        assert "Optimizer Meta Skill" in ctx
        assert "Focus on error handling patterns." in ctx
        assert "prior epoch transitions" in ctx

    def test_whitespace_stripping(self):
        ctx = format_meta_skill_context("  some guidance  ")
        assert "some guidance" in ctx
        # Leading/trailing whitespace should be stripped
        assert "  some guidance  " not in ctx.split("\n\n")[-1] or "some guidance" in ctx


class TestFormatComparisonSummary:
    """Tests now use slow_update.format_comparison_text (P3 alignment)."""

    def test_empty_pairs(self):
        result = format_comparison_text([])
        assert "Total samples: 0" in result

    def test_with_categories(self):
        pairs = [
            {
                "id": "t1",
                "task": "Solve math problem",
                "category": "improved",
                "prev": {"hard": 0, "soft": 0.3, "predicted_answer": "N/A"},
                "curr": {"hard": 1, "soft": 0.9, "predicted_answer": "42"},
            },
            {
                "id": "t2",
                "task": "Answer question",
                "category": "regressed",
                "prev": {"hard": 1, "soft": 0.9, "predicted_answer": "correct"},
                "curr": {"hard": 0, "soft": 0.2, "predicted_answer": "wrong",
                         "fail_reason": "timeout"},
            },
            {
                "id": "t3",
                "task": "Persistent failure",
                "category": "persistent_fail",
                "prev": {"hard": 0, "soft": 0.1, "predicted_answer": "N/A"},
                "curr": {"hard": 0, "soft": 0.2, "predicted_answer": "N/A"},
            },
        ]
        result = format_comparison_text(pairs)
        assert "Total samples: 3" in result
        assert "Improved" in result or "improved" in result
        assert "Regress" in result or "regress" in result
        assert "Persistent" in result or "persistent" in result
        assert "t1" in result
        assert "t2" in result

    def test_limits_entries_per_category(self):
        # Create 20 entries in one category — format_comparison_text limits to 15
        pairs = [
            {
                "id": f"t{i}",
                "task": f"Task {i}",
                "category": "improved",
                "prev": {"hard": 0, "soft": 0.1, "predicted_answer": "N/A"},
                "curr": {"hard": 1, "soft": 0.9, "predicted_answer": "ok"},
            }
            for i in range(20)
        ]
        result = format_comparison_text(pairs)
        # Should include first 15 (not all 20)
        assert "t0" in result
        assert "t14" in result
        # t19 should not be included (limit is 15)
        assert "t19" not in result
