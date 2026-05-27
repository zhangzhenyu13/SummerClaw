"""Unit tests for SkillOpt slow_update module (LLM-driven longitudinal analysis)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.slow_update import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    SlowUpdateResult,
    _format_trajectory,
    _strip_all_slow_update_fields,
    build_comparison_pairs,
    extract_slow_update_field,
    format_comparison_text,
    has_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    save_comparison_pairs,
)
from summerclaw.agent_trainer.types import RolloutResult


class TestSlowUpdateFieldManipulation:
    def test_has_slow_update_field(self):
        skill = f"# Skill\n{SLOW_UPDATE_START}\nguidance\n{SLOW_UPDATE_END}\n"
        assert has_slow_update_field(skill) is True

    def test_has_slow_update_field_missing(self):
        assert has_slow_update_field("# Skill\nNo markers") is False

    def test_extract_slow_update_field(self):
        skill = f"# Skill\n{SLOW_UPDATE_START}\nmy guidance\n{SLOW_UPDATE_END}\n"
        assert extract_slow_update_field(skill) == "my guidance"

    def test_extract_slow_update_field_missing(self):
        assert extract_slow_update_field("# Skill\n") is None

    def test_inject_empty_slow_update_field(self):
        skill = "# Skill\nContent"
        result = inject_empty_slow_update_field(skill)
        assert SLOW_UPDATE_START in result
        assert SLOW_UPDATE_END in result
        # Should not double-inject
        result2 = inject_empty_slow_update_field(result)
        assert result2 == result

    def test_replace_slow_update_field(self):
        skill = f"# Skill\n{SLOW_UPDATE_START}\nold\n{SLOW_UPDATE_END}\n"
        result = replace_slow_update_field(skill, "new guidance")
        assert "new guidance" in result
        assert "old" not in result
        assert result.count(SLOW_UPDATE_START) == 1
        assert result.count(SLOW_UPDATE_END) == 1

    def test_replace_slow_update_field_no_existing(self):
        skill = "# Skill\nContent"
        result = replace_slow_update_field(skill, "first guidance")
        assert "first guidance" in result
        assert SLOW_UPDATE_START in result

    def test_strip_all_slow_update_fields(self):
        skill = (
            f"# Skill\n"
            f"{SLOW_UPDATE_START}\nfirst\n{SLOW_UPDATE_END}\n"
            f"Content\n"
            f"{SLOW_UPDATE_START}\nsecond\n{SLOW_UPDATE_END}\n"
        )
        result = _strip_all_slow_update_fields(skill)
        assert SLOW_UPDATE_START not in result
        assert SLOW_UPDATE_END not in result
        assert "first" not in result
        assert "second" not in result
        assert "Content" in result


class TestBuildComparisonPairs:
    def test_basic_comparison(self):
        prev = [
            RolloutResult(id="t1", hard=0, soft=0.3),
            RolloutResult(id="t2", hard=1, soft=0.9),
            RolloutResult(id="t3", hard=0, soft=0.1),
            RolloutResult(id="t4", hard=1, soft=0.8),
        ]
        curr = [
            RolloutResult(id="t1", hard=1, soft=0.9),
            RolloutResult(id="t2", hard=0, soft=0.2),
            RolloutResult(id="t3", hard=0, soft=0.2),
            RolloutResult(id="t4", hard=1, soft=0.9),
        ]
        items = [
            {"id": "t1", "question": "Q1"},
            {"id": "t2", "question": "Q2"},
            {"id": "t3", "question": "Q3"},
            {"id": "t4", "question": "Q4"},
        ]

        pairs = build_comparison_pairs(prev, curr, items)
        assert len(pairs) == 4

        categories = [p["category"] for p in pairs]
        assert "improved" in categories  # t1: 0→1
        assert "regressed" in categories  # t2: 1→0
        assert "persistent_fail" in categories  # t3: 0→0
        assert "stable_success" in categories  # t4: 1→1

    def test_missing_ids(self):
        prev = [RolloutResult(id="t1", hard=1, soft=0.9)]
        curr = [RolloutResult(id="t2", hard=1, soft=0.9)]
        items = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
        pairs = build_comparison_pairs(prev, curr, items)
        # Only t1 and t2 have partial matches, but t3 has neither → skipped
        # t1 has prev but no curr, t2 has curr but no prev → both skipped
        assert len(pairs) == 0


class TestFormatComparisonText:
    def test_basic_formatting(self):
        pairs = [
            {
                "id": "t1",
                "task": "Task 1",
                "category": "regressed",
                "prev": {"hard": 1, "soft": 0.9, "predicted_answer": "A", "fail_reason": ""},
                "curr": {"hard": 0, "soft": 0.2, "predicted_answer": "B", "fail_reason": "wrong"},
                "prev_trajectory": "",
                "curr_trajectory": "[assistant] some output",
            },
        ]
        text = format_comparison_text(pairs)
        assert "Regressions" in text
        assert "t1" in text
        assert "wrong" in text

    def test_empty_pairs(self):
        text = format_comparison_text([])
        assert "Total samples: 0" in text


class TestSaveComparisonPairs:
    def test_save(self):
        pairs = [
            {
                "id": "t1",
                "task": "A" * 500,  # should be truncated to 300
                "category": "improved",
                "prev": {"hard": 0},
                "curr": {"hard": 1},
                "prev_trajectory": "should not be saved",
                "curr_trajectory": "should not be saved",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "pairs.json")
            save_comparison_pairs(pairs, out_path)
            assert os.path.exists(out_path)
            with open(out_path) as f:
                saved = json.load(f)
            assert len(saved) == 1
            assert "prev_trajectory" not in saved[0]
            assert "curr_trajectory" not in saved[0]
            assert len(saved[0]["task"]) <= 300


class TestFormatTrajectory:
    def test_assistant_messages(self):
        traj = [
            {"role": "assistant", "content": "I will search for that"},
            {"role": "user", "content": "Find me something"},
        ]
        text = _format_trajectory(traj)
        assert "assistant" in text
        assert "user" in text

    def test_tool_calls(self):
        traj = [
            {"type": "tool_call", "cmd": "search('test')", "obs": "found results"},
        ]
        text = _format_trajectory(traj)
        assert "search" in text
        assert "found results" in text

    def test_truncation(self):
        # Create multiple long messages to exceed max_chars
        traj = [
            {"role": "assistant", "content": "x" * 500},
            {"role": "user", "content": "y" * 500},
            {"role": "assistant", "content": "z" * 500},
        ]
        text = _format_trajectory(traj, max_chars=500)
        assert "[truncated]" in text


class TestSlowUpdateResult:
    def test_defaults(self):
        r = SlowUpdateResult()
        assert r.reasoning == ""
        assert r.guidance == ""
        assert r.action == ""
        assert r.prev_hard is None
        assert r.curr_hard is None

    def test_with_values(self):
        r = SlowUpdateResult(
            reasoning="improved",
            guidance="focus on X",
            action="improving",
            prev_hard=0.5,
            curr_hard=0.7,
        )
        assert r.guidance == "focus on X"
        assert r.curr_hard == 0.7
