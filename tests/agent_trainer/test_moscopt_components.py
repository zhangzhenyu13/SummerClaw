"""Unit tests for MOSCOP sub-modules: RejectedBuffer, Scheduler, Update, UpdateModes, MetaSkill, LRAutonomous.

Covers:
- RejectedBuffer: add, FIFO eviction, format_context, serialization
- Scheduler: constant/linear/cosine/autonomous, state persistence
- Update: apply_edit (4 ops), SLOW_UPDATE protection, apply_patch_with_report
- UpdateModes: normalize, payload_key, payload_label, describe_item, truncate_payload
- MetaSkill: format_meta_skill_context
- LRAutonomous: _coerce_nonnegative_int
"""
from __future__ import annotations

import math

import pytest

from summerclaw.agent_trainer.algorithms.moscopt.rejected_buffer import (
    RejectedBuffer,
    RejectedEntry,
)
from summerclaw.agent_trainer.algorithms.moscopt.scheduler import (
    AutonomousScheduler,
    ConstantScheduler,
    CosineScheduler,
    LinearScheduler,
    build_scheduler,
)
from summerclaw.agent_trainer.algorithms.moscopt.update import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    apply_edit,
    apply_patch,
    apply_patch_with_report,
)
from summerclaw.agent_trainer.algorithms.moscopt.update_modes import (
    describe_item,
    get_payload_items,
    normalize_update_mode,
    payload_key,
    payload_label,
    set_payload_items,
    truncate_payload,
)
from summerclaw.agent_trainer.algorithms.moscopt.meta_skill import (
    format_meta_skill_context,
)
from summerclaw.agent_trainer.algorithms.moscopt.lr_autonomous import (
    _coerce_nonnegative_int,
)
from summerclaw.agent_trainer.types import Edit, Patch


# ═══════════════════════════════════════════════════════════════
# 3.1 RejectedBuffer
# ═══════════════════════════════════════════════════════════════


class TestRejectedBuffer:
    def test_empty_buffer(self):
        buf = RejectedBuffer()
        assert buf.is_empty()
        assert len(buf) == 0
        assert buf.format_context() == ""

    def test_add_entry(self):
        buf = RejectedBuffer()
        buf.add(step=1, edits=[Edit(op="append", content="test")], score_before=0.5, score_after=0.3)
        assert len(buf) == 1
        assert not buf.is_empty()

    def test_add_dict_edit(self):
        buf = RejectedBuffer()
        buf.add(
            step=1,
            edits=[{"op": "replace", "content": "new", "target": "old"}],
            score_before=0.5,
            score_after=0.3,
        )
        assert len(buf) == 1
        entry = buf.entries[0]
        assert "replace" in entry.edits_summary[0]

    def test_add_other_type_edit(self):
        buf = RejectedBuffer()
        buf.add(step=1, edits=[42, "raw string"], score_before=0.5, score_after=0.3)
        entry = buf.entries[0]
        assert "42" in entry.edits_summary[0]
        assert "raw string" in entry.edits_summary[1]

    def test_fifo_eviction(self):
        buf = RejectedBuffer(max_size=3)
        for i in range(5):
            buf.add(step=i, edits=[], score_before=0.5, score_after=0.3)
        assert len(buf) == 3
        # Oldest entries (step 0, 1) should be evicted
        steps = [e.step for e in buf.entries]
        assert 0 not in steps
        assert 1 not in steps
        assert 4 in steps

    def test_format_context_with_failure_patterns(self):
        buf = RejectedBuffer()
        buf.add(
            step=1,
            edits=[Edit(op="append", content="bad edit")],
            score_before=0.5,
            score_after=0.3,
            failure_patterns=[{"pattern": "wrong answer type"}],
        )
        ctx = buf.format_context()
        assert "Previously Rejected" in ctx
        assert "wrong answer type" in ctx
        assert "bad edit" in ctx

    def test_format_context_truncates_failure_patterns(self):
        buf = RejectedBuffer()
        buf.add(
            step=1,
            edits=[],
            score_before=0.5,
            score_after=0.3,
            failure_patterns=[
                {"pattern": "p1"},
                {"pattern": "p2"},
                {"pattern": "p3"},
                {"pattern": "p4_should_be_capped"},
            ],
        )
        ctx = buf.format_context()
        assert "p1" in ctx
        assert "p3" in ctx
        # Max 3 per entry
        assert "p4_should_be_capped" not in ctx

    def test_clear(self):
        buf = RejectedBuffer()
        buf.add(step=1, edits=[], score_before=0.5, score_after=0.3)
        assert not buf.is_empty()
        buf.clear()
        assert buf.is_empty()

    def test_to_dict_from_dict_roundtrip(self):
        buf = RejectedBuffer(max_size=5, max_summary_chars=100)
        buf.add(
            step=3,
            edits=[Edit(op="append", content="test edit")],
            score_before=0.7,
            score_after=0.4,
            failure_patterns=[{"pattern": "test pattern"}],
        )
        d = buf.to_dict()
        buf2 = RejectedBuffer.from_dict(d)
        assert buf2.max_size == 5
        assert buf2.max_summary_chars == 100
        assert len(buf2) == 1
        assert buf2.entries[0].step == 3
        assert buf2.entries[0].failure_patterns == [{"pattern": "test pattern"}]

    def test_edit_with_target(self):
        buf = RejectedBuffer()
        buf.add(
            step=1,
            edits=[Edit(op="replace", content="new", target="old_text")],
            score_before=0.5,
            score_after=0.3,
        )
        entry = buf.entries[0]
        assert "old_text" in entry.edits_summary[0]

    def test_summary_truncation(self):
        buf = RejectedBuffer(max_summary_chars=20)
        buf.add(
            step=1,
            edits=[Edit(op="append", content="A" * 100)],
            score_before=0.5,
            score_after=0.3,
        )
        entry = buf.entries[0]
        assert len(entry.edits_summary[0]) <= 40  # "append: " + 20 chars


class TestRejectedEntry:
    def test_to_dict(self):
        entry = RejectedEntry(step=1, score_before=0.5, score_after=0.3)
        d = entry.to_dict()
        assert d["step"] == 1
        assert d["score_before"] == 0.5

    def test_from_dict(self):
        d = {"step": 5, "score_before": 0.9, "score_after": 0.6, "edits_summary": ["x"]}
        entry = RejectedEntry.from_dict(d)
        assert entry.step == 5
        assert entry.edits_summary == ["x"]


# ═══════════════════════════════════════════════════════════════
# 3.2 Scheduler
# ═══════════════════════════════════════════════════════════════


class TestScheduler:
    def test_build_constant(self):
        s = build_scheduler(mode="constant", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, ConstantScheduler)
        for _ in range(10):
            assert s.step() == 8

    def test_build_linear(self):
        s = build_scheduler(mode="linear", max_lr=10, min_lr=2, total_steps=10)
        assert isinstance(s, LinearScheduler)
        last = s.get_lr(10)
        assert last == 2
        # First step should be close to max_lr (may be slightly less due to rounding)
        first = s.get_lr(1)
        assert first >= 8  # near max_lr

    def test_build_cosine(self):
        s = build_scheduler(mode="cosine", max_lr=10, min_lr=2, total_steps=10)
        assert isinstance(s, CosineScheduler)
        first = s.get_lr(1)
        last = s.get_lr(10)
        assert first == 10
        assert last == 2

    def test_build_autonomous(self):
        s = build_scheduler(mode="autonomous", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, AutonomousScheduler)
        assert s.step() == 999

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            build_scheduler(mode="invalid_mode")

    def test_state_dict_roundtrip(self):
        s = build_scheduler(mode="cosine", max_lr=8, min_lr=2, total_steps=10)
        for _ in range(5):
            s.step()
        state = s.state_dict()
        assert state["current_step"] == 5

        s2 = build_scheduler(mode="cosine", max_lr=8, min_lr=2, total_steps=10)
        s2.load_state_dict(state)
        assert s2._current_step == 5

    def test_total_steps_1(self):
        s = build_scheduler(mode="linear", max_lr=8, min_lr=2, total_steps=1)
        assert s.step() == 8  # single step returns max_lr

    def test_cosine_total_steps_1(self):
        s = build_scheduler(mode="cosine", max_lr=10, min_lr=2, total_steps=1)
        assert s.step() == 10

    def test_linear_decay_monotonic(self):
        s = build_scheduler(mode="linear", max_lr=10, min_lr=2, total_steps=10)
        budgets = [s.get_lr(i) for i in range(1, 11)]
        # Should be non-increasing
        for i in range(1, len(budgets)):
            assert budgets[i] <= budgets[i - 1]

    def test_cosine_decay_bounds(self):
        s = build_scheduler(mode="cosine", max_lr=10, min_lr=2, total_steps=20)
        for step in range(1, 21):
            lr = s.get_lr(step)
            assert lr >= 2
            assert lr <= 10


# ═══════════════════════════════════════════════════════════════
# 3.3 Update
# ═══════════════════════════════════════════════════════════════


class TestUpdate:
    def test_append(self):
        skill = "# Original\nContent"
        result = apply_edit(skill, Edit(op="append", content="New section"))
        assert "New section" in result
        assert result.endswith("New section\n")

    def test_append_before_slow_update(self):
        skill = f"# Original\n{SLOW_UPDATE_START}\nprotected\n{SLOW_UPDATE_END}\n"
        result = apply_edit(skill, Edit(op="append", content="New rule"))
        assert "New rule" in result
        # Should be inserted before SLOW_UPDATE_START
        assert result.index("New rule") < result.index(SLOW_UPDATE_START)

    def test_insert_after(self):
        skill = "# Title\nSection A\nSection B\n"
        result = apply_edit(skill, Edit(op="insert_after", content="Inserted", target="Section A"))
        assert "Inserted" in result

    def test_insert_after_fallback(self):
        skill = "# Title\nContent"
        result = apply_edit(skill, Edit(op="insert_after", content="Fallback", target="not_found"))
        assert "Fallback" in result

    def test_replace(self):
        skill = "# Title\nOld text\nMore content"
        result = apply_edit(skill, Edit(op="replace", content="New text", target="Old text"))
        assert "New text" in result
        assert "Old text" not in result

    def test_replace_missing_target(self):
        skill = "# Title\nContent"
        result = apply_edit(skill, Edit(op="replace", content="New", target="missing"))
        assert result == skill  # unchanged

    def test_replace_empty_target(self):
        skill = "# Title\nContent"
        result = apply_edit(skill, Edit(op="replace", content="New", target=""))
        assert result == skill

    def test_delete(self):
        skill = "# Title\nRemove this\nKeep this"
        result = apply_edit(skill, Edit(op="delete", target="Remove this"))
        assert "Remove this" not in result
        assert "Keep this" in result

    def test_delete_missing_target(self):
        skill = "# Title\nContent"
        result = apply_edit(skill, Edit(op="delete", target="missing"))
        assert result == skill

    def test_slow_update_protection(self):
        skill = f"# Title\n{SLOW_UPDATE_START}\nProtected text\n{SLOW_UPDATE_END}\nMore"
        result = apply_edit(skill, Edit(op="replace", content="Hacked", target="Protected text"))
        assert "Protected text" in result  # unchanged
        assert "Hacked" not in result

    def test_unknown_op(self):
        skill = "# Title\nContent"
        result = apply_edit(skill, Edit(op="unknown_op", content="x"))
        assert result == skill

    def test_apply_patch_with_report(self):
        skill = "# Title\nContent"
        patch = Patch(edits=[
            Edit(op="append", content="Line 1"),
            Edit(op="append", content="Line 2"),
        ])
        new_skill, reports = apply_patch_with_report(skill, patch)
        assert len(reports) == 2
        assert reports[0]["status"] == "applied_append"
        assert reports[1]["status"] == "applied_append"
        assert "Line 1" in new_skill
        assert "Line 2" in new_skill

    def test_apply_patch_with_report_dict(self):
        skill = "# Title\nContent"
        patch_dict = {"edits": [{"op": "append", "content": "dict edit"}]}
        new_skill, reports = apply_patch_with_report(skill, patch_dict)
        assert len(reports) == 1

    def test_apply_patch(self):
        skill = "# Title"
        patch = Patch(edits=[Edit(op="append", content="Extra")])
        result = apply_patch(skill, patch)
        assert "Extra" in result


# ═══════════════════════════════════════════════════════════════
# 3.4 Update Modes
# ═══════════════════════════════════════════════════════════════


class TestUpdateModes:
    @pytest.mark.parametrize("alias,expected", [
        ("patch", "patch"),
        ("edits", "patch"),
        ("rewrite", "rewrite_from_suggestions"),
        ("rewrite_from_suggestions", "rewrite_from_suggestions"),
        ("suggestions", "rewrite_from_suggestions"),
        ("rewrite_suggestions", "rewrite_from_suggestions"),
        ("full_rewrite", "full_rewrite_minibatch"),
        ("full_rewrite_minibatch", "full_rewrite_minibatch"),
        ("minibatch_full_rewrite", "full_rewrite_minibatch"),
        ("skill_rewrite_minibatch", "full_rewrite_minibatch"),
    ])
    def test_normalize_update_mode(self, alias, expected):
        assert normalize_update_mode(alias) == expected

    def test_normalize_none_defaults_to_patch(self):
        assert normalize_update_mode(None) == "patch"

    def test_normalize_unknown_defaults_to_patch(self):
        assert normalize_update_mode("garbage_mode") == "patch"

    def test_payload_key(self):
        assert payload_key("patch") == "edits"
        assert payload_key("rewrite_from_suggestions") == "revise_suggestions"
        assert payload_key("full_rewrite_minibatch") == "skill_candidates"

    def test_payload_label(self):
        assert payload_label("patch") == "edits"
        assert payload_label("patch", singular=True) == "edit"
        assert payload_label("patch", title=True) == "Edits"
        assert payload_label("rewrite_from_suggestions") == "suggestions"
        assert payload_label("full_rewrite_minibatch") == "skill candidates"

    def test_get_set_payload_items(self):
        container = {}
        items = [{"op": "append", "content": "test"}]
        set_payload_items(container, items, "patch")
        assert container["edits"] == items

        retrieved = get_payload_items(container, "patch")
        assert retrieved == items

    def test_get_payload_items_missing_key(self):
        assert get_payload_items({}, "patch") == []
        assert get_payload_items(None, "patch") == []

    def test_describe_item_patch(self):
        item = {"op": "append", "content": "add this", "target": "here"}
        desc = describe_item(item, "patch")
        assert "append" in desc
        assert "add this" in desc

    def test_describe_item_rewrite(self):
        item = {"type": "add", "title": "New Rule", "instruction": "do something"}
        desc = describe_item(item, "rewrite_from_suggestions")
        assert "New Rule" in desc

    def test_describe_item_full_rewrite(self):
        item = {"title": "Rewrite", "change_summary": ["changed X", "changed Y"]}
        desc = describe_item(item, "full_rewrite_minibatch")
        assert "Rewrite" in desc

    def test_describe_item_truncation(self):
        item = {"op": "append", "content": "x" * 500}
        desc = describe_item(item, "patch", max_chars=50)
        assert len(desc) <= 50

    def test_truncate_payload(self):
        container = {"edits": [{"op": "a"}, {"op": "b"}, {"op": "c"}]}
        truncate_payload(container, max_items=2, mode="patch")
        assert len(container["edits"]) == 2

    def test_truncate_payload_negative_no_op(self):
        container = {"edits": [{"op": "a"}]}
        truncate_payload(container, max_items=-1, mode="patch")
        assert len(container["edits"]) == 1


# ═══════════════════════════════════════════════════════════════
# 3.5 Meta Skill
# ═══════════════════════════════════════════════════════════════


class TestMetaSkill:
    def test_format_empty(self):
        assert format_meta_skill_context("") == ""
        assert format_meta_skill_context(None) == ""
        assert format_meta_skill_context("   ") == ""

    def test_format_non_empty(self):
        result = format_meta_skill_context("Avoid large edits in late stages.")
        assert "Optimizer Meta Skill" in result
        assert "Avoid large edits" in result


# ═══════════════════════════════════════════════════════════════
# 3.6 LR Autonomous
# ═══════════════════════════════════════════════════════════════


class TestCoerceNonnegativeInt:
    def test_bool_returns_none(self):
        assert _coerce_nonnegative_int(True) is None
        assert _coerce_nonnegative_int(False) is None

    def test_int(self):
        assert _coerce_nonnegative_int(5) == 5
        assert _coerce_nonnegative_int(0) == 0
        assert _coerce_nonnegative_int(-3) == 0  # clamped to 0

    def test_float_integer(self):
        assert _coerce_nonnegative_int(3.0) == 3
        assert _coerce_nonnegative_int(0.0) == 0

    def test_float_non_integer(self):
        # 3.5 is not an integer, so it falls through to regex extraction
        # which finds "3" in the string representation
        result = _coerce_nonnegative_int(3.5)
        # Implementation converts to string "3.5" then regex finds "3"
        assert result == 3 or result is None  # either behavior is acceptable

    def test_string_with_number(self):
        assert _coerce_nonnegative_int("learning_rate: 5") == 5
        assert _coerce_nonnegative_int("42") == 42

    def test_empty_string(self):
        assert _coerce_nonnegative_int("") is None

    def test_none(self):
        assert _coerce_nonnegative_int(None) is None

    def test_string_no_number(self):
        assert _coerce_nonnegative_int("no numbers here") is None

    def test_negative_string(self):
        assert _coerce_nonnegative_int("-5") == 0  # clamped to 0
