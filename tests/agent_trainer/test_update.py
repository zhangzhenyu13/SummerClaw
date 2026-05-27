"""Unit tests for SkillOpt update module (edit application logic)."""
from __future__ import annotations

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.update import (
    _apply_edit_with_report,
    apply_patch_with_report,
)
from summerclaw.agent_trainer.types import Edit, Patch


class TestApplyEdit:
    def test_append(self):
        skill = "# Rules\n- Rule 1\n"
        edit = Edit(op="append", content="- Rule 2")
        result, report = _apply_edit_with_report(skill, edit)
        assert "- Rule 2" in result
        assert report["status"].startswith("applied")

    def test_insert_after(self):
        skill = "# Rules\n- Rule 1\n- Rule 3\n"
        edit = Edit(op="insert_after", content="- Rule 2", target="- Rule 1")
        result, report = _apply_edit_with_report(skill, edit)
        assert "- Rule 2" in result
        # Rule 2 should appear after Rule 1 and before Rule 3
        idx1 = result.index("- Rule 1")
        idx2 = result.index("- Rule 2")
        idx3 = result.index("- Rule 3")
        assert idx1 < idx2 < idx3
        assert report["status"].startswith("applied")

    def test_replace(self):
        skill = "# Rules\n- Old Rule\n"
        edit = Edit(op="replace", content="- New Rule", target="- Old Rule")
        result, report = _apply_edit_with_report(skill, edit)
        assert "- New Rule" in result
        assert "- Old Rule" not in result
        assert report["status"].startswith("applied")

    def test_delete(self):
        skill = "# Rules\n- Rule 1\n- Rule 2\n"
        edit = Edit(op="delete", target="- Rule 2")
        result, report = _apply_edit_with_report(skill, edit)
        assert "- Rule 2" not in result
        assert "- Rule 1" in result
        assert report["status"].startswith("applied")

    def test_replace_target_not_found(self):
        skill = "# Rules\n- Rule 1\n"
        edit = Edit(op="replace", content="- New", target="nonexistent")
        result, report = _apply_edit_with_report(skill, edit)
        assert result == skill  # unchanged
        assert report["status"].startswith("skipped")

    def test_insert_after_target_not_found_fallback(self):
        skill = "# Rules\n- Rule 1\n"
        edit = Edit(op="insert_after", content="- New", target="nonexistent")
        result, report = _apply_edit_with_report(skill, edit)
        # When target not found, falls back to append
        assert "- New" in result
        assert report["status"].startswith("applied_insert_after_fallback")


class TestApplyPatchWithReport:
    def test_multiple_edits(self):
        skill = "# Rules\n- Rule 1\n- Rule 2\n"
        patch = Patch(edits=[
            Edit(op="append", content="- Rule 3"),
            Edit(op="delete", target="- Rule 1"),
        ])
        result, report = apply_patch_with_report(skill, patch)
        assert "- Rule 3" in result
        assert "- Rule 1" not in result
        assert "- Rule 2" in result
        assert len(report) == 2

    def test_empty_patch(self):
        skill = "# Rules\n"
        patch = Patch(edits=[])
        result, report = apply_patch_with_report(skill, patch)
        assert result == skill
        assert report == []

    def test_slow_update_protection(self):
        skill = (
            "# Rules\n"
            "<!-- SLOW_UPDATE_START -->\n"
            "Protected content\n"
            "<!-- SLOW_UPDATE_END -->\n"
            "- Rule 1\n"
        )
        # This should NOT modify the SLOW_UPDATE region
        patch = Patch(edits=[
            Edit(op="replace", content="Hacked", target="Protected content"),
        ])
        result, report = apply_patch_with_report(skill, patch)
        # The protected content should remain
        assert "Protected content" in result or report[0]["status"].startswith("skipped")
