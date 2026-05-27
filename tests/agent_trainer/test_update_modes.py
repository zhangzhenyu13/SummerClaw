"""Unit tests for SkillOpt update_modes utility module."""
from __future__ import annotations

import pytest

from summerclaw.agent_trainer.algorithms.skillopt.update_modes import (
    PATCH_MODE,
    REWRITE_MODE,
    FULL_REWRITE_MINIBATCH_MODE,
    describe_item,
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    set_payload_items,
    short_item_summary,
    truncate_payload,
)


class TestNormalizeUpdateMode:
    def test_default_is_patch(self):
        assert normalize_update_mode(None) == PATCH_MODE
        assert normalize_update_mode("") == PATCH_MODE
        assert normalize_update_mode("patch") == PATCH_MODE

    def test_edits_alias(self):
        assert normalize_update_mode("edits") == PATCH_MODE

    def test_rewrite_aliases(self):
        assert normalize_update_mode("rewrite") == REWRITE_MODE
        assert normalize_update_mode("rewrite_from_suggestions") == REWRITE_MODE
        assert normalize_update_mode("suggestions") == REWRITE_MODE
        assert normalize_update_mode("rewrite_suggestions") == REWRITE_MODE

    def test_full_rewrite_aliases(self):
        assert normalize_update_mode("full_rewrite") == FULL_REWRITE_MINIBATCH_MODE
        assert normalize_update_mode("full_rewrite_minibatch") == FULL_REWRITE_MINIBATCH_MODE
        assert normalize_update_mode("minibatch_full_rewrite") == FULL_REWRITE_MINIBATCH_MODE
        assert normalize_update_mode("skill_rewrite_minibatch") == FULL_REWRITE_MINIBATCH_MODE

    def test_unknown_defaults_to_patch(self):
        assert normalize_update_mode("unknown_mode") == PATCH_MODE

    def test_case_insensitive(self):
        assert normalize_update_mode("PATCH") == PATCH_MODE
        assert normalize_update_mode("Rewrite") == REWRITE_MODE
        assert normalize_update_mode("FULL_REWRITE") == FULL_REWRITE_MINIBATCH_MODE


class TestModeChecks:
    def test_is_rewrite_mode(self):
        assert is_rewrite_mode("rewrite") is True
        assert is_rewrite_mode("rewrite_from_suggestions") is True
        assert is_rewrite_mode("patch") is False
        assert is_rewrite_mode("full_rewrite") is False

    def test_is_full_rewrite_minibatch_mode(self):
        assert is_full_rewrite_minibatch_mode("full_rewrite") is True
        assert is_full_rewrite_minibatch_mode("full_rewrite_minibatch") is True
        assert is_full_rewrite_minibatch_mode("patch") is False
        assert is_full_rewrite_minibatch_mode("rewrite") is False


class TestPayloadKey:
    def test_patch(self):
        assert payload_key("patch") == "edits"
        assert payload_key(None) == "edits"

    def test_rewrite(self):
        assert payload_key("rewrite") == "revise_suggestions"

    def test_full_rewrite(self):
        assert payload_key("full_rewrite") == "skill_candidates"


class TestPayloadLabel:
    def test_patch(self):
        assert payload_label("patch") == "edits"
        assert payload_label("patch", singular=True) == "edit"
        assert payload_label("patch", title=True) == "Edits"

    def test_rewrite(self):
        assert payload_label("rewrite") == "suggestions"
        assert payload_label("rewrite", singular=True) == "suggestion"

    def test_full_rewrite(self):
        assert payload_label("full_rewrite") == "skill candidates"
        assert payload_label("full_rewrite", singular=True) == "skill candidate"


class TestPayloadItems:
    def test_get_payload_items_patch(self):
        container = {"edits": [{"op": "append"}], "reasoning": "test"}
        assert get_payload_items(container, "patch") == [{"op": "append"}]

    def test_get_payload_items_rewrite(self):
        container = {"revise_suggestions": [{"type": "add"}]}
        assert get_payload_items(container, "rewrite") == [{"type": "add"}]

    def test_get_payload_items_full_rewrite(self):
        container = {"skill_candidates": [{"title": "v1"}]}
        assert get_payload_items(container, "full_rewrite") == [{"title": "v1"}]

    def test_get_payload_items_missing_key(self):
        assert get_payload_items({"reasoning": "test"}, "patch") == []

    def test_get_payload_items_none(self):
        assert get_payload_items(None, "patch") == []

    def test_set_payload_items(self):
        container: dict = {}
        set_payload_items(container, [{"op": "append"}], "patch")
        assert container == {"edits": [{"op": "append"}]}

    def test_truncate_payload(self):
        container = {"edits": [{"op": f"e{i}"} for i in range(10)]}
        truncate_payload(container, 3, "patch")
        assert len(container["edits"]) == 3

    def test_truncate_payload_no_limit(self):
        container = {"edits": [{"op": f"e{i}"} for i in range(10)]}
        truncate_payload(container, -1, "patch")
        assert len(container["edits"]) == 10


class TestDescribeItem:
    def test_patch_item(self):
        item = {"op": "append", "content": "new rule", "target": ""}
        desc = describe_item(item, "patch")
        assert "op=append" in desc
        assert "content=" in desc

    def test_rewrite_item(self):
        item = {"type": "add", "title": "Add rule", "instruction": "Add a new rule"}
        desc = describe_item(item, "rewrite")
        assert "type=add" in desc
        assert "title=" in desc

    def test_full_rewrite_item(self):
        item = {"title": "v1", "change_summary": ["fix bug"], "new_skill": "full skill..."}
        desc = describe_item(item, "full_rewrite")
        assert "title=" in desc
        assert "change_summary=" in desc

    def test_non_dict(self):
        assert describe_item("not a dict", "patch") == ""

    def test_truncation(self):
        item = {"op": "replace", "content": "x" * 500, "target": "y" * 200}
        desc = describe_item(item, "patch", max_chars=100)
        assert len(desc) <= 100


class TestShortItemSummary:
    def test_patch_summary(self):
        item = {"op": "append", "content": "rule text", "target": ""}
        s = short_item_summary(item, "patch")
        assert s["op"] == "append"

    def test_rewrite_summary(self):
        item = {"type": "modify", "title": "Fix", "instruction": "Fix this"}
        s = short_item_summary(item, "rewrite")
        assert s["type"] == "modify"

    def test_full_rewrite_summary(self):
        item = {"title": "v1", "change_summary": ["a", "b"], "source_type": "failure"}
        s = short_item_summary(item, "full_rewrite")
        assert s["title"] == "v1"
        assert s["source_type"] == "failure"
