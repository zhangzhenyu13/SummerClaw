"""Tests for MastraOM Observation Groups — wrapping, parsing, reconciliation.

Comprehensive coverage of the observation-groups module:
- wrap_in_observation_group, parse_observation_groups, strip_observation_groups
- combine_observation_group_ranges, build_message_range
- render_observation_groups_for_reflection
- reconcile_observation_groups_from_reflection
- generate_anchor_id uniqueness
- OBSERVATION_RETRIEVAL_INSTRUCTIONS content
"""

import pytest

from summerclaw.memory.mastra_om_memory.groups import (
    ObservationGroup,
    generate_anchor_id,
    wrap_in_observation_group,
    parse_observation_groups,
    strip_observation_groups,
    combine_observation_group_ranges,
    build_message_range,
    render_observation_groups_for_reflection,
    reconcile_observation_groups_from_reflection,
    OBSERVATION_RETRIEVAL_INSTRUCTIONS,
)


# ── Anchor ID ──────────────────────────────────────────────────────────────

class TestAnchorId:
    """Tests for anchor ID generation."""

    def test_generates_16_char_hex(self):
        anchor_id = generate_anchor_id()
        assert len(anchor_id) == 16
        assert all(c in '0123456789abcdef' for c in anchor_id)

    def test_generates_unique_ids(self):
        ids = {generate_anchor_id() for _ in range(100)}
        assert len(ids) == 100


# ── Wrap ───────────────────────────────────────────────────────────────────

class TestWrapObservationGroup:
    """Tests for wrap_in_observation_group."""

    def test_basic_wrap(self):
        result = wrap_in_observation_group(
            observations="* 🔴 User prefers dark mode",
            range_spec="msg_001:msg_010",
            group_id="abc123",
        )
        assert '<observation-group id="abc123" range="msg_001:msg_010">' in result
        assert '* 🔴 User prefers dark mode' in result
        assert '</observation-group>' in result

    def test_generates_id_if_not_provided(self):
        result = wrap_in_observation_group(
            observations="* 🔴 Test observation",
            range_spec="msg_001:msg_005",
        )
        assert 'id="' in result
        assert 'range="msg_001:msg_005"' in result

    def test_includes_kind_attribute(self):
        result = wrap_in_observation_group(
            observations="* 🔴 Test",
            range_spec="msg_001:msg_003",
            group_id="abc",
            kind="reflection",
        )
        assert 'kind="reflection"' in result

    def test_trims_content(self):
        result = wrap_in_observation_group(
            observations="  * 🔴 Test  ",
            range_spec="msg_001:msg_001",
            group_id="abc",
        )
        assert "\n* 🔴 Test\n" in result


# ── Parse ──────────────────────────────────────────────────────────────────

class TestParseObservationGroups:
    """Tests for parse_observation_groups."""

    def test_parses_single_group(self):
        text = '<observation-group id="abc123" range="msg_001:msg_010">\n* 🔴 Test observation\n</observation-group>'
        groups = parse_observation_groups(text)
        assert len(groups) == 1
        assert groups[0].id == "abc123"
        assert groups[0].range == "msg_001:msg_010"
        assert "* 🔴 Test observation" in groups[0].content
        assert groups[0].kind is None

    def test_parses_multiple_groups(self):
        text = (
            '<observation-group id="g1" range="msg_001:msg_005">\n* 🔴 First\n</observation-group>\n'
            '<observation-group id="g2" range="msg_006:msg_010">\n* 🔴 Second\n</observation-group>'
        )
        groups = parse_observation_groups(text)
        assert len(groups) == 2
        assert groups[0].id == "g1"
        assert groups[1].id == "g2"

    def test_parses_group_with_kind(self):
        text = '<observation-group id="abc" range="msg_001:msg_005" kind="reflection">\n* 🔴 Test\n</observation-group>'
        groups = parse_observation_groups(text)
        assert len(groups) == 1
        assert groups[0].kind == "reflection"

    def test_skips_groups_without_id(self):
        text = '<observation-group range="msg_001:msg_005">\n* 🔴 Test\n</observation-group>'
        groups = parse_observation_groups(text)
        assert len(groups) == 0

    def test_skips_groups_without_range(self):
        text = '<observation-group id="abc">\n* 🔴 Test\n</observation-group>'
        groups = parse_observation_groups(text)
        assert len(groups) == 0

    def test_returns_empty_list_for_empty_input(self):
        assert parse_observation_groups("") == []
        assert parse_observation_groups(None) == []  # type: ignore[arg-type]

    def test_returns_empty_for_no_groups(self):
        text = "Just some plain observations\n* 🔴 No groups here"
        assert parse_observation_groups(text) == []

    def test_case_insensitive_tag_matching(self):
        text = '<OBSERVATION-GROUP id="abc" range="msg_001:msg_005">\n* 🔴 Test\n</OBSERVATION-GROUP>'
        groups = parse_observation_groups(text)
        assert len(groups) == 1
        assert groups[0].id == "abc"

    def test_interleaved_text_with_groups(self):
        text = (
            "Date: Jan 1, 2026\n"
            '<observation-group id="g1" range="msg_001:msg_005">\n* 🔴 G1\n</observation-group>\n'
            "Some free text\n"
            '<observation-group id="g2" range="msg_006:msg_010">\n* 🔴 G2\n</observation-group>'
        )
        groups = parse_observation_groups(text)
        assert len(groups) == 2


# ── Strip ──────────────────────────────────────────────────────────────────

class TestStripObservationGroups:
    """Tests for strip_observation_groups."""

    def test_strips_single_group(self):
        text = '<observation-group id="abc" range="msg_001:msg_010">\n* 🔴 Test\n</observation-group>'
        result = strip_observation_groups(text)
        assert '<observation-group' not in result
        assert '* 🔴 Test' in result

    def test_strips_multiple_groups(self):
        text = (
            '<observation-group id="g1" range="msg_001:msg_005">\n* 🔴 First\n</observation-group>\n'
            '<observation-group id="g2" range="msg_006:msg_010">\n* 🔴 Second\n</observation-group>'
        )
        result = strip_observation_groups(text)
        assert '<observation-group' not in result
        assert '* 🔴 First' in result
        assert '* 🔴 Second' in result

    def test_preserves_non_group_text(self):
        text = (
            "Date: Jan 1, 2026\n"
            '<observation-group id="g1" range="msg_001:msg_005">\n* 🔴 Test\n</observation-group>\n'
            "More free text"
        )
        result = strip_observation_groups(text)
        assert 'Date: Jan 1, 2026' in result
        assert 'More free text' in result

    def test_handles_empty_input(self):
        assert strip_observation_groups("") == ""
        assert strip_observation_groups(None) is None  # type: ignore[arg-type]

    def test_collapses_multiple_blank_lines(self):
        text = (
            '<observation-group id="g1" range="msg_001:msg_005">\n* 🔴 A\n</observation-group>\n\n\n\n'
            '<observation-group id="g2" range="msg_006:msg_010">\n* 🔴 B\n</observation-group>'
        )
        result = strip_observation_groups(text)
        assert '\n\n\n' not in result


# ── Range Utilities ────────────────────────────────────────────────────────

class TestCombineRanges:
    """Tests for combine_observation_group_ranges."""

    def test_single_group(self):
        groups = [ObservationGroup(id="g1", range="msg_001:msg_010", content="test")]
        result = combine_observation_group_ranges(groups)
        assert result == "msg_001:msg_010"

    def test_multiple_groups_first_last(self):
        groups = [
            ObservationGroup(id="g1", range="msg_001:msg_005", content="a"),
            ObservationGroup(id="g2", range="msg_006:msg_010", content="b"),
        ]
        result = combine_observation_group_ranges(groups)
        assert result == "msg_001:msg_010"

    def test_empty_groups(self):
        assert combine_observation_group_ranges([]) == ""

    def test_comma_separated_ranges(self):
        groups = [
            ObservationGroup(id="g1", range="msg_001:msg_003,msg_007:msg_009", content="a"),
        ]
        result = combine_observation_group_ranges(groups)
        assert result == "msg_001:msg_009"

    def test_single_message_range(self):
        groups = [
            ObservationGroup(id="g1", range="msg_005:msg_005", content="a"),
        ]
        result = combine_observation_group_ranges(groups)
        assert result == "msg_005:msg_005"


class TestBuildMessageRange:
    """Tests for build_message_range."""

    def test_builds_range_from_messages(self):
        messages = [
            {"id": "msg_001", "role": "user", "content": "Hello"},
            {"id": "msg_002", "role": "assistant", "content": "Hi"},
            {"id": "msg_003", "role": "user", "content": "How are you?"},
        ]
        result = build_message_range(messages)
        assert result == "msg_001:msg_003"

    def test_single_message(self):
        messages = [{"id": "msg_001", "role": "user", "content": "Hello"}]
        result = build_message_range(messages)
        assert result == "msg_001:msg_001"

    def test_skips_messages_without_id(self):
        messages = [
            {"role": "user", "content": "No ID"},
            {"id": "msg_001", "role": "assistant", "content": "Has ID"},
            {"role": "user", "content": "No ID again"},
        ]
        result = build_message_range(messages)
        assert result == "msg_001:msg_001"

    def test_returns_none_for_no_ids(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = build_message_range(messages)
        assert result is None

    def test_returns_none_for_empty(self):
        assert build_message_range([]) is None


# ── Reflection Rendering ───────────────────────────────────────────────────

class TestRenderForReflection:
    """Tests for render_observation_groups_for_reflection."""

    def test_renders_groups_as_markdown_headings(self):
        text = '<observation-group id="abc123" range="msg_001:msg_010">\n* 🔴 Test observation\n</observation-group>'
        result = render_observation_groups_for_reflection(text)
        assert result is not None
        assert '## Group `abc123`' in result
        assert '_range: `msg_001:msg_010`_' in result
        assert '* 🔴 Test observation' in result
        assert '<observation-group' not in result

    def test_returns_none_for_no_groups(self):
        result = render_observation_groups_for_reflection("* 🔴 Just a plain observation")
        assert result is None

    def test_renders_multiple_groups(self):
        text = (
            '<observation-group id="g1" range="msg_001:msg_005">\n* 🔴 First\n</observation-group>\n'
            '<observation-group id="g2" range="msg_006:msg_010">\n* 🔴 Second\n</observation-group>'
        )
        result = render_observation_groups_for_reflection(text)
        assert result is not None
        assert '## Group `g1`' in result
        assert '## Group `g2`' in result


# ── Reconciliation ─────────────────────────────────────────────────────────

class TestReconcileFromReflection:
    """Tests for reconcile_observation_groups_from_reflection."""

    def test_reconciles_reflected_content(self):
        source = '<observation-group id="abc" range="msg_001:msg_010">\n* 🔴 Original observation\n</observation-group>'
        reflected = (
            '## Group `abc`\n'
            '_range: `msg_001:msg_010`_\n\n'
            '* 🔴 Original observation (condensed)\n'
        )
        result = reconcile_observation_groups_from_reflection(reflected, source)
        assert result is not None
        assert '<observation-group' in result
        assert 'kind="reflection"' in result

    def test_returns_none_when_no_source_groups(self):
        result = reconcile_observation_groups_from_reflection(
            content="Some reflected text",
            source_observations="Plain text without groups",
        )
        assert result is None

    def test_returns_empty_string_for_empty_content(self):
        source = '<observation-group id="abc" range="msg_001:msg_010">\n* 🔴 Test\n</observation-group>'
        result = reconcile_observation_groups_from_reflection("", source)
        assert result == ""

    def test_fallback_wraps_entire_content(self):
        source = '<observation-group id="abc" range="msg_001:msg_010">\n* 🔴 Test\n</observation-group>'
        reflected = 'Just some plain reflected text without ## Group headings'
        result = reconcile_observation_groups_from_reflection(reflected, source)
        assert result is not None
        assert '<observation-group' in result
        assert 'kind="reflection"' in result


# ── Retrieval Instructions ─────────────────────────────────────────────────

class TestRetrievalInstructions:
    """Verify OBSERVATION_RETRIEVAL_INSTRUCTIONS contains all sections."""

    def test_contains_recall_title(self):
        assert 'Recall — looking up source messages' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_contains_observation_group_explanation(self):
        assert '<observation-group>' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'startId:endId' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_contains_when_to_use(self):
        assert 'When to use recall' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'repeat, show, or reproduce' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_contains_how_to_use(self):
        assert 'How to use recall' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'cursor' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_contains_detail_levels(self):
        assert 'Detail levels' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'low' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'high' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_contains_truncated_parts_guidance(self):
        assert 'Following up on truncated parts' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'truncated' in OBSERVATION_RETRIEVAL_INSTRUCTIONS.lower()

    def test_contains_when_not_needed(self):
        assert 'When recall is NOT needed' in OBSERVATION_RETRIEVAL_INSTRUCTIONS

    def test_contains_pagination_hints(self):
        assert 'hasNextPage' in OBSERVATION_RETRIEVAL_INSTRUCTIONS
        assert 'paginate' in OBSERVATION_RETRIEVAL_INSTRUCTIONS.lower()
