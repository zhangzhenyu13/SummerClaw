"""Tests for MastraOM auto-recall module — LLM-judged session history injection."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from summerclaw.memory.mastra_om_memory.recall import (
    RecallConfig,
    build_cycle_summaries,
    build_recall_prompt,
    fetch_recalled_entries,
    format_recall_section,
    judge_and_recall,
    parse_history_cursors,
    parse_recall_response,
)
from summerclaw.memory.mastra_om_memory.store import MastraOMStore


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return MastraOMStore(tmp_path)


@pytest.fixture
def sample_observations():
    return """\
# Observational Memory

## Observation Cycle a1b2c3d4 — 2025-05-09 14:30 history_cursor="42:47"
Date: May 9, 2025
* 🔴 (14:30) User prefers Python over Java
* 🟡 (14:32) User is building a REST API
* ✅ (14:35) REST API scaffold created

## Observation Cycle e5f6g7h8 — 2025-05-09 15:00 history_cursor="48:53"
Date: May 9, 2025
* 🔴 (15:00) User wants authentication module
* 🟡 (15:05) Discussed JWT vs OAuth
* ✅ (15:10) Auth module designed

## Observation Cycle i9j0k1l2 — 2025-05-10 10:00 history_cursor="60:65"
Date: May 10, 2025
* 🔴 (10:00) User needs database migration help
* 🟡 (10:05) PostgreSQL schema discussed
"""


@pytest.fixture
def sample_history_entries():
    return [
        {"cursor": 42, "timestamp": "2025-05-09 14:28", "content": "[14:28] USER: I need help with the API"},
        {"cursor": 43, "timestamp": "2025-05-09 14:29", "content": "[14:29] ASSISTANT: Sure, let me look"},
        {"cursor": 47, "timestamp": "2025-05-09 14:35", "content": "[14:35] ASSISTANT: Scaffold created"},
        {"cursor": 48, "timestamp": "2025-05-09 14:58", "content": "[14:58] USER: Auth module please"},
        {"cursor": 53, "timestamp": "2025-05-09 15:10", "content": "[15:10] ASSISTANT: Auth designed"},
        {"cursor": 60, "timestamp": "2025-05-10 09:58", "content": "[09:58] USER: Database migration"},
        {"cursor": 65, "timestamp": "2025-05-10 10:10", "content": "[10:10] ASSISTANT: Migration done"},
        {"cursor": 70, "timestamp": "2025-05-10 11:00", "content": "[11:00] USER: Unrelated topic"},
    ]


# ── TestParseHistoryCursors ─────────────────────────────────────────────────

class TestParseHistoryCursors:
    """Tests for parsing history_cursor from OBSERVATIONS.md headers."""

    def test_parse_single_cycle(self):
        text = '## Observation Cycle a1b2c3d4 — 2025-05-09 14:30 history_cursor="42:47"\n'
        result = parse_history_cursors(text)
        assert len(result) == 1
        assert result[0]["cycle_id"] == "a1b2c3d4"
        assert result[0]["cursor_start"] == 42
        assert result[0]["cursor_end"] == 47
        assert "2025-05-09" in result[0]["timestamp"]

    def test_parse_multiple_cycles(self, sample_observations):
        result = parse_history_cursors(sample_observations)
        assert len(result) == 3
        assert result[0]["cursor_start"] == 42
        assert result[1]["cursor_start"] == 48
        assert result[2]["cursor_start"] == 60

    def test_skip_cycles_without_cursor(self):
        text = """\
## Observation Cycle abc12345 — 2025-05-09 14:30
* Some observation without cursor
## Observation Cycle def67890 — 2025-05-09 15:00 history_cursor="10:15"
* Another observation
"""
        result = parse_history_cursors(text)
        assert len(result) == 1
        assert result[0]["cycle_id"] == "def67890"
        assert result[0]["cursor_start"] == 10

    def test_empty_input(self):
        assert parse_history_cursors("") == []
        assert parse_history_cursors("   ") == []

    def test_no_cycles(self):
        text = "# Observational Memory\nSome content without cycle headers\n"
        assert parse_history_cursors(text) == []

    def test_malformed_cursor_skipped(self):
        text = '## Observation Cycle abc — 2025-05-09 history_cursor="invalid"\n'
        result = parse_history_cursors(text)
        assert result == []


# ── TestBuildCycleSummaries ─────────────────────────────────────────────────

class TestBuildCycleSummaries:
    """Tests for building cycle summaries from observations text."""

    def test_build_summaries(self, sample_observations):
        cycles = parse_history_cursors(sample_observations)
        summaries = build_cycle_summaries(cycles, sample_observations, max_facts=2)

        assert len(summaries) == 3
        assert summaries[0]["cycle_id"] == "a1b2c3d4"
        assert len(summaries[0]["facts"]) == 2
        assert "Python" in summaries[0]["facts"][0]
        assert "summary_line" in summaries[0]

    def test_empty_cycles(self, sample_observations):
        summaries = build_cycle_summaries([], sample_observations)
        assert summaries == []

    def test_empty_observations(self):
        cycles = [{"cycle_id": "abc", "timestamp": "2025-05-09", "cursor_start": 1, "cursor_end": 5}]
        summaries = build_cycle_summaries(cycles, "")
        assert summaries == []

    def test_max_facts_limit(self, sample_observations):
        cycles = parse_history_cursors(sample_observations)
        summaries = build_cycle_summaries(cycles, sample_observations, max_facts=1)
        for s in summaries:
            assert len(s["facts"]) <= 1


# ── TestBuildRecallPrompt ───────────────────────────────────────────────────

class TestBuildRecallPrompt:
    """Tests for building the LLM recall prompt."""

    def test_basic_prompt(self, sample_observations):
        cycles = parse_history_cursors(sample_observations)
        summaries = build_cycle_summaries(cycles, sample_observations)

        session_tail = [
            {"role": "user", "content": "Help me with the database", "timestamp": "2025-05-10 09:50"},
            {"role": "assistant", "content": "Sure, what database?", "timestamp": "2025-05-10 09:51"},
        ]

        prompt = build_recall_prompt(session_tail, summaries)
        assert "Recent Conversation" in prompt
        assert "Available Observation Cycles" in prompt
        assert "database" in prompt
        assert "a1b2c3d4" in prompt

    def test_empty_session_tail(self, sample_observations):
        cycles = parse_history_cursors(sample_observations)
        summaries = build_cycle_summaries(cycles, sample_observations)
        prompt = build_recall_prompt([], summaries)
        assert "(no recent messages)" in prompt

    def test_long_messages_truncated(self, sample_observations):
        cycles = parse_history_cursors(sample_observations)
        summaries = build_cycle_summaries(cycles, sample_observations)

        long_msg = [{"role": "user", "content": "x" * 1000}]
        prompt = build_recall_prompt(long_msg, summaries)
        # Content should be truncated to 200 chars
        assert "x" * 1000 not in prompt


# ── TestParseRecallResponse ─────────────────────────────────────────────────

class TestParseRecallResponse:
    """Tests for parsing LLM recall response."""

    def test_valid_json_array(self):
        response = '[{"start": 42, "end": 47}, {"start": 55, "end": 60}]'
        result = parse_recall_response(response)
        assert result == [(42, 47), (55, 60)]

    def test_empty_array(self):
        assert parse_recall_response("[]") == []

    def test_empty_string(self):
        assert parse_recall_response("") == []

    def test_invalid_json(self):
        assert parse_recall_response("not json") == []
        assert parse_recall_response("{invalid}") == []

    def test_not_a_list(self):
        assert parse_recall_response('{"start": 1, "end": 5}') == []

    def test_markdown_code_fence(self):
        response = '```json\n[{"start": 10, "end": 20}]\n```'
        result = parse_recall_response(response)
        assert result == [(10, 20)]

    def test_invalid_range_skipped(self):
        response = '[{"start": 50, "end": 40}, {"start": 10, "end": 20}]'
        result = parse_recall_response(response)
        assert result == [(10, 20)]  # first item has start > end

    def test_missing_fields_skipped(self):
        response = '[{"start": 10}, {"end": 20}, {"start": 5, "end": 8}]'
        result = parse_recall_response(response)
        assert result == [(5, 8)]

    def test_non_dict_items_skipped(self):
        response = '[1, 2, {"start": 1, "end": 5}]'
        result = parse_recall_response(response)
        assert result == [(1, 5)]


# ── TestFetchRecalledEntries ────────────────────────────────────────────────

class TestFetchRecalledEntries:
    """Tests for filtering history entries by cursor ranges."""

    def test_fetch_single_range(self, sample_history_entries):
        result = fetch_recalled_entries(sample_history_entries, [(42, 47)])
        assert len(result) == 3
        assert all(42 <= e["cursor"] <= 47 for e in result)

    def test_fetch_multiple_ranges(self, sample_history_entries):
        result = fetch_recalled_entries(sample_history_entries, [(42, 43), (60, 65)])
        assert len(result) == 4
        cursors = [e["cursor"] for e in result]
        assert 42 in cursors and 43 in cursors
        assert 60 in cursors and 65 in cursors

    def test_no_matching_entries(self, sample_history_entries):
        result = fetch_recalled_entries(sample_history_entries, [(100, 200)])
        assert result == []

    def test_empty_entries(self):
        result = fetch_recalled_entries([], [(1, 5)])
        assert result == []

    def test_empty_ranges(self, sample_history_entries):
        result = fetch_recalled_entries(sample_history_entries, [])
        assert result == []

    def test_sorted_by_cursor(self, sample_history_entries):
        result = fetch_recalled_entries(sample_history_entries, [(42, 65)])
        cursors = [e["cursor"] for e in result]
        assert cursors == sorted(cursors)

    def test_no_duplicates_with_overlapping_ranges(self, sample_history_entries):
        result = fetch_recalled_entries(sample_history_entries, [(42, 50), (45, 55)])
        cursors = [e["cursor"] for e in result]
        assert len(cursors) == len(set(cursors))  # no duplicates


# ── TestFormatRecallSection ─────────────────────────────────────────────────

class TestFormatRecallSection:
    """Tests for formatting recalled entries into injection text."""

    def test_basic_format(self, sample_history_entries):
        entries = fetch_recalled_entries(sample_history_entries, [(42, 47)])
        section = format_recall_section(entries, max_bytes=10000)

        assert "## Recent Session Context" in section
        assert "Raw conversation logs" in section
        assert "[Session —" in section
        assert "USER" in section or "ASSISTANT" in section

    def test_empty_entries(self):
        assert format_recall_section([], max_bytes=10000) == ""

    def test_byte_budget_respected(self, sample_history_entries):
        entries = fetch_recalled_entries(sample_history_entries, [(42, 65)])
        section = format_recall_section(entries, max_bytes=200)

        # Section should exist but be truncated
        assert "## Recent Session Context" in section
        # Total bytes should be within budget
        assert len(section.encode("utf-8")) <= 200 + 100  # some header overhead

    def test_chronological_order(self, sample_history_entries):
        entries = fetch_recalled_entries(sample_history_entries, [(42, 65)])
        section = format_recall_section(entries, max_bytes=50000)

        # Find positions of session headers
        lines = section.split("\n")
        session_lines = [l for l in lines if l.startswith("[Session —")]
        if len(session_lines) >= 2:
            # Earlier session should come first
            assert "2025-05-09" in session_lines[0]


# ── TestJudgeAndRecall (Integration) ────────────────────────────────────────

class TestJudgeAndRecall:
    """Integration tests for the full judge_and_recall pipeline."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '[{"start": 42, "end": 47}]'
        provider.chat_with_retry = AsyncMock(return_value=mock_response)
        return provider

    @pytest.fixture
    def mock_session(self):
        session = MagicMock()
        session.messages = [
            {"role": "user", "content": "Help with API", "timestamp": "2025-05-10 09:50"},
            {"role": "assistant", "content": "Sure", "timestamp": "2025-05-10 09:51"},
        ]
        return session

    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path, mock_provider, mock_session, sample_observations):
        store = MastraOMStore(tmp_path)
        store.write_observations(sample_observations)

        # Write history entries with cursors 42-47 to match the mock LLM response
        # We need to manually write to history.jsonl with specific cursors
        import json
        for i in range(42, 48):
            entry = {"cursor": i, "timestamp": f"2025-05-09 14:{i}", "content": f"[14:{i}] USER: message {i}"}
            with open(store.history_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        # Update cursor file
        store._cursor_file.write_text("47")

        config = RecallConfig(enabled=True, max_cycles=3)

        await judge_and_recall(store, mock_provider, "test-model", mock_session, config)

        assert len(store._recalled_entries_cache) > 0
        mock_provider.chat_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_observations(self, tmp_path, mock_provider, mock_session):
        store = MastraOMStore(tmp_path)
        config = RecallConfig(enabled=True)

        await judge_and_recall(store, mock_provider, "test-model", mock_session, config)

        assert store._recalled_entries_cache == []
        mock_provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_returns_empty(self, tmp_path, mock_session, sample_observations):
        store = MastraOMStore(tmp_path)
        store.write_observations(sample_observations)

        provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "[]"
        provider.chat_with_retry = AsyncMock(return_value=mock_response)

        config = RecallConfig(enabled=True)
        await judge_and_recall(store, provider, "test-model", mock_session, config)

        assert store._recalled_entries_cache == []

    @pytest.mark.asyncio
    async def test_llm_call_fails(self, tmp_path, mock_session, sample_observations):
        store = MastraOMStore(tmp_path)
        store.write_observations(sample_observations)

        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=Exception("API error"))

        config = RecallConfig(enabled=True)
        await judge_and_recall(store, provider, "test-model", mock_session, config)

        assert store._recalled_entries_cache == []

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, tmp_path, mock_session, sample_observations):
        store = MastraOMStore(tmp_path)
        store.write_observations(sample_observations)

        # Provider that raises exception
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("boom"))

        config = RecallConfig(enabled=True)
        # Should not raise
        await judge_and_recall(store, provider, "test-model", mock_session, config)
        assert store._recalled_entries_cache == []


# ── TestStoreIntegration ────────────────────────────────────────────────────

class TestStoreIntegration:
    """Tests for store integration with recall."""

    def test_recall_config_default(self, tmp_path):
        store = MastraOMStore(tmp_path)
        assert hasattr(store, "_recall_config")
        assert store._recall_config.enabled is True

    def test_recall_config_custom(self, tmp_path):
        config = RecallConfig(enabled=False, max_cycles=2)
        store = MastraOMStore(tmp_path, recall_config=config)
        assert store._recall_config.enabled is False
        assert store._recall_config.max_cycles == 2

    def test_recalled_cache_initialized(self, tmp_path):
        store = MastraOMStore(tmp_path)
        assert hasattr(store, "_recalled_entries_cache")
        assert store._recalled_entries_cache == []

    def test_get_memory_context_without_recall(self, tmp_path):
        store = MastraOMStore(tmp_path)
        store.write_observations("# Observational Memory\n* Some fact")
        context = store.get_memory_context()
        # Should not contain recall section when cache is empty
        assert "Recent Session Context" not in context

    def test_get_memory_context_with_recall(self, tmp_path):
        store = MastraOMStore(tmp_path)
        store.write_observations("# Observational Memory\n* Some fact")
        store._recalled_entries_cache = [
            {"cursor": 1, "timestamp": "2025-05-09 14:30", "content": "[14:30] USER: Hello"},
        ]
        context = store.get_memory_context()
        assert "Recent Session Context" in context
        assert "USER: Hello" in context
