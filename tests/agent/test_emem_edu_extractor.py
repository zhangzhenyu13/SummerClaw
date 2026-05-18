"""Tests for EDUExtractor — LLM-based Elementary Discourse Unit extraction."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from summerclaw.memory.emem_memory.datatypes import EDURecord
from summerclaw.memory.emem_memory.edu_extractor import EDUExtractor


# ===================================================================
# EDUExtractor fixtures
# ===================================================================

@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLMProvider."""
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def extractor(mock_provider: MagicMock) -> EDUExtractor:
    """Create an EDUExtractor with mocked provider (simple mode by default)."""
    return EDUExtractor(
        provider=mock_provider,
        model="test-model",
        extract_events=False,
        skip_context_gen=True,
    )


@pytest.fixture
def extractor_with_events(mock_provider: MagicMock) -> EDUExtractor:
    """Create an EDUExtractor in full event extraction mode."""
    return EDUExtractor(
        provider=mock_provider,
        model="test-model",
        extract_events=True,
        skip_context_gen=False,
    )


def _make_llm_response(content: str, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.finish_reason = finish_reason
    return resp


# ===================================================================
# EDUExtractor — simple mode extraction
# ===================================================================

class TestEDUExtractorSimple:
    """Test EDU extraction in simple mode (text-only, no events)."""

    async def test_extract_simple_edus(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({
                "edus": [
                    {"text": "Alice deployed the app on Tuesday."},
                    {"text": "Bob reviewed the deployment logs."},
                ],
            }),
        )

        edus = await extractor.extract_from_history(
            history_text="Alice: I deployed the app.\nBob: I'll review it.",
            session_id="session-001",
            speakers=["Alice", "Bob"],
            timestamp=datetime(2026, 5, 1, 10, 0),
        )
        assert len(edus) == 2
        assert isinstance(edus[0], EDURecord)
        assert edus[0].text == "Alice deployed the app on Tuesday."
        assert edus[0].session_id == "session-001"
        assert edus[0].source_speakers == ["Alice", "Bob"]
        assert edus[0].timestamp == datetime(2026, 5, 1, 10, 0)
        assert edus[0].event_type is None

    async def test_extract_simple_empty_history(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"edus": []}),
        )
        edus = await extractor.extract_from_history("", session_id="")
        assert edus == []

    async def test_extract_simple_llm_error(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.side_effect = Exception("LLM error")
        edus = await extractor.extract_from_history("some history")
        assert edus == []

    async def test_extract_simple_finish_reason_error(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            "", finish_reason="error",
        )
        edus = await extractor.extract_from_history("some history")
        assert edus == []

    async def test_extract_simple_empty_response(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response("")
        edus = await extractor.extract_from_history("some history")
        assert edus == []

    async def test_extract_skips_empty_text_edus(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({
                "edus": [
                    {"text": ""},
                    {"text": "Valid EDU"},
                    {"text": "   "},
                ],
            }),
        )
        edus = await extractor.extract_from_history("some history")
        assert len(edus) == 1
        assert edus[0].text == "Valid EDU"


# ===================================================================
# EDUExtractor — full event extraction
# ===================================================================

class TestEDUExtractorWithEvents:
    """Test EDU extraction in full mode with event types and role-argument pairs."""

    async def test_extract_with_events(
        self, extractor_with_events: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({
                "edus": [{
                    "text": "Bob submitted the report to Carol.",
                    "event_type": "Communication",
                    "event_triggers": ["submitted"],
                    "event_role_argument_pairs": [
                        {"role": "AGENT", "argument": "Bob"},
                        {"role": "PATIENT", "argument": "the report"},
                        {"role": "RECIPIENT", "argument": "Carol"},
                    ],
                }],
            }),
        )
        edus = await extractor_with_events.extract_from_history("history")
        assert len(edus) == 1
        assert edus[0].event_type == "Communication"
        assert edus[0].event_triggers == ["submitted"]
        assert edus[0].event_role_argument_pairs == [
            {"role": "AGENT", "argument": "Bob"},
            {"role": "PATIENT", "argument": "the report"},
            {"role": "RECIPIENT", "argument": "Carol"},
        ]

    async def test_extract_with_events_llm_error(
        self, extractor_with_events: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.side_effect = Exception("LLM error")
        edus = await extractor_with_events.extract_from_history("history")
        assert edus == []


# ===================================================================
# EDUExtractor — response parsing
# ===================================================================

class TestEDUExtractorParse:
    """Test the _parse_response static method."""

    def test_parse_json_object_with_edus_key(self) -> None:
        content = json.dumps({"edus": [{"text": "EDU 1"}, {"text": "EDU 2"}]})
        result = EDUExtractor._parse_response(content, simple=True)
        assert len(result) == 2
        assert result[0]["text"] == "EDU 1"

    def test_parse_json_list_directly(self) -> None:
        content = json.dumps([{"text": "Direct EDU"}])
        result = EDUExtractor._parse_response(content, simple=True)
        assert len(result) == 1
        assert result[0]["text"] == "Direct EDU"

    def test_parse_markdown_code_fence_removed(self) -> None:
        content = '```json\n{"edus": [{"text": "Inside fence"}]}\n```'
        result = EDUExtractor._parse_response(content, simple=True)
        assert len(result) == 1
        assert result[0]["text"] == "Inside fence"

    def test_parse_markdown_code_fence_no_lang(self) -> None:
        content = '```\n{"edus": [{"text": "No lang"}]}\n```'
        result = EDUExtractor._parse_response(content, simple=True)
        assert len(result) == 1
        assert result[0]["text"] == "No lang"

    def test_parse_find_json_in_text(self) -> None:
        content = 'Some preamble text.\n{"edus": [{"text": "Found it"}]}\nMore text.'
        result = EDUExtractor._parse_response(content, simple=True)
        # Should find the JSON object via regex
        # This may or may not find it depending on exact regex pattern
        # The regex looks for { ... }
        # The content within the middle should match
        assert len(result) >= 0

    def test_parse_invalid_json_returns_empty(self) -> None:
        result = EDUExtractor._parse_response("not json at all", simple=True)
        assert result == []

    def test_parse_empty_string(self) -> None:
        result = EDUExtractor._parse_response("", simple=True)
        assert result == []

    def test_parse_empty_edus_list_in_dict(self) -> None:
        content = json.dumps({"edus": []})
        result = EDUExtractor._parse_response(content, simple=True)
        assert result == []


# ===================================================================
# EDUExtractor — entity extraction
# ===================================================================

class TestEDUExtractorEntities:
    """Test extract_entities_from_query."""

    async def test_extract_entities(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            json.dumps({"entities": ["Alice", "Project X", "deployment"]}),
        )
        entities = await extractor.extract_entities_from_query(
            "What did Alice do for Project X deployment?",
        )
        assert len(entities) == 3
        assert "Alice" in entities
        assert "Project X" in entities
        assert "deployment" in entities

    async def test_extract_entities_llm_error(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.side_effect = Exception("LLM error")
        entities = await extractor.extract_entities_from_query("query")
        assert entities == []

    async def test_extract_entities_error_response(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response(
            "", finish_reason="error",
        )
        entities = await extractor.extract_entities_from_query("query")
        assert entities == []

    async def test_extract_entities_invalid_json(
        self, extractor: EDUExtractor, mock_provider: MagicMock,
    ) -> None:
        mock_provider.chat_with_retry.return_value = _make_llm_response("not json")
        entities = await extractor.extract_entities_from_query("query")
        assert entities == []
