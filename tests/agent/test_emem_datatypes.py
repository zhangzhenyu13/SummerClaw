"""Tests for EMem data types — EDURecord, ArgumentRecord, SessionRecord, EMemConfig, utilities."""

from __future__ import annotations

from datetime import datetime
from hashlib import md5

import numpy as np
import pytest

from summerclaw.memory.emem_memory.datatypes import (
    ArgumentRecord,
    EDURecord,
    EMemConfig,
    SessionRecord,
    compute_mdhash_id,
    min_max_normalize,
)


# ===================================================================
# EDURecord tests
# ===================================================================

class TestEDURecord:
    """Test EDURecord creation, ID computation, and context string formatting."""

    def test_create_basic_edu(self) -> None:
        edu = EDURecord(
            edu_id="edu-test123",
            text="Alice bought a red car on Tuesday.",
            source_speakers=["Alice"],
            timestamp=datetime(2026, 5, 1, 10, 30),
            session_id="session-001",
        )
        assert edu.edu_id == "edu-test123"
        assert edu.text == "Alice bought a red car on Tuesday."
        assert edu.source_speakers == ["Alice"]
        assert edu.session_id == "session-001"
        assert edu.event_type is None
        assert edu.event_triggers is None
        assert edu.event_role_argument_pairs is None
        assert edu.context_text is None
        assert edu.metadata == {}

    def test_default_values(self) -> None:
        edu = EDURecord(
            edu_id="edu-defaults",
            text="Simple statement.",
        )
        assert edu.source_speakers == []
        assert edu.timestamp is None
        assert edu.session_id == ""
        assert edu.event_type is None
        assert edu.event_triggers is None
        assert edu.event_role_argument_pairs is None
        assert edu.context_text is None
        assert edu.metadata == {}

    def test_compute_id_deterministic(self) -> None:
        text = "The project deadline was moved to Friday."
        id1 = EDURecord.compute_id(text)
        id2 = EDURecord.compute_id(text)
        assert id1 == id2
        assert id1.startswith("edu-")

    def test_compute_id_different_texts_different_ids(self) -> None:
        id1 = EDURecord.compute_id("First text")
        id2 = EDURecord.compute_id("Second text")
        assert id1 != id2

    def test_compute_id_custom_prefix(self) -> None:
        text = "Custom prefix test"
        edu_id = EDURecord.compute_id(text, prefix="custom-")
        assert edu_id.startswith("custom-")
        assert edu_id != EDURecord.compute_id(text)  # Different prefix => different id

    def test_compute_id_equals_md5_of_text(self) -> None:
        text = "Verify md5 computation"
        expected = "edu-" + md5(text.encode()).hexdigest()
        assert EDURecord.compute_id(text) == expected

    def test_edu_with_full_event_info(self) -> None:
        edu = EDURecord(
            edu_id="edu-full",
            text="Bob submitted the report to Carol.",
            source_speakers=["Bob", "Carol"],
            timestamp=datetime(2026, 5, 1, 14, 0),
            session_id="session-002",
            event_type="Communication",
            event_triggers=["submitted"],
            event_role_argument_pairs=[
                {"role": "AGENT", "argument": "Bob"},
                {"role": "PATIENT", "argument": "the report"},
                {"role": "RECIPIENT", "argument": "Carol"},
            ],
            context_text="Bob was working on the quarterly report.",
            metadata={"source_turn": 3},
        )
        assert edu.event_type == "Communication"
        assert edu.event_triggers == ["submitted"]
        assert len(edu.event_role_argument_pairs) == 3
        assert edu.event_role_argument_pairs[0] == {"role": "AGENT", "argument": "Bob"}
        assert edu.context_text == "Bob was working on the quarterly report."
        assert edu.metadata == {"source_turn": 3}

    def test_to_context_string_iso_format(self) -> None:
        edu = EDURecord(
            edu_id="edu-ctx",
            text="The server was deployed at 3pm.",
            source_speakers=["Alice"],
            timestamp=datetime(2026, 5, 1, 15, 0, 0),
        )
        ctx = edu.to_context_string(date_format="iso")
        assert "The server was deployed at 3pm." in ctx
        assert "2026-05-01T15:00:00" in ctx
        assert "Alice" in ctx

    def test_to_context_string_locomo_format(self) -> None:
        edu = EDURecord(
            edu_id="edu-ctx2",
            text="Deployment completed.",
            source_speakers=["Bob"],
            timestamp=datetime(2026, 5, 1, 15, 0, 0),
        )
        ctx = edu.to_context_string(date_format="locomo")
        assert "Deployment completed." in ctx
        assert "Bob" in ctx
        # locomo format should contain AM/PM and month name
        assert "PM" in ctx or "May" in ctx or "15:00" in ctx

    def test_to_context_string_no_speakers(self) -> None:
        edu = EDURecord(
            edu_id="edu-nospk",
            text="Anonymous observation.",
            timestamp=datetime(2026, 5, 1, 12, 0),
        )
        ctx = edu.to_context_string()
        assert "Unknown" in ctx

    def test_to_context_string_no_timestamp(self) -> None:
        edu = EDURecord(
            edu_id="edu-nots",
            text="Timeless fact.",
            source_speakers=["Alice"],
        )
        ctx = edu.to_context_string()
        assert "unknown date" in ctx

    def test_to_context_string_default_date_format_is_iso(self) -> None:
        edu = EDURecord(
            edu_id="edu-default",
            text="Check default format.",
            timestamp=datetime(2026, 5, 1, 12, 0, 0),
        )
        ctx = edu.to_context_string()
        # Default is iso format
        assert "2026-05-01T12:00:00" in ctx


# ===================================================================
# ArgumentRecord tests
# ===================================================================

class TestArgumentRecord:
    """Test ArgumentRecord creation and ID computation."""

    def test_create_basic_argument(self) -> None:
        arg = ArgumentRecord(
            arg_id="arg-001",
            text="Alice",
            source_edu_ids=["edu-001", "edu-002"],
        )
        assert arg.arg_id == "arg-001"
        assert arg.text == "Alice"
        assert arg.source_edu_ids == ["edu-001", "edu-002"]

    def test_default_source_edu_ids(self) -> None:
        arg = ArgumentRecord(arg_id="arg-002", text="Project X")
        assert arg.source_edu_ids == []

    def test_compute_id_deterministic(self) -> None:
        text = "Project Alpha"
        id1 = ArgumentRecord.compute_id(text)
        id2 = ArgumentRecord.compute_id(text)
        assert id1 == id2
        assert id1.startswith("argument-")

    def test_compute_id_different_texts_different_ids(self) -> None:
        id1 = ArgumentRecord.compute_id("Entity A")
        id2 = ArgumentRecord.compute_id("Entity B")
        assert id1 != id2

    def test_compute_id_equals_md5_of_text(self) -> None:
        text = "Verify argument md5"
        expected = "argument-" + md5(text.encode()).hexdigest()
        assert ArgumentRecord.compute_id(text) == expected


# ===================================================================
# SessionRecord tests
# ===================================================================

class TestSessionRecord:
    """Test SessionRecord creation and ID computation."""

    def test_create_basic_session(self) -> None:
        session = SessionRecord(
            session_id="session-001",
            turns=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            summary="Greeting exchange.",
            date=datetime(2026, 5, 1),
        )
        assert session.session_id == "session-001"
        assert len(session.turns) == 2
        assert session.summary == "Greeting exchange."
        assert session.date == datetime(2026, 5, 1)

    def test_default_values(self) -> None:
        session = SessionRecord(session_id="session-empty")
        assert session.turns == []
        assert session.summary is None
        assert session.date is None

    def test_compute_id_deterministic(self) -> None:
        key = "channel:cli:user123"
        id1 = SessionRecord.compute_id(key)
        id2 = SessionRecord.compute_id(key)
        assert id1 == id2
        assert id1.startswith("session-")

    def test_compute_id_different_keys_different_ids(self) -> None:
        id1 = SessionRecord.compute_id("key-a")
        id2 = SessionRecord.compute_id("key-b")
        assert id1 != id2


# ===================================================================
# EMemConfig tests
# ===================================================================

class TestEMemConfig:
    """Test EMemConfig defaults and attribute access."""

    def test_default_values(self) -> None:
        config = EMemConfig()
        assert config.linking_top_k == 5
        assert config.retrieval_top_k == 200
        assert config.qa_top_k == 5
        assert config.damping == 0.5
        assert config.synonymy_edge_topk == 2047
        assert config.synonymy_edge_sim_threshold == 0.8
        assert config.synonymy_edge_query_batch_size == 1000
        assert config.synonymy_edge_key_batch_size == 10000
        assert config.skip_ppr is False
        assert config.skip_edu_context_gen is True
        assert config.passage_node_weight == 0.05
        assert config.embedding_batch_size == 16
        assert config.embedding_return_as_normalized is True
        assert config.force_reindex is False
        assert config.date_format_type == "iso"

    def test_custom_values(self) -> None:
        config = EMemConfig(
            linking_top_k=10,
            retrieval_top_k=100,
            damping=0.8,
            skip_ppr=True,
        )
        assert config.linking_top_k == 10
        assert config.retrieval_top_k == 100
        assert config.damping == 0.8
        assert config.skip_ppr is True
        # Defaults for the rest
        assert config.qa_top_k == 5

    def test_is_dataclass(self) -> None:
        """EMemConfig is a dataclass so equality works by value."""
        config1 = EMemConfig(linking_top_k=10)
        config2 = EMemConfig(linking_top_k=10)
        assert config1 == config2
        config3 = EMemConfig(linking_top_k=5)
        assert config1 != config3


# ===================================================================
# Utility function tests
# ===================================================================

class TestComputeMdhashId:
    """Test compute_mdhash_id utility."""

    def test_empty_prefix(self) -> None:
        result = compute_mdhash_id("test content")
        assert len(result) == 32  # md5 hex is 32 chars
        assert result == md5("test content".encode()).hexdigest()

    def test_with_prefix(self) -> None:
        result = compute_mdhash_id("test content", prefix="pre-")
        assert result.startswith("pre-")
        assert len(result) == 32 + 4  # prefix len + md5 len

    def test_deterministic(self) -> None:
        result1 = compute_mdhash_id("same content", prefix="pfx-")
        result2 = compute_mdhash_id("same content", prefix="pfx-")
        assert result1 == result2

    def test_different_content_different_hash(self) -> None:
        h1 = compute_mdhash_id("content A")
        h2 = compute_mdhash_id("content B")
        assert h1 != h2


class TestMinMaxNormalize:
    """Test min_max_normalize utility."""

    def test_basic_normalization(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = min_max_normalize(x)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.5)

    def test_all_same_values_returns_ones(self) -> None:
        x = np.array([3.0, 3.0, 3.0])
        result = min_max_normalize(x)
        assert np.all(result == 1.0)

    def test_negative_values(self) -> None:
        x = np.array([-10.0, 0.0, 10.0])
        result = min_max_normalize(x)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_single_value(self) -> None:
        x = np.array([42.0])
        result = min_max_normalize(x)
        assert result[0] == 1.0

    def test_large_array(self) -> None:
        x = np.arange(100.0)
        result = min_max_normalize(x)
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_non_contiguous_data(self) -> None:
        x = np.array([5.0, 1.0, 9.0, 3.0])
        result = min_max_normalize(x)
        assert result[0] == pytest.approx(0.5)   # (5-1)/(9-1)
        assert result[1] == pytest.approx(0.0)   # (1-1)/(9-1)
        assert result[2] == pytest.approx(1.0)   # (9-1)/(9-1)
        assert result[3] == pytest.approx(0.25)  # (3-1)/(9-1)
