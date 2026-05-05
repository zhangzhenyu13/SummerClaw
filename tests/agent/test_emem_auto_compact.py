"""Tests for EMemAutoCompact — proactive compression of idle sessions with EDU archiving."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.memory.emem_memory.auto_compact import EMemAutoCompact
from nanobot.memory.emem_memory.consolidator import EMemConsolidator
from nanobot.memory.emem_memory.store import EMemStore
from nanobot.session.manager import Session


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def mock_embedder() -> MagicMock:
    m = MagicMock()
    m.batch_encode = MagicMock(return_value=[])
    return m


@pytest.fixture
def store(tmp_path, mock_embedder: MagicMock) -> EMemStore:
    return EMemStore(workspace=tmp_path, embedding_model=mock_embedder)


@pytest.fixture
def mock_sessions(tmp_path) -> MagicMock:
    sm = MagicMock()
    sm.save = MagicMock()
    sm.invalidate = MagicMock()

    def _get_or_create(key: str) -> Session:
        return Session(key=key)

    sm.get_or_create = MagicMock(side_effect=_get_or_create)
    sm.list_sessions = MagicMock(return_value=[])
    return sm


@pytest.fixture
def mock_consolidator(store: EMemStore, mock_sessions: MagicMock) -> MagicMock:
    """Mock EMemConsolidator for auto compact tests."""
    c = MagicMock(spec=EMemConsolidator)
    c.archive = AsyncMock(return_value="Summary.")
    return c


@pytest.fixture
def auto_compact(
    mock_sessions: MagicMock,
    mock_consolidator: MagicMock,
) -> EMemAutoCompact:
    return EMemAutoCompact(
        sessions=mock_sessions,
        consolidator=mock_consolidator,
        session_ttl_minutes=15,
    )


def _add_turns(session: Session, turns: int, *, prefix: str = "msg") -> None:
    for i in range(turns):
        session.add_message("user", f"{prefix} user {i}")
        session.add_message("assistant", f"{prefix} assistant {i}")


# ===================================================================
# EMemAutoCompact — TTL configuration
# ===================================================================

class TestEMemAutoCompactTTL:
    """Test TTL and expiration logic."""

    def test_default_ttl_is_zero(self, mock_sessions: MagicMock, mock_consolidator: MagicMock) -> None:
        ac = EMemAutoCompact(sessions=mock_sessions, consolidator=mock_consolidator)
        assert ac._ttl == 0

    def test_custom_ttl_stored(
        self, mock_sessions: MagicMock, mock_consolidator: MagicMock,
    ) -> None:
        ac = EMemAutoCompact(
            sessions=mock_sessions,
            consolidator=mock_consolidator,
            session_ttl_minutes=30,
        )
        assert ac._ttl == 30

    def test_is_expired_when_ttl_zero(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact._ttl = 0
        ts = datetime.now() - timedelta(minutes=100)
        assert auto_compact._is_expired(ts) is False

    def test_is_expired_boundary(self, auto_compact: EMemAutoCompact) -> None:
        ts = datetime.now() - timedelta(minutes=15)
        assert auto_compact._is_expired(ts) is True
        ts2 = datetime.now() - timedelta(minutes=14, seconds=59)
        assert auto_compact._is_expired(ts2) is False

    def test_is_expired_string_timestamp(self, auto_compact: EMemAutoCompact) -> None:
        ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        assert auto_compact._is_expired(ts) is True

    def test_is_expired_none(self, auto_compact: EMemAutoCompact) -> None:
        assert auto_compact._is_expired(None) is False

    def test_is_expired_empty_string(self, auto_compact: EMemAutoCompact) -> None:
        assert auto_compact._is_expired("") is False

    def test_is_expired_custom_now(self, auto_compact: EMemAutoCompact) -> None:
        now = datetime(2026, 5, 5, 12, 0, 0)
        ts = datetime(2026, 5, 5, 11, 44, 0)
        assert auto_compact._is_expired(ts, now=now) is True
        ts2 = datetime(2026, 5, 5, 11, 46, 0)
        assert auto_compact._is_expired(ts2, now=now) is False


# ===================================================================
# EMemAutoCompact — format_summary
# ===================================================================

class TestEMemAutoCompactFormatSummary:
    """Test _format_summary static method."""

    def test_format_summary(self) -> None:
        last_active = datetime.now() - timedelta(minutes=10)
        summary = EMemAutoCompact._format_summary(
            "User discussed deployment.", last_active,
        )
        assert "Inactive for" in summary
        assert "User discussed deployment." in summary


# ===================================================================
# EMemAutoCompact — _split_unconsolidated
# ===================================================================

class TestEMemAutoCompactSplit:
    """Test _split_unconsolidated logic."""

    def test_split_empty_session(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        archive, kept = auto_compact._split_unconsolidated(session)
        assert archive == []
        assert kept == []

    def test_split_splits_older_from_recent(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"user msg {i}")
            session.add_message("assistant", f"assistant msg {i}")
        archive, kept = auto_compact._split_unconsolidated(session)
        assert len(kept) == auto_compact._RECENT_SUFFIX_MESSAGES
        assert len(archive) == 24 - len(kept)
        assert len(archive) > 0

    def test_split_respects_last_consolidated(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        for i in range(20):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")
        session.last_consolidated = 30
        archive, kept = auto_compact._split_unconsolidated(session)
        assert len(kept) == auto_compact._RECENT_SUFFIX_MESSAGES
        assert len(archive) == 10 - len(kept)
        assert len(archive) > 0


# ===================================================================
# EMemAutoCompact — _archive
# ===================================================================

class TestEMemAutoCompactArchive:
    """Test _archive method."""

    @pytest.mark.asyncio
    async def test_archive_empty_session(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)
        archive_called = False

        async def _fake_archive(msgs):
            nonlocal archive_called
            archive_called = True
            return "Summary."

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")
        assert not archive_called
        assert "cli:test" not in auto_compact._archiving

    @pytest.mark.asyncio
    async def test_archive_stores_summary(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"user msg {i}")
            session.add_message("assistant", f"assistant msg {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _fake_archive(msgs):
            return "User said hello many times."

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")

        entry = auto_compact._summaries.get("cli:test")
        assert entry is not None
        assert entry[0] == "User said hello many times."
        assert "cli:test" not in auto_compact._archiving

    @pytest.mark.asyncio
    async def test_archive_nothing_summary_not_stored(
        self, auto_compact: EMemAutoCompact,
    ) -> None:
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _fake_archive(msgs):
            return "(nothing)"

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")
        assert "cli:test" not in auto_compact._summaries

    @pytest.mark.asyncio
    async def test_archive_empty_summary_not_stored(
        self, auto_compact: EMemAutoCompact,
    ) -> None:
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _fake_archive(msgs):
            return ""

        auto_compact.consolidator.archive = _fake_archive
        await auto_compact._archive("cli:test")
        assert "cli:test" not in auto_compact._summaries

    @pytest.mark.asyncio
    async def test_archive_error_is_caught(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _failing_archive(msgs):
            raise RuntimeError("LLM down")

        auto_compact.consolidator.archive = _failing_archive
        await auto_compact._archive("cli:test")
        assert "cli:test" not in auto_compact._archiving

    @pytest.mark.asyncio
    async def test_archive_keeps_recent_suffix_after_error(
        self, auto_compact: EMemAutoCompact,
    ) -> None:
        session = Session(key="cli:test")
        for i in range(12):
            session.add_message("user", f"msg {i}")
            session.add_message("assistant", f"resp {i}")

        auto_compact.sessions.invalidate = MagicMock()
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)

        async def _failing_archive(msgs):
            raise RuntimeError("API down")

        auto_compact.consolidator.archive = _failing_archive
        await auto_compact._archive("cli:test")
        assert len(session.messages) == 24
        assert "cli:test" not in auto_compact._archiving


# ===================================================================
# EMemAutoCompact — prepare_session
# ===================================================================

class TestEMemAutoCompactPrepareSession:
    """Test prepare_session for summary recovery."""

    def test_prepare_session_no_summary(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        result_session, summary = auto_compact.prepare_session(session, "cli:test")
        assert result_session is session
        assert summary is None

    def test_prepare_session_from_in_memory(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        last_active = datetime.now() - timedelta(minutes=20)
        auto_compact._summaries["cli:test"] = ("User discussed auth.", last_active)
        session.metadata["_last_summary"] = {"text": "old", "last_active": "2026-01-01T00:00:00"}

        result_session, summary = auto_compact.prepare_session(session, "cli:test")
        assert summary is not None
        assert "User discussed auth." in summary
        assert "Inactive for" in summary
        assert "cli:test" not in auto_compact._summaries
        assert "_last_summary" not in result_session.metadata

    def test_prepare_session_from_metadata(self, auto_compact: EMemAutoCompact) -> None:
        session = Session(key="cli:test")
        last_active = datetime.now() - timedelta(minutes=10)
        session.metadata["_last_summary"] = {
            "text": "User prefers Go language.",
            "last_active": last_active.isoformat(),
        }

        result_session, summary = auto_compact.prepare_session(session, "cli:test")
        assert summary is not None
        assert "User prefers Go language." in summary
        assert "_last_summary" not in result_session.metadata

    def test_prepare_session_metadata_consumed_once(
        self, auto_compact: EMemAutoCompact,
    ) -> None:
        session = Session(key="cli:test")
        session.metadata["_last_summary"] = {
            "text": "Summary.",
            "last_active": datetime.now().isoformat(),
        }
        _, summary1 = auto_compact.prepare_session(session, "cli:test")
        assert summary1 is not None
        _, summary2 = auto_compact.prepare_session(session, "cli:test")
        assert summary2 is None


# ===================================================================
# EMemAutoCompact — check_expired
# ===================================================================

class TestEMemAutoCompactCheckExpired:
    """Test check_expired scheduling."""

    def test_noop_when_ttl_zero(
        self, mock_sessions: MagicMock, mock_consolidator: MagicMock,
    ) -> None:
        ac = EMemAutoCompact(
            sessions=mock_sessions,
            consolidator=mock_consolidator,
            session_ttl_minutes=0,
        )
        ac.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=30)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        ac.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_schedules_expired_sessions(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 1
        assert "cli:test" in auto_compact._archiving

    def test_skips_active_session_keys(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule, active_session_keys={"cli:test"})
        assert len(scheduled) == 0

    def test_skips_already_archiving(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact._archiving.add("cli:test")
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_skips_recent_sessions(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:test", "updated_at": datetime.now().isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_skips_empty_key(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 0

    def test_multiple_sessions_partial_expired(self, auto_compact: EMemAutoCompact) -> None:
        auto_compact.sessions.list_sessions.return_value = [
            {"key": "cli:expired", "updated_at": (datetime.now() - timedelta(minutes=20)).isoformat()},
            {"key": "cli:active", "updated_at": datetime.now().isoformat()},
            {"key": "cli:also_expired", "updated_at": (datetime.now() - timedelta(minutes=30)).isoformat()},
        ]
        scheduled = []

        def _schedule(coro):
            scheduled.append(coro)

        auto_compact.check_expired(_schedule)
        assert len(scheduled) == 2


# ===================================================================
# EMemAutoCompact — summary persistence
# ===================================================================

class TestEMemAutoCompactPersistence:
    """Test that summary survives restart via session metadata."""

    @pytest.mark.asyncio
    async def test_summary_persisted_in_session_metadata(
        self, auto_compact: EMemAutoCompact,
    ) -> None:
        session = Session(key="cli:test")
        _add_turns(session, 6, prefix="hello")
        session.updated_at = datetime.now() - timedelta(minutes=20)
        auto_compact.sessions.save(session)

        async def _fake_archive(messages):
            return "User said hello."

        auto_compact.consolidator.archive = _fake_archive
        auto_compact.sessions.get_or_create = MagicMock(return_value=session)
        auto_compact.sessions.invalidate = MagicMock()

        await auto_compact._archive("cli:test")

        meta = session.metadata.get("_last_summary")
        assert meta is not None
        assert meta["text"] == "User said hello."
        assert "last_active" in meta
