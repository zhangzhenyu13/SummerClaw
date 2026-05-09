"""Tests for MastraOM AutoCompact — idle session compression via Observer."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from nanobot.memory.mastra_om_memory.auto_compact import MastraOMAutoCompact


@pytest.fixture
def mock_sessions():
    s = MagicMock()
    s.save = MagicMock()
    s.invalidate = MagicMock()
    s.list_sessions = MagicMock(return_value=[])
    return s


@pytest.fixture
def mock_consolidator():
    c = MagicMock()
    c.observe_and_store = AsyncMock(return_value="Observed summary")
    return c


@pytest.fixture
def auto_compact(mock_sessions, mock_consolidator):
    return MastraOMAutoCompact(
        sessions=mock_sessions,
        consolidator=mock_consolidator,
        session_ttl_minutes=30,
    )


# ------------------------------------------------------------------
# is_expired
# ------------------------------------------------------------------


class TestIsExpired:

    def test_not_expired_when_ttl_zero(self):
        ac = MastraOMAutoCompact(MagicMock(), MagicMock(), session_ttl_minutes=0)
        assert ac._is_expired(datetime.now()) is False

    def test_not_expired_when_recent(self, auto_compact):
        recent = datetime.now() - timedelta(minutes=10)
        assert auto_compact._is_expired(recent) is False

    def test_expired_when_old(self, auto_compact):
        old = datetime.now() - timedelta(minutes=60)
        assert auto_compact._is_expired(old) is True

    def test_expired_at_boundary(self, auto_compact):
        exactly_ttl = datetime.now() - timedelta(minutes=30)
        assert auto_compact._is_expired(exactly_ttl) is True

    def test_not_expired_just_before_boundary(self, auto_compact):
        just_before = datetime.now() - timedelta(minutes=29, seconds=59)
        assert auto_compact._is_expired(just_before) is False

    def test_string_timestamp(self, auto_compact):
        old_str = (datetime.now() - timedelta(minutes=60)).isoformat()
        assert auto_compact._is_expired(old_str) is True

    def test_none_timestamp(self, auto_compact):
        assert auto_compact._is_expired(None) is False


# ------------------------------------------------------------------
# format_summary
# ------------------------------------------------------------------


class TestFormatSummary:

    def test_includes_idle_minutes(self):
        last_active = datetime.now() - timedelta(minutes=45)
        result = MastraOMAutoCompact._format_summary("Summary text", last_active)
        assert "Inactive for " in result
        assert "45 minutes" in result or "44 minutes" in result
        assert "Summary text" in result


# ------------------------------------------------------------------
# split_unconsolidated
# ------------------------------------------------------------------


class TestSplitUnconsolidated:

    def test_split_returns_two_parts(self, auto_compact):
        from nanobot.session.manager import Session

        session = Session(
            key="test:key",
            messages=[
                {"role": "user", "content": f"m{i}"} for i in range(20)
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            last_consolidated=0,
        )
        archiveable, kept = auto_compact._split_unconsolidated(session)
        assert len(archiveable) > 0
        assert len(kept) == auto_compact._RECENT_SUFFIX_MESSAGES
        assert len(archiveable) + len(kept) == 20

    def test_split_all_consolidated(self, auto_compact):
        from nanobot.session.manager import Session

        session = Session(
            key="test:key",
            messages=[{"role": "user", "content": f"m{i}"} for i in range(10)],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            last_consolidated=10,  # everything consolidated
        )
        archiveable, kept = auto_compact._split_unconsolidated(session)
        assert archiveable == []
        assert kept == []

    def test_split_small_tail_all_kept(self, auto_compact):
        from nanobot.session.manager import Session

        session = Session(
            key="test:key",
            messages=[{"role": "user", "content": "only message"}],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            last_consolidated=0,
        )
        archiveable, kept = auto_compact._split_unconsolidated(session)
        assert archiveable == []
        assert len(kept) == 1  # fewer than RECENT_SUFFIX, keep all


# ------------------------------------------------------------------
# check_expired
# ------------------------------------------------------------------


class TestCheckExpired:

    def test_schedules_archival_for_expired(self, auto_compact, mock_sessions):
        old_ts = (datetime.now() - timedelta(minutes=60)).isoformat()
        mock_sessions.list_sessions.return_value = [
            {"key": "old:session", "updated_at": old_ts},
        ]
        scheduled = []

        def capture(coro):
            scheduled.append(coro)

        auto_compact.check_expired(schedule_background=capture)
        assert len(scheduled) == 1
        assert "old:session" in auto_compact._archiving

    def test_skips_active_sessions(self, auto_compact, mock_sessions):
        old_ts = (datetime.now() - timedelta(minutes=60)).isoformat()
        mock_sessions.list_sessions.return_value = [
            {"key": "active:session", "updated_at": old_ts},
        ]
        scheduled = []

        def capture(coro):
            scheduled.append(coro)

        auto_compact.check_expired(
            schedule_background=capture,
            active_session_keys={"active:session"},
        )
        assert len(scheduled) == 0

    def test_skips_already_archiving(self, auto_compact, mock_sessions):
        old_ts = (datetime.now() - timedelta(minutes=60)).isoformat()
        auto_compact._archiving.add("already:archiving")
        mock_sessions.list_sessions.return_value = [
            {"key": "already:archiving", "updated_at": old_ts},
        ]
        scheduled = []

        def capture(coro):
            scheduled.append(coro)

        auto_compact.check_expired(schedule_background=capture)
        assert len(scheduled) == 0

    def test_skips_not_expired(self, auto_compact, mock_sessions):
        recent_ts = (datetime.now() - timedelta(minutes=10)).isoformat()
        mock_sessions.list_sessions.return_value = [
            {"key": "recent:session", "updated_at": recent_ts},
        ]
        scheduled = []

        def capture(coro):
            scheduled.append(coro)

        auto_compact.check_expired(schedule_background=capture)
        assert len(scheduled) == 0


# ------------------------------------------------------------------
# prepare_session
# ------------------------------------------------------------------


class TestPrepareSession:

    def test_returns_session_and_none_when_no_summary(self, auto_compact, mock_sessions):
        session = MagicMock()
        session.updated_at = datetime.now()
        session.metadata = {}
        result_session, summary = auto_compact.prepare_session(session, "test:key")
        assert result_session is session
        assert summary is None

    def test_returns_summary_from_in_memory(self, auto_compact, mock_sessions):
        session = MagicMock()
        session.updated_at = datetime.now()
        session.metadata = {}
        last_active = datetime.now() - timedelta(minutes=45)
        auto_compact._summaries["test:key"] = ("Cached summary", last_active)

        result_session, summary = auto_compact.prepare_session(session, "test:key")
        assert "Inactive" in summary
        assert "Cached summary" in summary

    def test_returns_summary_from_metadata(self, auto_compact, mock_sessions):
        session = MagicMock()
        session.updated_at = datetime.now()
        last_active = datetime.now() - timedelta(minutes=45)
        session.metadata = {
            "_last_summary": {
                "text": "Metadata summary",
                "last_active": last_active.isoformat(),
            }
        }

        result_session, summary = auto_compact.prepare_session(session, "test:key")
        assert "Inactive" in summary
        assert "Metadata summary" in summary
        assert "_last_summary" not in session.metadata
        mock_sessions.save.assert_called_once()

    def test_reloads_session_when_expired(self, auto_compact, mock_sessions):
        old_session = MagicMock()
        old_ts = datetime.now() - timedelta(minutes=60)
        old_session.updated_at = old_ts
        old_session.metadata = {}

        new_session = MagicMock()
        new_session.metadata = {}
        mock_sessions.get_or_create.return_value = new_session

        result_session, summary = auto_compact.prepare_session(old_session, "test:key")
        assert result_session is new_session
        mock_sessions.get_or_create.assert_called_once_with("test:key")
