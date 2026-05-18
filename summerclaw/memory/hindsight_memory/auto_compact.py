"""Hindsight AutoCompact — idle session compression with Hindsight retention.

When a session has been idle for ``session_ttl_minutes``, archives old messages
and retains summaries to both the file store and the Hindsight server.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from summerclaw.memory.naive_memory.consolidator import Consolidator
    from summerclaw.session.manager import Session, SessionManager


_RECENT_SUFFIX_MESSAGES = 8


class HindsightAutoCompact:
    """Idle session compression with Hindsight server retention.

    Behaves identically to naive AutoCompact for message archival and session
    trimming.  Additionally, when a Hindsight server is available, summaries
    are retained via the server API for long-term semantic search.
    """

    def __init__(
        self,
        sessions: "SessionManager",
        consolidator: "Consolidator",
        session_ttl_minutes: int,
        *,
        hindsight_store: Any = None,
    ):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set[str] = set()
        self._summaries: dict[str, tuple[str, datetime]] = {}
        self._hindsight_store = hindsight_store

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    @property
    def has_hindsight(self) -> bool:
        return self._hindsight_store is not None and self._hindsight_store.hindsight_enabled

    # -- time helpers ---------------------------------------------------------

    def _is_expired(
        self, ts: datetime | str | None, now: datetime | None = None,
    ) -> bool:
        if self._ttl <= 0 or not ts:
            return False
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ((now or datetime.now()) - ts).total_seconds() >= self._ttl * 60

    @staticmethod
    def _format_summary(text: str, last_active: datetime) -> str:
        idle_min = int((datetime.now() - last_active).total_seconds() / 60)
        return f"Inactive for {idle_min} minutes.\nPrevious conversation summary: {text}"

    # -- unconsolidated split ------------------------------------------------

    def _split_unconsolidated(
        self, session: "Session",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tail = list(session.messages[session.last_consolidated:])
        if not tail:
            return [], []

        from summerclaw.session.manager import Session as SessCls
        probe = SessCls(
            key=session.key,
            messages=tail.copy(),
            created_at=session.created_at,
            updated_at=session.updated_at,
            metadata={},
            last_consolidated=0,
        )
        probe.retain_recent_legal_suffix(_RECENT_SUFFIX_MESSAGES)
        kept = probe.messages
        cut = len(tail) - len(kept)
        return tail[:cut], kept

    # -- agent-loop interface -------------------------------------------------

    def check_expired(
        self,
        schedule_background: Callable[[Coroutine], None],
        active_session_keys: Collection[str] = (),
    ) -> None:
        if not self.enabled:
            return
        now = datetime.now()
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            if not key or key in self._archiving:
                continue
            if key in active_session_keys:
                continue
            if self._is_expired(info.get("updated_at"), now):
                self._archiving.add(key)
                schedule_background(self._archive(key))

    def prepare_session(
        self, session: "Session", key: str,
    ) -> tuple["Session", str | None]:
        if key in self._archiving or self._is_expired(session.updated_at):
            logger.info(
                "HindsightAutoCompact: reloading session {} (archiving={})",
                key, key in self._archiving,
            )
            session = self.sessions.get_or_create(key)
        entry = self._summaries.pop(key, None)
        if entry:
            session.metadata.pop("_last_summary", None)
            return session, self._format_summary(entry[0], entry[1])
        if "_last_summary" in session.metadata:
            meta = session.metadata.pop("_last_summary")
            self.sessions.save(session)
            return session, self._format_summary(
                meta["text"], datetime.fromisoformat(meta["last_active"]),
            )
        return session, None

    # -- archival -------------------------------------------------------------

    async def _archive(self, key: str) -> None:
        try:
            self.sessions.invalidate(key)
            session = self.sessions.get_or_create(key)
            archive_msgs, kept_msgs = self._split_unconsolidated(session)
            if not archive_msgs and not kept_msgs:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return

            last_active = session.updated_at
            summary = ""
            if archive_msgs:
                summary = await self.consolidator.archive(archive_msgs) or ""
            if summary and summary != "(nothing)":
                self._summaries[key] = (summary, last_active)
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                }

            session.messages = kept_msgs
            session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)
            if archive_msgs:
                logger.info(
                    "HindsightAutoCompact: archived {} (archived={}, kept={}, summary={})",
                    key,
                    len(archive_msgs),
                    len(kept_msgs),
                    bool(summary),
                )
        except Exception:
            logger.exception("HindsightAutoCompact: _archive failed for {}", key)
        finally:
            self._archiving.discard(key)
