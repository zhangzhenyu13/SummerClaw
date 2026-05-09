"""Mem0V3 AutoCompact — idle session compression for mem0 v3.

When a session has been idle for session_ttl_minutes, the auto-compact
triggers consolidation and extracts vector memories from the conversation
before trimming the session.

Follows the same interface pattern as naive AutoCompact and EMemAutoCompact:
  - ``check_expired(schedule_background, active_session_keys)``
  - ``prepare_session(session, key) → (session, summary_or_none)``
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory.mem0v3_memory.consolidator import Mem0V3Consolidator
    from nanobot.session.manager import Session, SessionManager


_RECENT_SUFFIX_MESSAGES = 8


class Mem0V3AutoCompact:
    """Idle session compression — triggers vector memory extraction and session cleanup.

    When ``session_ttl_minutes > 0``, this component monitors session
    idle time and triggers vector memory extraction when a session has
    been inactive for too long.
    """

    def __init__(
        self,
        sessions: "SessionManager",
        consolidator: "Mem0V3Consolidator",
        session_ttl_minutes: int,
    ):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set[str] = set()
        self._summaries: dict[str, tuple[str, datetime]] = {}
        self._running = False

    # -- time helpers ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def _is_expired(self, ts: datetime | str | None, now: datetime | None = None) -> bool:
        if self._ttl <= 0 or not ts:
            return False
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ((now or datetime.now()) - ts).total_seconds() >= self._ttl * 60

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        if not self.enabled:
            logger.debug("Mem0V3 AutoCompact disabled (session_ttl_minutes=0)")
            return
        if self._running:
            return
        self._running = True
        logger.info("Mem0V3 AutoCompact started (ttl={}min)", self._ttl)

    async def stop(self) -> None:
        self._running = False
        logger.debug("Mem0V3 AutoCompact stopped")

    # -- agent-loop interface --------------------------------------------------

    def check_expired(
        self,
        schedule_background: Callable[[Coroutine], None],
        active_session_keys: Collection[str] = (),
    ) -> None:
        """Schedule archival for idle sessions, skipping those with in-flight agent tasks."""
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
        """Check for archiving or expired session; return summary if available."""
        if key in self._archiving or self._is_expired(session.updated_at):
            logger.info(
                "Mem0V3 AutoCompact: reloading session {} (archiving={})",
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

    def _split_unconsolidated(
        self, session: "Session",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split unconsolidated messages into archive prefix and recent suffix."""
        tail = list(session.messages[session.last_consolidated:])
        if not tail:
            return [], []

        # Build a probe session so we can use retain_recent_legal_suffix
        from nanobot.session.manager import Session as SessCls
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

    async def _archive(self, key: str) -> None:
        """Extract vector memories from unconsolidated messages and trim session."""
        try:
            self.sessions.invalidate(key)
            session = self.sessions.get_or_create(key)
            all_msgs = list(session.messages)
            archive_msgs, kept_msgs = self._split_unconsolidated(session)

            if not archive_msgs and not kept_msgs:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return

            last_active = session.updated_at
            summary = ""

            # Extract vector memories from all unconsolidated messages
            unprocessed = list(session.messages[session.last_consolidated:])
            if unprocessed:
                try:
                    extracted = await self.consolidator.extract_and_store(
                        unprocessed, session,
                    )
                    if extracted:
                        summary = f"Extracted {len(extracted)} memories"
                        logger.info(
                            "Mem0V3 AutoCompact: extracted {} memories for {}",
                            len(extracted), key,
                        )
                except Exception:
                    logger.exception(
                        "Mem0V3 AutoCompact: extraction failed for {}", key,
                    )

            if summary:
                self._summaries[key] = (summary, last_active)
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                }

            # Trim session to recent suffix only
            session.messages = kept_msgs
            session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)

            logger.info(
                "Mem0V3 AutoCompact: archived {} (unprocessed={}, kept={})",
                key, len(unprocessed), len(kept_msgs),
            )
        except Exception:
            logger.exception("Mem0V3 AutoCompact: _archive failed for {}", key)
        finally:
            self._archiving.discard(key)

    @staticmethod
    def _format_summary(text: str, last_active: datetime) -> str:
        idle_min = int((datetime.now() - last_active).total_seconds() / 60)
        return f"Inactive for {idle_min} minutes.\nPrevious conversation summary: {text}"
