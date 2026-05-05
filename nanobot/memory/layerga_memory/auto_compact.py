"""Layerga auto compact — idle session compression with long-term memory triggering.

Extends the naive AutoCompact with:
1. L0-guided long-term memory trigger detection
2. L4 session archive writing on compression
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from nanobot.memory.naive_memory.auto_compact import AutoCompact
from nanobot.memory.layerga_memory.decision_tree import (
    L0DecisionTree,
    MemoryLayer,
    VerifiedFact,
)

if TYPE_CHECKING:
    from nanobot.memory.layerga_memory.consolidator import LayergaConsolidator
    from nanobot.memory.layerga_memory.store import LayergaStore
    from nanobot.session.manager import Session


class LayergaAutoCompact(AutoCompact):
    """Auto-compact with L0-guided long-term memory triggering.

    On top of standard idle session compression, this class:
    1. Appends compression summaries to L4 session archives.
    2. Detects significant patterns that warrant long-term memory updates.
    """

    def __init__(
        self,
        sessions,
        consolidator: LayergaConsolidator,
        session_ttl_minutes: int = 0,
        decision_tree: L0DecisionTree | None = None,
        enable_l4_archive: bool = True,
    ):
        super().__init__(
            sessions=sessions,
            consolidator=consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.layered_consolidator = consolidator
        self.decision_tree = decision_tree
        self.enable_l4_archive = enable_l4_archive

    # ------------------------------------------------------------------
    # Override: archive to L4
    # ------------------------------------------------------------------

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
                summary = await self.layered_consolidator.archive(archive_msgs) or ""

            if summary and summary != "(nothing)":
                self._summaries[key] = (summary, last_active)
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                }

                # Write to L4 session archives
                if self.enable_l4_archive:
                    try:
                        store: LayergaStore = self.layered_consolidator.layered_store
                        store.append_archive(summary)
                    except Exception:
                        logger.debug("L4 archive write skipped for {}", key)

                # Check for long-term memory triggers
                await self._maybe_trigger_long_term_update(
                    session, archive_msgs, summary
                )

            session.messages = kept_msgs
            session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)

            if archive_msgs:
                logger.info(
                    "Layerga auto-compact: archived {} (archived={}, kept={}, summary={})",
                    key,
                    len(archive_msgs),
                    len(kept_msgs),
                    bool(summary),
                )
        except Exception:
            logger.exception("Layerga auto-compact: failed for {}", key)
        finally:
            self._archiving.discard(key)

    # ------------------------------------------------------------------
    # Long-term memory trigger detection
    # ------------------------------------------------------------------

    async def _maybe_trigger_long_term_update(
        self, session: Session, messages: list[dict], summary: str
    ) -> bool:
        """Detect whether the session contains patterns worth long-term memory.

        Uses the L0 decision tree to evaluate extracted facts from the
        archived messages. If significant patterns are found, they will
        be processed by the Consolidator's classification pipeline.

        Returns True if any patterns were detected and classified.
        """
        if not self.decision_tree:
            return False

        # Extract facts from archived messages
        verified_facts = self.layered_consolidator._extract_verified_facts(messages)
        if not verified_facts:
            return False

        # Classify each fact
        stats = {"L1": 0, "L2": 0, "L3": 0, "dropped": 0}
        store: LayergaStore = self.layered_consolidator.layered_store

        for fact in verified_facts:
            result = self.decision_tree.classify(fact)
            if result.layer == MemoryLayer.L1_RULES:
                stats["L1"] += 1
            elif result.layer == MemoryLayer.L2:
                stats["L2"] += 1
            elif result.layer in (MemoryLayer.L3_SOP, MemoryLayer.L3_SCRIPT):
                stats["L3"] += 1
            else:
                stats["dropped"] += 1

            if result.layer != MemoryLayer.DROP:
                logger.debug(
                    "Layerga auto-compact: LTM trigger — {} → {} ({})",
                    fact.source_tool,
                    result.layer.value,
                    result.reason[:80],
                )

        return stats["L1"] > 0 or stats["L2"] > 0 or stats["L3"] > 0
