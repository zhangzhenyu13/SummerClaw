"""EMem consolidator — token-budget consolidation with EDU extraction and indexing.

Extends the naive Consolidator's token-budget management with EMem-specific
EDU extraction: when messages are archived, EDUs are extracted via LLM and
indexed into the EMemStore for later retrieval.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from summerclaw.memory.naive_memory.consolidator import Consolidator
from summerclaw.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain
from summerclaw.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from summerclaw.memory.emem_memory.edu_extractor import EDUExtractor
    from summerclaw.memory.emem_memory.store import EMemStore
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import Session, SessionManager


class EMemConsolidator(Consolidator):
    """Token-budget consolidation with EMem EDU extraction.

    Extends the naive Consolidator to additionally extract Elementary Discourse
    Units (EDUs) from archived conversation chunks and index them into the
    EMemStore for later dense retrieval and PPR graph propagation.

    Attributes:
        edu_extractor: EDUExtractor for extracting atomic propositions.
        emem_store: EMemStore where EDUs and arguments are indexed.
    """

    def __init__(
        self,
        store: Any,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        edu_extractor: "EDUExtractor",
        emem_store: "EMemStore",
        max_completion_tokens: int = 4096,
    ):
        super().__init__(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
        )
        self.edu_extractor = edu_extractor
        self.emem_store = emem_store

    async def archive(self, messages: list[dict]) -> str | None:
        """Summarize messages via LLM, then extract EDUs and index them.

        Extends the parent archive() by adding EDU extraction after the
        LLM summary is generated and appended to history.jsonl.

        Returns the summary text on success, None if nothing to archive.
        """
        if not messages:
            return None

        # First, perform the standard consolidation (LLM summary + history append)
        summary = await super().archive(messages)

        # Then, extract EDUs from the evicted messages asynchronously
        try:
            formatted = self._format_messages_for_edu(messages)
            if formatted.strip():
                edus = await self.edu_extractor.extract_from_history(
                    history_text=formatted,
                    session_id="",
                )
                if edus:
                    # Index new EDUs into the EMem store
                    self.emem_store.edu_store.insert_content(edus)
                    logger.debug(
                        "EMem: indexed {} EDUs from consolidation chunk", len(edus)
                    )
        except Exception:
            logger.exception("EMem EDU extraction during consolidation failed")

        return summary

    @staticmethod
    def _format_messages_for_edu(messages: list[dict]) -> str:
        """Format messages into a plain text representation for EDU extraction.

        Converts message dicts with role/content into a readable conversation
        format suitable for the EDUExtractor LLM prompt.
        """
        from summerclaw.memory.naive_memory.store import MemoryStore

        return MemoryStore._format_messages(messages)
