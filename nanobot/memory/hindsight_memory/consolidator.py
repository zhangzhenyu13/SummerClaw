"""Hindsight consolidator — token-budget consolidation + Hindsight server retention.

Extends the naive Consolidator with optional Hindsight server-backed memory
retention.  When a Hindsight server is available, consolidated summaries are
also retained via the server's API for multi-strategy semantic recall (TEMPR).
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.memory.naive_memory.consolidator import Consolidator as _NaiveConsolidator
from nanobot.memory.naive_memory.store import MemoryStore
from nanobot.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


class HindsightConsolidator(_NaiveConsolidator):
    """Token-budget consolidator with optional Hindsight server retention.

    Inherits all token-budget logic from the naive Consolidator.  Additionally,
    when a Hindsight server is configured, summarised content is retained via
    :meth:`hindsight_store.aretain` for server-side semantic search.

    Hermes Mode:
        When ``skill_autogen.enable: true``, the mid-turn skill distillation
        triggers ``extract_and_store()`` on the recent conversation to capture
        facts before skill generation.  The Hindsight consolidator sends these
        facts to both the file store and the Hindsight server.
    """

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_CHUNK_MESSAGES = 60
    _SAFETY_BUFFER = 1024

    def __init__(
        self,
        store: MemoryStore,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        *,
        hindsight_store: Any = None,
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
        self._hindsight_store = hindsight_store

    @property
    def has_hindsight(self) -> bool:
        return self._hindsight_store is not None and self._hindsight_store.hindsight_enabled

    # ------------------------------------------------------------------
    # Override archive to also retain on Hindsight server
    # ------------------------------------------------------------------

    async def archive(self, messages: list[dict]) -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        If a Hindsight server is available, also retain the summary there.
        """
        summary = await super().archive(messages)
        if summary and self.has_hindsight:
            try:
                await self._hindsight_store.aretain(
                    content=summary,
                    context=f"nanobot consolidation: {len(messages)} messages",
                )
                logger.debug("Hindsight retain: {} chars summary", len(summary))
            except Exception:
                logger.exception("Hindsight retain failed in archive()")
        return summary

    # ------------------------------------------------------------------
    # Hermes-Autogen: extract_and_store (mid-turn skill distillation)
    # ------------------------------------------------------------------

    async def extract_and_store(
        self,
        messages: list[dict],
        session: "Session",
        *,
        custom_instructions: str | None = None,
    ) -> list[dict]:
        """Extract facts from messages and store to both file and Hindsight.

        Used by Hermes-Autogen mid-turn skill distillation to capture facts
        from the recent conversation before generating skills.

        Args:
            messages: Recent conversation messages to extract from.
            session: The current session.
            custom_instructions: Optional extraction override instructions.

        Returns:
            List of extracted fact dicts with ``memory`` and ``event`` keys.
        """
        if not messages:
            return []

        # Build a summary first (reuse archive logic)
        formatted = MemoryStore._format_messages(messages)
        extracted_facts: list[dict] = []

        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/consolidator_archive.md",
                            strip=True,
                        ),
                    },
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"LLM returned error: {response.content}")

            summary = response.content or ""
            if summary:
                # Store in local history
                self.store.append_history(summary)

                # Build extracted fact records
                facts = self._parse_summary_into_facts(summary)
                for fact_text in facts:
                    extracted_facts.append({
                        "memory": fact_text,
                        "event": "ADD",
                        "id": "",
                    })

                # Retain on Hindsight server
                if self.has_hindsight and extracted_facts:
                    try:
                        for fact in extracted_facts:
                            await self._hindsight_store.aretain(
                                content=fact["memory"],
                                context="nanobot hermes extraction",
                            )
                        logger.debug(
                            "Hindsight hermes retain: {} facts",
                            len(extracted_facts),
                        )
                    except Exception:
                        logger.exception("Hindsight hermes retain failed")
        except Exception:
            logger.warning("Hermes extraction LLM call failed, raw-dumping")
            self.store.raw_archive(messages)

        return extracted_facts

    @staticmethod
    def _parse_summary_into_facts(summary: str) -> list[str]:
        """Heuristically split a summary into individual fact lines."""
        facts: list[str] = []
        for line in summary.strip().split("\n"):
            line = line.strip()
            if not line or len(line) < 5:
                continue
            # Strip common bullet markers
            for prefix in ("- ", "* ", "• ", "1. ", "2. ", "3. "):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                facts.append(line)
        return facts or [summary.strip()]
