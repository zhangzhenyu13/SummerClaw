"""EMem EDU extractor — LLM-based Elementary Discourse Unit extraction.

Extracts atomic propositions (EDUs) from conversation turns, optionally
with event types, triggers, and role-argument pairs.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from nanobot.memory.emem_memory.datatypes import EDURecord

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class RoleArgument(BaseModel):
    role: str = Field(description="Semantic role of the argument (e.g. AGENT, PATIENT)")
    argument: str = Field(description="The argument text (entity, value, etc.)")


class ExtractedEDU(BaseModel):
    text: str = Field(description="The EDU text — a single atomic proposition")
    event_type: str | None = Field(
        default=None,
        description="Optional event type classification",
    )
    event_triggers: list[str] = Field(
        default_factory=list,
        description="Trigger words/phrases for the event",
    )
    event_role_argument_pairs: list[RoleArgument] = Field(
        default_factory=list,
        description="Semantic role-argument pairs",
    )


class EDUExtractionResult(BaseModel):
    edus: list[ExtractedEDU] = Field(
        description="List of extracted EDUs from the conversation"
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EDU_EXTRACTION_SYSTEM = """You are an expert conversation analyst. Your task is to decompose \
conversation history into Elementary Discourse Units (EDUs) — atomic, self-contained \
propositions that represent individual facts, statements, events, or pieces of information.

For each EDU:
1. Write a **self-contained sentence** that can be understood without surrounding context.
2. Include **entity names** and **specific details** rather than pronouns or references.
3. Optionally identify the **event type** and **trigger words**.
4. Optionally extract **semantic role-argument pairs** (e.g. who did what to whom).

Rules:
- Each EDU must be a single proposition (one fact per EDU).
- Prefer specificity over generality ("Alice bought a red car on Tuesday" not "Someone bought something").
- Include temporal information when available.
- Preserve named entities exactly as they appear.
- Output valid JSON matching the specified schema."""

_EDU_EXTRACTION_USER = """## Conversation History
{conversation_text}

## Instructions
Extract all EDUs from the conversation above. For each EDU:
- Write the text as a self-contained proposition.
- Identify the event type if applicable.
- List trigger words.
- Extract role-argument pairs.

Output the result as a JSON object with an "edus" array."""

# Simplified prompt for quick extraction (no events/arguments)
_EDU_SIMPLE_SYSTEM = """You are a conversation analyst. Break down conversation history into \
atomic, self-contained propositions (EDUs). Each EDU must be understandable on its own, \
with explicit entity names and specific details instead of pronouns."""

_EDU_SIMPLE_USER = """## Conversation
{conversation_text}

Extract all atomic propositions as a JSON array of {{"text": "..."}} objects in an "edus" field."""


class EDUExtractor:
    """Extracts Elementary Discourse Units from conversation history using LLM.

    Attributes:
        provider: nanobot LLMProvider for model calls.
        model: The model name to use.
        extract_events: Whether to extract event types, triggers, and role-argument pairs.
        skip_context_gen: If True, skip generating surrounding context for each EDU.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        model: str,
        extract_events: bool = False,
        skip_context_gen: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.extract_events = extract_events
        self.skip_context_gen = skip_context_gen

    async def extract_from_history(
        self,
        history_text: str,
        session_id: str = "",
        speakers: list[str] | None = None,
        timestamp: datetime | None = None,
    ) -> list[EDURecord]:
        """Extract EDUs from formatted conversation history text.

        Args:
            history_text: Formatted conversation text.
            session_id: The session this history belongs to.
            speakers: Known speaker names.
            timestamp: Approximate timestamp of the conversation.

        Returns:
            List of EDURecord objects.
        """
        if self.extract_events:
            return await self._extract_with_events(
                history_text, session_id, speakers, timestamp,
            )
        return await self._extract_simple(
            history_text, session_id, speakers, timestamp,
        )

    async def _extract_simple(
        self,
        history_text: str,
        session_id: str,
        speakers: list[str] | None,
        timestamp: datetime | None,
    ) -> list[EDURecord]:
        """Extract EDUs without event/role-argument details."""
        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EDU_SIMPLE_SYSTEM},
                    {"role": "user", "content": _EDU_SIMPLE_USER.format(
                        conversation_text=history_text,
                    )},
                ],
                tools=None,
                tool_choice=None,
            )
        except Exception:
            logger.exception("EDU extraction LLM call failed")
            return []

        if response.finish_reason == "error" or not response.content:
            logger.warning("EDU extraction returned error or empty")
            return []

        edus = self._parse_response(response.content, simple=True)
        records: list[EDURecord] = []
        for edu in edus:
            text = edu.get("text", "") if isinstance(edu, dict) else str(edu)
            if not text.strip():
                continue
            edu_id = EDURecord.compute_id(text)
            records.append(EDURecord(
                edu_id=edu_id,
                text=text.strip(),
                source_speakers=speakers or [],
                timestamp=timestamp,
                session_id=session_id,
            ))
        return records

    async def _extract_with_events(
        self,
        history_text: str,
        session_id: str,
        speakers: list[str] | None,
        timestamp: datetime | None,
    ) -> list[EDURecord]:
        """Extract EDUs with full event/role-argument extraction."""
        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EDU_EXTRACTION_SYSTEM},
                    {"role": "user", "content": _EDU_EXTRACTION_USER.format(
                        conversation_text=history_text,
                    )},
                ],
                tools=None,
                tool_choice=None,
            )
        except Exception:
            logger.exception("EDU extraction (with events) LLM call failed")
            return []

        if response.finish_reason == "error" or not response.content:
            return []

        edus = self._parse_response(response.content, simple=False)
        records: list[EDURecord] = []
        for edu_data in edus:
            text = edu_data.get("text", "")
            if not text.strip():
                continue
            edu_id = EDURecord.compute_id(text)

            role_arg_pairs = None
            raw_pairs = edu_data.get("event_role_argument_pairs", [])
            if raw_pairs:
                role_arg_pairs = [
                    {"role": p.get("role", ""), "argument": p.get("argument", "")}
                    for p in raw_pairs
                    if p.get("argument", "").strip()
                ]

            records.append(EDURecord(
                edu_id=edu_id,
                text=text.strip(),
                source_speakers=speakers or [],
                timestamp=timestamp,
                session_id=session_id,
                event_type=edu_data.get("event_type"),
                event_triggers=edu_data.get("event_triggers", []),
                event_role_argument_pairs=role_arg_pairs,
            ))
        return records

    @staticmethod
    def _parse_response(content: str, simple: bool = False) -> list[dict[str, Any]]:
        """Parse LLM response into EDU list."""
        # Try to extract JSON from the response
        content = content.strip()

        # Remove markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            import re
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse EDU response: {content[:200]}")
                    return []
            else:
                logger.warning(f"No JSON found in EDU response: {content[:200]}")
                return []

        if isinstance(data, list):
            # Direct list of EDU dicts
            return data
        if isinstance(data, dict):
            return data.get("edus", [])
        return []

    async def extract_entities_from_query(self, query: str) -> list[str]:
        """Extract named entities from a query string for retrieval."""
        prompt = (
            'Extract all named entities (people, places, organizations, dates, '
            'topics, key terms) from the query below. '
            'Return them as a JSON array of strings in {"entities": [...]} format.\n\n'
            f"Query: {query}"
        )
        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract entities from queries."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
        except Exception:
            logger.exception("Entity extraction failed")
            return []

        if response.finish_reason == "error" or not response.content:
            return []

        try:
            data = json.loads(response.content.strip())
            return data.get("entities", [])
        except json.JSONDecodeError:
            return []
