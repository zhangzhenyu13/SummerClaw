"""Batch message segmentation into topic-coherent episode groups.

Ported from nemori (https://github.com/nemori-ai/nemori).
Uses LLM-powered boundary detection to group messages into episodes.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from summerclaw.memory.nemori_memory.models import Message
from summerclaw.memory.nemori_memory.prompts import PromptTemplates

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider

logger = logging.getLogger("nemori")

# Max messages per segmentation call to keep LLM output manageable
_SEGMENT_CHUNK_SIZE = 80


class BatchSegmenter:
    """Segments a batch of messages into coherent episode groups using LLM."""

    def __init__(self, provider: "LLMProvider", model: str) -> None:
        self._provider = provider
        self._model = model

    async def segment(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Segment messages into groups. Returns list of {messages, topic}.

        For large batches (>_SEGMENT_CHUNK_SIZE), splits into chunks and
        segments each chunk independently.
        """
        if not messages:
            return []

        if len(messages) <= _SEGMENT_CHUNK_SIZE:
            return await self._segment_chunk(messages, offset=0)

        # Split into manageable chunks
        all_groups: list[dict[str, Any]] = []
        for start in range(0, len(messages), _SEGMENT_CHUNK_SIZE):
            chunk = messages[start : start + _SEGMENT_CHUNK_SIZE]
            groups = await self._segment_chunk(chunk, offset=start)
            all_groups.extend(groups)

        return all_groups if all_groups else [{"messages": messages, "topic": "conversation"}]

    async def _segment_chunk(
        self, messages: list[Message], offset: int = 0
    ) -> list[dict[str, Any]]:
        """Segment a single chunk of messages via LLM."""
        formatted_lines: list[str] = []
        for i, msg in enumerate(messages, 1):
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else ""
            content = msg.text_content()
            if ts:
                formatted_lines.append(f"{i}. [{ts}] {msg.role}: {content}")
            else:
                formatted_lines.append(f"{i}. {msg.role}: {content}")
        formatted = "\n".join(formatted_lines)

        prompt = PromptTemplates.get_batch_segmentation_prompt(
            count=len(messages), messages=formatted
        )

        try:
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent conversation segmentation expert. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            parsed = self._parse_json(response.content or "")
            return self._build_groups(parsed, messages)

        except Exception as e:
            logger.warning("Segmentation failed, returning single group: %s", e)
            return [{"messages": messages, "topic": "conversation"}]

    def _build_groups(
        self, parsed: dict[str, Any], messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Build message groups from LLM segmentation response."""
        groups: list[dict[str, Any]] = []
        for ep in parsed.get("episodes", []):
            indices = ep.get("indices", [])
            topic = ep.get("topic", "")
            group_messages: list[Message] = []
            for idx in indices:
                if 1 <= idx <= len(messages):
                    group_messages.append(messages[idx - 1])
            if group_messages:
                groups.append({"messages": group_messages, "topic": topic})

        return groups if groups else [{"messages": messages, "topic": "conversation"}]

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
