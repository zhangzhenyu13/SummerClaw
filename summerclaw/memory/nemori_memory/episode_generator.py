"""Episode generation from conversation messages.

Ported from nemori (https://github.com/nemori-ai/nemori).
Converts message groups into structured episodic narratives via LLM.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from summerclaw.memory.nemori_memory.models import Episode, Message
from summerclaw.memory.nemori_memory.prompts import PromptTemplates

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider

logger = logging.getLogger("nemori")


class EpisodeGenerator:
    """Constructs episode generation prompts and parses LLM responses."""

    def __init__(self, provider: "LLMProvider", model: str) -> None:
        self._provider = provider
        self._model = model

    async def generate(
        self,
        user_id: str,
        agent_id: str,
        messages: list[Message],
        boundary_reason: str,
    ) -> Episode:
        """Generate an episodic memory from a message group.

        Args:
            user_id: User identifier.
            agent_id: Agent namespace.
            messages: The message group to convert.
            boundary_reason: Why these messages were grouped together (from segmenter).

        Returns:
            A structured Episode dataclass.
        """
        msg_dicts = [m.to_dict() for m in messages]
        conversation = PromptTemplates.format_conversation(msg_dicts)
        prompt = PromptTemplates.get_episode_generation_prompt(conversation, boundary_reason)

        has_images = any(m.has_images() for m in messages)

        if has_images:
            user_content = self._build_multimodal_prompt(messages, boundary_reason)
        else:
            user_content = prompt

        try:
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are an episodic memory generation expert."},
                    {"role": "user", "content": user_content},
                ],
                tools=None,
                tool_choice=None,
            )
            parsed = self._parse_response(response.content or "")

            timestamp = datetime.now(timezone.utc)
            if parsed.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(parsed["timestamp"])
                except (ValueError, TypeError):
                    pass

            return Episode(
                user_id=user_id,
                title=parsed["title"],
                content=parsed["content"],
                source_messages=msg_dicts,
                agent_id=agent_id,
                metadata={"boundary_reason": boundary_reason},
                created_at=timestamp,
                updated_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning("Episode generation failed, creating fallback: %s", e)
            return self._create_fallback(user_id, agent_id, messages, boundary_reason)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse JSON response from LLM, handling markdown code fences."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

    def _build_multimodal_prompt(
        self, messages: list[Message], boundary_reason: str
    ) -> str:
        """Build a text prompt that includes image markers."""
        conversation = self._format_with_image_markers(messages)
        prompt = PromptTemplates.get_episode_generation_prompt(conversation, boundary_reason)
        prompt += "\n\n" + PromptTemplates.get_multimodal_guidance()
        return prompt

    @staticmethod
    def _format_with_image_markers(messages: list[Message]) -> str:
        """Format conversation text, marking image positions."""
        lines: list[str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                lines.append(f"{msg.role}: {msg.content}")
            else:
                msg_parts: list[str] = []
                for part in msg.content:
                    if part.get("type") == "text":
                        msg_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        msg_parts.append("[Image attached]")
                lines.append(f"{msg.role}: {' '.join(msg_parts)}")
        return "\n".join(lines)

    def _create_fallback(
        self,
        user_id: str,
        agent_id: str,
        messages: list[Message],
        boundary_reason: str,
    ) -> Episode:
        """Create a raw episode when LLM generation fails."""
        conversation = "\n".join(f"{m.role}: {m.text_content()}" for m in messages)
        return Episode(
            user_id=user_id,
            title=f"Conversation ({len(messages)} messages)",
            content=conversation,
            source_messages=[m.to_dict() for m in messages],
            agent_id=agent_id,
            metadata={"boundary_reason": boundary_reason, "fallback": True},
        )
