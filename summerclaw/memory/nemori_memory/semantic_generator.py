"""Semantic memory extraction from episodes.

Ported from nemori (https://github.com/nemori-ai/nemori).

Implements the Predict-Calibrate learning loop:
  1. Predict: hypothesise episode content from existing semantic knowledge
  2. Calibrate: extract high-value facts from discrepancies
  3. Direct Extraction: fallback when no existing semantics exist

Knowledge classification: identity / preference / relationship / goal / belief / habit
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from summerclaw.memory.nemori_memory.models import Episode, SemanticMemory
from summerclaw.memory.nemori_memory.prompts import PromptTemplates

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider

logger = logging.getLogger("nemori")


def _extract_text(msg_dict: dict[str, Any]) -> str:
    """Extract text from a source_message dict, handling both str and content array."""
    content = msg_dict.get("content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            parts.append("[image]")
    return " ".join(parts)


class SemanticGenerator:
    """Extracts semantic memories from episodes using Predict-Calibrate.

    Two modes:
    - Predict-Calibrate (when existing_semantics provided and enabled):
        1. Predict episode from existing knowledge → 2. Extract deltas
    - Direct Extraction (fallback when no existing semantics):
        Single-step extraction from episode content
    """

    def __init__(
        self,
        provider: "LLMProvider",
        model: str,
        enable_prediction_correction: bool = True,
    ) -> None:
        self._provider = provider
        self._model = model
        self._enable_pc = enable_prediction_correction

    async def generate(
        self,
        user_id: str,
        agent_id: str,
        episode: Episode,
        existing_semantics: list[SemanticMemory],
    ) -> list[SemanticMemory]:
        """Extract semantic memories from an episode.

        Args:
            user_id: User identifier.
            agent_id: Agent namespace.
            episode: The episode to extract knowledge from.
            existing_semantics: Previously extracted semantic memories for context.

        Returns:
            List of new SemanticMemory objects.
        """
        try:
            if self._enable_pc and existing_semantics:
                statements = await self._prediction_correction(episode, existing_semantics)
            else:
                statements = await self._direct_extraction(episode)

            if not statements:
                return []

            memories: list[SemanticMemory] = []
            for stmt in statements:
                memories.append(SemanticMemory(
                    user_id=user_id,
                    content=stmt,
                    memory_type=self._classify_type(stmt),
                    agent_id=agent_id,
                    embedding=None,  # File backend: no embeddings by default
                    source_episode_id=episode.id,
                ))
            return memories

        except Exception as e:
            logger.warning("Semantic generation failed: %s", e)
            return []

    # ── Predict-Calibrate ──────────────────────────────────────────────

    async def _prediction_correction(
        self, episode: Episode, existing: list[SemanticMemory]
    ) -> list[str]:
        """Two-step: predict from knowledge, then extract deltas."""
        knowledge = [s.content for s in existing]

        # Step 1: Predict
        predict_prompt = PromptTemplates.get_prediction_prompt(episode.title, knowledge)
        try:
            predict_resp = await self._provider.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": predict_prompt}],
                tools=None,
                tool_choice=None,
            )
            predicted = predict_resp.content or ""
        except Exception as e:
            logger.warning("Prediction step failed, falling back to direct extraction: %s", e)
            return await self._direct_extraction(episode)

        # Step 2: Extract deltas
        original = "\n".join(
            f"{m.get('role', 'unknown')}: {_extract_text(m)}"
            for m in episode.source_messages
        )
        extract_prompt = PromptTemplates.EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT.format(
            original_messages=original,
            predicted_episode=predicted,
        )
        try:
            extract_resp = await self._provider.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": extract_prompt}],
                tools=None,
                tool_choice=None,
            )
            return self._parse_statements(extract_resp.content or "")
        except Exception as e:
            logger.warning("Extract step failed: %s", e)
            return []

    # ── Direct Extraction ───────────────────────────────────────────────

    async def _direct_extraction(self, episode: Episode) -> list[str]:
        """Single-step extraction from episode content."""
        ep_text = f"Episode 1:\nTitle: {episode.title}\nContent: {episode.content}"
        prompt = PromptTemplates.get_semantic_generation_prompt(ep_text)
        try:
            resp = await self._provider.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
            )
            return self._parse_statements(resp.content or "")
        except Exception as e:
            logger.warning("Direct extraction failed: %s", e)
            return []

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_statements(content: str) -> list[str]:
        """Parse JSON statements from LLM response."""
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(text)
            return data.get("statements", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse semantic statements")
            return []

    @staticmethod
    def _classify_type(statement: str) -> str:
        """Simple keyword-based classification of semantic memory type."""
        lower = statement.lower()
        if any(w in lower for w in ["name is", "works at", "job", "profession", "role is"]):
            return "identity"
        if any(w in lower for w in ["likes", "prefers", "favorite", "enjoys"]):
            return "preference"
        if any(w in lower for w in ["family", "friend", "colleague", "partner", "wife", "husband"]):
            return "relationship"
        if any(w in lower for w in ["goal", "plan", "wants to", "aims to", "intends"]):
            return "goal"
        if any(w in lower for w in ["believes", "thinks that", "values"]):
            return "belief"
        if any(w in lower for w in ["every", "always", "usually", "routine", "habit"]):
            return "habit"
        return "identity"
