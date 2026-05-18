"""Episode merger for consolidating similar episodes.

Ported from nemori (https://github.com/nemori-ai/nemori).

Uses similarity search (text-based in file mode, vector-based in PG+Qdrant mode)
to find candidate episodes, then asks LLM to decide merge vs. keep-separate.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from summerclaw.memory.nemori_memory.models import Episode
from summerclaw.memory.nemori_memory.prompts import PromptTemplates

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.memory.nemori_memory.store import NemoriStore

logger = logging.getLogger("nemori")


class EpisodeMerger:
    """Checks and merges similar episodes to avoid semantic duplication."""

    def __init__(
        self,
        provider: "LLMProvider",
        model: str,
        store: "NemoriStore",
        similarity_threshold: float = 0.85,
        merge_top_k: int = 5,
    ) -> None:
        self._provider = provider
        self._model = model
        self._store = store
        self._similarity_threshold = similarity_threshold
        self._merge_top_k = merge_top_k

    async def check_and_merge(
        self, episode: Episode, agent_id: str
    ) -> tuple[bool, Episode | None, str | None]:
        """Check if episode should merge with existing ones.

        Returns:
            (merged, final_episode, old_episode_id_to_delete)
            If not merged, returns (False, None, None).
        """
        try:
            # 1. Find similar episodes via text search
            candidates = await self._find_similar(episode, agent_id)
            if not candidates:
                return False, None, None

            # 2. LLM decides whether to merge
            should_merge, target_id, reason = await self._decide_merge(episode, candidates)
            if not should_merge or not target_id:
                return False, None, None

            # 3. Find target episode
            target = next((c for c in candidates if c.id == target_id), None)
            if not target:
                return False, None, None

            # 4. Generate merged content
            merged = await self._merge_contents(target, episode, agent_id)
            logger.info(
                "Episode merge: %s + %s -> %s",
                target.id[:8], episode.id[:8], merged.id[:8],
            )
            return True, merged, target.id

        except Exception as e:
            logger.warning("Episode merge check failed: %s", e)
            return False, None, None

    async def _find_similar(self, episode: Episode, agent_id: str) -> list[Episode]:
        """Find similar episodes using text search (file backend)."""
        if episode.embedding:
            # Try vector search first
            try:
                results = self._store.search_episodes_by_vector(
                    episode.embedding, episode.user_id, agent_id,
                    self._merge_top_k + 1,
                )
                ids = [r["id"] for r in results if r["id"] != episode.id][:self._merge_top_k]
                if ids:
                    return self._store.get_episodes_batch(ids, episode.user_id, agent_id)
            except Exception:
                pass

        # Fallback to text search
        all_eps = self._store.list_episodes(episode.user_id, agent_id, limit=50)
        # Filter self, return top_k
        return [e for e in all_eps if e.id != episode.id][:self._merge_top_k]

    async def _decide_merge(
        self, new_episode: Episode, candidates: list[Episode]
    ) -> tuple[bool, str | None, str]:
        """Use LLM to decide if merging is appropriate."""
        candidates_text = self._format_candidates(candidates)
        ts = (
            new_episode.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if new_episode.created_at else "unknown"
        )
        new_time_range = f"{ts} ({len(new_episode.source_messages)} messages)"

        prompt = PromptTemplates.get_merge_decision_prompt(
            new_time_range=new_time_range,
            new_content=new_episode.content,
            candidates=candidates_text,
        )
        try:
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
            )
            parsed = self._parse_json(response.content or "")
        except Exception as e:
            logger.warning("Merge decision LLM call failed: %s", e)
            return False, None, ""

        decision = parsed.get("decision", "new")
        target_id = parsed.get("merge_target_id")
        reason = parsed.get("reason", "")
        return decision == "merge" and target_id is not None, target_id, reason

    async def _merge_contents(
        self, target: Episode, new_episode: Episode, agent_id: str
    ) -> Episode:
        """Generate merged episode content via LLM."""
        target_ts = (
            target.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if target.created_at else "unknown"
        )
        new_ts = (
            new_episode.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if new_episode.created_at else "unknown"
        )

        prompt = PromptTemplates.get_merge_content_prompt(
            original_time_range=f"{target_ts} ({len(target.source_messages)} messages)",
            original_title=target.title,
            original_content=target.content,
            new_time_range=f"{new_ts} ({len(new_episode.source_messages)} messages)",
            new_title=new_episode.title,
            new_content=new_episode.content,
            combined_events=(
                f"Original: {target.content}\n\nNew: {new_episode.content}"
            ),
        )
        try:
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
            )
            parsed = self._parse_json(response.content or "")
        except Exception as e:
            logger.warning("Merge content LLM call failed: %s", e)
            # Fallback: simple concatenation
            return Episode(
                user_id=new_episode.user_id,
                agent_id=agent_id,
                title=f"Merged: {target.title}",
                content=f"{target.content}\n\n{new_episode.content}",
                source_messages=target.source_messages + new_episode.source_messages,
                metadata={
                    "merged_from": [target.id, new_episode.id],
                    "merge_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                created_at=min(target.created_at or datetime.now(timezone.utc),
                               new_episode.created_at or datetime.now(timezone.utc)),
                updated_at=datetime.now(timezone.utc),
            )

        merged_messages = target.source_messages + new_episode.source_messages

        # Use earliest timestamp
        ts_target = target.created_at or datetime.now(timezone.utc)
        ts_new = new_episode.created_at or datetime.now(timezone.utc)
        if ts_target.tzinfo is None:
            ts_target = ts_target.replace(tzinfo=timezone.utc)
        if ts_new.tzinfo is None:
            ts_new = ts_new.replace(tzinfo=timezone.utc)
        merged_ts = min(ts_target, ts_new)

        if parsed.get("timestamp"):
            try:
                parsed_ts = datetime.fromisoformat(parsed["timestamp"])
                if parsed_ts.tzinfo is None:
                    parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
                merged_ts = parsed_ts
            except (ValueError, TypeError):
                pass

        return Episode(
            user_id=new_episode.user_id,
            agent_id=agent_id,
            title=parsed.get("title", f"Merged: {target.title}"),
            content=parsed.get("content", f"{target.content}\n\n{new_episode.content}"),
            source_messages=merged_messages,
            metadata={
                "merged_from": [target.id, new_episode.id],
                "merge_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            created_at=merged_ts,
            updated_at=datetime.now(timezone.utc),
        )

    def _format_candidates(self, candidates: list[Episode]) -> str:
        """Format candidate episodes for the merge decision prompt."""
        lines: list[str] = []
        for i, ep in enumerate(candidates, 1):
            ts = ep.created_at.strftime("%Y-%m-%d %H:%M:%S") if ep.created_at else "unknown"
            lines.append(
                f"{i}. Candidate ID: {ep.id}\n"
                f"   Time: {ts} ({len(ep.source_messages)} messages)\n"
                f"   Title: {ep.title}\n"
                f"   Content: {ep.content[:200]}..."
            )
        return "\n\n".join(lines)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON, stripping markdown fences if present."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
