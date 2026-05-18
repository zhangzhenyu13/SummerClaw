"""EMem rerank — LLM-based filtering of candidate EDUs and arguments.

Provides two-stage LLM reranking:
1. **EDU filtering**: Given a query and candidate EDUs, select the most relevant ones.
2. **Argument filtering**: Given a query and candidate arguments, select the most relevant ones.

Uses fuzzy matching (difflib) to align LLM output back to original candidates.
"""

from __future__ import annotations

import difflib
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EDU_FILTER_SYSTEM = """You are a memory retrieval filter. Given a user's question and a list \
of candidate memory items (EDUs — Elementary Discourse Units), select ONLY the items that \
are directly relevant to answering the question.

Each EDU is a self-contained proposition from past conversations. Keep EDUs that:
1. Contain information needed to answer the question.
2. Provide context about entities, events, or facts mentioned in the question.
3. Offer temporal or relational information relevant to the query.

Discard EDUs that are unrelated, redundant, or too vague to help.

Return the selected EDUs as a JSON array in the "selected_edus" field, preserving \
the exact text of each selected EDU."""

_EDU_FILTER_USER = """## Question
{question}

## Candidate Memory Items
{edu_list}

Select the EDUs relevant to answering the question. Return as:
{{"selected_edus": ["edu text 1", "edu text 2", ...]}}"""

_ARGUMENT_FILTER_SYSTEM = """You are a memory filter. Given a question and candidate arguments \
(entities, values, concepts extracted from past conversations), select the arguments that \
are relevant to the question.

Keep arguments that:
1. Refer to entities or concepts mentioned in the question.
2. Provide context about people, places, things, or events in the question.

Return the selected arguments as a JSON array in "selected_arguments" field."""

_ARGUMENT_FILTER_USER = """## Question
{question}

## Candidate Arguments
{argument_list}

Select the relevant arguments. Return as:
{{"selected_arguments": ["arg 1", "arg 2", ...]}}"""


class EDUReranker:
    """LLM-based EDU reranker for memory retrieval.

    Filters candidate EDUs by relevance to a query, using an LLM
    to select the most useful memory items.
    """

    def __init__(self, provider: "LLMProvider", model: str):
        self.provider = provider
        self.model = model

    async def rerank(
        self,
        query: str,
        candidate_items: list[str],
        candidate_indices: list[int],
        max_after_rerank: int | None = None,
    ) -> tuple[list[int], list[str], dict[str, Any]]:
        """Rerank candidate EDUs by relevance to the query.

        Args:
            query: The user's question/query.
            candidate_items: List of candidate EDU text strings.
            candidate_indices: Original indices for each candidate.
            max_after_rerank: Max number of EDUs to keep.

        Returns:
            Tuple of (filtered_indices, filtered_items, metadata).
        """
        if not candidate_items:
            return [], [], {"num_candidates": 0, "num_selected": 0}

        # Format candidates as numbered list for the LLM
        edu_list = "\n".join(
            f"{i + 1}. {item}" for i, item in enumerate(candidate_items)
        )

        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EDU_FILTER_SYSTEM},
                    {"role": "user", "content": _EDU_FILTER_USER.format(
                        question=query,
                        edu_list=edu_list,
                    )},
                ],
                tools=None,
                tool_choice=None,
            )
        except Exception:
            logger.exception("EDU rerank LLM call failed")
            return candidate_indices, candidate_items, {
                "error": "LLM call failed",
                "num_candidates": len(candidate_items),
            }

        if response.finish_reason == "error" or not response.content:
            logger.warning("EDU rerank returned error, keeping all candidates")
            return candidate_indices, candidate_items, {
                "error": "LLM returned error",
                "num_candidates": len(candidate_items),
                "num_selected": len(candidate_items),
            }

        selected_edus = self._parse_edu_response(response.content)

        # Match selected EDUs back to candidate items using fuzzy matching
        result_indices: list[int] = []
        matched_items: list[str] = []

        for selected in selected_edus:
            closest = difflib.get_close_matches(
                str(selected),
                [str(item) for item in candidate_items],
                n=1,
                cutoff=0.85,
            )
            if closest:
                try:
                    idx = candidate_items.index(closest[0])
                    if idx not in result_indices:
                        result_indices.append(idx)
                        matched_items.append(candidate_items[idx])
                except ValueError:
                    pass

        # Map back to original candidate indices
        filtered_indices = [
            candidate_indices[i]
            for i in result_indices
            if i < len(candidate_indices)
        ]

        if max_after_rerank is not None:
            filtered_indices = filtered_indices[:max_after_rerank]
            matched_items = matched_items[:max_after_rerank]

        if not filtered_indices:
            # Fallback: keep all candidates
            logger.debug("No EDUs matched after rerank, keeping all")
            return (
                candidate_indices[:max_after_rerank] if max_after_rerank else candidate_indices,
                candidate_items[:max_after_rerank] if max_after_rerank else candidate_items,
                {"num_candidates": len(candidate_items), "num_selected": len(candidate_items), "fallback": True},
            )

        return filtered_indices, matched_items, {
            "num_candidates": len(candidate_items),
            "num_selected": len(filtered_indices),
        }

    @staticmethod
    def _parse_edu_response(content: str) -> list[str]:
        """Parse LLM response to extract selected EDUs."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
            return data.get("selected_edus", [])
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("selected_edus", [])
                except json.JSONDecodeError:
                    pass
        return []


class ArgumentReranker:
    """LLM-based argument reranker for memory retrieval.

    Filters candidate arguments (entities/concepts) by relevance to a query.
    """

    def __init__(self, provider: "LLMProvider", model: str):
        self.provider = provider
        self.model = model

    async def rerank(
        self,
        query: str,
        candidate_arguments: list[str],
        candidate_arg_keys: list[str],
        candidate_arg_scores: list[float],
        max_after_rerank: int | None = None,
    ) -> tuple[list[str], list[str], list[float], dict[str, Any]]:
        """Rerank candidate arguments by relevance.

        Args:
            query: The user's question.
            candidate_arguments: List of argument text strings.
            candidate_arg_keys: Corresponding argument node keys.
            candidate_arg_scores: Corresponding similarity scores.
            max_after_rerank: Max arguments to keep.

        Returns:
            Tuple of (filtered_keys, filtered_arguments, filtered_scores, metadata).
        """
        if not candidate_arguments:
            return [], [], [], {"num_candidates": 0, "num_selected": 0}

        arg_list = "\n".join(
            f"{i + 1}. {arg}" for i, arg in enumerate(candidate_arguments)
        )

        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": _ARGUMENT_FILTER_SYSTEM},
                    {"role": "user", "content": _ARGUMENT_FILTER_USER.format(
                        question=query,
                        argument_list=arg_list,
                    )},
                ],
                tools=None,
                tool_choice=None,
            )
        except Exception:
            logger.exception("Argument rerank LLM call failed")
            return (
                candidate_arg_keys,
                candidate_arguments,
                candidate_arg_scores,
                {"error": "LLM call failed"},
            )

        if response.finish_reason == "error" or not response.content:
            return (
                candidate_arg_keys,
                candidate_arguments,
                candidate_arg_scores,
                {"error": "LLM returned error"},
            )

        selected_args = self._parse_arg_response(response.content)

        # Match back using fuzzy matching
        result_keys: list[str] = []
        result_args: list[str] = []
        result_scores: list[float] = []

        for selected in selected_args:
            closest = difflib.get_close_matches(
                str(selected),
                [str(a) for a in candidate_arguments],
                n=1,
                cutoff=0.85,
            )
            if closest:
                try:
                    idx = candidate_arguments.index(closest[0])
                    if candidate_arg_keys[idx] not in result_keys:
                        result_keys.append(candidate_arg_keys[idx])
                        result_args.append(candidate_arguments[idx])
                        result_scores.append(candidate_arg_scores[idx])
                except (ValueError, IndexError):
                    pass

        if max_after_rerank is not None:
            result_keys = result_keys[:max_after_rerank]
            result_args = result_args[:max_after_rerank]
            result_scores = result_scores[:max_after_rerank]

        return result_keys, result_args, result_scores, {
            "num_candidates": len(candidate_arguments),
            "num_selected": len(result_args),
        }

    @staticmethod
    def _parse_arg_response(content: str) -> list[str]:
        """Parse LLM response to extract selected arguments."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
            return data.get("selected_arguments", [])
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("selected_arguments", [])
                except json.JSONDecodeError:
                    pass
        return []
