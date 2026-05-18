"""Naive memory algorithm — the default file-based memory implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from summerclaw.memory.base import MemoryAlgorithm, MemoryComponents
from summerclaw.memory.naive_memory.store import MemoryStore
from summerclaw.memory.naive_memory.consolidator import Consolidator
from summerclaw.memory.naive_memory.dream import Dream
from summerclaw.memory.naive_memory.auto_compact import AutoCompact

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import SessionManager


class NaiveMemoryAlgorithm(MemoryAlgorithm):
    """Default naive_memory algorithm.

    Uses file-based storage (MEMORY.md, history.jsonl, SOUL.md, USER.md),
    token-budget-triggered consolidation, and cron-scheduled Dream processing.
    """

    name = "naive_memory"

    def build(
        self,
        workspace: Path,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages,
        get_tool_definitions,
        max_completion_tokens: int,
        session_ttl_minutes: int,
        max_batch_size: int,
        max_iterations: int,
        max_tool_result_chars: int,
        annotate_line_ages: bool,
        embedding_config: Any = None,
    ) -> MemoryComponents:
        store = MemoryStore(workspace, algo_name=self.name)

        consolidator = Consolidator(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
        )

        dream = Dream(
            store=store,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=self.name,
        )

        auto_compact = AutoCompact(
            sessions=sessions,
            consolidator=consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )

        return MemoryComponents(
            store=store,
            consolidator=consolidator,
            dream=dream,
            auto_compact=auto_compact,
        )
