"""ReMe memory algorithm — memory implementation backed by ReMeLight (reme-ai)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.memory.base import MemoryAlgorithm, MemoryComponents
from nanobot.memory.remem_memory.auto_compact import ReMeAutoCompact
from nanobot.memory.remem_memory.consolidator import ReMeConsolidator
from nanobot.memory.remem_memory.dream import ReMeDream
from nanobot.memory.remem_memory.store import ReMeStore

if TYPE_CHECKING:
    from pathlib import Path

    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


class ReMeMemoryAlgorithm(MemoryAlgorithm):
    """ReMe-backed memory algorithm using ReMeLight from the ``reme`` package.

    Uses ReMeLight for semantic memory search, automatic compaction,
    and long-term memory summarisation.
    """

    name = "remem_memory"

    def build(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Any,
        get_tool_definitions: Any,
        max_completion_tokens: int,
        session_ttl_minutes: int,
        max_batch_size: int,
        max_iterations: int,
        max_tool_result_chars: int,
        annotate_line_ages: bool,
        embedding_config: Any = None,
    ) -> MemoryComponents:
        import asyncio
        import sys

        from loguru import logger
        from reme.reme_light import ReMeLight

        default_llm_config: dict[str, Any] = {
            "model": model,
        }
        if hasattr(provider, "generation") and provider.generation:
            default_llm_config["temperature"] = provider.generation.temperature
            default_llm_config["max_tokens"] = provider.generation.max_tokens

        reme_light = ReMeLight(
            working_dir=str(workspace),
            llm_api_key=getattr(provider, "api_key", None),
            llm_base_url=getattr(provider, "api_base", None),
            default_as_llm_config=default_llm_config,
            enable_load_env=False,
        )
        # ReMeLight.start() → Application.start() → init_logger(log_to_console=False)
        # calls logger.remove() which wipes *all* handlers including nanobot's
        # default stderr sink.  We restore console output afterwards.
        try:
            asyncio.run(reme_light.start())
        finally:
            logger.add(sys.stderr, level="INFO", colorize=True)

        store = ReMeStore(reme_light=reme_light, workspace=workspace, algo_name=self.name)

        consolidator = ReMeConsolidator(
            store=store,
            reme_light=reme_light,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
        )

        dream = ReMeDream(
            store=store,
            reme_light=reme_light,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=self.name,
        )

        auto_compact = ReMeAutoCompact(
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
