"""ReMe consolidator — adapter wrapping ReMeLight's compaction capabilities."""

from __future__ import annotations

import asyncio
import inspect
import weakref
from typing import TYPE_CHECKING, Any, Callable

from agentscope.message import Msg
from loguru import logger

from nanobot.memory.remem_memory.store import ReMeStore
from nanobot.utils.helpers import estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


class ReMeConsolidator:
    """Adapter that wraps ReMeLight ``compact_memory`` and ``pre_reasoning_hook``."""

    _SAFETY_BUFFER = 1024

    def __init__(
        self,
        store: ReMeStore,
        reme_light: Any,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = store
        self.reme_light = reme_light
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    # -- message conversion ---------------------------------------------------

    @staticmethod
    def _to_msg(msg_dict: dict[str, Any]) -> Msg:
        """Convert a nanobot message dict to an AgentScope Msg.

        Nanobot messages use ``role`` and ``content`` keys; this helper
        normalises the role and constructs an AgentScope-compatible Msg.
        """
        role = msg_dict.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        return Msg(
            name=msg_dict.get("name", role),
            role=role,
            content=msg_dict.get("content", ""),
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (
            session.key.split(":", 1) if ":" in session.key else (None, None)
        )
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive(self, messages: list[dict]) -> str | None:
        """Summarize messages via ReMeLight ``compact_memory`` and append to history.

        Returns the summary text on success, ``None`` if nothing to archive.
        """
        if not messages:
            return None
        try:
            # Append raw messages to the companion history so ReMeLight
            # can see them if it reads dialog files directly.
            self.store.raw_archive(messages)

            # Trigger ReMeLight's compaction with converted messages.
            msgs = [self._to_msg(m) for m in messages]
            result = self.reme_light.compact_memory(messages=msgs)
            if inspect.isawaitable(result):
                summary = await result
            else:
                summary = result

            if summary and isinstance(summary, str):
                self.store.append_history(summary)
                return summary
            return None
        except Exception:
            logger.warning("ReMe compact_memory failed, raw-dumping to history")
            self.store.raw_archive(messages)
            return None

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Let ReMeLight ``pre_reasoning_hook`` handle token-budget compression."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = (
                self.context_window_tokens
                - self.max_completion_tokens
                - self._SAFETY_BUFFER
            )
            try:
                estimated, source = self.estimate_session_prompt_tokens(session)
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "ReMe consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            logger.info(
                "ReMe consolidation for {}: {}/{} via {}, msgs={}",
                session.key,
                estimated,
                self.context_window_tokens,
                source,
                len(session.messages),
            )

            try:
                msgs = [self._to_msg(m) for m in session.messages]
                result = self.reme_light.pre_reasoning_hook(messages=msgs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "ReMe pre_reasoning_hook failed for {}", session.key
                )

            # Mark everything as consolidated so we don't repeatedly trigger.
            session.last_consolidated = len(session.messages)
            self.sessions.save(session)
