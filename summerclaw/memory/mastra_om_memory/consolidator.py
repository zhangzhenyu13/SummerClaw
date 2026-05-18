"""MastraOM Consolidator — Observer/Reflector pipeline triggered by token budget.

Based on Mastra's ObservationalMemory processor pipeline.
The Consolidator is the "online" component that runs during agent execution:

1. When unobserved message tokens exceed ``message_tokens_threshold``, the
   Observer converts raw messages into observations.
2. When observation tokens exceed ``observation_tokens_threshold``, the
   Reflector condenses the observation log.
3. ``extract_and_store()`` provides the Hermes-Autogen integration point,
   extracting facts from recent conversation for skill distillation.

This is the summerclaw-adapted version that works with SessionManager, file-based
storage, and the standard MemoryAlgorithm build interface.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from summerclaw.memory.mastra_om_memory.groups import (
    build_message_range,
    wrap_in_observation_group,
    strip_observation_groups,
    OBSERVATION_RETRIEVAL_INSTRUCTIONS,
)
from summerclaw.memory.mastra_om_memory.buffering import (
    BufferingCoordinator,
    BufferingStore,
    async_buffer_observe,
    activate_buffered_observations,
)
from summerclaw.memory.mastra_om_memory.observer import (
    build_observer_prompt,
    build_observer_system_prompt,
    parse_observer_output,
    optimize_observations_for_context,
)
from summerclaw.memory.mastra_om_memory.reflector import (
    build_reflector_prompt,
    build_reflector_system_prompt,
    parse_reflector_output,
)
from summerclaw.memory.mastra_om_memory.store import MastraOMStore
from summerclaw.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import Session, SessionManager


class MastraOMConsolidator:
    """Observer/Reflector pipeline: token-budget-triggered observation + reflection.

    Default thresholds (from Mastra):
    - message_tokens_threshold: 30_000  (trigger Observer)
    - observation_tokens_threshold: 40_000  (trigger Reflector)
    """

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_CHUNK_MESSAGES = 60
    _SAFETY_BUFFER = 1024
    _MAX_REFLECTION_RETRIES = 4  # up to compression level 4

    def __init__(
        self,
        store: MastraOMStore,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        message_tokens_threshold: int = 30_000,
        observation_tokens_threshold: int = 40_000,
        buffer_tokens: float | None = 0.2,
        buffer_activation: float | None = 0.5,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.message_tokens_threshold = message_tokens_threshold
        self.observation_tokens_threshold = observation_tokens_threshold
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        # Buffering
        self._buffer_coordinator = BufferingCoordinator(
            buffer_tokens=buffer_tokens,
            buffer_activation=buffer_activation,
        )
        self._buffering_store = BufferingStore()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_session_prompt_tokens(self, session: "Session") -> tuple[int, str]:
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

    # ------------------------------------------------------------------
    # Boundary picking (same algorithm as naive Consolidator)
    # ------------------------------------------------------------------

    def pick_consolidation_boundary(
        self,
        session: "Session",
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def _cap_consolidation_boundary(
        self,
        session: "Session",
        end_idx: int,
    ) -> int | None:
        """Clamp the chunk size without breaking the user-turn boundary."""
        start = session.last_consolidated
        if end_idx - start <= self._MAX_CHUNK_MESSAGES:
            return end_idx

        capped_end = start + self._MAX_CHUNK_MESSAGES
        for idx in range(capped_end, start, -1):
            if session.messages[idx].get("role") == "user":
                return idx
        return None

    # ------------------------------------------------------------------
    # Observer call (convert messages → observations)
    # ------------------------------------------------------------------

    async def _observe_messages(
        self,
        messages: list[dict],
        existing_observations: str = "",
    ) -> dict[str, Any] | None:
        """Call the Observer LLM to convert messages into observations.

        Returns parsed observer result, or None on failure.
        """
        if not messages:
            return None

        try:
            formatted_prompt = build_observer_prompt(
                existing_observations=existing_observations or None,
                messages_to_observe=messages,
            )
            logger.info(
                "[OM:observer] calling Observer LLM for {} messages (existing_obs={} chars)",
                len(messages), len(existing_observations),
            )
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": build_observer_system_prompt(),
                    },
                    {"role": "user", "content": formatted_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"Observer LLM returned error: {response.content}")

            result = parse_observer_output(response.content or "")
            obs_len = len(result.get("observations", "")) if result else 0
            logger.info(
                "[OM:observer] Observer LLM response: {} chars observations, task={}",
                obs_len, result.get("current_task", "n/a") if result else "null",
            )
            return result
        except Exception:
            logger.exception("Observer LLM call failed")
            return None

    # ------------------------------------------------------------------
    # Reflector call (condense observations)
    # ------------------------------------------------------------------

    async def _reflect_observations(
        self,
        observations: str,
        compression_level: int = 0,
    ) -> dict[str, Any] | None:
        """Call the Reflector LLM to condense observations.

        Returns parsed reflector result, or None on failure.
        """
        if not observations:
            return None

        try:
            prompt = build_reflector_prompt(
                observations=observations,
                compression_level=compression_level,
            )
            logger.info(
                "[OM:reflector] calling Reflector LLM ({} chars obs, compression={})",
                len(observations), compression_level,
            )
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": build_reflector_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"Reflector LLM returned error: {response.content}")

            result = parse_reflector_output(response.content or "", observations)
            refl_len = len(result.get("observations", "")) if result else 0
            logger.info(
                "[OM:reflector] Reflector LLM response: {}→{} chars (degenerate={})",
                len(observations), refl_len,
                result.get("degenerate", False) if result else True,
            )
            return result
        except Exception:
            logger.exception("Reflector LLM call failed")
            return None

    # ------------------------------------------------------------------
    # Main consolidation: archive messages via Observer
    # ------------------------------------------------------------------

    async def observe_and_store(
        self,
        messages: list[dict],
    ) -> str | None:
        """Observe messages, store raw to history and observations to OBSERVATIONS.md.

        Raw messages → history.jsonl (for Dream analysis).
        Observer output → OBSERVATIONS.md.
        OM summary → om-ops.jsonl (for pipeline debugging).

        Returns the new observations text, or None if nothing was observed.
        """
        if not messages:
            return None

        try:
            # Always write raw messages to history.jsonl for Dream analysis
            self.store.append_history(
                self.store._format_messages(messages)
            )

            existing = self.store.read_observations()
            result = await self._observe_messages(messages, existing)

            if not result or result.get("degenerate"):
                logger.warning("Observer returned empty/degenerate result, raw-dumping")
                self.store.raw_archive(messages)
                return None

            observations_text = result.get("observations", "")
            if not observations_text.strip():
                self.store.raw_archive(messages)
                return None

            # Wrap with observation group if we have message IDs
            message_range = build_message_range(messages)
            if message_range:
                observations_text = wrap_in_observation_group(
                    observations=observations_text,
                    range_spec=message_range,
                )

            # Append to observation log
            self.store.append_observations(observations_text)

            # Write OM operation summary to om-ops.jsonl (not history anymore)
            summary = (
                f"[OM-OBSERVED] {len(messages)} messages → "
                f"{len(observations_text)} chars of observations"
            )
            self.store.append_om_ops(summary)

            logger.info(
                "Observer: {} messages → {} chars of observations",
                len(messages), len(observations_text),
            )
            return observations_text
        except Exception:
            logger.warning("Observation failed, raw-dumping to history")
            self.store.raw_archive(messages)
            return None

    async def reflect_and_condense(self) -> bool:
        """Condense observations via Reflector if they exceed threshold.

        Returns True if reflection was performed.
        """
        observations = self.store.read_observations()
        if not observations:
            return False

        # Estimate tokens
        obs_tokens = len(observations) // 4  # rough estimate: ~4 chars/token

        if obs_tokens < self.observation_tokens_threshold:
            return False

        logger.info(
            "Reflector triggered: {} estimated observation tokens (threshold={})",
            obs_tokens, self.observation_tokens_threshold,
        )

        # Try with progressive compression
        for level in range(self._MAX_REFLECTION_RETRIES + 1):
            result = await self._reflect_observations(observations, compression_level=level)

            if not result or result.get("degenerate"):
                logger.warning("Reflector returned degenerate result at level {}", level)
                continue

            reflected = result.get("observations", "")
            reflected_tokens = len(reflected) // 4

            if reflected_tokens < self.observation_tokens_threshold or level == self._MAX_REFLECTION_RETRIES:
                # Accept the result
                if reflected.strip():
                    self.store.replace_observations(reflected)
                    self.store.increment_generation()
                    logger.info(
                        "Reflector: condensed observations (level={}, {}→{} tokens)",
                        level, obs_tokens, reflected_tokens,
                    )
                return True

            # Result still too large, try higher compression
            logger.debug(
                "Reflector level {} still too large ({} tokens), retrying...",
                level, reflected_tokens,
            )

        return False

    # ------------------------------------------------------------------
    # Token-budget-triggered consolidation (online, during agent loop)
    # ------------------------------------------------------------------

    async def maybe_consolidate_by_tokens(self, session: "Session") -> None:
        """Loop: observe old messages until prompt fits within safe budget.

        When message tokens exceed ``message_tokens_threshold``, run Observer
        on the excess. Also check if observations need reflection.

        Async buffering: before the sync threshold is reached, background
        Observer calls pre-compute observations at regular token intervals
        (default: every 20% of threshold). Buffered chunks are activated
        when the sync threshold triggers.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        lock_key = self._buffer_coordinator.get_lock_key(thread_id=session.key)

        async with lock:
            budget = (
                self.context_window_tokens
                - self.max_completion_tokens
                - self._SAFETY_BUFFER
            )
            target = budget // 2

            try:
                estimated, source = self.estimate_session_prompt_tokens(session)
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                return

            # ── Async buffering check (before sync threshold) ──────────
            if self._buffer_coordinator.is_async_observation_enabled():
                # Calculate unobserved message tokens
                unobserved = session.messages[session.last_consolidated:]
                unobserved_tokens = sum(
                    len(str(m.get("content", ""))) // 4 for m in unobserved
                )

                if self._buffer_coordinator.should_trigger_async_observation(
                    current_tokens=unobserved_tokens,
                    lock_key=lock_key,
                    message_tokens_threshold=self.message_tokens_threshold,
                    db_boundary=0,
                ):
                    logger.info(
                        "[OM:buffer] triggering async observation for {} ({} tokens)",
                        lock_key, unobserved_tokens,
                    )
                    # Launch background task (fire-and-forget)
                    asyncio.create_task(
                        async_buffer_observe(
                            consolidator=self,
                            messages=unobserved,
                            buffer_coordinator=self._buffer_coordinator,
                            lock_key=lock_key,
                            buffering_store=self._buffering_store,
                        )
                    )

            if estimated < budget:
                logger.debug(
                    "MastraOM consolidation idle {}: {}/{} via {}",
                    session.key, estimated, self.context_window_tokens, source,
                )
                return

            # ── Sync consolidation (threshold crossed) ──────────────────
            logger.info(
                "MastraOM consolidation triggered {}: {}/{} tokens via {}",
                session.key, estimated, self.context_window_tokens, source,
            )
            # First, activate any buffered observations
            buffered_obs = await activate_buffered_observations(
                buffer_coordinator=self._buffer_coordinator,
                lock_key=lock_key,
                buffering_store=self._buffering_store,
                consolidator=self,
            )
            # Inject buffered observations into session so downstream
            # consumers (SkillAutogen, etc.) can read them transparently.
            if buffered_obs:
                obs_records = self.store._observations_as_records(buffered_obs)
                if obs_records:
                    session.add_message("system", f"[Memory]\n{obs_records}")

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(
                    session, max(1, estimated - target)
                )
                if boundary is None:
                    logger.info(
                        "MastraOM consolidation: no safe boundary for {} (round {})",
                        session.key, round_num,
                    )
                    return

                end_idx = boundary[0]
                end_idx = self._cap_consolidation_boundary(session, end_idx)
                if end_idx is None:
                    logger.info(
                        "MastraOM consolidation: no capped boundary for {} (round {})",
                        session.key, round_num,
                    )
                    return

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "MastraOM observation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num, session.key, estimated,
                    self.context_window_tokens, source, len(chunk),
                )

                # Observe the message chunk
                observations_text = await self.observe_and_store(chunk)
                if not observations_text:
                    return

                # Inject observations as session messages so downstream
                # consumers (SkillAutogen, etc.) can read them transparently
                # without knowing about Observer internals.
                obs_records = self.store._observations_as_records(observations_text)
                if obs_records:
                    session.add_message("system", f"[Memory]\n{obs_records}")

                session.last_consolidated = end_idx
                self.sessions.save(session)

                try:
                    estimated, source = self.estimate_session_prompt_tokens(session)
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                if estimated <= 0:
                    return

        # After consolidation, check if observations need reflection
        await self.reflect_and_condense()

    # ------------------------------------------------------------------
    # extract_and_store (Hermes-Autogen integration)
    # ------------------------------------------------------------------

    async def extract_and_store(
        self,
        messages: list[dict],
    ) -> list[str]:
        """Extract facts from recent conversation for Hermes-Autogen.

        Called by SkillAutogen to capture facts before skill generation.
        Uses the Observer to extract observations, stores raw messages to
        history.jsonl and observations to OBSERVATIONS.md, then returns
        facts as a list of strings.

        Returns:
            List of extracted fact strings.
        """
        if not messages:
            return []

        logger.info("[OM:extract] extracting facts from {} messages", len(messages))

        try:
            # Write raw messages to history.jsonl for Dream analysis
            self.store.append_history(
                self.store._format_messages(messages)
            )

            result = await self._observe_messages(messages, existing_observations="")

            if not result or result.get("degenerate"):
                logger.debug("[OM:extract] observer returned degenerate/empty result")
                self.store.append_om_ops(
                    f"[OM-EXTRACT-DEGENERATE] {len(messages)} messages, observer returned empty"
                )
                return []

            observations_text = result.get("observations", "")
            if not observations_text.strip():
                logger.debug("[OM:extract] no observations extracted")
                self.store.append_om_ops(
                    f"[OM-EXTRACT-EMPTY] {len(messages)} messages, no observations extracted"
                )
                return []

            # Store the observations
            self.store.append_observations(observations_text)

            # Write OM operation summary to om-ops.jsonl
            self.store.append_om_ops(
                f"[OM-EXTRACT] {len(messages)} messages → "
                f"{len(observations_text)} chars of observations"
            )

            # Extract individual facts (lines starting with *)
            facts = [
                line.strip()
                for line in observations_text.split("\n")
                if line.strip().startswith("*")
            ]

            logger.info(
                "[OM:extract] extracted {} facts from {} messages ({} chars observations)",
                len(facts), len(messages), len(observations_text),
            )
            return facts
        except Exception:
            logger.exception("[OM:extract] extract_and_store failed for {} messages", len(messages))
            return []

    # ------------------------------------------------------------------
    # Legacy archive (for AutoCompact compatibility)
    # ------------------------------------------------------------------

    async def archive(self, messages: list[dict]) -> str | None:
        """Archive messages by observing them. For AutoCompact compatibility."""
        return await self.observe_and_store(messages)

    # ------------------------------------------------------------------
    # Context injection (build context system messages)
    # ------------------------------------------------------------------

    def build_context_system_messages(
        self,
        thread_id: str = "",
        resource_id: str = "",
    ) -> list[str]:
        """Build system messages that inject observations into the agent context.

        The observations block is injected as a system message so the agent
        sees its memory of past interactions.

        Returns:
            List of system message strings to inject into context.
        """
        from summerclaw.memory.mastra_om_memory.observer import (
            OBSERVATION_CONTEXT_PROMPT,
            OBSERVATION_CONTEXT_INSTRUCTIONS,
            OBSERVATION_CONTINUATION_HINT,
        )
        from summerclaw.memory.mastra_om_memory.groups import parse_observation_groups

        observations = self.store.read_observations()
        if not observations.strip():
            logger.debug("[OM:context] no observations found, returning continuation hint only")
            return [
                f"<system-reminder>{OBSERVATION_CONTINUATION_HINT}</system-reminder>"
            ]

        # Check if observations contain groups — if so, strip for compact context
        # but include retrieval instructions
        groups = parse_observation_groups(observations)
        context_obs = observations
        retrieval_block = ""
        if groups:
            retrieval_block = f"\n\n{OBSERVATION_RETRIEVAL_INSTRUCTIONS}"
            logger.info(
                "[OM:context] {} observation groups found, retrieval instructions added",
                len(groups),
            )

        # Build the full observation context
        msg = (
            f"{OBSERVATION_CONTEXT_PROMPT}\n\n"
            f"<observations>\n{context_obs}\n</observations>\n\n"
            f"{OBSERVATION_CONTEXT_INSTRUCTIONS}"
            f"{retrieval_block}"
        )

        logger.info(
            "[OM:context] built context message: {} chars observations → {} chars system msg{}",
            len(observations), len(msg),
            f" ({len(groups)} groups)" if groups else "",
        )

        return [msg]
