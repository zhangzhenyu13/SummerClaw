"""Mem0V3 consolidator — single-pass ADD-only memory extraction pipeline.

Implements the full mem0 v3 extraction pipeline adapted for summerclaw.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
import weakref
from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from summerclaw.memory.mem0v3_memory.prompts import (
    MEM0V3_EXTRACTION_SYSTEM_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    build_extraction_user_prompt,
    extract_entities,
    lemmatize_text,
    normalize_facts,
    remove_code_blocks,
    extract_json as extract_json_from_response,
)
from summerclaw.memory.mem0v3_memory.store import Mem0V3Store

import math as _math

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import Session, SessionManager


class Mem0V3Consolidator:
    """Single-pass ADD-only memory extraction and storage."""

    _DEFAULT_CONTEXT_TOP_K = 10
    _DEFAULT_LAST_K = 10

    def __init__(
        self,
        store: Mem0V3Store,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages,
        get_tool_definitions,
        max_completion_tokens: int = 4096,
        *,
        context_top_k: int = _DEFAULT_CONTEXT_TOP_K,
        last_k_messages: int = _DEFAULT_LAST_K,
        embedding_model: str | None = None,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.embedding_model = embedding_model or model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self.context_top_k = context_top_k
        self.last_k_messages = last_k_messages
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------

    async def extract_and_store(
        self,
        messages: list[dict],
        session: "Session",
        *,
        custom_instructions: str | None = None,
    ) -> list[dict]:
        if not messages:
            return []

        session_scope = self._build_session_scope(session)
        last_messages = self.store.get_last_messages(session_scope, limit=self.last_k_messages)
        parsed = self._parse_messages(messages)

        # Phase 1: Semantic search
        query_embedding = self._embed(parsed)
        existing_results = []
        if query_embedding:
            existing_results = self.store.search_semantic(
                query_embedding, top_k=self.context_top_k, threshold=0.0,
            )

        uuid_mapping: dict[str, str] = {}
        existing_memories: list[dict] = []
        for idx, mem in enumerate(existing_results):
            short_id = str(idx)
            orig_id = mem.get("id", "")
            uuid_mapping[short_id] = orig_id
            payload = mem.get("payload", {})
            existing_memories.append({
                "id": short_id,
                "text": payload.get("text", payload.get("data", "")),
            })

        # Phase 2: LLM extraction
        is_agent_scoped = bool(getattr(session, "agent_id", None)) and not getattr(session, "user_id", None)
        system_prompt = MEM0V3_EXTRACTION_SYSTEM_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        user_prompt = build_extraction_user_prompt(
            existing_memories=existing_memories,
            new_messages=parsed,
            last_k_messages=last_messages if last_messages else None,
            custom_instructions=custom_instructions,
        )

        extraction_response = await self._call_llm(system_prompt, user_prompt)
        if extraction_response is None:
            self.store.save_messages(messages, session_scope)
            return []

        extracted = self._parse_extraction_response(extraction_response)
        if not extracted:
            self.store.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embedding
        mem_texts = [m.get("text", "") for m in extracted if m.get("text")]
        if not mem_texts:
            self.store.save_messages(messages, session_scope)
            return []

        embeddings_map = self._embed_batch(mem_texts)

        # Phase 4: Hash dedup + prepare records
        existing_hashes: set[str] = set()
        for mem in existing_results:
            payload = mem.get("payload", {})
            h = payload.get("hash", "")
            if h:
                existing_hashes.add(h)

        records: list[dict] = []
        seen_hashes: set[str] = set()

        for mem in extracted:
            text = mem.get("text", "")
            if not text or text not in embeddings_map:
                continue
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                continue
            seen_hashes.add(mem_hash)
            lemmatized = lemmatize_text(text)
            records.append({
                "text": text, "hash": mem_hash, "lemmatized": lemmatized,
                "embedding": embeddings_map[text],
                "metadata": {
                    "attributed_to": mem.get("attributed_to", "unknown"),
                    "linked_memory_ids_raw": mem.get("linked_memory_ids", []),
                },
            })

        if not records:
            self.store.save_messages(messages, session_scope)
            return []

        # Phase 5: Batch persist
        inserted_ids = self.store.insert_memories_batch(records)

        # Phase 6: Entity linking
        for idx, mem_id in enumerate(inserted_ids):
            if idx >= len(records):
                break
            rec = records[idx]
            text = rec["text"]
            embedding = rec.get("embedding")
            entities = extract_entities(text)
            seen_entity: set[str] = set()
            for entity_type, entity_text in entities[:8]:
                key = entity_text.strip().lower()
                if not key or key in seen_entity:
                    continue
                seen_entity.add(key)
                try:
                    self.store.upsert_entity(
                        entity_text=entity_text, entity_type=entity_type,
                        memory_id=mem_id, embedding=embedding,
                    )
                except Exception as e:
                    logger.debug(f"Entity link failed: {e}")

        # Phase 7: Save messages
        self.store.save_messages(messages, session_scope)

        returned = []
        for idx, mem_id in enumerate(inserted_ids):
            rec = records[idx] if idx < len(records) else {}
            returned.append({"id": mem_id, "memory": rec.get("text", ""), "event": "ADD"})

        logger.info("Mem0V3: {} msgs → {} new memories", len(messages), len(returned))
        return returned

    # ------------------------------------------------------------------
    # Multi-signal search
    # ------------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 20, threshold: float = 0.1, 
               enable_temporal_decay: bool = True) -> list[dict]:
        """Multi-signal search with optional temporal decay.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            threshold: Minimum score threshold.
            enable_temporal_decay: Whether to apply Memory Decay (default True).

        Returns:
            List of search results with temporal decay applied.
        """
        query_lemmatized = lemmatize_text(query)
        query_embedding = self._embed(query)
        if query_embedding is None:
            return []

        # Over-fetch to allow temporal decay re-ranking
        internal_limit = max(top_k * 4, 60)
        semantic_results = self.store.search_semantic(query_embedding, top_k=internal_limit, threshold=0.0)

        query_tokens = query_lemmatized.split()
        keyword_results = self.store.search_keyword(query_tokens, top_k=internal_limit)
        bm25_scores: dict[str, float] = {}
        if keyword_results:
            midpoint, steepness = _get_bm25_params(query, lemmatized=query_lemmatized)
            for kr in keyword_results:
                mem_id = kr.get("id", "")
                raw_score = kr.get("score", 0.0)
                if raw_score > 0:
                    bm25_scores[mem_id] = _normalize_bm25(raw_score, midpoint, steepness)

        query_entities = extract_entities(query)
        entity_boosts: dict[str, float] = {}
        if query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, query_embedding)

        scored = _score_and_rank(
            semantic_results=semantic_results, bm25_scores=bm25_scores,
            entity_boosts=entity_boosts, threshold=threshold, top_k=internal_limit,
        )

        # Apply temporal decay (Memory Decay)
        if enable_temporal_decay and scored:
            scored = _apply_temporal_decay(
                scored,
                max_boost=1.5,
                min_dampen=0.3,
                decay_rate=0.1,
                access_history_enabled=True,
            )

        # Save updated access_history back to store
        if enable_temporal_decay:
            for result in scored:
                mem_id = result.get("id")
                payload = result.get("payload", {})
                if mem_id and "metadata" in payload:
                    self.store.update_memory_metadata(mem_id, payload.get("metadata", {}))

        formatted = []
        for s in scored:
            payload = s.get("payload", {})
            formatted.append({
                "id": s["id"],
                "memory": payload.get("text", payload.get("data", "")),
                "score": s["score"],
                "hash": payload.get("hash"),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "decay_factor": s.get("decay_factor"),
                "original_score": s.get("original_score"),
            })
        
        # Return top_k after temporal decay
        return formatted[:top_k]

    async def consolidate(self, session: "Session") -> int:
        messages = list(session.messages)
        if not messages:
            return 0
        extracted = await self.extract_and_store(messages, session)
        return len(extracted)

    async def maybe_consolidate_by_tokens(self, session: "Session") -> None:
        """Token-budget-aware consolidation entry point required by AgentLoop.

        mem0v3 uses ADD-only vector extraction instead of message archiving.
        New unconsolidated messages trigger memory extraction via the LLM pipeline.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            last_consolidated = getattr(session, "last_consolidated", 0)
            new_messages = list(session.messages[last_consolidated:])
            if not new_messages:
                return

            try:
                extracted = await self.extract_and_store(new_messages, session)
                # Always advance the pointer to avoid re-processing the same
                # messages when extraction yields no memories (e.g. LLM returned
                # non-JSON text). Messages are already saved by extract_and_store
                # regardless of extraction outcome.
                session.last_consolidated = len(session.messages)
                if self.sessions is not None:
                    self.sessions.save(session)
                if extracted:
                    logger.debug(
                        "mem0v3 extracted {} memories from {} new messages for {}",
                        len(extracted), len(new_messages), session.key,
                    )
            except Exception:
                logger.exception("mem0v3 maybe_consolidate_by_tokens failed for {}", session.key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session_scope(self, session: "Session") -> str:
        parts = []
        for attr in ("user_id", "agent_id"):
            val = getattr(session, attr, None)
            if val:
                parts.append(f"{attr}={val}")
        return "&".join(parts) if parts else "session=default"

    @staticmethod
    def _parse_messages(messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue
            name = msg.get("name", "")
            prefix = f"{name} ({role})" if name else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _embed(self, text: str) -> list[float] | None:
        try:
            embeddings = self.provider.embed([text], self.embedding_model)
            return embeddings[0] if embeddings else None
        except NotImplementedError:
            return None
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _embed_batch(self, texts: list[str]) -> dict[str, list[float]]:
        if not texts:
            return {}
        result: dict[str, list[float]] = {}
        try:
            embeddings = self.provider.embed(texts, self.embedding_model)
            for text, emb in zip(texts, embeddings):
                result[text] = emb
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            for text in texts:
                emb = self._embed(text)
                if emb:
                    result[text] = emb
        return result

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                max_tokens=self.max_completion_tokens,
            )
            if response is None:
                logger.warning("LLM extraction returned None response")
                return None
            content = getattr(response, "content", None)
            if not content or not str(content).strip():
                logger.warning("LLM extraction returned empty content")
                return None
            return str(content)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None

    def _parse_extraction_response(self, response: str) -> list[dict]:
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                return []
            try:
                data = json.loads(response, strict=False)
            except json.JSONDecodeError:
                json_str = extract_json_from_response(response)
                if not json_str or not json_str.strip():
                    logger.warning(
                        "Failed to parse extraction response: empty JSON after extraction. "
                        "Raw response (first 500 chars): {}",
                        str(response)[:500],
                    )
                    return []
                data = json.loads(json_str, strict=False)
            memory_list = data.get("memory", [])
            if not isinstance(memory_list, list):
                return []
            result = []
            for item in memory_list:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "")
                if not text:
                    continue
                result.append({
                    "text": text,
                    "attributed_to": item.get("attributed_to", "unknown"),
                    "linked_memory_ids": item.get("linked_memory_ids", []),
                    "id": item.get("id", ""),
                })
            return result
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse extraction response: {}. "
                "Raw response (first 500 chars): {}",
                e,
                str(response)[:500],
            )
            raise e
            return []
        except Exception as e:
            logger.warning(
                "Failed to parse extraction response: {}. "
                "Raw response (first 500 chars): {}",
                e,
                str(response)[:500],
            )
            return []

    def _compute_entity_boosts(
        self, query_entities: list[tuple[str, str]], query_embedding: list[float],
    ) -> dict[str, float]:
        ENTITY_BOOST_WEIGHT = 0.5
        seen: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for etype, etext in query_entities[:8]:
            key = etext.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((etype, etext))
        if not deduped:
            return {}
        memory_boosts: dict[str, float] = {}
        for _, entity_text in deduped:
            try:
                entity_embedding = self._embed(entity_text)
                if entity_embedding is None:
                    continue
                matches = self.store.search_entities(entity_embedding, top_k=500, threshold=0.5)
                for match in matches:
                    similarity = match.get("score", 0.0)
                    if similarity < 0.5:
                        continue
                    linked_ids = match.get("linked_memory_ids", [])
                    if not linked_ids:
                        continue
                    num_linked = max(len(linked_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
                    for memory_id in linked_ids:
                        if memory_id:
                            memory_boosts[memory_id] = max(memory_boosts.get(memory_id, 0.0), boost)
            except Exception as e:
                logger.debug(f"Entity boost failed for '{entity_text}': {e}")
        return memory_boosts


# ---------------------------------------------------------------------------
# Temporal Decay utilities
# ---------------------------------------------------------------------------

def _apply_temporal_decay(
    scored: list[dict],
    *,
    max_boost: float = 1.5,
    min_dampen: float = 0.3,
    decay_rate: float = 0.1,
    access_history_enabled: bool = True,
) -> list[dict]:
    """Apply explicit time decay to search results (Mem0 Memory Decay).

    Implements exponential decay based on memory recency:
    - Fresh memories (recently accessed/created) get up to max_boost (1.5×)
    - Stale memories (idle for weeks) get dampened toward min_dampen (0.3×)
    - Formula: factor = min_dampen + (max_boost - min_dampen) × e^(-rate × days)

    Args:
        scored: Search results from multi-signal fusion scoring.
        max_boost: Maximum boost factor for fresh memories (default 1.5).
        min_dampen: Minimum dampening factor for stale memories (default 0.3).
        decay_rate: Exponential decay rate per day (default 0.1).
        access_history_enabled: Whether to track access timestamps.

    Returns:
        Re-sorted results with temporal decay applied.
    """
    now = datetime.now(timezone.utc)
    decayed_results = []

    for result in scored:
        payload = result.get("payload", {})
        mem_id = result.get("id", "")

        # Track access time (if enabled)
        if access_history_enabled:
            access_history = payload.get("metadata", {}).get("access_history", [])
            access_history.append(now.isoformat())
            # Keep only last 20 accesses (matches official implementation)
            if len(access_history) > 20:
                access_history = access_history[-20:]
            payload.setdefault("metadata", {})["access_history"] = access_history

        # Determine reference timestamp
        # Priority: most recent access > updated_at > created_at
        access_history = payload.get("metadata", {}).get("access_history", [])
        if access_history:
            # Use most recent access time
            try:
                last_access = datetime.fromisoformat(access_history[-1])
                if last_access.tzinfo is None:
                    last_access = last_access.replace(tzinfo=timezone.utc)
                reference_time = last_access
            except (ValueError, IndexError):
                reference_time = None
        else:
            reference_time = None

        # Fallback to updated_at or created_at
        if reference_time is None:
            updated_at_str = payload.get("updated_at")
            created_at_str = payload.get("created_at")
            
            if updated_at_str:
                try:
                    reference_time = datetime.fromisoformat(updated_at_str)
                    if reference_time.tzinfo is None:
                        reference_time = reference_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    reference_time = None
            
            if reference_time is None and created_at_str:
                try:
                    reference_time = datetime.fromisoformat(created_at_str)
                    if reference_time.tzinfo is None:
                        reference_time = reference_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    reference_time = None

        # Calculate decay factor
        if reference_time is not None:
            days_since_reference = (now - reference_time).total_seconds() / 86400.0
            # Exponential decay: factor = min_dampen + (max_boost - min_dampen) × e^(-rate × days)
            decay_factor = min_dampen + (max_boost - min_dampen) * math.exp(-decay_rate * days_since_reference)
            # Clamp to [min_dampen, max_boost]
            decay_factor = max(min_dampen, min(max_boost, decay_factor))
        else:
            # No timestamp available: neutral factor (1.0)
            decay_factor = 1.0
            logger.debug("Memory {} has no timestamp, using neutral decay factor", mem_id)

        # Apply decay to combined score
        original_score = result.get("score", 0.0)
        decayed_score = original_score * decay_factor
        # Clamp final score to [0, 1]
        decayed_score = min(max(decayed_score, 0.0), 1.0)

        decayed_results.append({
            **result,
            "score": decayed_score,
            "original_score": original_score,
            "decay_factor": decay_factor,
        })

    # Re-sort by decayed score
    decayed_results.sort(key=lambda x: x["score"], reverse=True)
    return decayed_results


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _get_bm25_params(query: str, *, lemmatized: str | None = None) -> tuple[float, float]:
    if lemmatized is None:
        lemmatized = lemmatize_text(query)
    num_terms = len(lemmatized.split()) if lemmatized else 1
    if num_terms <= 3:
        return (5.0, 0.7)
    elif num_terms <= 6:
        return (7.0, 0.6)
    elif num_terms <= 9:
        return (9.0, 0.5)
    elif num_terms <= 15:
        return (10.0, 0.5)
    else:
        return (12.0, 0.5)


def _normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + _math.exp(-steepness * (raw_score - midpoint)))


def _score_and_rank(
    semantic_results: list[dict],
    bm25_scores: dict[str, float],
    entity_boosts: dict[str, float],
    threshold: float,
    top_k: int,
) -> list[dict]:
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)
    ENTITY_BOOST_WEIGHT = 0.5
    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT
    scored: list[dict] = []
    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue
        semantic_score = result.get("score", 0.0)
        if semantic_score < threshold:
            continue
        mem_id_str = str(mem_id)
        bm25 = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)
        raw_combined = semantic_score + bm25 + entity_boost
        combined = min(raw_combined / max_possible, 1.0)
        scored.append({"id": mem_id_str, "score": combined, "payload": result.get("payload", {})})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
