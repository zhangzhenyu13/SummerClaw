"""Supermemory consolidator — chunk-based token-budget consolidation with memory graph updates.

Extends the naive Consolidator with Supermemory-specific processing:
- Chunk-based ingestion: decomposes large sessions into semantic blocks
- Contextual memory generation: LLM-powered atomic memory extraction with
  reference resolution, static/dynamic classification, and temporal grounding
- Embedding-based relationship detection: uses cosine similarity on embeddings
  rather than simple word overlap
- Temporal grounding: extracts documentDate and eventDate
- Relational versioning: establishes updates/extends/derives relationships
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger

from summerclaw.memory.embedding_store import batch_cosine_np
from summerclaw.memory.naive_memory.consolidator import Consolidator
from summerclaw.memory.supermemory_memory.store import (
    MemoryEdge,
    MemoryNode,
    MemoryRelation,
    SourceChunk,
    SupermemoryStore,
)
from summerclaw.utils.helpers import estimate_message_tokens

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import SessionManager

# ------------------------------------------------------------------
# LLM extraction prompt — Supermemory-style atomic memory generation
# ------------------------------------------------------------------

_SUPERMEMORY_EXTRACTION_SYSTEM_PROMPT = """You are an expert memory extraction system based on the Supermemory architecture.
Your task is to extract atomic, self-contained memories from conversation chunks.

## Core Principles
1. **Atomic**: Each memory should be a single, self-contained fact. No compound facts.
2. **Self-Contained**: Resolve all pronouns and ambiguous references. "He likes it" → "John likes Python".
3. **Temporal Grounding**: Identify when the described event actually happened (eventDate), distinct from the conversation time.
4. **Static vs Dynamic**: Classify each memory as static (enduring knowledge: preferences, identity, skills) or dynamic (transient state: current task, temporary context).

## Output Format
Return a JSON object with a "memories" array. Each memory has:
- "text": The atomic, self-contained memory text (required)
- "is_static": true for enduring facts, false for transient context (required)
- "event_date": When the described event occurred, in YYYY-MM-DD format (optional)
- "forget_after": ISO date when this dynamic memory should expire, e.g. "2026-06-01" (optional, only for dynamic memories)

Example:
```json
{
  "memories": [
    {"text": "User prefers dark mode for coding", "is_static": true},
    {"text": "User is currently working on the authentication module", "is_static": false, "forget_after": "2026-06-01"},
    {"text": "User started learning Rust in January 2025", "is_static": true, "event_date": "2025-01-01"}
  ]
}
```

Return ONLY valid JSON, no other text."""


class SupermemoryConsolidator(Consolidator):
    """Chunk-based consolidator with memory graph integration.

    When archiving messages:
    1. Summarize via LLM (inherited)
    2. Split messages into semantic chunks
    3. For each chunk, resolve contextual references and generate atomic memories
    4. Store chunks for hybrid search
    5. Detect relationships with existing memories
    """

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for empty vectors, mismatched lengths, or zero-norm vectors.
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __init__(
        self,
        store: SupermemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        embedding_model: str | None = None,
    ) -> None:
        super().__init__(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
        )
        self._super_store: SupermemoryStore = store
        self.embedding_model = embedding_model or model

    # ------------------------------------------------------------------
    # Chunk-based ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_messages(
        messages: list[dict[str, Any]],
        max_chunk_tokens: int = 2000,
    ) -> list[list[dict[str, Any]]]:
        """Split messages into semantic chunks based on user-turn boundaries.

        Supermemory's key technique: decompose large sessions into manageable
        semantic blocks before generating memories.
        """
        if not messages:
            return []

        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        current_tokens = 0

        for msg in messages:
            msg_tokens = estimate_message_tokens(msg)

            # Start new chunk at user turn boundaries when current chunk is large enough
            if (msg.get("role") == "user"
                    and current_chunk
                    and current_tokens >= max_chunk_tokens // 2):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(msg)
            current_tokens += msg_tokens

            # Force chunk split if exceeding max
            if current_tokens >= max_chunk_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def _format_chunk_for_memory_generation(
        chunk: list[dict[str, Any]],
    ) -> str:
        """Format a message chunk for contextual memory generation.

        Uses the same formatting as the store's _format_messages but adds
        temporal context for the chunk.
        """
        from summerclaw.memory.naive_memory.store import MemoryStore as NaiveStore

        # Get chunk timestamp
        ts = chunk[0].get("timestamp", "") if chunk else ""
        if isinstance(ts, str) and len(ts) > 16:
            ts = ts[:16]

        formatted = NaiveStore._format_messages(chunk)
        if ts:
            return f"[Chunk timestamp: {ts}]\n{formatted}"
        return formatted

    async def _generate_memories_from_chunk(
        self,
        chunk: list[dict[str, Any]],
        chunk_id: str,
    ) -> list[MemoryNode]:
        """Generate atomic memories from a single conversation chunk using LLM.

        Uses the Supermemory extraction prompt to:
        1. Resolve ambiguous references (contextual retrieval)
        2. Extract atomic facts as memory nodes
        3. Classify as static vs dynamic
        4. Identify temporal information (documentDate, eventDate)
        5. Suggest forget_after for dynamic memories

        Falls back to heuristic extraction if LLM call fails.
        """
        if not chunk:
            return []

        formatted = self._format_chunk_for_memory_generation(chunk)
        if not formatted.strip():
            return []

        doc_date = datetime.now().strftime("%Y-%m-%d")

        # Try LLM-based extraction first
        try:
            extracted = await self._extract_memories_llm(formatted)
            if extracted:
                memories: list[MemoryNode] = []
                for item in extracted:
                    text = item.get("text", "").strip()
                    if not text:
                        continue
                    node = MemoryNode(
                        id=str(uuid.uuid4()),
                        memory=text,
                        content=formatted,
                        document_date=doc_date,
                        event_date=item.get("event_date"),
                        is_static=item.get("is_static", False),
                        forget_after=item.get("forget_after"),
                    )
                    memories.append(node)
                if memories:
                    logger.debug(
                        "LLM extracted {} atomic memories from chunk {}",
                        len(memories), chunk_id[:8],
                    )
                    return memories
        except Exception:
            logger.warning(
                "LLM extraction failed for chunk {}, falling back to heuristic",
                chunk_id[:8],
            )

        # Fallback: heuristic extraction
        return self._generate_memories_heuristic(chunk, formatted, doc_date)

    def _generate_memories_heuristic(
        self,
        chunk: list[dict[str, Any]],
        formatted: str,
        doc_date: str,
    ) -> list[MemoryNode]:
        """Heuristic fallback for memory extraction when LLM is unavailable."""
        memories: list[MemoryNode] = []
        for msg in chunk:
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue

            role = msg.get("role", "")
            if role == "user":
                text = content.strip()
                if len(text) > 10 and len(text) < 500:
                    node = MemoryNode(
                        id=str(uuid.uuid4()),
                        memory=f"User stated: {text[:200]}",
                        content=formatted,
                        document_date=doc_date,
                    )
                    memories.append(node)

        return memories

    # ------------------------------------------------------------------
    # LLM extraction helpers
    # ------------------------------------------------------------------

    async def _extract_memories_llm(self, formatted_chunk: str) -> list[dict[str, Any]] | None:
        """Call the LLM to extract atomic memories from a formatted chunk.

        Returns a list of memory dicts, or None if extraction failed.
        """
        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": _SUPERMEMORY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": formatted_chunk},
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
            return self._parse_extraction_response(str(content))
        except NotImplementedError:
            logger.debug("LLM provider does not support chat(), using heuristic")
            return None
        except Exception:
            logger.exception("LLM extraction failed")
            return None

    @staticmethod
    def _parse_extraction_response(response: str) -> list[dict[str, Any]] | None:
        """Parse the LLM's JSON response into a list of memory dicts."""
        if not response or not response.strip():
            return None

        # Try to extract JSON from response (handles markdown code blocks)
        text = response.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove opening fence (```json or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text, strict=False)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            import re
            match = re.search(r'\{[^{}]*"memories"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
            if not match:
                logger.warning(
                    "Failed to parse extraction response: no JSON found. "
                    "Raw (first 300 chars): {}",
                    response[:300],
                )
                return None
            try:
                data = json.loads(match.group(0), strict=False)
            except json.JSONDecodeError:
                logger.warning("Failed to parse extracted JSON fragment")
                return None

        memories = data.get("memories", [])
        if not isinstance(memories, list):
            return None

        result: list[dict[str, Any]] = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            text_val = item.get("text", "")
            if not text_val or not str(text_val).strip():
                continue
            result.append({
                "text": str(text_val).strip(),
                "is_static": bool(item.get("is_static", False)),
                "event_date": item.get("event_date"),
                "forget_after": item.get("forget_after"),
            })

        return result if result else None

    async def _detect_relationships(
        self,
        new_nodes: list[MemoryNode],
    ) -> None:
        """Detect relationships between new memories and existing ones using embeddings.

        Uses cosine similarity on embedding vectors when available, falling back
        to Jaccard word overlap for nodes without embeddings.

        Relationship thresholds (cosine similarity):
        - updates: ≥ 0.75 (high semantic overlap → contradiction/update)
        - extends: ≥ 0.50 (moderate overlap → detail addition)
        - derives: ≥ 0.30 (low overlap → inferred connection)

        Falls back to Jaccard with thresholds:
        - updates: ≥ 0.60
        - extends: ≥ 0.30
        """
        existing_nodes = self._super_store.get_latest_nodes()
        if not existing_nodes:
            return

        for new_node in new_nodes:
            best_match: MemoryNode | None = None
            best_score = 0.0
            best_relation = MemoryRelation.EXTENDS

            for existing in existing_nodes:
                # Skip self-comparison
                if existing.id == new_node.id:
                    continue
                # Prefer embedding-based similarity when both have embeddings
                new_emb = new_node.embedding
                existing_emb = self._super_store._embeddings.get(existing.id)
                if new_emb and existing_emb:
                    score = float(batch_cosine_np(
                        np.array(new_emb, dtype=np.float32),
                        np.array([existing_emb], dtype=np.float32),
                    )[0])
                else:
                    # Fallback to Jaccard word overlap
                    score = self._jaccard_similarity(
                        new_node.memory, existing.memory,
                    )

                if score > best_score:
                    best_score = score
                    best_match = existing
                    # Determine relation type based on similarity
                    if score >= 0.75:
                        best_relation = MemoryRelation.UPDATES
                    elif score >= 0.50:
                        best_relation = MemoryRelation.EXTENDS
                    elif score >= 0.30:
                        best_relation = MemoryRelation.DERIVES
                    else:
                        continue  # Below threshold, don't use this match

            if best_match and best_score >= 0.30:
                edge = MemoryEdge(
                    id=str(uuid.uuid4()),
                    source_id=new_node.id,
                    target_id=best_match.id,
                    edge_type=best_relation,
                )
                try:
                    self._super_store.add_edge(edge)
                    logger.debug(
                        "Detected {} relationship (score={:.2f}): '{}' -> '{}'",
                        best_relation.value,
                        best_score,
                        new_node.memory[:50],
                        best_match.memory[:50],
                    )
                except ValueError:
                    pass  # Target node might have been removed

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Compute Jaccard word overlap similarity (fallback)."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        union = words_a | words_b
        return len(overlap) / len(union) if union else 0.0

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> list[float] | None:
        """Generate embedding for a single text string.

        Returns the embedding vector or None if embeddings are not supported.
        """
        try:
            embeddings = self.provider.embed([text], self.embedding_model)
            return embeddings[0] if embeddings else None
        except NotImplementedError:
            return None
        except Exception as e:
            logger.warning("Embedding failed for text (len={}): {}", len(text), e)
            return None

    def _embed_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Generate embeddings for a batch of texts.

        Returns a dict mapping each text to its embedding vector.
        Falls back to single-text embedding on batch failure.
        """
        if not texts:
            return {}
        result: dict[str, list[float]] = {}
        try:
            embeddings = self.provider.embed(texts, self.embedding_model)
            for text, emb in zip(texts, embeddings):
                result[text] = emb
        except Exception as e:
            logger.warning("Batch embedding failed ({}), falling back to single: {}", len(texts), e)
            for text in texts:
                emb = self._embed_text(text)
                if emb:
                    result[text] = emb
        return result

    # ------------------------------------------------------------------
    # Override archive
    # ------------------------------------------------------------------

    async def archive(self, messages: list[dict]) -> str | None:
        """Archive messages: summarize via LLM, then chunk and generate memories.

        Extends the naive archive with Supermemory-specific processing:
        1. Call parent archive to get LLM summary and store history
        2. Split messages into semantic chunks
        3. Generate atomic memories from each chunk
        4. Store chunks for hybrid search
        5. Detect relationships
        """
        # Step 1: Standard consolidation (LLM summary → history.jsonl)
        summary = await super().archive(messages)

        if not messages:
            return summary

        # Step 2: Chunk-based memory generation
        try:
            logger.debug(
                "Supermemory archive: processing {} messages",
                len(messages),
            )
            chunks = self._chunk_messages(messages)
            logger.debug(
                "Supermemory: split {} messages into {} chunks",
                len(messages), len(chunks),
            )

            doc_date = datetime.now().strftime("%Y-%m-%d")
            all_new_nodes: list[MemoryNode] = []

            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())

                # Store the source chunk for hybrid search
                formatted_chunk = self._format_chunk_for_memory_generation(chunk)
                source_chunk = SourceChunk(
                    id=chunk_id,
                    content=formatted_chunk,
                    document_date=doc_date,
                )
                self._super_store.add_chunk(source_chunk)

                # Generate atomic memories from this chunk
                mem_nodes = await self._generate_memories_from_chunk(chunk, chunk_id)
                all_new_nodes.extend(mem_nodes)

                # Link memories to their source chunk
                for node in mem_nodes:
                    source_chunk.memory_ids.append(node.id)
                    self._super_store.add_node(node)

                # Update chunk with memory IDs
                self._super_store.add_chunk(source_chunk)

            # Step 3: Detect relationships
            if all_new_nodes:
                # Embed the new memories for semantic search and relationship detection
                mem_texts = [n.memory for n in all_new_nodes]
                embeddings_map = self._embed_batch(mem_texts)
                for node in all_new_nodes:
                    emb = embeddings_map.get(node.memory)
                    if emb:
                        self._super_store.set_node_embedding(node.id, emb)
                        node.embedding = emb  # transient for _detect_relationships

                await self._detect_relationships(all_new_nodes)
                embedded_count = len([n for n in all_new_nodes if n.embedding])
                logger.info(
                    "Supermemory: generated {} memory nodes ({} with embeddings) "
                    "from {} chunks",
                    len(all_new_nodes), embedded_count, len(chunks),
                )

        except Exception:
            logger.exception("Supermemory chunk processing failed, summary preserved")

        return summary
