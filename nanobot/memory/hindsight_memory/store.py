"""Hindsight store — local TEMPR multi-strategy memory engine.

Implements Hindsight's TEMPR (Temporal + Embedding + Metadata + Probabilistic
+ Relational) multi-strategy retrieval entirely locally, with zero external
dependencies beyond the standard library and the existing LLM provider's
``embed()`` API for optional semantic search.

Fusion uses Reciprocal Rank Fusion (RRF) instead of linear weights, matching
the official Hindsight algorithm.  Graph expansion simulates link-expansion
retrieval via shared-entity co-occurrence, semantic kNN precomputation, and
causal-chain boosting.  Temporal retrieval uses time-window proximity scoring
with spreading propagation.

All memories are persisted to ``memory/hindsight_memories.json`` in the
workspace, alongside the standard naive file store (MEMORY.md, history.jsonl,
SOUL.md, USER.md).
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.memory.naive_memory.store import MemoryStore as _NaiveStore
from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


# ============================================================================
# Cosine similarity (pure Python — no numpy dependency)
# ============================================================================

def _dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = _dot(a, b)
    na = _norm(a)
    nb = _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _batch_cosine(query: list[float], candidates: list[list[float]]) -> list[float]:
    return [_cosine_similarity(query, c) for c in candidates]


# ============================================================================
# Simple BM25 inverted index (pure Python)
# ============================================================================

class _BM25Index:
    """Minimal TF-IDF inverted index for keyword search."""

    def __init__(self) -> None:
        self._inverted: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_count = 0
        self._avg_dl = 0.0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        self.remove(doc_id)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            self._inverted.setdefault(term, {})[doc_id] = freq
        self._doc_lengths[doc_id] = len(tokens)
        self._doc_count += 1
        if self._doc_count > 0:
            self._avg_dl = sum(self._doc_lengths.values()) / self._doc_count

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_lengths:
            return
        for term_dict in self._inverted.values():
            term_dict.pop(doc_id, None)
        self._doc_lengths.pop(doc_id, None)
        self._doc_count -= 1
        if self._doc_count > 0:
            self._avg_dl = sum(self._doc_lengths.values()) / self._doc_count

    def search(self, query_tokens: list[str], top_k: int = 60) -> list[tuple[str, float]]:
        if not query_tokens or self._doc_count == 0:
            return []
        k1, b = 1.2, 0.75
        scores: dict[str, float] = {}
        for term in set(query_tokens):
            postings = self._inverted.get(term, {})
            if not postings:
                continue
            idf = math.log(
                1 + (self._doc_count - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for doc_id, tf in postings.items():
                dl = self._doc_lengths.get(doc_id, 1)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / max(self._avg_dl, 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================================
# Tokenizer (pure Python)
# ============================================================================

def _tokenize(text: str) -> list[str]:
    """Naive lowercase word tokenizer.  Keeps CJK characters."""
    return re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower())


# ============================================================================
# Reciprocal Rank Fusion (RRF) — matches official Hindsight algorithm
# ============================================================================

def _rrf_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> dict[str, float]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum_over_lists(1 / (k + rank(d)))

    This is the same algorithm used by the official Hindsight server to merge
    semantic, BM25, graph, and temporal retrieval results.  It requires no
    manual weight tuning and is robust across different score distributions.

    Args:
        result_lists: Each inner list is [(doc_id, raw_score), ...] sorted desc.
        k: Constant for RRF formula (default 60, matches official).

    Returns:
        Dict mapping doc_id → RRF score.
    """
    rrf_scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (doc_id, _score) in enumerate(results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return rrf_scores


# ============================================================================
# Entity extraction (simple token overlap for co-occurrence)
# ============================================================================

def _extract_entities(text: str) -> set[str]:
    """Extract potential entity tokens (longer words, capitalized)."""
    tokens = re.findall(r"[A-Z][a-z]+|[a-z]{4,}|[\u4e00-\u9fff]{2,}", text)
    return {t.lower() for t in tokens}


# ============================================================================
# HindsightStore
# ============================================================================

class HindsightStore(_NaiveStore):
    """File-based store with built-in local TEMPR multi-strategy memory engine.

    Always provides the full naive MemoryStore interface (MEMORY.md,
    history.jsonl, SOUL.md, USER.md).  Additionally maintains a local memory
    bank with TEMPR retrieval capabilities using RRF fusion:

    - **T**emporal:  time-window proximity + spreading propagation
    - **E**mbedding: cosine similarity via provider.embed() (optional)
    - **M**etadata:  fact_type tagging (world / experience / observation)
    - **P**robabilistic: BM25 keyword scoring
    - **R**elational: graph expansion (entity co-occurrence + semantic kNN +
      causal-chain boosting)

    Memories are persisted to ``memory/hindsight_memories.json``.

    Fact types follow the official Hindsight hierarchy::

        world       — objective facts (default)
        experience  — agent's own actions and interactions
        observation — auto-consolidated knowledge from multiple facts
        opinion     — subjective statements
    """

    # Allowed fact types
    FACT_TYPES = frozenset({"world", "experience", "observation", "opinion"})

    # Temporal decay half-life in days
    _TEMPORAL_HALF_LIFE_DAYS = 30.0

    # RRF constant (matches official Hindsight)
    _RRF_K = 60

    # Graph expansion budget (max nodes to expand from seeds)
    _GRAPH_EXPANSION_BUDGET = 30

    # Semantic kNN: number of neighbours to keep per memory
    _SEMANTIC_KNN_K = 5
    _SEMANTIC_KNN_THRESHOLD = 0.70

    # Boosts for relational scoring
    _ENTITY_BOOST = 0.5          # shared-entity co-occurrence
    _SEMANTIC_KNN_BOOST = 0.7    # precomputed semantic neighbour
    _CAUSAL_BOOST = 2.0          # explicit causal link
    _CONTEXT_BOOST = 0.3         # same-context proximity

    _DEFAULT_MAX_MEMORIES = 10_000

    # Recency/temporal boost alphas (multiplicative, matches official)
    _RECENCY_ALPHA = 0.20
    _TEMPORAL_ALPHA = 0.20
    _PROOF_COUNT_ALPHA = 0.10

    def __init__(
        self,
        workspace: Path,
        *,
        max_history_entries: int = 1000,
        provider: "LLMProvider | None" = None,
        embedding_model: str | None = None,
        max_memories: int = _DEFAULT_MAX_MEMORIES,
        algo_name: str | None = None,
    ):
        # Ensure hindsight-memory logs are always visible even when the CLI's
        # ``logger.disable("nanobot")`` silences the parent namespace.
        logger.enable("nanobot.memory.hindsight_memory")

        super().__init__(workspace, max_history_entries=max_history_entries, algo_name=algo_name)
        self._provider = provider
        self._embedding_model = embedding_model
        self._max_memories = max_memories

        # Memory bank
        self._memories_path = self.memory_dir / "hindsight_memories.json"

        # Migrate legacy hindsight-specific files if needed
        if algo_name:
            self._migrate_hindsight_legacy()

        self._memories: dict[str, dict] = {}  # mem_id → record
        self._bm25 = _BM25Index()
        # Entity → set of memory IDs (for graph expansion)
        self._entity_index: dict[str, set[str]] = defaultdict(set)
        # Causal links: from_mem_id → [(to_mem_id, link_type, weight)]
        self._links: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

        self._load_memories()

    def _migrate_hindsight_legacy(self) -> None:
        """Migrate hindsight-specific files from the legacy location."""
        from nanobot.memory.migrate import maybe_migrate_legacy_files
        old_memory_dir = self.workspace / "memory"
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=self.workspace,
            files=["hindsight_memories.json"],
        )

    # -- properties -----------------------------------------------------------

    @property
    def hindsight_enabled(self) -> bool:
        """Local TEMPR engine is always available (no server needed)."""
        return True

    @property
    def memory_count(self) -> int:
        return len(self._memories)

    # -- persistence ----------------------------------------------------------

    def _load_memories(self) -> None:
        if not self._memories_path.exists():
            return
        try:
            with open(self._memories_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._memories = data.get("memories", {})
            # Rebuild BM25 index and entity index
            for mem_id, rec in self._memories.items():
                tokens = _tokenize(rec.get("content", ""))
                if tokens:
                    for t in tokens:
                        self._bm25._inverted.setdefault(t, {})[mem_id] = (
                            self._bm25._inverted.setdefault(t, {}).get(mem_id, 0) + 1
                        )
                    self._bm25._doc_lengths[mem_id] = len(tokens)
                # Rebuild entity index
                entities = rec.get("entities", [])
                for ent in entities:
                    self._entity_index[ent].add(mem_id)
                # Rebuild semantic kNN links
                knn = rec.get("_knn", [])
                for neighbour_id in knn:
                    self._links[mem_id].append(
                        (neighbour_id, "semantic", self._SEMANTIC_KNN_BOOST)
                    )
                    self._links[neighbour_id].append(
                        (mem_id, "semantic", self._SEMANTIC_KNN_BOOST)
                    )
                # Rebuild causal links
                for link in rec.get("_causal_links", []):
                    if isinstance(link, dict):
                        self._links[mem_id].append(
                            (link["to"], link.get("type", "causes"), link.get("weight", 0.5))
                        )
                    elif isinstance(link, list) and len(link) >= 2:
                        self._links[mem_id].append(tuple(link) if len(link) == 3 else (link[0], link[1], 0.5))
            self._bm25._doc_count = len(self._memories)
            if self._bm25._doc_count > 0:
                self._bm25._avg_dl = (
                    sum(self._bm25._doc_lengths.values()) / self._bm25._doc_count
                )
            logger.info("HindsightStore: loaded {} memories", self.memory_count)
        except Exception:
            logger.exception("Failed to load Hindsight memories; starting fresh")
            self._memories = {}
            self._bm25 = _BM25Index()
            self._entity_index = defaultdict(set)
            self._links = defaultdict(list)

    def _save_memories(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "memories": self._memories,
            "version": 2,
            "entity_count": len(self._entity_index),
        }
        tmp = self._memories_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._memories_path)

    def _trim_memories(self) -> None:
        """Remove oldest memories if exceeding max."""
        if len(self._memories) <= self._max_memories:
            return
        sorted_ids = sorted(
            self._memories.keys(),
            key=lambda mid: self._memories[mid].get("timestamp", ""),
        )
        to_remove = sorted_ids[: len(self._memories) - self._max_memories]
        for mid in to_remove:
            self._bm25.remove(mid)
            rec = self._memories.get(mid, {})
            for ent in rec.get("entities", []):
                self._entity_index.get(ent, set()).discard(mid)
            self._links.pop(mid, None)
            del self._memories[mid]
        logger.info("HindsightStore: trimmed {} old memories", len(to_remove))

    # -- embedding helpers ----------------------------------------------------

    def _can_embed(self) -> bool:
        return self._provider is not None and self._embedding_model is not None

    def _try_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via provider. Returns empty list on failure."""
        if not self._can_embed() or not texts:
            return []
        try:
            embeddings = self._provider.embed(texts, self._embedding_model)
            return [list(e) for e in embeddings]
        except Exception:
            logger.debug("HindsightStore: embedding generation failed (non-fatal)")
            return []

    # -- server operations (local TEMPR implementation) -----------------------

    async def aretain(
        self,
        content: str,
        context: str | None = None,
        *,
        fact_type: str = "world",
    ) -> Any:
        """Store a memory into the local TEMPR bank.

        Returns a response object with ``.id`` and ``.text`` on success;
        None on failure.
        """
        if not content or not content.strip():
            return None

        if fact_type not in self.FACT_TYPES:
            logger.warning("Unknown fact_type '{}', falling back to 'world'", fact_type)
            fact_type = "world"

        mem_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Generate embedding (best-effort)
        embedding: list[float] | None = None
        emb_results = self._try_embed([content])
        if emb_results:
            embedding = emb_results[0]

        # Extract entities for graph expansion
        entities = sorted(_extract_entities(content))

        # Find causal links (heuristic: detect cause-effect patterns)
        causal_links = self._detect_causal_links(content, context)

        record: dict[str, Any] = {
            "id": mem_id,
            "content": content.strip(),
            "context": context,
            "timestamp": now,
            "embedding": embedding,
            "fact_type": fact_type,
            "source_type": "manual",
            "entities": entities,
            "proof_count": 1 if fact_type == "observation" else None,
            "_knn": [],
            "_causal_links": causal_links,
        }

        self._memories[mem_id] = record

        # Update BM25
        tokens = _tokenize(content)
        if tokens:
            self._bm25.add(mem_id, tokens)

        # Update entity index
        for ent in entities:
            self._entity_index[ent].add(mem_id)

        # Precompute semantic kNN links (best-effort)
        if embedding:
            self._compute_knn_links(mem_id, embedding)

        self._trim_memories()
        self._save_memories()
        logger.info(
            "HindsightStore aretain: {} chars (id={}, type={})",
            len(content), mem_id, fact_type,
        )

        return _RetainResponse(mem_id, content.strip())

    async def arecall(
        self,
        query: str,
        max_tokens: int = 4096,
        budget: str = "mid",
    ) -> Any:
        """TEMPR multi-strategy search over local memories with RRF fusion.

        Returns an object with ``.text`` containing the top results formatted
        as a readable summary.
        """
        if not self._memories:
            logger.info("HindsightStore arecall: no memories available")
            return _RecallResponse("")

        results = self._tempr_search(query, budget=budget, top_k=20)
        text = self._format_search_results(results, max_tokens)
        logger.info(
            "HindsightStore arecall: {} results, {} chars (budget={})",
            len(results), len(text), budget,
        )
        return _RecallResponse(text)

    async def areflect(
        self,
        query: str,
        budget: str = "low",
        context: str | None = None,
    ) -> Any:
        """Deep reasoning over local memories.

        Performs TEMPR search, then (if a provider is available) asks the LLM
        to synthesize a structured reflective analysis::
        
        1. Key facts found
        2. Patterns & connections
        3. Contradictions or gaps
        4. Actionable insights

        Returns an object with ``.text`` containing the analysis.
        """
        if not self._memories:
            logger.info("HindsightStore areflect: no memories available")
            return _ReflectResponse("")

        logger.info(
            "HindsightStore areflect: entering with budget={}, mems={}",
            budget, len(self._memories),
        )

        # Step 1: TEMPR search
        results = self._tempr_search(query, budget=budget, top_k=30)
        search_text = self._format_search_results(results, max_tokens=6000)

        # Step 2: Structured LLM synthesis
        if self._provider and search_text:
            try:
                response = await self._synthesize_reflect(query, context, search_text)
                logger.info(
                    "HindsightStore areflect (LLM synthesis): {} chars",
                    len(response.text),
                )
                return response
            except Exception:
                logger.exception("HindsightStore areflect LLM synthesis failed")

        # Fallback: return raw TEMPR search results
        logger.info(
            "HindsightStore areflect (raw fallback): {} results, {} chars",
            len(results), len(search_text),
        )
        return _ReflectResponse(search_text)

    async def _synthesize_reflect(
        self,
        query: str,
        context: str | None,
        search_text: str,
    ) -> _ReflectResponse:
        """Two-step synthesis: first classify and structure, then reason."""
        synthesis = await self._provider.chat_with_retry(
            model=self._embedding_model or "gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory analysis engine.  Analyze the retrieved "
                        "memories and the provided context.  Structure your response "
                        "in four sections:\n"
                        "1. **Key Facts**: extracted factual statements from the memories\n"
                        "2. **Patterns & Connections**: recurring themes, relationships\n"
                        "3. **Contradictions & Gaps**: conflicting information, missing data\n"
                        "4. **Actionable Insights**: what should be added/updated in long-term memory\n"
                        "Be concise and evidence-based.  Do not fabricate facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Query\n{query}\n\n"
                        f"## Context\n{context or '(none)'}\n\n"
                        f"## Retrieved Memories\n{search_text}"
                    ),
                },
            ],
            tools=None,
            tool_choice=None,
        )
        if synthesis and getattr(synthesis, "content", None):
            text = synthesis.content or ""
            logger.debug(
                "HindsightStore areflect (LLM synthesis): {} chars",
                len(text),
            )
            return _ReflectResponse(text)
        return _ReflectResponse(search_text)

    # -- TEMPR search engine (RRF fusion) -------------------------------------

    def _tempr_search(
        self,
        query: str,
        budget: str = "mid",
        top_k: int = 20,
    ) -> list[dict]:
        """Multi-strategy TEMPR search with RRF fusion.

        Runs up to 4 parallel retrieval strategies, merges via Reciprocal Rank
        Fusion (matching the official Hindsight algorithm), then applies
        multiplicative recency and temporal-proximity boosts.

        Budget affects which strategies are used:
        - ``low``:  keyword only (fastest)
        - ``mid``:  keyword + embedding + temporal + graph (RRF fusion)
        - ``high``: all strategies with max expansion budget
        """
        query_tokens = _tokenize(query)

        # ---- 1. Keyword (BM25) retrieval ----
        kw_raw = self._bm25.search(query_tokens, top_k=60)
        kw_results: list[tuple[str, float]] = [(did, s) for did, s in kw_raw]

        # ---- 2. Embedding (semantic) retrieval ----
        emb_results: list[tuple[str, float]] = []
        if budget in ("mid", "high") and self._can_embed():
            query_embs = self._try_embed([query])
            if query_embs:
                q_emb = query_embs[0]
                mem_ids: list[str] = []
                embeddings: list[list[float]] = []
                for mid, rec in self._memories.items():
                    emb = rec.get("embedding")
                    if emb and len(emb) > 0:
                        mem_ids.append(mid)
                        embeddings.append(emb)
                if embeddings:
                    scores = _batch_cosine(q_emb, embeddings)
                    emb_results = sorted(
                        [
                            (mid, s)
                            for mid, s in zip(mem_ids, scores)
                            if s > 0
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:60]

        # ---- 3. Graph (link-expansion) retrieval ----
        graph_results: list[tuple[str, float]] = []
        if budget in ("mid", "high"):
            # Seeds: top keyword + top embedding results
            seed_ids: set[str] = set()
            for did, _ in kw_results[:10]:
                seed_ids.add(did)
            for did, _ in emb_results[:10]:
                seed_ids.add(did)
            graph_results = self._graph_expand(list(seed_ids), budget)

        # ---- 4. Temporal retrieval (time-window proximity) ----
        temp_results: list[tuple[str, float]] = []
        if budget in ("mid", "high"):
            temp_results = self._temporal_retrieve(query, budget)

        # ---- Log per-strategy result counts ----
        logger.info(
            "TEMPR search (budget={}): kw={}, emb={}, graph={}, temp={}",
            budget,
            len(kw_results),
            len(emb_results),
            len(graph_results),
            len(temp_results),
        )

        # ---- RRF Fusion ----
        result_lists: list[list[tuple[str, float]]] = []
        if kw_results:
            result_lists.append(kw_results)
        if emb_results:
            result_lists.append(emb_results)
        if graph_results:
            result_lists.append(graph_results)
        if temp_results:
            result_lists.append(temp_results)

        if not result_lists:
            return []

        rrf_scores = _rrf_fusion(result_lists, k=self._RRF_K)

        # ---- Log RRF fusion summary ----
        logger.info(
            "TEMPR RRF fusion: {} lists → {} unique docs merged",
            len(result_lists), len(rrf_scores),
        )

        # ---- Post-RRF boosts (recency + temporal proximity + proof_count) ----
        now = datetime.now(timezone.utc)
        final_scores: dict[str, float] = {}
        for mid, rrf_score in rrf_scores.items():
            rec = self._memories.get(mid, {})
            boost = 1.0

            # Recency boost: linear decay over 365 days → [0.9, 1.1]
            ts_str = rec.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    days_ago = (now - ts).total_seconds() / 86400
                    recency = max(0.1, min(1.0, 1.0 - days_ago / 365))
                    boost *= 1.0 + self._RECENCY_ALPHA * (recency - 0.5)
                except Exception:
                    pass

            # Temporal proximity is already factored into RRF via temp_results list.
            # Proof count boost (for observations)
            proof_count = rec.get("proof_count")
            if proof_count is not None and proof_count >= 1:
                proof_norm = min(1.0, max(0.0, 0.5 + math.log(proof_count) / 10.0))
                boost *= 1.0 + self._PROOF_COUNT_ALPHA * (proof_norm - 0.5)

            final_scores[mid] = rrf_score * boost

        # Sort and return top-k
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        results: list[dict] = []
        for mid, score in ranked[:top_k]:
            rec = self._memories.get(mid, {})
            results.append({
                "id": mid,
                "score": round(score, 4),
                "content": rec.get("content", ""),
                "timestamp": rec.get("timestamp", ""),
                "context": rec.get("context"),
                "fact_type": rec.get("fact_type", "world"),
            })
        return results

    # -- Graph expansion (simulates official link-expansion retrieval) --------

    def _graph_expand(
        self,
        seed_ids: list[str],
        budget: str = "mid",
    ) -> list[tuple[str, float]]:
        """Expand from seeds through 3 signal graphs.

        1. **Entity co-occurrence**: shared entities between seeds and candidates.
           Score = tanh(count * 0.5) maps to [0, 1].
        2. **Semantic kNN**: precomputed embedding neighbours at retain time.
           Score = link weight ∈ [0.7, 1.0].
        3. **Causal links**: heuristic cause/effect, enables/prevents patterns.
           Score = link weight + 1.0 (highest-quality signal).

        Facts appearing in multiple signals accumulate higher scores.
        """
        expansion_budget = self._GRAPH_EXPANSION_BUDGET
        if budget == "high":
            expansion_budget *= 2

        # Collect all entities from seeds
        seed_entities: set[str] = set()
        for mid in seed_ids:
            rec = self._memories.get(mid, {})
            for ent in rec.get("entities", []):
                seed_entities.add(ent)

        entity_scores: dict[str, float] = defaultdict(float)
        semantic_scores: dict[str, float] = defaultdict(float)
        causal_scores: dict[str, float] = defaultdict(float)
        context_scores: dict[str, float] = defaultdict(float)

        # Collect seed contexts for relational boosting
        seed_contexts: set[str] = set()
        for mid in seed_ids:
            ctx = self._memories.get(mid, {}).get("context", "")
            if ctx:
                seed_contexts.add(ctx)

        # 1. Entity co-occurrence expansion
        if seed_entities:
            # Count shared entities per candidate memory
            candidate_counts: dict[str, int] = defaultdict(int)
            for ent in seed_entities:
                for cand_id in self._entity_index.get(ent, set()):
                    if cand_id not in seed_ids:
                        candidate_counts[cand_id] += 1
            for cand_id, count in candidate_counts.items():
                entity_scores[cand_id] = math.tanh(count * 0.5)

        # 2. Semantic kNN + 3. Causal links (from precomputed _links)
        for mid in seed_ids:
            for to_id, link_type, weight in self._links.get(mid, []):
                if to_id in seed_ids:
                    continue
                if link_type == "semantic":
                    semantic_scores[to_id] = max(
                        semantic_scores.get(to_id, 0.0), weight
                    )
                elif link_type in ("causes", "caused_by", "enables", "prevents"):
                    causal_scores[to_id] = max(
                        causal_scores.get(to_id, 0.0), weight + 1.0
                    )

        # 4. Context-based relational boost
        if seed_contexts:
            for cand_id in set(entity_scores) | set(semantic_scores) | set(causal_scores):
                ctx = self._memories.get(cand_id, {}).get("context", "")
                if ctx and ctx in seed_contexts:
                    context_scores[cand_id] = self._CONTEXT_BOOST

        # Merge: additive intra-score across signals
        all_ids = (
            set(entity_scores)
            | set(semantic_scores)
            | set(causal_scores)
            | set(context_scores)
        )
        score_map: dict[str, float] = {}
        for fid in all_ids:
            score_map[fid] = (
                entity_scores.get(fid, 0.0)
                + semantic_scores.get(fid, 0.0)
                + causal_scores.get(fid, 0.0)
                + context_scores.get(fid, 0.0)
            )

        return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:expansion_budget]

    # -- Temporal retrieval ---------------------------------------------------

    def _temporal_retrieve(
        self,
        query: str,
        budget: str = "mid",
    ) -> list[tuple[str, float]]:
        """Temporal retrieval with time-window proximity + spreading.

        Computes temporal proximity scores relative to the current time as the
        reference window center.  For memories with explicit date entities in
        the query, a focused time window would be used; here we default to a
        365-day window centered at "now".

        Spreading: top-scored temporal memories act as seeds for graph expansion
        through causal links (simulating the official temporal spreading).
        """
        now = datetime.now(timezone.utc)
        window_days = 365.0  # default: 1-year window
        mid_date = now

        # 1. Compute temporal proximity for all memories
        proximity: dict[str, float] = {}
        for mid, rec in self._memories.items():
            ts_str = rec.get("timestamp", "")
            if not ts_str:
                proximity[mid] = 0.5
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                days_from_mid = abs((ts - mid_date).total_seconds() / 86400)
                if window_days > 0:
                    proximity[mid] = 1.0 - min(days_from_mid / (window_days / 2), 1.0)
                else:
                    proximity[mid] = 1.0
            except Exception:
                proximity[mid] = 0.5

        # Sort by proximity descending
        sorted_temp = sorted(proximity.items(), key=lambda x: x[1], reverse=True)

        # 2. Spreading through causal links (top-10 temporal seeds)
        seed_ids = [mid for mid, _ in sorted_temp[:10]]
        spread_budget = 20 if budget == "high" else 10

        spread_scores: dict[str, float] = {}
        visited: set[str] = set(seed_ids)
        frontier = list(seed_ids)
        iteration = 0
        max_iterations = 3

        while frontier and len(spread_scores) < spread_budget and iteration < max_iterations:
            iteration += 1
            new_frontier: list[str] = []
            for src_id in frontier[:10]:
                parent_prox = proximity.get(src_id, 0.5)
                for to_id, link_type, weight in self._links.get(src_id, []):
                    if to_id in visited:
                        continue
                    visited.add(to_id)
                    # Causal boost (matches official: causes×2.0, enables×1.5)
                    if link_type in ("causes", "caused_by"):
                        causal_boost = 2.0
                    elif link_type in ("enables", "prevents"):
                        causal_boost = 1.5
                    else:
                        causal_boost = 1.0
                    propagated = parent_prox * weight * causal_boost * 0.7
                    neighbour_prox = proximity.get(to_id, 0.3)
                    combined = max(neighbour_prox, propagated)
                    spread_scores[to_id] = max(spread_scores.get(to_id, 0.0), combined)
                    if combined > 0.2:
                        new_frontier.append(to_id)
            frontier = new_frontier

        # Merge: original proximity + spread scores
        all_temp: dict[str, float] = {}
        for mid, prox in proximity.items():
            all_temp[mid] = max(prox, spread_scores.get(mid, 0.0))

        return sorted(all_temp.items(), key=lambda x: x[1], reverse=True)[:40]

    # -- Causal link detection heuristic --------------------------------------

    def _detect_causal_links(
        self,
        content: str,
        context: str | None,
    ) -> list[dict]:
        """Detect potential causal links from content text patterns.

        Heuristic patterns:
        - "because"/"due to" → causes link
        - "enables"/"allows" → enables link
        - "prevents"/"blocks" → prevents link
        """
        links: list[dict] = []
        content_lower = content.lower()
        combined = f"{context or ''} {content_lower}"

        # Check for causal language patterns
        causal_patterns = [
            (r"\b(because|due to|caused by|as a result of|resulting in)\b", "causes", 0.7),
            (r"\b(enables?|allows?|facilitates?|supports?)\b", "enables", 0.5),
            (r"\b(prevents?|blocks?|inhibits?|stops?)\b", "prevents", 0.5),
        ]

        for pattern, link_type, weight in causal_patterns:
            if re.search(pattern, combined):
                # Find target by looking at adjacent memories with shared entities
                entities = set(_extract_entities(content))
                for mid, rec in self._memories.items():
                    rec_entities = set(rec.get("entities", []))
                    if entities & rec_entities and mid:
                        links.append({
                            "to": mid,
                            "type": link_type,
                            "weight": weight,
                        })
                        if len(links) >= 3:  # cap per memory
                            break
        return links[:3]

    # -- Semantic kNN precomputation ------------------------------------------

    def _compute_knn_links(self, new_id: str, embedding: list[float]) -> None:
        """Compute top-K semantic neighbours for a new memory at retain time.

        Matches the official Hindsight approach of precomputing kNN links at
        insert time (similarity >= 0.7, top-5 neighbours).
        """
        if not embedding:
            return

        similarities: list[tuple[str, float]] = []
        for mid, rec in self._memories.items():
            if mid == new_id:
                continue
            emb = rec.get("embedding")
            if not emb:
                continue
            sim = _cosine_similarity(embedding, emb)
            if sim >= self._SEMANTIC_KNN_THRESHOLD:
                similarities.append((mid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[: self._SEMANTIC_KNN_K]

        rec = self._memories.get(new_id, {})
        rec["_knn"] = [mid for mid, _ in top_k]

        # Add bidirectional links
        for neighbour_id, sim in top_k:
            self._links[new_id].append((neighbour_id, "semantic", sim))
            self._links[neighbour_id].append((new_id, "semantic", sim))

    # -- Observation consolidation --------------------------------------------

    async def consolidate(self) -> int:
        """Auto-consolidate world facts into observations.

        Groups world facts sharing entities and merges them into observation
        records with proof_count tracking.  This simulates the official
        Hindsight observation consolidation.

        Returns the number of new observations created.
        """
        world_facts = {
            mid: rec
            for mid, rec in self._memories.items()
            if rec.get("fact_type") == "world"
        }
        if len(world_facts) < 3:
            return 0

        # Group by shared entities
        groups: dict[str, list[str]] = defaultdict(list)
        for mid, rec in world_facts.items():
            entities = rec.get("entities", [])
            if entities:
                key = "|".join(sorted(entities[:5]))  # top-5 entities as group key
                groups[key].append(mid)

        new_obs_count = 0
        for key, mem_ids in groups.items():
            if len(mem_ids) < 2:
                continue

            # Check if we already have an observation for this group
            existing_obs = [
                mid for mid, rec in self._memories.items()
                if rec.get("fact_type") == "observation"
                and rec.get("_group_key") == key
            ]
            if existing_obs:
                continue

            # Build observation text
            contents = [
                self._memories[mid].get("content", "")
                for mid in mem_ids
            ]
            merged_text = " | ".join(contents[:10])  # cap at 10 facts

            # Create observation
            obs_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            observation: dict[str, Any] = {
                "id": obs_id,
                "content": merged_text,
                "context": f"auto-consolidated from {len(mem_ids)} world facts",
                "timestamp": now,
                "embedding": None,
                "fact_type": "observation",
                "source_type": "consolidation",
                "entities": sorted(set(
                    e for mid in mem_ids
                    for e in self._memories[mid].get("entities", [])
                )),
                "proof_count": len(mem_ids),
                "_knn": [],
                "_causal_links": [],
                "_group_key": key,
                "_source_ids": mem_ids,
            }

            # Generate embedding for observation
            emb_results = self._try_embed([merged_text[:2000]])
            if emb_results:
                observation["embedding"] = emb_results[0]

            self._memories[obs_id] = observation
            tokens = _tokenize(merged_text)
            if tokens:
                self._bm25.add(obs_id, tokens)
            for ent in observation.get("entities", []):
                self._entity_index[ent].add(obs_id)
            if observation.get("embedding"):
                self._compute_knn_links(obs_id, observation["embedding"])

            new_obs_count += 1

        if new_obs_count:
            self._trim_memories()
            self._save_memories()
            logger.info(
                "HindsightStore: consolidated {} new observations",
                new_obs_count,
            )

        return new_obs_count

    # -- Formatting -----------------------------------------------------------

    def _format_search_results(
        self, results: list[dict], max_tokens: int = 4096,
    ) -> str:
        """Format search results into a readable text block."""
        if not results:
            return ""
        lines: list[str] = []
        char_budget = max_tokens * 3  # rough char estimate
        used = 0
        for i, r in enumerate(results):
            ts = r.get("timestamp", "")[:10]
            content = r.get("content", "")
            score = r.get("score", 0)
            fact_type = r.get("fact_type", "world")
            line = f"[{i+1}] ({ts}, {fact_type}, score={score:.3f}) {content}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    # -- Synchronous wrappers (convenience for non-async code) ---------------

    def _run_async(self, coro):
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def retain(self, content: str, context: str | None = None) -> Any:
        return self._run_async(self.aretain(content, context))

    def recall(self, query: str, max_tokens: int = 4096, budget: str = "mid") -> Any:
        return self._run_async(self.arecall(query, max_tokens, budget))

    def reflect(self, query: str, budget: str = "low", context: str | None = None) -> Any:
        return self._run_async(self.areflect(query, budget, context))


# ============================================================================
# Response wrappers (mimic hindsight_client response objects)
# ============================================================================

class _RetainResponse:
    """Mimics hindsight_client aretain response."""

    def __init__(self, mem_id: str, text: str) -> None:
        self.id = mem_id
        self.text = text

    def __repr__(self) -> str:
        return f"<RetainResponse id={self.id}>"


class _RecallResponse:
    """Mimics hindsight_client arecall response."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"<RecallResponse {len(self.text)} chars>"


class _ReflectResponse:
    """Mimics hindsight_client areflect response."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"<ReflectResponse {len(self.text)} chars>"