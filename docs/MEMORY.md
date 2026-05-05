# Memory in UnionClaw

UnionClaw's memory is built on a simple belief: memory should feel alive, but it should not feel chaotic.

Good memory is not a pile of notes. It is a quiet system of attention. It notices what is worth keeping, lets go of what no longer needs the spotlight, and turns lived experience into something calm, durable, and useful.

That is the shape of memory in UnionClaw.

## Pluggable Architecture

UnionClaw inherits nanobot's layered memory design and extends it with a **pluggable memory algorithm** system. Different algorithms implement different strategies for storage, consolidation, and retrieval, all behind a unified `MemoryAlgorithm` interface.

```
   ┌──────────────────────────────────────────────────┐
   │              MemoryRegistry                       │
   │  naive_memory | layerga_memory | emem_memory | nemori_memory | ... │
   └──────────────┬───────────────────────────────────┘
                  │ build(workspace, provider, ...)
                  ▼
   ┌──────────────────────────────────────────────────┐
   │           MemoryComponents                        │
   │  store  │  consolidator  │  dream  │ auto_compact │
   └──────────────────────────────────────────────────┘
```

Choose an algorithm via `memoryAlgorithm` in config:

```json
{
  "agents": {
    "defaults": {
      "memoryAlgorithm": "naive_memory"
    }
  }
}
```

Available algorithms are registered in `MemoryRegistry`. The default is always `naive_memory`.

## The Design

UnionClaw does not treat memory as one giant file.

It separates memory into layers, because different kinds of remembering deserve different tools:

- `session.messages` holds the living short-term conversation.
- `memory/history.jsonl` is the running archive of compressed past turns.
- `SOUL.md`, `USER.md`, and `memory/MEMORY.md` are the durable knowledge files.
- `GitStore` records how those durable files change over time.

This keeps the system light in the moment, but reflective over time.

## The Flow

Memory moves through UnionClaw in two stages.

### Stage 1: Consolidator

When a conversation grows large enough to pressure the context window, UnionClaw does not try to carry every old message forever.

Instead, the `Consolidator` summarizes the oldest safe slice of the conversation and appends that summary to `memory/history.jsonl`.

This file is:

- append-only
- cursor-based
- optimized for machine consumption first, human inspection second

Each line is a JSON object:

```json
{"cursor": 42, "timestamp": "2026-04-03 00:02", "content": "- User prefers dark mode\n- Decided to use PostgreSQL"}
```

It is not the final memory. It is the material from which final memory is shaped.

### Stage 2: Dream

`Dream` is the slower, more thoughtful layer. It runs on a cron schedule by default and can also be triggered manually.

Dream reads:

- new entries from `memory/history.jsonl`
- the current `SOUL.md`
- the current `USER.md`
- the current `memory/MEMORY.md`

Then it works in two phases:

1. It studies what is new and what is already known.
2. It edits the long-term files surgically, not by rewriting everything, but by making the smallest honest change that keeps memory coherent.

This is why UnionClaw's memory is not just archival. It is interpretive.

## Memory Algorithms

### 1. `naive_memory` — Default File-Based Memory

The default and simplest algorithm. Pure file I/O with zero extra dependencies.

**Storage Structure:**

```
{workspace}/
├── SOUL.md                    # Agent's long-term voice and communication style
├── USER.md                    # Stable knowledge about the user
└── memory/
    ├── MEMORY.md              # Project facts, decisions, and durable context
    ├── history.jsonl          # Append-only history summaries (JSONL)
    ├── .cursor               # Consolidator write cursor
    ├── .dream_cursor         # Dream consumption cursor
    └── .git/                 # GitStore version history for long-term files
```

**Files in detail:**

| File | Purpose |
|------|---------|
| `SOUL.md` | Remembers how the agent should sound. Defines personality, tone, and behavioral rules. |
| `USER.md` | Remembers who the user is and what they prefer. Static preferences and identity. |
| `memory/MEMORY.md` | Durable project facts, decisions, technical notes, and long-lived context. |
| `memory/history.jsonl` | Append-only cursor-based conversation archive. Raw material for Dream. |
| `memory/.cursor` | Auto-incrementing integer counter for history entries. |
| `memory/.dream_cursor` | Last cursor position consumed by Dream. |
| `.git/` | Git history of SOUL.md, USER.md, and MEMORY.md — enables audit and rollback. |

---

### 2. `emem_memory` — Structured Memory with EDU Extraction

EMem (Elementary Discourse Unit Memory) adds structured proposition extraction and embedding-based retrieval on top of the naive file-based layer.

**Key features:**
- **EDU extraction**: LLM decomposes conversation turns into atomic propositions (EDUs) with event types, triggers, and role-argument pairs.
- **Dense retrieval**: Embedding-based KNN search for relevant EDUs via Parquet-stored vectors.
- **LLM rerank**: Semantic filtering of candidate EDUs and argument entities.
- **Heterogeneous graph**: Session-EDU-Argument graph with optional Personalized PageRank (PPR) for associative recall.
- **Token-budget consolidation**: Online compression with EDU archiving.

**Storage Structure:**

```
{workspace}/
├── SOUL.md
├── USER.md
└── memory/
    ├── MEMORY.md              # Long-term memory (compatible with naive)
    ├── history.jsonl          # Conversation history (compatible with naive)
    ├── .cursor
    ├── .dream_cursor
    └── emem/                  # ★ EMem structured storage
        ├── edu_storage/
        │   ├── content_edu.pkl         # EDU records (pickle): hash_ids + contents
        │   └── embeddings_edu.parquet  # EDU vectors (Parquet): hash_id + embedding
        ├── argument_storage/
        │   ├── content_argument.pkl    # Entity/argument records (pickle)
        │   └── embeddings_argument.parquet
        └── session_storage/
            └── content_session.pkl     # Session records (no embeddings)
```

**Data models:**

| Model | Description |
|-------|-------------|
| `EDURecord` | Atomic proposition (`edu_id`, `text`, `source_speakers`, `timestamp`, `session_id`, `event_type`, `event_triggers`, `event_role_argument_pairs`) |
| `ArgumentRecord` | Entity/argument node (`arg_id`, `text`, `source_edu_ids`) |
| `SessionRecord` | Conversation session batch (`session_id`, `turns`, `summary`, `date`) |

Records are deduplicated by MD5 hash ID. Embeddings are stored as Parquet files for efficient columnar access.

**Optional dependencies:** `pip install nanobot-ai[emem]` for `igraph` (PPR), `sentence-transformers` (local embeddings), `torch`, `scipy`.

**Configuration** (via `EMemConfig`):
- `skip_ppr`: disable PPR graph propagation (dense-only mode)
- `linking_top_k` / `retrieval_top_k`: candidate counts
- `damping`: PPR damping factor (0–1)

---

### 3. `layerga_memory` — L0-L4 Hierarchical Layered Memory

Based on the GenericAgent multi-layer memory architecture, this algorithm organises memory into five hierarchical layers, each with distinct roles and constraints.

**Key features:**
- **L0 Decision Tree**: A meta-rules constitution governs all memory write decisions — information is classified by the L0 decision tree before storage.
- **Action Verification**: "No Execution, No Memory" — only action-verified facts are stored, preventing hallucinated knowledge.
- **Minimum Sufficient Pointer**: Higher layers (L1) only keep the shortest locator pointing to detailed content in lower layers (L2–L4).
- **Self-Evolution**: The Agent autonomously decides what, where, and how to remember — acting as both executor and memory librarian.
- **Three-Phase Dream**: Phase 1 consolidates recent history into layered storage; Phase 2 crystallises reusable patterns into skills; Phase 3 performs L1 ROI-based cleanup.
- **L4 Session Archives**: Compressed conversation history with automatic archive management.

**Storage Structure:**

```
{workspace}/
├── SOUL.md                    # Agent personality (same as naive)
├── USER.md                    # User profile (same as naive)
├── layerga/
│   └── constitution.md        # ★ L0: Meta-rules constitution (the "memory law")
└── memory/
    ├── MEMORY.md              # Long-term memory (compatible with naive)
    ├── history.jsonl          # Conversation history (same as naive)
    ├── .cursor
    ├── .dream_cursor
    ├── layer_insight.txt      # ★ L1: Minimal insight index (≤30 lines hard cap)
    ├── layer_facts.txt        # ★ L2: Environment fact base (## [SECTION] blocks)
    ├── sop/                   # ★ L3: Task SOP library (*.md + *.py)
    │   └── ...
    └── archives/              # ★ L4: Session archives
        └── all_histories.txt
```

**Layer hierarchy:**

| Layer | Name | Storage | Constraint | Purpose |
|-------|------|---------|------------|---------|
| **L0** | Constitution | `layerga/constitution.md` | Core axioms (immutable) | Meta-rules governing all memory write decisions |
| **L1** | Insight Index | `memory/layer_insight.txt` | ≤30 lines hard cap | Minimal navigation index; ROI-based cleanup |
| **L2** | Fact Base | `memory/layer_facts.txt` | `## [SECTION]` blocks | Environment-specific facts an LLM cannot infer |
| **L3** | Task SOPs | `memory/sop/*.md` + `*.py` | Per-task files | Reusable workflows and utility scripts |
| **L4** | Archives | `memory/archives/` | Auto-managed | Compressed session histories |

**Dependencies:** Zero external dependencies — pure Python implementation using the standard nanobot file I/O stack.

**Decision flow:**
1. Consolidator classifies each conversation segment via the L0 decision tree
2. Facts are written to L2 (`layer_facts.txt`) with minimal patch-only modifications
3. L1 (`layer_insight.txt`) is auto-synced to point to the most valuable L2/L3/L4 entries
4. Dream Phase 1 consolidates history into layered storage; Phase 2 crystallises skills; Phase 3 enforces L1 ≤30-line cap
5. L4 archives are automatically managed by AutoCompact on idle sessions

---

### 4. `nemori_memory` — Self-Organising Long-Term Memory

Based on [nemori](https://github.com/nemori-ai/nemori), this algorithm implements two coupled control loops for self-organising memory.

**Key features:**
- **Two-Step Alignment**: LLM-powered topic segmentation → episode narrative generation
- **Predict-Calibrate Learning**: Hypothesise from existing knowledge → extract high-value facts from discrepancies
- **Episode merging**: Avoid duplication across episodes
- **Unified search**: Keyword + vector search across episodes and semantic memories

**Storage Structure:**

```
{workspace}/
└── memory/
    └── nemori/
        ├── episodes.json          # Episode memories (JSON array)
        │                          #   {id, user_id, title, content, source_messages,
        │                          #    embedding, metadata, created_at, updated_at}
        ├── semantic_memories.json # Semantic knowledge facts (JSON array)
        │                          #   {id, user_id, content, memory_type, confidence,
        │                          #    embedding, source_episode_id, created_at}
        └── message_buffer.jsonl   # Unprocessed message buffer (JSONL, append-only)
                                   #   {message_id, role, content, timestamp,
                                   #    metadata, processed: false}
```

**Data models:**

| Model | Description |
|-------|-------------|
| `Message` | Single conversation message (`role`, `content`, `timestamp`, `message_id`, `metadata`). Can contain multimodal content. |
| `Episode` | Structured episodic memory (`user_id`, `title`, `content`, `source_messages`, `embedding`, `metadata`). |
| `SemanticMemory` | Extracted knowledge fact (`user_id`, `content`, `memory_type`, `confidence`, `embedding`, `source_episode_id`). |

**Backend options:**
- **`"file"`** (default): Zero extra dependencies. JSON files for episodes and semantics, JSONL for message buffer.
- **`"postgres"`**: PostgreSQL + Qdrant for production-grade vector search (requires `asyncpg` + `qdrant_client`).

**Pipeline:**
`message_buffer.jsonl` → `BatchSegmenter` (topic boundaries) → `EpisodeGenerator` (narrative) → `SemanticGenerator` (predict-calibrate) → `EpisodeMerger` (dedup) → `episodes.json` / `semantic_memories.json`

---

### 5. `remem_memory` — ReMeLight-Backed Memory

Adapter wrapping [ReMeLight](https://github.com/nousresearch/reme) for semantic memory search, automatic compaction, and long-term summarisation.

**Key features:**
- ReMeLight handles internal storage (dialog files, semantic index)
- Nanobot companion JSONL for cursor-based Dream/Consolidator compatibility
- Git-tracked MEMORY.md for interop with other algorithms

**Storage Structure:**

```
{workspace}/
├── MEMORY.md                   # Long-term memory (managed by ReMeLight + Dream)
└── memory/
    ├── remem_history.jsonl     # Companion history (JSONL) for cursor tracking
    ├── .remem_cursor           # Cursor counter
    └── .remem_dream_cursor     # Dream cursor
```

ReMeLight manages its own internal directory under `{workspace}` (dialog files, index files). The `remem_history.jsonl` file is a nanobot-side adapter that ensures cursor-based pipelines work unchanged.

---

### Algorithm Comparison

| Algorithm | Core Storage | Format | Vector Search | Extra Dependencies | Best For |
|-----------|-------------|--------|:---:|--------------------|----------|
| **naive_memory** | MEMORY.md + history.jsonl | Markdown + JSONL | ❌ | None | Simple setups, minimal resource usage |
| **emem_memory** | EDU / Argument / Session Store | Pickle + Parquet | ✅ | `igraph`, `sentence-transformers` | Structured fact extraction, entity tracking |
| **layerga_memory** | L0-L4 layered (constitution + insight + facts + SOP + archives) | Markdown + TXT + MD | ❌ | None | Self-organising hierarchical knowledge, zero-dependency |
| **nemori_memory** | Episode + SemanticMemory | JSON + JSONL | ✅ (file or PG+Qdrant) | None (file), `asyncpg`+`qdrant_client` (PG) | Self-organising long-term knowledge |
| **remem_memory** | ReMeLight + companion JSONL | ReMeLight + JSONL | ✅ | `reme-ai` | External memory engine integration |

## The Files

These files play different roles:

- `SOUL.md` remembers how the agent should sound.
- `USER.md` remembers who the user is and what they prefer.
- `MEMORY.md` remembers what remains true about the work itself.
- `history.jsonl` remembers what happened on the way there.

## Why `history.jsonl`

The old `HISTORY.md` format was pleasant for casual reading, but it was too fragile as an operational substrate.

`history.jsonl` gives the agent:

- stable incremental cursors
- safer machine parsing
- easier batching
- cleaner migration and compaction
- a better boundary between raw history and curated knowledge

You can still search it with familiar tools:

```bash
# grep
grep -i "keyword" memory/history.jsonl

# jq
cat memory/history.jsonl | jq -r 'select(.content | test("keyword"; "i")) | .content' | tail -20

# Python
python -c "import json; [print(json.loads(l).get('content','')) for l in open('memory/history.jsonl','r',encoding='utf-8') if l.strip() and 'keyword' in l.lower()][-20:]"
```

The difference is philosophical as much as technical:

- `history.jsonl` is for structure
- `SOUL.md`, `USER.md`, and `MEMORY.md` are for meaning

## Commands

Memory is not hidden behind the curtain. Users can inspect and guide it.

| Command | What it does |
|---------|--------------|
| `/dream` | Run Dream immediately |
| `/dream-log` | Show the latest Dream memory change |
| `/dream-log <sha>` | Show a specific Dream change |
| `/dream-restore` | List recent Dream memory versions |
| `/dream-restore <sha>` | Restore memory to the state before a specific change |

These commands exist for a reason: automatic memory is powerful, but users should always retain the right to inspect, understand, and restore it.

## Versioned Memory

After Dream changes long-term memory files, nanobot can record that change with `GitStore`.

This gives memory a history of its own:

- you can inspect what changed
- you can compare versions
- you can restore a previous state

That turns memory from a silent mutation into an auditable process.

## Configuration

### Memory Algorithm Selection

```json
{
  "agents": {
    "defaults": {
      "memoryAlgorithm": "naive_memory"
    }
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memoryAlgorithm` | string | `"naive_memory"` | Registered algorithm name: `naive_memory`, `layerga_memory`, `emem_memory`, `nemori_memory`, or `remem_memory` |

### Dream Configuration

Dream is configured under `agents.defaults.dream`:

```json
{
  "agents": {
    "defaults": {
      "dream": {
        "intervalH": 2,
        "modelOverride": null,
        "maxBatchSize": 20,
        "maxIterations": 15,
        "annotateLineAges": true
      }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `intervalH` | How often Dream runs, in hours |
| `modelOverride` | Optional Dream-specific model override |
| `maxBatchSize` | How many history entries Dream processes per run |
| `maxIterations` | The tool budget for Dream's editing phase |
| `annotateLineAges` | Annotate each memory line with its age to guide future consolidation |

### Embedding Configuration (for EMem / Nemori)

```json
{
  "agents": {
    "defaults": {
      "embedding": {
        "model": "text-embedding-3-small",
        "provider": "auto",
        "apiKey": null,
        "apiBase": null,
        "batchSize": 16,
        "normalize": true
      }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `model` | Embedding model name (OpenAI-compatible or HuggingFace for `provider: "local"`) |
| `provider` | `"auto"` (inherit LLM provider) or `"local"` (Sentence-Transformers) |
| `apiKey` | Optional override for embedding API key |
| `apiBase` | Optional override for embedding API base URL |

In practical terms:

- `modelOverride: null` means Dream uses the same model as the main agent. Set it only if you want Dream to run on a different model.
- `maxBatchSize` controls how many new `history.jsonl` entries Dream consumes in one run. Larger batches catch up faster; smaller batches are lighter and steadier.
- `maxIterations` limits how many read/edit steps Dream can take while updating `SOUL.md`, `USER.md`, and `MEMORY.md`. It is a safety budget, not a quality score.
- `intervalH` is the normal way to configure Dream. Internally it runs as an `every` schedule, not as a cron expression.

Legacy note:

- Older source-based configs may still contain `dream.cron`. nanobot continues to honor it for backward compatibility, but new configs should use `intervalH`.
- Older source-based configs may still contain `dream.model`. nanobot continues to honor it for backward compatibility, but new configs should use `modelOverride`.

## In Practice

What this means in daily use is simple:

- conversations can stay fast without carrying infinite context
- durable facts can become clearer over time instead of noisier
- the user can inspect and restore memory when needed
- different memory algorithms can be swapped in to match different needs — from lightweight file storage to structured, embedding-powered long-term memory

Memory should not feel like a dump. It should feel like continuity.

That is what this design is trying to protect.
