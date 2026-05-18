# SummerClaw

SummerClaw is developed based on [Nanobot](https://github.com/HKUDS/nanobot), with the following key features:

- **Browser-Enhanced Planning** — Plans on any task even with a relatively weak model by leveraging real-time web search.
- **DAG-Based Subagent Pools** — Directed Acyclic Graph scheduling with configurable recursion depths.
- **ReAct Worker & Re-planner** — Self-improves the planner DAG based on environment feedback.
- **[Enhanced Memory Consolidation & Auto-Skill Learning](docs/MEMORY.md)** — Pluggable memory algorithms (naive / Layerga / EMem / Nemori / Mem0V3 / Supermemory / Hindsight / MastraOM) with automatic skill distillation inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Architecture

```
   ╔═════════════════════════════════════════════════════════════════╗
   ║                    SummerClaw  Runtime Stack                      ║
   ╠═════════════════════════════════════════════════════════════════╣
   │  ①  Public Support Layer                                        │
   │     Config Center · State Hub · Monitoring · Exception Guard    │
   │     Tool Pool  (incl. Web Search Tools)                        │
   ├─────────────────────────────────────────────────────────────────┤
   │  ②  Web-Enhanced Decision Layer                       ★ NEW    │
   │     Search Decision Engine · Retrieval Executor                │
   │     Information Purifier & Injector                            │
   ├─────────────────────────────────────────────────────────────────┤
   │  ③  Closed-Loop Control Layer                         ★ NEW    │
   │     Feedback Collector · Task Evaluator · Re-planning Engine   │
   ├─────────────────────────────────────────────────────────────────┤
   │  ④  Planning & Scheduling Layer                                  │
   │     Top-Level Master Planner · Recursive Depth Controller      │
   ├─────────────────────────────────────────────────────────────────┤
   │  ⑤  Execution Engine Layer                                       │
   │     Reactive Event Bus (ReAct Core) · DAG Dependency Scheduler │
   ├─────────────────────────────────────────────────────────────────┤
   │  ⑥  Execution Unit Layer                                         │
   │     SubAgent Pool · Recursive Decomposition (Configurable)     │
   ├─────────────────────────────────────────────────────────────────┤
   │  ⑦  Memory Layer                                      ★ NEW    │
   │     Consolidator (context pressure) · AutoCompact (session TTL) │
   │     MemoryStore → SOUL.md · USER.md · MEMORY.md · history.jsonl│
   │     ↑  Pluggable: naive / layerga / emem / nemori / mem0v3 /    │
   │     │             supermemory / hindsight / mastra_om  │
   ╠════════════════════ ▼ async · observes history ═══════════════════╣
   │  ⑧  Skill Distillation Layer                          ★ NEW    │
   │     Dream:  cron 2h · /dream  →  dreamed-* skills              │
   │     Hermes: ×10 calls · /skill-autogen  →  hermes-* skills    │
   │     ↑  Skills fed back into ① Tool Pool                        │
   ╚═════════════════════════════════════════════════════════════════╝
```

## Quick Start
1. Install summerclaw:
   ```bash
   pip install -e .
   ```

2. Copy the example config:
  Configure your models and apis, then
   ```bash
   cp config.json.example ~/.summerclaw/config.json
   ```
3. Start the gateway:
   ```bash
   summerclaw gateway
   ```

## SummerClaw Advantages

SummerClaw extends [Nanobot](https://github.com/HKUDS/nanobot) with four architectural innovations that related frameworks lack or only partially implement. It also adopts [GenericAgent](https://github.com/Generative-AI-Research-Company/GenericAgent)'s L0-L4 layered memory as one of its pluggable backends (`layerga_memory`).

| Capability | SummerClaw | Nanobot | GenericAgent | Hermes | OpenClaw |
|---|---|---|---|---|---|
| **Browser-Enhanced Planning** | ✓ Independent web-enhanced decision layer — search → purify → inject, fully automated closed loop | ✗ No planning layer; tool calls are single-step with no pre-planning enhancement | ✗ No web-enhanced planning layer; offline reasoning only | ✗ Manual search-tool calls only; no automatic search injection during planning | ✗ Pure offline planning; no internet augmentation |
| **DAG Subagent Pools** | ✓ Built-in DAG scheduler in the execution engine; `max_subagent_depth` 0~N freely configurable | ✗ Pure single-agent; no scheduling layer, no DAG, no subagent pool | ✗ Single-agent execution; no DAG scheduler or parallel subagent pool | ✗ Single-agent sequential execution; no task decomposition, no concurrency | △ Basic task splitting, but no recursive depth config and no concurrent agent pool |
| **ReAct Re-planner** | ✓ Independent closed-loop control — feedback → evaluate → replan, fully automatic | ✗ Single-step streaming; wrong = retry from scratch with no plan correction | ✗ No closed-loop replanning; fixed execution path | △ Dynamic in-step correction, but no DAG structure and no standardized replanning engine | ✗ Static one-time PTS plan; locked during execution with no feedback loop |
| **Memory + Auto-Skill** | ✓ Dual-mode: Dream (cron scheduled) + Hermes-Autogen (tool-count triggered); skills auto-injected into tool pool | △ Dream-only weak mode — shell skills from batch summarization, no trajectory distillation | △ L0-L4 layered memory consolidation (constitution → insight → facts → SOP → archives); no Hermes-style trajectory distillation | ✓ Trajectory-driven real-time skill generation only; no Dream-style batch consolidation | ✗ No memory purification, no automatic skills; all skills must be hand-written |

> ✓ = Full native implementation &nbsp; △ = Partial / limited &nbsp; ✗ = Not available

## Memory

SummerClaw ships with a **pluggable memory algorithm** system. Choose from eight backends via the `memoryAlgorithm` config key:

| Algorithm | Strategy | Best For |
|-----------|----------|----------|
| `naive_memory` | File-based (MEMORY.md + history.jsonl) | Simple setups, zero extra deps |
| `emem_memory` | EDU extraction + embedding vectors | Structured fact & entity tracking |
| `layerga_memory` | L0-L4 layered (constitution → insight → facts → SOP → archives) | Self-organising hierarchical knowledge |
| `nemori_memory` | Episode + semantic self-organising | Long-term knowledge evolution |
| `mem0v3_memory` | ADD-only single-pass extraction + entity linking + multi-signal fusion | Token-efficient LLM-native memory |
| `supermemory_memory` | Chunk-based + relational versioning + temporal grounding + hybrid search + static/dynamic classification + auto-forgetting | SOTA agent memory with version tracking |
| `hindsight_memory` | Built-in local TEMPR engine (RRF fusion + fact types + graph expansion) | Zero-dependency multi-strategy retrieval |
| `mastra_om_memory` | Observer/Reflector pipeline + async buffering + observation groups + prompt-cache friendly context | High-density observational memory (94.87% LongMemEval) |

See the **[Memory Documentation](docs/MEMORY.md)** for storage structures, configuration, and algorithm details.

## Tools

SummerClaw provides a set of **atomic tool primitives** that form the agent's interaction surface with the world — every capability is a self-contained `Tool` instance with typed parameters, validation, and safety constraints, all managed by a unified `ToolRegistry`.

| Category | Tools | Gate |
|----------|-------|------|
| **Filesystem** | `read_file` · `write_file` · `edit_file` · `list_dir` | Always on |
| **Search** | `glob` · `grep` | Always on |
| **Notebook** | `notebook_edit` | Always on |
| **Shell** | `exec` — sandboxed command execution | `tools.exec.enable` |
| **Web** | `web_search` · `web_fetch` — 6 search backends | `tools.web.enable` |
| **Browser** | `browser_search` · `browser_fetch` · `browser_navigate` · `browser_snapshot` · `browser_execute_js` | `tools.browser.enable` |
| **Interaction** | `message` · `ask_user` · `spawn` · `cron` | Always on |
| **Self** | `my` — runtime introspection & tuning | `tools.my.enable` |
| **MCP** | Dynamic tools via Model Context Protocol | `mcpServers` config |

> Each tool declares `read_only` (safe to parallelize) and `exclusive` (must run alone) flags. Parameter validation uses JSON Schema with automatic type casting. MCP tools are dynamically discovered and registered with the `mcp_<server>_` prefix.

See the **[Tools Documentation](docs/TOOLS.md)** for full tool descriptions, parameters, safety classification, and registration flow.

## Skill Distillation

SummerClaw supports two orthogonal skill distillation modes that can run independently or simultaneously.

### Mode 1 — Dream  *(Memory Consolidation)*

Consolidates conversation history into long-term memory files (`MEMORY.md`, `SOUL.md`, `USER.md`) and extracts reusable skills prefixed with `dreamed-`.

| Trigger | Condition | Notification |
|---|---|---|
| `/dream` command | Always | `Dream completed in Xs.` / `Dream: nothing to process.` |
| Cron auto (every 2 h) | When changes exist | `[Dream] Memory consolidation complete. MEMORY / SOUL / USER.md updated.` |

- **Default state:** enabled
- **Skill output:** `workspace/skills/dreamed-<name>/SKILL.md`

### Mode 2 — Hermes-Autogen  *(Tool-Usage Distillation)*

Watches tool-call history and distills recurring patterns into reusable skills prefixed with `hermes-`, inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent).

| Trigger | Condition | Notification |
|---|---|---|
| `/skill-autogen` command | Always | `Skill-Autogen completed in Xs: a new skill was created...` |
| Threshold auto (every 10 tool calls) | When a new skill is identified | `[Skill-Autogen] Background distillation complete. New skill saved to workspace/skills/` |

- **Default state:** disabled — enable via `skill_autogen.enable = true` in config
- **Skill output:** `workspace/skills/hermes-<name>/SKILL.md`

> **Note:** All channel notifications are routed to the last active session.
> Silent (log-only) when no active session exists.

## Task Persistence

| Command | Description |
|---|---|
| `/tasks` | List currently running subagents and pending interrupted tasks |
| `/tasks resume <id>` | Re-spawn a subagent task; re-deliver a main-flow task to the bus |
| `/tasks discard <id>` | Remove from registry; mark main-flow task as errored |

**Key behaviors after service restart:**

- Within 3 seconds, a notification is pushed to each affected channel listing all interrupted tasks with `resume`/`discard` options.
- `/tasks resume <id>`: Subagent restarts from the beginning (original task description); main-flow task resumes from the persisted `runtime_checkpoint`.
- `/tasks discard <id>`: Clears the persistence record; subagent is dropped entirely; main-flow task leaves an error marker in the session.

## Documentation

- [Tools — Available Tools & Configuration](docs/TOOLS.md)
- [Memory — Algorithms & Storage](docs/MEMORY.md)
- [SummerClaw Extended Docs](summerclaw.md)
- [Upstream SummerClaw](https://github.com/HKUDS/summerclaw)

## Key Configuration

Copy `config.json.example` to `~/.summerclaw/config.json` and edit as needed.
All keys below live under `agents.defaults` unless noted otherwise.

### Plan-and-Solve + DAG Subagents

| Key | Type | Default | Description |
|---|---|---|---|
| `plan_and_solve` | bool | `true` | Enable ReAct→Plan-and-Solve upgrade; the agent builds a DAG before execution |
| `max_subagent_depth` | int | `2` | Maximum recursion depth for nested SubAgent spawning (0 = flat, no nesting) |

### Search-Enhanced Planning  *(② Web-Enhanced Decision Layer)*

| Key | Type | Default | Description |
|---|---|---|---|
| `search_enhanced_planning.enable` | bool | `true` | Gate for the whole web-search pre-planning pipeline |
| `search_enhanced_planning.max_results` | int | `5` | Max search results fetched per planning query |
| `search_enhanced_planning.timeout` | int | `120` | Seconds before a single search call times out |
| `search_enhanced_planning.search_on_replan` | bool | `false` | Re-run web search each time the planner triggers a re-plan cycle |
| `search_enhanced_planning.max_purified_chars` | int | `2000` | Max characters of distilled search content injected into the planning prompt |
| `tools.web.enable` | bool | `false` | **Must be `true`** to activate web search tools used by this layer |
| `tools.web.search.provider` | string | `"duckduckgo"` | Search backend: `duckduckgo` \| `bing` \| `google` \| `searxng` |
| `tools.web.search.apiKey` | string | `""` | API key for providers that require one |
| `tools.web.search.maxResults` | int | `5` | Hard cap on results returned by the search tool |
| `tools.web.search.timeout` | int | `30` | Per-request timeout for the search tool (seconds) |

### Memory Algorithm  *(Pluggable Backend)*

| Key | Type | Default | Description |
|---|---|---|---|
| `memoryAlgorithm` | string | `"naive_memory"` | Select the memory backend: `naive_memory`, `layerga_memory`, `emem_memory`, `nemori_memory`, `mem0v3_memory`, `supermemory_memory`, `hindsight_memory`, or `mastra_om_memory` |

### Skill Distillation — Dream  *(⑦ Mode 1)*

| Key | Type | Default | Description |
|---|---|---|---|
| `dream.intervalH` | float | `2` | Hours between automatic Dream cron runs (set to `0` to disable cron) |
| `dream.modelOverride` | string\|null | `null` | Use a different model for Dream; falls back to agent default when `null` |
| `dream.maxBatchSize` | int | `20` | Max conversation turns processed per Dream run |
| `dream.maxIterations` | int | `15` | Max LLM iterations allowed inside a single Dream run |
| `dream.annotateLineAges` | bool | `true` | Annotate each memory line with its age to guide future consolidation |

### Skill Distillation — Hermes-Autogen  *(⑦ Mode 2)*

| Key | Type | Default | Description |
|---|---|---|---|
| `skill_autogen.enable` | bool | `false` | Enable Hermes-Autogen; **disabled by default** — must opt in explicitly |
| `skill_autogen.nudge_interval` | int | `10` | Tool-call count threshold that triggers a background distillation run |
| `skill_autogen.max_iterations` | int | `8` | Max LLM iterations allowed inside a single Hermes-Autogen run |
