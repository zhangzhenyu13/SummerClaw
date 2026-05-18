# Memory Management Constitution (L0 Meta-Rules)

> **"No Execution, No Memory"** — Only action-verified information is worth remembering.

## 0. Core Axioms (Highest Priority)

### Axiom 1: Action-Verified Only
**Definition**: Any information written to L1/L2/L3 MUST originate from a **successful tool-call result** (exit_code=0, status="success", confirmed by read_file, etc.).

**Forbidden**: Do NOT write the model's "inherent knowledge", "inferred guesses", "unexecuted plans", or "unverified hypotheses" as facts.

**Slogan**: **No Execution, No Memory.**

### Axiom 2: Sanctity of Verified Data
**Definition**: Any action-verified configuration, pitfall guide, or critical path MUST NOT be discarded during refactoring/GC.

**Operation**: You may compress text, migrate between layers (L2 → L3), but MUST NOT lose accuracy or traceability. When editing memory, use only minimal patch operations — never overwrite entire files. If a patch fails, prefer leaving it alone over forcing a rewrite.

### Axiom 3: No Volatile State
**Definition**: DO NOT store data that changes frequently across time/sessions.

**Examples**: Current timestamps, temporary Session IDs, running PIDs, specific absolute paths that vary per machine, connected device info.

### Axiom 4: Minimum Sufficient Pointer
**Definition**: Upper layers only keep the shortest identifier that can locate the lower layer. One extra word is one extra word of waste.

---
## Memory Layer Architecture

```
L1: layer_insight.txt (Minimal Index — strict ≤30 lines)
    ↓ navigates to (Pointer)
L2: layer_facts.txt (Fact Base — grows with usage)
    ↓ detailed reference (Reference)
L3: memory/sop/ (Task Records — .md SOPs + .py scripts)
L4: memory/archives/ (Session Archives — compressed history summaries)
```

---
## Layer Responsibilities

### L1: Global Memory Insight (layer_insight.txt)
- **Volume**: ≤30 lines (hard constraint), <1K tokens expected
- **Content**: Two-tier "scenario keyword → memory location" mapping + RULES section
  - Tier 1: High-frequency scenarios → direct pointer (key→value)
  - Tier 2: Low-frequency scenarios → keyword only (Agent searches L2/L3 on demand)
  - RULES: Compressed pitfall guidelines (1 sentence each)
- **Updates**: When L2/L3 gain/lose entries, assess frequency and sync to appropriate tier.
- **Forbidden**: Passwords, API Keys, How-to details, task-specific technical details, log records.

### L2: Global Fact Base (layer_facts.txt)
- **Content**: Environment-specific facts organized as `## [SECTION]` blocks
- **What goes here**: Paths, credentials, configurations, API endpoints, proxy ports, account IDs — anything an LLM cannot infer zero-shot
- **Forbidden**: Volatile state, guesses, common-sense knowledge

### L3: Task-Level SOP Library (memory/sop/)
- **Purpose**: Supplement L1/L2 with task-specific details essential for future reuse
- **SOP files (*_sop.md)**: ONLY record hidden preconditions and typical pitfalls
- **Script files (*.py)**: ONLY encapsulate highly-reusable complex logic
- **DO NOT record**: Ordinary operation steps, paths recoverable in a few probes

### L4: Session Archives (memory/archives/)
- **Purpose**: Compressed historical conversation summaries for cross-session recall
- **Managed by**: Dream (cron-scheduled) and AutoCompact (idle-triggered)

---
## L1 ↔ L2/L3 Synchronization Rules

| Operation | L1 Sync Required |
|-----------|-----------------|
| L2/L3 new entry | Default to low-frequency → add filename to L3 list. Only add parenthetical trigger word (2-4 chars) for counter-intuitive scenarios. |
| L2/L3 delete entry | Remove corresponding keyword/mapping line |
| L2/L3 modify value | If scenario location unchanged, leave L1 alone |
| Discover universal pitfall | Compress into 1 sentence and append to RULES |

> **Sync Red Line**: L1 only writes keywords/names — DO NOT copy details. Parentheses only for counter-intuitive trigger words (2-4 chars). DO NOT write mechanism/method/step descriptions.

---
## Information Classification Decision Tree

```
"Which layer does this information belong in?"

Is it an 'environment-specific fact'?
(IP, non-standard path, credential, ID, API key — LLM zero-shot cannot generate accurately)
  ├─ YES → L2 (layer_facts.txt)
  │        Then → Assess frequency:
  │          High-frequency → L1 Tier 1 (key→value)
  │          Low-frequency → L1 Tier 2 (keyword only)
  │
  └─ NO
       ↓
       Is it a 'universal operating rule'?
       (Cross-task pitfall guide, troubleshooting method, not task-specific)
       ├─ YES → L1 [RULES] (1 compressed sentence only)
       │
       └─ NO
            ↓
            Is it 'task-specific technique'?
            (Hard-won through struggle, reusable in future tasks)
            ├─ YES → L3 (sop/ — specialized SOP or script)
            │
            └─ NO → Classify as 'common knowledge' or 'redundant':
                    DO NOT store — discard immediately
```

---
## Quick Reference Table

| Layer | Trigger Condition | Key Words | Typical Examples |
|-------|------------------|-----------|-----------------|
| **L2 → L1** | Environment fact LLM can't infer | path, credential, ID, port | `proxy_port=7890`, `data_directory` |
| **L1 RULES** | Cross-task pitfall | global, 1 sentence | `Do NOT unconditionally kill python (would kill self)` |
| **L3** | Task-specific hard-won experience | future reuse, non-disposable | `discord_input_box_special_handling` |
| **Discard** | Common knowledge or temporary state | inferrable, one-time | timestamps, PIDs, ordinary operation steps |

---
## Cleanup Principles (L1 Maintenance)

**ROI Formula**: ROI = (mistake probability without this word × cost) / per-turn token cost

- **Keep**: Counter-intuitive trigger words — without them you wouldn't think to check the SOP
- **Delete**: Name translations, content descriptions, intuitive abilities, redundant lines
- **Compress**: Multiple related entries under a single parent scenario name
- **Hard Rule**: After cleanup, total lines MUST be ≤30
