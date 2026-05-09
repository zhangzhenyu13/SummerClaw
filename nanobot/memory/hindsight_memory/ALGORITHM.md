# Hindsight 记忆算法

> 内置本地 TEMPR 多策略检索引擎的增强型记忆算法，在 naive 文件存储基础上叠加语义搜索、RRF 融合、图扩展与关联推理。算法核心对标官方 Hindsight（https://hindsight.vectorize.io/），使用 Reciprocal Rank Fusion 替代线性加权，用图扩展模拟 link-expansion retrieval，并支持事实类型分层与观察自动合并。

## 核心思想

naive_memory 的局限在于**检索环节**：Dream 做反思时只能读取 `history.jsonl` 的原始文本和当前 `MEMORY.md`，无法从海量历史记忆中**搜索**最相关的上下文。这导致：
- "用户上次说的那个配置参数是什么？"——需要人工翻 MEMORY.md
- "这个 bug 和三个月前那个有关系吗？"——纯文本文件无法关联
- MEMORY.md 越来越大后，"Lost in the Middle" 效应加剧

Hindsight 用**本地双写策略**解决这个问题：
1. **文件层**（始终可用）：完整的 naive 文件存储（MEMORY.md + history.jsonl）
2. **TEMPR 引擎层**（内置，零外部依赖）：每次 consolidate/dream 的同时，将摘要 retain 到本地 JSON 记忆库，获得多策略检索能力

### TEMPR 多策略检索

TEMPR（Temporal + Embedding + Metadata + Probabilistic + Relational）是五引擎联合检索框架：

| 引擎 | 策略 | 解决的问题 |
|------|------|-----------|
| **T**emporal | 时间窗口邻近度 + 因果链扩散传播 | "上周讨论的那个方案" |
| **E**mbedding | 语义向量余弦相似度 | "和这类似的问题" |
| **M**etadata | fact_type 标签（world/experience/observation/opinion） | "只找客观事实" |
| **P**robabilistic | BM25 关键词匹配（纯 Python 倒排索引） | 精确术语查询 |
| **R**elational | 图扩展（实体共现 + 语义 kNN + 因果链 boosting） | "这个 bug 相关的后续修复" |

### Reciprocal Rank Fusion (RRF)

**不使用权重复合，而是采用 RRF 融合**——与官方 Hindsight 算法一致：

```
score(d) = Σ (1 / (k + rank_i(d)))    k = 60
```

优势：
- **无需手动调权**：对各引擎的分数分布鲁棒
- **排名即信号**：每条结果列表中的排序位置直接决定贡献
- **自然融合**：同一记忆出现在多个引擎结果中时自动累积更高得分

RRF 融合后，再施加乘性后增强（post-RRF boosts）：

```
final_score = rrf_score × recency_boost × proof_count_boost

recency_boost    = 1 + α_rec × (recency - 0.5)    α_rec = 0.20
                    recency ∈ [0.1, 1.0]，线性衰减 365 天
proof_count_boost = 1 + α_proof × (proof_norm - 0.5)  α_proof = 0.10
                    proof_norm = log(proof_count) 归一化
```

### 事实类型 (fact_type)

对标官方 Hindsight 的事实分层：

| 类型 | 说明 | 来源 |
|------|------|------|
| `world` | 客观事实（默认） | aretain() 默认 |
| `experience` | Agent 自身行动与交互 | hook 中轮提取 |
| `observation` | 多条事实自动合并的知识 | consolidate() 生成 |
| `opinion` | 主观陈述 | 手动标记 |

### 图扩展 (Graph Expansion)

模拟官方 Hindsight 的 link-expansion retrieval，通过三种信号构建关联图：

```
1. 实体共现 (Entity Co-occurrence)
   - 基于 entity_index (entity → memory_ids)
   - score = tanh(shared_entity_count × 0.5)

2. 语义 kNN (Semantic kNN)
   - aretain 时预计算：cosine ≥ 0.70，top-5 近邻
   - 存为双向链接

3. 因果链 (Causal Links)
   - 启发式检测：because/due to/enables/prevents 等模式
   - causes/caused_by → +2.0x boost
   - enables/prevents → +1.5x boost

4. 上下文关联 (Context Proximity)
   - 同一 context 的记忆获得 +0.3x boost
```

### 时态检索 (Temporal Retrieval)

```
1. 时间窗口邻近度
   proximity = 1 - min(days_from_mid / (window_days / 2), 1.0)
   默认 365 天窗口，以"现在"为中心

2. 因果链扩散传播 (Spreading)
   - top-10 时态种子通过 causal links BFS 扩散
   - 最多 3 轮迭代
   - causes × 2.0, enables × 1.5 增强
   - propagated = parent_prox × weight × causal_boost × 0.7
```

### 检索预算 (budget)

| 预算 | 启用的引擎 | 适用场景 |
|------|-----------|---------|
| `low` | BM25 关键词 | 精确术语查找 |
| `mid` | BM25 + Embedding + Temporal + Graph (RRF 融合) | 日常 Dream 分析 |
| `high` | 全部策略，最大图扩展预算（×2） | 深度推理、跨会话关联 |

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                 Hindsight 记忆算法流水线                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  对话消息 ──► Consolidator ──► LLM 摘要 ──► history.jsonl         │
│    │              │                    │                          │
│    │      令牌预算检查          ┌──────┴──────┐                   │
│    │      超过窗口则归档        │  本地 TEMPR  │ retain 摘要       │
│    │                           │  引擎        │←──(JSON 文件)────│
│    │                           └──────┬──────┘                   │
│    │                                  │                          │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  Cron 定时 ──► Dream (Phase 1) ──► store.areflect()               │
│                        │               │                         │
│                        │          ┌────┴────┐                    │
│                        │          │ 成功?     │                   │
│                        │          │Yes  No   │                   │
│                        │          ▼    ▼     │                   │
│                        │     reflect  LLM    │                   │
│                        │     analysis fallback                   │
│                        │               │                         │
│                        ▼               ▼                         │
│               Dream (Phase 2) ──► AgentRunner 编辑 MEMORY.md      │
│                                    SOUL.md / USER.md              │
│                                    skills/dreamed-*/SKILL.md     │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  空闲会话 ──► AutoCompact ──► 压缩 + TEMPR retain                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Hermes-Autogen 模式

```
  对话中轮次 ──► extract_and_store()
                      │
                      ▼
              LLM 提取事实列表
              "- Fact one"
              "- Fact two"
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    history.jsonl          TEMPR.aretain()
    (逐条事实写入)          (逐条事实写入本地JSON)
```

## 存储布局

```
workspace/
├── SOUL.md                    # Agent 身份/个性/行为准则
├── USER.md                    # 用户画像/偏好
└── memory/
    ├── MEMORY.md              # 长期记忆（LLM 上下文注入）
    ├── history.jsonl          # 对话历史（append-only JSONL）
    ├── .cursor                # 归档游标
    ├── .dream_cursor          # Dream 处理游标
    └── hindsight_memories.json # 本地 TEMPR 记忆库

本地 TEMPR 记忆库（hindsight_memories.json）：
    └── 每个条目:
        ├── id: UUID
        ├── content: 记忆文本
        ├── embedding: float[] (通过 provider.embed() 生成)
        ├── fact_type: "world" | "experience" | "observation" | "opinion"
        ├── entities: str[] (提取的实体标记，用于图扩展)
        ├── proof_count: int (观察类记忆的证据计数，自动合并时递增)
        ├── _knn: str[] (预计算的语义 kNN 近邻 ID)
        ├── _causal_links: [{to, type, weight}] (检测到的因果链接)
        ├── timestamp: ISO datetime
        ├── context: str (来源上下文)
        └── source_type: "manual" | "consolidation" (来源标记)

内部索引结构（内存中）：
    ├── _entity_index: defaultdict[str, set[str]]  ← entity → memory_ids
    └── _links: defaultdict[str, list[(to, type, weight)]]  ← 双向关联图
```

## 四大组件

### HindsightStore

继承自 naive `MemoryStore`，在完整文件 I/O 基础上新增本地 TEMPR 检索引擎。

**文件层 API**（继承自 MemoryStore，始终可用）：

| 方法 | 功能 |
|------|------|
| `read_memory()` / `write_memory()` | 长期记忆读写 |
| `read_soul()` / `write_soul()` | Agent 身份读写 |
| `read_user()` / `write_user()` | 用户画像读写 |
| `append_history()` | 追加到 history.jsonl，返回游标 |
| `read_unprocessed_history()` | 按游标增量读取 |
| `compact_history()` | 按条目数上限裁剪 |
| `get_memory_context()` | 构建 LLM 上下文注入文本 |
| `git` (GitStore) | 行龄标注 + 自动 commit |

**TEMPR 引擎 API**（内置，始终可用）：

| 方法 | 功能 | 实现 |
|------|------|------|
| `aretain(content, context, *, fact_type)` | 异步存储记忆到本地 JSON 库 | 生成 embedding + 实体提取 + kNN 预计算 + 因果链检测 + 持久化 |
| `arecall(query, max_tokens, budget)` | 异步多策略搜索 (RRF 融合) | TEMPR 四引擎 → RRF 融合 → post-RRF 增强 → 排序 |
| `areflect(query, budget, context)` | 异步推理式检索 | TEMPR 搜索 + LLM 四段式结构化综合推理（可选） |
| `consolidate()` | 自动合并 world 事实为 observation | 共享实体分组 → 合并 → proof_count 跟踪 |
| `retain()` / `recall()` / `reflect()` | 同步包装器 | 内部调用 `asyncio.run()` |

**关键属性**：

- `hindsight_enabled` → 始终 `True`（内置实现，无外部依赖）
- `memory_count` → 当前本地记忆库条目数

**检索核心流程 (`arecall()`)**

```
arecall(query, budget="mid")
    │
    ├──► _tokenize(query)                  # 中文分词 / 英文空格分词
    │
    ├──► [budget >= low]
    │       BM25 关键词搜索 (纯 Python 倒排索引)
    │       TF-IDF 加权，短词归一化
    │
    ├──► [budget >= mid]
    │       Embedding 余弦相似度 (query vs 所有记忆)
    │       (需 provider.embed()，无 provider 则跳过)
    │
    │       Graph 图扩展检索
    │       - 从 top-10 keyword + top-10 embedding 取种子
    │       - 实体共现 (tanh 评分)
    │       - 语义 kNN (预计算链接)
    │       - 因果链 boosting (×2.0 / ×1.5)
    │       - 上下文关联 (+0.3)
    │
    │       Temporal 时态检索
    │       - 时间窗口邻近度 (365 天窗口)
    │       - BFS 因果链扩散传播 (3 轮迭代)
    │
    ├──► RRF 融合排序
    │       rrf_score = Σ (1 / (60 + rank))
    │
    ├──► Post-RRF 增强
    │       final = rrf_score × recency_boost × proof_count_boost
    │
    └──► 截断到 max_tokens，格式化为文本返回 (含 fact_type)
```

**嵌入生成 (`aretain()`)**

```python
async def aretain(content, context=None, *, fact_type="world"):
    # 1. 验证 fact_type 合法性（world/experience/observation/opinion）
    # 2. 构建记忆条目 (ID, content, metadata_tags, timestamp, context)
    # 3. 提取实体 (entity extraction) → 更新 entity_index
    # 4. 检测因果链 (causal link detection) → 启发式模式匹配
    # 5. 如果 provider 可用: embedding = await provider.embed(model, [content])
    #    无 provider: embedding = None (语义检索不可用，其他引擎仍正常)
    # 6. 预计算语义 kNN 链接 (cosine ≥ 0.70, top-5 neighbours)
    # 7. 写入 JSON 文件，更新 BM25 倒排索引
    # 8. 如果超出 max_memories: 按时间裁剪最旧的条目
```

### HindsightConsolidator

继承 naive `Consolidator`，扩展了本地 TEMPR 保留和 Hermes 中轮提取。

**archive() 增强**：

```
archive(messages)
    │
    ├──► LLM 摘要（与 naive 相同）
    ├──► history.jsonl 写入（与 naive 相同）
    └──► if has_hindsight:
            hindsight_store.aretain(摘要文本)
            (失败不影响文件写入)
```

**Hermes-Autogen 模式** (`extract_and_store()`)：

1. LLM 从对话中提取事实列表（格式：`- Fact one\n- Fact two`）
2. 解析为独立事实 (`_parse_summary_into_facts()`)
3. 每条事实写入 history.jsonl + TEMPR retain
4. LLM 失败时回退到原始文本归档

**Hermes 触发**：在 `hook.post_turn()` 中，满足条件时调用 `extract_and_store()`：
- provider 支持 Hermes（`provider.supports_hermes`）
- 或者配置了 `hermes_model`

### HindsightDream

Cron 定时触发的深度记忆处理器，两阶段流水线。

**Phase 1 — 分析（TEMPR 优先 + LLM 兜底）**：

```
has_hindsight?
    │
    ├── Yes ──► hindsight_store.areflect(
    │              query="分析历史+当前记忆...",
    │              context=phase1_prompt[:8000],
    │              budget="mid"
    │           )
    │              │
    │         ┌────┴────┐
    │         │ 成功?     │
    │         │Yes  No   │
    │         ▼    ▼     │
    │    有内容?  异常    │
    │    │   │    │      │
    │    ▼   ▼    ▼      │
    │  使用  LLM  LLM    │
    │       fallback     │
    │                    │
    └── No ──► LLM 分析 ─┘
```

- **TEMPR reflect**：将当前 MEMORY.md + 历史条目作为 context，通过本地 TEMPR 多引擎做深度关联分析。如 provider 可用，LLM 输出结构化四段分析：① Key Facts ② Patterns & Connections ③ Contradictions & Gaps ④ Actionable Insights。LLM 不可用时回退到原始 TEMPR 搜索结果
- **LLM fallback**：当 reflect 返回空、或 TEMPR 检索失败时，回退到标准 LLM 调用
- **两者都失败**：`run()` 返回 `False`，Dream 终止

**Phase 2 — 执行**：

- 将 Phase 1 分析结果传给 `AgentRunner`
- 工具集：`read_file` / `edit_file` / `SkillPrefixWriteFileTool`（前缀 `dreamed-`）
- AgentRunner 自主决定如何编辑文件，可创建 dreamed-* 技能
- 处理完推进 `.dream_cursor`，自动 `compact_history()`

**行龄标注**（`annotate_line_ages=True` 时）：

```
# MEMORY.md 渲染时，超过 14 天的行追加 ← Nd 标记
- User prefers dark mode  ← 30d
- Project uses React 18   ← 20d
- fresh item
```

实现细节：
- 基于 Git 的 `line_ages()` 获取每行年龄
- 仅标注 MEMORY.md（SOUL.md 和 USER.md 永远不标注——它们是永久配置）
- 行数与 ages 数量不匹配时跳过标注（防止错位）

**技能发现**：

- Phase 2 prompt 注入 `## Existing Skills` 上下文（workspace skills + BUILTIN_SKILLS_DIR）
- Phase 2 system prompt 包含 `skill-creator` 技能的使用说明
- AgentRunner 可通过 `SkillPrefixWriteFileTool` 创建 `skills/dreamed-*/SKILL.md`

### HindsightAutoCompact

空闲会话压缩器。

- 定时扫描超过 TTL 的非活跃会话
- 调用 `HindsightConsolidator.archive()` 压缩消息
- `archive()` 内部已有 TEMPR retain 逻辑（无需额外处理）
- `has_hindsight` 属性反映 TEMPR 引擎可用性

## 数据流全景

```
   用户消息
      │
      ▼
   ┌─────────────────────┐
   │  Consolidator        │  ← 每轮检查 token 预算
   │  estimate tokens     │
   │       │              │
   │       ▼              │
   │  超预算? ──Yes──► archive() ──► history.jsonl
   │       │              │              │
   │      No              │     TEMPR.aretain() ← (本地 JSON)
   │       │              │
   │       ▼              │
   │  继续正常对话         │
   └─────────────────────┘
          │
          │ (hook.post_turn)
          ▼
   ┌──────────────────────────────┐
   │  Hermes (可选)                 │
   │  extract_and_store()          │
   │    → LLM 提取事实              │
   │    → history.jsonl            │
   │    → TEMPR.aretain() 逐条     │
   └──────────────────────────────┘

                              (Cron 定时)
                                   │
                                   ▼
   ┌──────────────────────────────────────────┐
   │  Dream                                    │
   │                                           │
   │  Phase 1:                                 │
   │    has_hindsight?                         │
   │      Yes → areflect(phase1_prompt)        │
   │        ├─ 成功+有内容 → 使用 reflect 分析  │
   │        └─ 失败/空 → LLM fallback          │
   │      No  → LLM 直接分析                   │
   │                                           │
   │  Phase 2:                                 │
   │    AgentRunner(analysis)                  │
   │    + read_file / edit_file / SkillPrefix   │
   │    → 编辑 MEMORY.md / SOUL.md 等           │
   │    → 创建 dreamed-* 技能                   │
   │    → advance cursor + compact history     │
   │    → git commit                           │
   └──────────────────────────────────────────┘

   ┌──────────────────────────────────────────┐
   │  AutoCompact (定时)                       │
   │  扫描空闲会话 → Consolidator.archive()     │
   │  → history.jsonl + TEMPR retain           │
   └──────────────────────────────────────────┘
```

## 使用方式

```python
from pathlib import Path
from nanobot.memory.registry import MemoryRegistry
from nanobot.memory.hindsight_memory import HindsightMemoryAlgorithm

registry = MemoryRegistry()
registry.register(HindsightMemoryAlgorithm())

algo = registry.get("hindsight_memory")
components = algo.build(
    workspace=Path("./agent_workspace"),
    provider=llm_provider,        # 提供 embed() 用于语义检索
    model="gpt-4o",
    sessions=session_manager,
    context_window_tokens=128_000,
    build_messages=build_messages_fn,
    get_tool_definitions=get_tool_definitions_fn,
    max_completion_tokens=4096,
    session_ttl_minutes=60,
    max_batch_size=20,
    max_iterations=10,
    max_tool_result_chars=16_000,
    annotate_line_ages=True,
)

store = components.store
assert store.hindsight_enabled  # 始终 True，内置实现

# 手动 retain（同步包装器）
store.retain("用户喜欢用 TypeScript", context="session-2024-001")

# 搜索记忆（异步）
import asyncio
result = asyncio.run(store.arecall("TypeScript 相关记忆", max_tokens=2048))

# 深度推理（异步）
reflection = asyncio.run(store.areflect(
    "分析这些历史与当前记忆的关系",
    budget="mid",
    context="...历史上下文..."
))
```

### 构造参数

```python
store = HindsightStore(
    workspace=Path("./workspace"),
    provider=llm_provider,          # 可选：提供 embed() 用于语义检索
    embedding_model="text-embedding-3-small",  # embedding 模型名
    max_memories=10_000,            # 本地记忆库最大条目数
    max_history_entries=1000,       # history.jsonl 最大条目数
)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `provider` | None | LLM Provider，提供 `embed()` 用于语义检索 |
| `embedding_model` | 同 chat model | embedding 模型名称 |
| `max_memories` | 10,000 | 本地 TEMPR 记忆库最大条目数 |
| `max_history_entries` | 1000 | history.jsonl 最大条目数 |
| `context_window_tokens` | — | LLM 上下文窗口大小 |
| `max_completion_tokens` | 4096 | 最大输出 token 数 |
| `session_ttl_minutes` | 60 | AutoCompact 空闲会话 TTL |
| `max_batch_size` | 20 | Dream 每次处理的最大历史条目数 |
| `max_iterations` | 10 | Dream Phase 2 AgentRunner 最大迭代次数 |
| `max_tool_result_chars` | 16,000 | 工具结果最大字符数 |
| `annotate_line_ages` | True | 是否在 MEMORY.md 中标注陈旧行 |
| `stale_threshold_days` | 14 | 行龄标注阈值（超过此天数标注 `← Nd`） |

**TEMPR 内部参数**（硬编码，可调）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `_RRF_K` | 60 | RRF 融合常数（与官方一致） |
| `_TEMPORAL_HALF_LIFE_DAYS` | 30.0 | 时间衰减半衰期（legacy） |
| `_GRAPH_EXPANSION_BUDGET` | 30 | 图扩展种子扩散上限 |
| `_SEMANTIC_KNN_K` | 5 | 每条记忆保留的 kNN 近邻数 |
| `_SEMANTIC_KNN_THRESHOLD` | 0.70 | kNN 链接的最低余弦相似度 |
| `_RECENCY_ALPHA` | 0.20 | Post-RRF 时新性增强系数 |
| `_PROOF_COUNT_ALPHA` | 0.10 | Post-RRF 证据计数增强系数 |
| `_CAUSAL_BOOST` | 2.0 | 因果链 (causes/caused_by) 图扩展增强 |
| `_ENTITY_BOOST` | 0.5 | 实体共现 图扩展增强 |
| `_SEMANTIC_KNN_BOOST` | 0.7 | 语义 kNN 图扩展增强 |
| `_CONTEXT_BOOST` | 0.3 | 上下文关联 图扩展增强 |

## 与 naive_memory 的对比

| 特性 | naive_memory | hindsight_memory |
|------|:---:|:---:|
| 存储格式 | MEMORY.md 纯文本 | 同 naive + 本地 JSON TEMPR 库 |
| 记忆检索 | 全文读取 MEMORY.md | TEMPR 四引擎 RRF 融合检索 |
| 融合算法 | — | Reciprocal Rank Fusion (k=60) |
| 图扩展 | ❌ | ✅ 实体共现 + 语义 kNN + 因果链 |
| 时态检索 | ❌ | ✅ 时间窗口邻近度 + 因果链扩散 |
| 事实类型分层 | ❌ | ✅ world/experience/observation/opinion |
| 观察自动合并 | ❌ | ✅ consolidate() 共享实体分组 |
| Dream Phase 1 | 纯 LLM 分析 | TEMPR reflect + LLM fallback |
| 语义搜索 | ❌ | ✅ 向量余弦相似度（需 provider.embed） |
| 关键词搜索 | 人工 grep | ✅ 纯 Python BM25 |
| 时间检索 | ❌ | ✅ 时间窗口邻近度 |
| 跨会话关联 | ❌ | ✅ 图扩展 + 因果链推理 |
| Hermes 中轮提取 | ❌ | ✅ 逐条事实提取 + retain |
| Git 行龄标注 | ✅ | ✅ 完全继承 |
| 外部依赖 | 零 | 零（纯 Python 实现） |
| 离线可用性 | ✅ | ✅ 始终可用 |

## 设计哲学

1. **零依赖内置**：无需外部服务器、无需 `hindsight_client`、纯 Python 标准库 + provider.embed()
2. **静默降级**：无 provider 时语义搜索不可用，其他引擎正常；provider 恢复后自动补全
3. **双写共存**：文件层是真理源（source of truth），TEMPR 层是检索加速器
4. **RRF 融合**：用 Reciprocal Rank Fusion 替代手动调权，排名即信号，对各引擎分数分布鲁棒
5. **图扩展增强**：实体共现 + 语义 kNN 预计算 + 因果链启发式检测，模拟官方 link-expansion retrieval
6. **渐进检索**：budget 控制启用哪些引擎，按需在精度和性能间权衡
7. **Dream 优先 TEMPR**：Phase 1 优先用 TEMPR reflect 做深度分析，比纯 LLM 推理能发现更多跨会话关联

## 参考文献

1. Hindsight Documentation. https://hindsight.vectorize.io/
2. Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
3. Cormack, Clarke & Buettcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. *SIGIR*.
4. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
5. Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.
