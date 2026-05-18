# Supermemory 记忆算法

> 基于 [Supermemory Research](https://supermemory.ai/research/) 论文架构实现的本地化长期记忆引擎，零外部依赖。

## 核心思想

标准 RAG（检索增强生成）在长期对话场景中容易失效，因为检索到的原始片段脱离对话上下文后存在歧义。LLM 对嵌入在长上下文中间的信息也容易遗忘（"Lost in the Middle"）。

Supermemory 通过以下五个核心机制解决上述问题，在 LongMemEval_s 基准测试中达到 85.2%（gemini-3-pro）的 SOTA 准确率：

| 类别 | 全上下文(gpt-4o) | Zep(gpt-4o) | Supermemory(gpt-4o) | Supermemory(gemini-3-pro) |
|------|:-:|:-:|:-:|:-:|
| 单会话-用户 | 81.4% | 92.9% | **97.14%** | **98.57%** |
| 单会话-助手 | 94.6% | 80.4% | **96.43%** | **98.21%** |
| 单会话-偏好 | 20.0% | 56.7% | **70.00%** | **70.00%** |
| 知识更新 | 78.2% | 83.3% | **88.46%** | **89.74%** |
| 时序推理 | 45.1% | 62.4% | **76.69%** | **81.95%** |
| 多会话 | 44.3% | 57.9% | **71.43%** | **76.69%** |
| **总体** | **60.2%** | **71.2%** | **81.6%** | **85.2%** |

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                   Supermemory 记忆算法流水线                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  对话消息 ──► Consolidator ──► history.jsonl                     │
│                    │                                             │
│                    ▼                                             │
│              分块（Chunking）                                     │
│                    │                                             │
│                    ▼                                             │
│              原子记忆生成 ──► Memory Graph (nodes + edges)        │
│                    │              │                              │
│                    ▼              ▼                              │
│              Chunks/ 目录    关系检测 (updates/extends/derives)    │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  Cron 定时 ──► Dream (Phase 1) ──► LLM 分析                      │
│                        │                                         │
│                        ▼                                         │
│               Dream (Phase 2) ──► AgentRunner 编辑 MEMORY.md      │
│                                    SOUL.md / USER.md              │
│                                    skills/dreamed-*/SKILL.md     │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  空闲会话 ──► AutoCompact ──► 压缩 + 记忆图更新                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 五大核心机制

### 1. 分块摄入与上下文消歧 (Chunk-based Ingestion)

**问题**：标准 RAG 检索到的原始片段脱离对话上下文后存在指代歧义（如 "它"、"那个项目" 等）。

**方案**：将大型会话按用户轮次边界分解为可管理的语义块，然后为每个块生成**原子记忆**——单个信息单元，在块内完成指代消解。

```python
# 分块：在用户轮次边界处切割，保持语义完整
messages = [
    {"role": "user", "content": "我喜欢用 Python"},      # ← 新块开始
    {"role": "assistant", "content": "Python 是个好选择"},
    {"role": "user", "content": "帮我部署到生产环境"},     # ← 新块开始
    {"role": "assistant", "content": "部署完成"},
]
# → 两个语义块，分别生成原子记忆
```

### 2. 关系版本化 (Relational Versioning)

三种语义关系追踪记忆间的演化：

| 关系类型 | 含义 | 示例 |
|----------|------|------|
| `updates` | 状态变更（创建版本链） | "喜欢的颜色是蓝色" → "喜欢的颜色是绿色" |
| `extends` | 补充细节（不矛盾） | "在 Acme 工作" → "职位是高级工程师" |
| `derives` | 推理（组合多记忆） | "生于巴黎" + "讲法语" → "可能是法国国籍" |

当一条事实被更新时，旧版本**保留为历史记录**而非删除：

```
[Memory v1: "喜欢蓝色"] ← is_latest=False
        ↑ updates
[Memory v2: "喜欢绿色"] ← is_latest=True, parent=v1, root=v1
```

### 3. 时序锚定 (Temporal Grounding)

双重时间戳让系统能够进行时序推理：

- **documentDate**：对话发生的时间（来自消息时间戳）
- **eventDate**：所描述事件实际发生的时间（从对话内容中提取）

```python
node = MemoryNode(
    memory="用户在 2024 年搬到了旧金山",
    document_date="2026-05-09",   # 这条对话今天的
    event_date="2024-03-15",      # 搬家发生在 2024 年
)
```

这直接驱动了 LongMemEval 中 76.69% 的时序推理得分。

### 4. 混合搜索 (Hybrid Search)

搜索流程分两步：

1. **语义搜索**：对原子记忆做语义匹配——原子记忆是高信号、低噪声的，搜索精度远高于直接搜原始对话块
2. **源块注入**：找到匹配记忆后，将其关联的**原始对话块**一并返回，让 LLM 能获取完整上下文

```
用户查询："我上次说的那个 bug 怎么样了？"
     │
     ▼
语义搜索记忆节点 → 命中: "User reported auth bug on 2026-04-15"
     │
     ▼
注入源块 → "User: 登录接口有个竞态条件... Assistant: 我看到了..."
     │
     ▼
LLM 获得完整上下文，精准回答
```

### 5. 会话级摄入 (Session-Based Processing)

与逐轮处理不同，Supermemory 按**会话**摄入对话历史。这让系统能够：
- 跨越多轮对话检测模式
- 在一次处理中完成整个会话的上下文消歧
- 减少重复的关系检测开销

## 数据模型

### MemoryNode — 原子记忆节点

```python
@dataclass
class MemoryNode:
    id: str                    # 唯一标识
    memory: str                # 记忆文本（消歧后的原子事实）
    content: str               # 原始源块内容（用于混合搜索）
    document_date: str         # 对话时间
    event_date: str | None     # 事件实际时间
    version: int               # 版本号
    is_latest: bool            # 是否最新版本
    parent_memory_id: str | None  # 版本链中的前驱
    root_memory_id: str | None    # 版本链的根
    embedding: list[float] | None # 向量嵌入（可选）
```

### MemoryEdge — 关系边

```python
@dataclass
class MemoryEdge:
    source_id: str      # 源节点
    target_id: str      # 目标节点
    edge_type: MemoryRelation  # updates | extends | derives
```

### SourceChunk — 源对话块

```python
@dataclass
class SourceChunk:
    id: str                # 唯一标识
    content: str           # 原始块文本
    document_date: str     # 时间戳
    memory_ids: list[str]  # 关联的记忆节点 ID
```

## 存储布局

```
workspace/
├── SOUL.md                    # 灵魂配置
├── USER.md                    # 用户配置
└── memory/
    ├── MEMORY.md              # 格式化长期记忆（供 LLM 上下文注入）
    ├── history.jsonl          # 对话历史（append-only JSONL）
    ├── memory_graph.json      # 记忆图（节点 + 边）
    ├── .cursor                # 历史游标
    ├── .dream_cursor          # Dream 处理游标
    └── chunks/                # 源对话块
        ├── chunk_<uuid1>.json
        └── chunk_<uuid2>.json
```

## 四大组件

### SupermemoryStore

继承自 `MemoryStore`，扩展了图存储、块存储和关系管理。

**关键方法**：
| 方法 | 功能 |
|------|------|
| `add_node()` / `get_node()` | 记忆节点 CRUD |
| `add_edge()` | 添加关系边 |
| `create_new_version()` | 创建新版本（updates 链） |
| `extend_memory()` | 扩展记忆（extends） |
| `derive_memory()` | 推理新记忆（derives） |
| `add_chunk()` / `get_chunks_for_memory()` | 源块管理 |
| `search_memories_by_keyword()` | 关键词搜索 |
| `get_memory_context()` | 构建 LLM 上下文注入文本 |
| `stats()` | 图统计（节点数、边数、版本链数等） |

### SupermemoryConsolidator

在线令牌预算触发的压缩器，在 naive Consolidator 基础上增加了：

1. **分块归档**：`_chunk_messages()` 将消息按用户轮次边界切分为语义块
2. **记忆生成**：`_generate_memories_from_chunk()` 从每块生成原子记忆节点
3. **关系检测**：`_detect_relationships()` 比较新记忆与现有记忆，自动建立 updates/extends/derives 边
4. **源块存储**：每块作为 `SourceChunk` 保存到 `chunks/` 目录

### SupermemoryDream

Cron 定时触发的深度记忆处理器，两阶段流水线：

**Phase 1 — 分析**：
- 读取 `history.jsonl` 中未处理的条目
- 注入当前 `MEMORY.md` / `SOUL.md` / `USER.md` 内容
- **注入记忆图上下文**（最新记忆、关系统计）
- 注入行龄标注（`← 30d` 标记陈旧信息）
- LLM 分析生成摘要

**Phase 2 — 执行**：
- 通过 `AgentRunner` 调用 `read_file` / `edit_file` 工具
- 增量编辑 `MEMORY.md`（不替换整个文件）
- 可创建 `skills/dreamed-*/SKILL.md`
- 自动 Git 提交变更

### SupermemoryAutoCompact

空闲会话的主动压缩器：
- 检测超过 TTL 的非活跃会话
- 调用 `SupermemoryConsolidator.archive()` 进行分块压缩+记忆图更新
- 保留最近 8 条消息作为上下文尾巴

## 使用方式

```python
from summerclaw.memory.registry import MemoryRegistry
from summerclaw.memory.supermemory_memory import SupermemoryMemoryAlgorithm

registry = MemoryRegistry()
registry.register(SupermemoryMemoryAlgorithm())

algo = registry.get("supermemory_memory")
components = algo.build(
    workspace=Path("./agent_workspace"),
    provider=llm_provider,
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

# 组件
store = components.store          # SupermemoryStore
consolidator = components.consolidator  # SupermemoryConsolidator
dream = components.dream           # SupermemoryDream
auto_compact = components.auto_compact  # SupermemoryAutoCompact
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_batch_size` | 20 | Dream 每次处理的最大历史条目数 |
| `max_iterations` | 10 | Dream Phase 2 AgentRunner 最大迭代次数 |
| `max_tool_result_chars` | 16,000 | 工具结果最大字符数 |
| `annotate_line_ages` | True | 是否在 MEMORY.md 中标注行龄 |
| `session_ttl_minutes` | 60 | AutoCompact 会话过期时间（分钟） |
| `max_nodes` | 5,000 | 记忆图最大节点数（超限时遗忘节点优先移除） |

## 与 naive_memory 的对比

| 特性 | naive_memory | supermemory_memory |
|------|:---:|:---:|
| 存储格式 | MEMORY.md 纯文本 | MEMORY.md + 记忆图 + 源块 |
| 记忆粒度 | 段落级 | 原子级（单个事实） |
| 版本追踪 | 无（覆盖写入） | 版本链（保留历史） |
| 关系语义 | 无 | updates / extends / derives |
| 时序信息 | 仅对话时间 | documentDate + eventDate |
| 搜索方式 | 全文读取 | 关键词 + 语义（可选嵌入） |
| 指代消歧 | 依赖 LLM | 块内上下文消歧 |
| 知识冲突 | LLM 自行判断 | 版本链明确标识 |

## 参考文献

1. Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.
2. Wu et al. (2024). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. *arXiv:2410.10813*.
3. Rasmussen et al. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. *arXiv:2501.13956*.
4. Keluskar et al. (2024). Do LLMs Understand Ambiguity in Text? *IEEE BigData*.
5. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
6. Barnett et al. (2024). Seven Failure Points When Engineering a Retrieval Augmented Generation System. *IEEE/ACM AI Engineering*.
7. Supermemory Research. https://supermemory.ai/research/
