# Nemori 记忆算法

> 基于 [nemori](https://github.com/nemori-ai/nemori) 的自组织长期记忆系统，零外部依赖（文件模式）或可选 PostgreSQL + Qdrant 后端。

## 核心思想

传统对话记忆系统面临两大难题：**如何切分无限长的对话流**（边界对齐）和**如何从已知知识中提取增量价值**（预测-校准）。

Nemori 用两个耦合的控制回路解决：

1. **Two-Step Alignment**（边界对齐 + 表征对齐）：先用 LLM 将消息按**话题**分割为 episode 组，再为每个 episode 生成结构化的**情景叙事**（谁、什么时候、做了什么、为什么）
2. **Predict-Calibrate Learning**（预测-校准学习）：先让 LLM 根据已有知识**预测**当前 episode 中可能发生什么，再将预测与实际对话**对比**，从差异中提取高价值的**新语义知识**

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    Nemori 记忆算法流水线                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  消息 ──► message_buffer.jsonl                                    │
│                │                                                  │
│                ▼                                                  │
│         BatchSegmenter ──► LLM 话题边界检测                        │
│                │                                                  │
│                ▼                                                  │
│         [{messages, topic}, ...]  ← 消息组                         │
│                │                                                  │
│                ▼                                                  │
│         EpisodeGenerator ──► LLM 情景叙事生成                      │
│                │                       │                          │
│                ▼                       ▼                          │
│         episodes.json         EpisodeMerger                       │
│                │              合并重复 episode                     │
│                ▼                                                  │
│         SemanticGenerator                                           │
│         ├── Predict: 用已有语义知识预测 episode                      │
│         ├── Calibrate: 对比预测与实际，提取差异                       │
│         └── Direct: 无已有知识时直接提取                             │
│                │                                                  │
│                ▼                                                  │
│         semantic_memories.json                                    │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  Cron 定时 ──► NemoriDream                                       │
│     Phase 1: 读取 episodes + semantics → LLM 分析                 │
│     Phase 2: AgentRunner 编辑 MEMORY.md + 创建 dreamed-* 技能      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 数据模型

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│    Message    │     │     Episode       │     │  SemanticMemory   │
├──────────────┤     ├──────────────────┤     ├───────────────────┤
│ message_id   │     │ id               │     │ id                │
│ role         │────►│ user_id          │────►│ user_id           │
│ content      │     │ title            │     │ content           │
│ timestamp    │     │ content          │     │ memory_type       │
│ metadata     │     │ source_messages  │     │ source_episode_id │
│              │     │ embedding        │     │ confidence        │
│              │     │ created_at       │     │ embedding         │
└──────────────┘     └──────────────────┘     └───────────────────┘
                                                      │
                                              知识分类：identity /
                                              preference / relationship /
                                              goal / belief / habit
```

## 核心机制

### 1. 边界对齐 — LLM 话题分割

`BatchSegmenter` 将消息按**话题边界**分割为独立的 episode 组。每 80 条消息为一个 chunk 发送给 LLM，LLM 返回 `{episodes: [{indices: [1,5,9], topic: "..."}]}` 格式的 JSON 分割方案。

这是 Nemori 最核心的创新：**不依赖固定长度或固定时间窗口，而是用 LLM 理解对话语义来判断"话题切换"的时机**。

### 2. 表征对齐 — 情景叙事生成

`EpisodeGenerator` 将每条消息组转化为结构化的情景叙事（episodic narrative）：

```
Title: "讨论 Python 项目部署"
Content: "用户在 2024-03-15 提出要将 Python 服务部署到生产环境。
         Assistant 建议使用 Docker + nginx 方案。用户同意但担心
         端口冲突。Assistant 通过 web_search 发现 8080 端口已被
         占用，建议改用 8081..."
```

叙事包含了：**参与者、时间锚定、动作序列、工具使用、决策理由**——这些是后续语义提取的基础。

### 3. 预测-校准学习

`SemanticGenerator` 实现了 Nemori 论文中的 Predict-Calibrate 循环：

```
Step 1 - Predict (预测):
  Prompt: "已知以下用户信息：[已有知识列表]
           请预测 episode '讨论 Python 项目部署' 中可能发生的事"
  → LLM: "用户可能会要求部署 Python 服务..."

Step 2 - Calibrate (校准):
  Prompt: "原始对话：[...]\n预测：[...]\n请提取预测与实际不符之处"
  → LLM: ["用户偏好使用 Docker Compose", "端口冲突是常见问题"]
```

**关键点**：预测-校准比直接提取更能发现"意料之外"但有价值的信息——因为 LLM 的预测本身就是一种注意力引导机制。

### 4. Episode 去重合并

`EpisodeMerger` 通过**相似搜索 + LLM 决策**避免语义重复：

- 文本搜索（或向量搜索，如 PG+Qdrant 后端可用）找到候选相似 episode
- LLM 判断：merge（合并为一个）还是 keep-separate（保持独立）
- 如果合并，LLM 生成合并后的标题和内容，删除旧 episode，保存合并版本

### 5. 统一搜索

`UnifiedSearch` 提供跨 episode 和 semantic memory 的统一检索接口，支持文本搜索和向量搜索两种模式。

## 存储布局

```
workspace/
├── SOUL.md
├── USER.md
├── MEMORY.md
└── memory/
    └── nemori/
        ├── episodes.json          # 所有 episode（JSON 数组）
        ├── semantic_memories.json # 所有语义记忆（JSON 数组）
        └── message_buffer.jsonl   # 消息缓冲（processed 标记）
```

可选 PG+Qdrant 后端（设置 `backend="postgres"`）：
- PostgreSQL：存储 episode 和 semantic memory 的结构化数据
- Qdrant：存储向量嵌入，支持语义搜索

## 六大组件

| 组件 | 职责 | LLM 调用 |
|------|------|:---:|
| `NemoriStore` | 文件/数据库读写 + 向量搜索 | — |
| `BatchSegmenter` | 话题边界分割 | ✓ |
| `EpisodeGenerator` | 情景叙事生成 | ✓ |
| `SemanticGenerator` | 预测-校准语义提取 | ✓✓ (两步) |
| `EpisodeMerger` | Episode 去重合并 | ✓ |
| `UnifiedSearch` | 跨数据源搜索 | — |

### NemoriConsolidator — 管线编排器

作为在线处理的核心，`NemoriConsolidator` 协调上述组件的执行顺序：

1. `ingest()` — 消息入缓冲，达到阈值（buffer_size_min=2）触发后台处理
2. 后台 `_process()` — 依次调用 segment → generate episodes → extract semantics → merge → 标记已处理
3. `flush()` — 强制立即处理所有缓冲消息

注意 Nemori 不使用传统的 `AutoCompact`（`auto_compact=None`），而是通过缓冲区清理（`compacted_buffer()`）自行管理。

### NemoriDream — 离线反思

Cron 定时从 `episodes.json` 和 `semantic_memories.json` 中读取最新数据，通过标准的 **Phase 1 分析 + Phase 2 AgentRunner 编辑** 流程写入 `MEMORY.md`。

与 naive Dream 的区别：Phase 1 的输入不再是 history.jsonl 的文本摘要，而是结构化的 episode 标题+内容和语义记忆列表，信息密度更高。

## 使用方式

```python
from nanobot.memory.registry import MemoryRegistry
from nanobot.memory.nemori_memory import NemoriMemoryAlgorithm

registry = MemoryRegistry()
registry.register(NemoriMemoryAlgorithm())

algo = registry.get("nemori_memory")
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
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `buffer_size_min` | 2 | 触发后台处理的最小缓冲消息数 |
| `batch_threshold` | 20 | 触发 BatchSegmenter 的最小消息数（否则单组） |
| `episode_min_messages` | 2 | 单 episode 最少消息数 |
| `enable_semantic` | True | 是否启用语义提取 |
| `enable_merging` | True | 是否启用 episode 去重合并 |
| `enable_prediction_correction` | True | 是否启用预测-校准（否则直接提取） |

## 与 naive_memory 的对比

| 特性 | naive_memory | nemori_memory |
|------|:---:|:---:|
| 消息组织 | 等长批量归档 | LLM 话题分割 |
| 记忆粒度 | LLM 自由摘要 | 结构化 Episode + SemanticMemory |
| 知识提取 | Dream Phase 1 LLM | Predict-Calibrate 双步提取 |
| 去重 | 无 | EpisodeMerger 自动合并 |
| 搜索 | 全文读取 | UnifiedSearch 文本/向量 |
| 存储 | 纯文本文件 | JSON 文件 / PG+Qdrant |
| 外部依赖 | 无 | 文件模式零依赖 |

## 参考文献

- Nemori: Self-Organising Long-Term Memory for AI Agents. https://github.com/nemori-ai/nemori
- Two-Step Alignment framework for episodic memory formation
- Predict-Calibrate learning loop for incremental semantic knowledge extraction
