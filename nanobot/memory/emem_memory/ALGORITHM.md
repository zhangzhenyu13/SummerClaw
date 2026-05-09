# EMem 记忆算法

> EMem（Elementary Discourse Unit Memory）——基于基本话语单元的图结构化对话记忆，融合 OpenIE 信息抽取 + 稠密向量检索 + 异构图 PPR 推理。

## 核心思想

大多数记忆系统将对话历史压缩为文本摘要，这在问答时依赖 LLM 自己"回忆"相关上下文。EMem 另辟蹊径：**将对话分解为不可再分的原子命题（EDU），再通过图结构将它们关联起来，使记忆检索变成精确的数据查询而非模糊的上下文注入。**

核心创新：
1. **EDU 分解**：用 LLM（OpenIE 范式）将对话轮次拆解为 Elementary Discourse Units——独立的、完整的事实陈述
2. **异构图记忆**：构建 Session → EDU → Argument 三层异构图，EDU 之间通过共引的 Argument 节点间接关联
3. **PPR 联想召回**：Personalized PageRank 在图上游走，自动发现"表面上无关但语义上关联"的记忆

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                      EMem 记忆算法流水线                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  对话消息                                                         │
│     │                                                            │
│     ├──► Consolidator (在线)                                      │
│     │        │                                                   │
│     │        ▼                                                   │
│     │    EDUExtractor ──► LLM OpenIE                             │
│     │        │                                                   │
│     │        ▼                                                   │
│     │    EDURecord[] ──► ContentStore[EDU]                       │
│     │        │            (pickle + parquet 嵌入)                 │
│     │        ▼                                                   │
│     │    EMemGraph.add_nodes(edu_ids, "EDU")                     │
│     │                                                                  │
│     └──► Dream (离线 Cron)                                        │
│              │                                                   │
│              ▼                                                   │
│          Phase 1: LLM 分析 history.jsonl                          │
│              │                                                   │
│              ├──► 提取 EDUs ──► ContentStore                      │
│              ├──► 更新 EMemGraph (节点+边)                         │
│              ├──► 构建近义边 (synonymy edges via KNN)              │
│              │                                                   │
│              ▼                                                   │
│          Phase 2: AgentRunner 编辑 MEMORY.md                      │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  检索时 (实时):                                                    │
│    查询 ──► EMemEmbedder ──► 稠密检索 KNN                         │
│                │                                                  │
│                ▼                                                  │
│            PPR 图传播 ──► 联想扩展                                  │
│                │                                                  │
│                ▼                                                  │
│            EDUReranker ──► LLM 语义过滤                            │
│                │                                                  │
│                ▼                                                  │
│            ArgumentReranker ──► LLM 参数过滤                       │
│                │                                                  │
│                ▼                                                  │
│            排序后的记忆注入 LLM 上下文                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 数据模型

### 三层数据实体

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ SessionRecord │     │    EDURecord      │     │  ArgumentRecord    │
├──────────────┤     ├──────────────────┤     ├───────────────────┤
│ session_id   │     │ edu_id (MD5)      │     │ arg_id (MD5)       │
│ turns[]      │────►│ text (原子命题)    │────►│ text (实体/值)      │
│ summary      │     │ source_speakers   │     │ source_edu_ids[]   │
│ date         │     │ timestamp         │     └───────────────────┘
│              │     │ session_id        │
└──────────────┘     │ event_type        │
                     │ event_triggers    │
                     │ role_argument_pairs│  ← [{role, argument}]
                     │ context_text      │
                     └──────────────────┘
```

### EDURecord — 原子命题

```
对话: "我已经在 /home/prod 部署了 v2.1，监听 8080 端口"
            ↓ EDUExtractor (OpenIE)
EDU 1: 部署位置是 /home/prod
  event_type: "deployment"
  role_argument_pairs: [{"role": "location", "argument": "/home/prod"}]

EDU 2: 版本是 v2.1
  event_type: "version_info"
  role_argument_pairs: [{"role": "version", "argument": "v2.1"}]

EDU 3: 监听端口是 8080
  event_type: "network_config"
  role_argument_pairs: [{"role": "port", "argument": "8080"}]
```

### ArgumentRecord — 实体锚点

EDU 中的每个 `argument`（实体值）被提取为图中的独立节点：

```
Argument "/home/prod"  ← 连接所有提及此路径的 EDU
Argument "8080"        ← 连接分布式端口引用
Argument "v2.1"        ← 版本号踪迹
```

## 五大核心组件

### 1. EMemEmbedder

向量嵌入生成器——支持两种模式：
- **OpenAI API**：`text-embedding-3-small` 等在线模型
- **本地模型**：通过 `sentence-transformers` 离线嵌入

### 2. ContentStore — 通用内容存储

通用的 pickle + parquet 混合存储：

```
emem_storage/
├── content_edu.pkl         # EDU 对象 (pickle)
├── embeddings_edu.parquet  # EDU 嵌入向量 (parquet)
├── content_argument.pkl
├── embeddings_argument.parquet
├── content_session.pkl
└── (session 无嵌入)
```

特点：
- 基于 MD5 的自动去重（相同文本只存一份）
- 批量嵌入生成以减少 API 调用
- 索引内存常驻以加速检索

### 3. EDUExtractor

LLM 驱动的 OpenIE 提取器——使用 OpenAI Responses API 的结构化输出，将对话文本分解为原子命题+事件类型+角色-参数对。

可配置 `skip_edu_context_gen` 跳过上下文生成（减少 token 消耗）。

### 4. EMemGraph — 异构图 + PPR

三层节点类型的异构图：

```
        Session: "session-abc123"
             │
    ┌────────┼────────┐
    │        │        │
  EDU-1    EDU-2    EDU-3     ← 共现于同一会话
    │        │        │
    ├────────┤        │
    │  port:8080      │        ← Argument 节点 (EDU 间桥梁)
    │        │        │
    └── path:/home    │
             │        │
           EDU-5    EDU-8      ← 不同会话中提及同一路径
```

**PPR (Personalized PageRank)** 在图上游走以发现"联想记忆"：

```
查询: "部署配置"
  → 检索到 EDU-2 (版本 v2.1) 和 EDU-3 (端口 8080)
  → PPR 从这些节点出发，通过 Argument 桥接
  → 发现 EDU-8 (另一个会话中也用了 8080)
  → 返回完整的部署上下文链
```

实现双引擎：
- **igraph**（优先）：C 实现，速度快
- **scipy**（回退）：纯 Python 幂迭代，无额外依赖

### 5. Reranker — LLM 语义过滤

检索得到的候选 EDU 和 Argument 可能很多（`retrieval_top_k=200`），需要 LLM 做最终语义过滤：

- **EDUReranker**：判断每个 EDU 是否真的与查询相关
- **ArgumentReranker**：判断每个参数是否为查询关注的实体

## 存储布局

```
workspace/
├── SOUL.md
├── USER.md
├── MEMORY.md
└── memory/
    ├── history.jsonl
    ├── .cursor
    ├── .dream_cursor
    └── emem/
        ├── edu_storage/
        │   ├── content_edu.pkl
        │   └── embeddings_edu.parquet
        ├── argument_storage/
        │   ├── content_argument.pkl
        │   └── embeddings_argument.parquet
        ├── session_storage/
        │   └── content_session.pkl
        └── emem_graph.pkl
```

## 四大生命周期组件

### EMemConsolidator

扩展 naive 的 `archive()`：在 LLM 摘要之后额外调用 `EDUExtractor.extract_from_history()` 对被归档的消息进行 EDU 提取并索引。

### EMemDream

Cron 触发的离线处理——除了标准 Phase 1/2（分析 + AgentRunner 编辑），还包含：
- EDU 提取并索引到 ContentStore
- 更新 EMemGraph（新增 Session/EDU 节点、边）
- 构建 Argument 近义边（synonymy edges via KNN）
- 保存图 pickle

### EMemAutoCompact

标准的 idle session 压缩器。

## 图近义边构建

当 Argument 节点积累到一定数量，系统通过 KNN 余弦相似度构建近义边：

```
Argument "8080"   ←(sim=0.92)→  Argument "port 8080"
Argument "prod"   ←(sim=0.88)→  Argument "production"
```

这些边使得 PPR 传播时可以跨越文本表面差异找到语义等价的实体。

## 使用方式

```python
from nanobot.memory.registry import MemoryRegistry
from nanobot.memory.emem_memory import EMemMemoryAlgorithm
from nanobot.memory.emem_memory.datatypes import EMemConfig

config = EMemConfig(
    retrieval_top_k=200,
    damping=0.5,
    skip_ppr=False,    # 启用 PPR 图传播
)

registry = MemoryRegistry()
registry.register(EMemMemoryAlgorithm(config=config))

algo = registry.get("emem_memory")
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

## 配置参数 (EMemConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `linking_top_k` | 5 | 链接后保留的 top 候选项数 |
| `retrieval_top_k` | 200 | 每次查询检索的 EDU 数 |
| `damping` | 0.5 | PPR 阻尼因子 (0–1) |
| `synonymy_edge_topk` | 2047 | KNN 近义边的 K 值 |
| `synonymy_edge_sim_threshold` | 0.8 | 近义边相似度阈值 |
| `skip_ppr` | False | 跳过 PPR（仅用稠密检索） |
| `skip_edu_context_gen` | True | 跳过 EDU 上下文生成 |
| `force_reindex` | False | 强制重建索引 |

## 可选依赖

```bash
pip install nanobot-ai[emem]    # igraph, sentence-transformers, torch, scipy
```

## 与其他算法的对比

| 特性 | naive | nemori | emem |
|------|:---:|:---:|:---:|
| 记忆粒度 | 文本级 | Episode 级 | 原子命题级 |
| 结构化 | 无 | JSON | 图 (pickle+parquet) |
| 检索方式 | 全文读取 | 文本/向量搜索 | 稠密 KNN + PPR 图 |
| 关系推理 | 依赖 LLM | 依赖 LLM | 图算法保证 |
| 引入开销 | 最低 | 中 (多次 LLM 调用) | 高 (嵌入+图构建) |
| 外部依赖 | 无 | 文件模式无 | igraph/scipy/pandas |
