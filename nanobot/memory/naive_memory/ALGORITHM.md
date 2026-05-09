# NaiveMemory 算法

> nanobot 的默认记忆算法——基于文件的最简实现，零外部依赖，所有高级算法的基准线。

## 核心思想

NaiveMemory 是"记忆算法的 Hello World"——用最直接的工程直觉解决对话记忆问题，没有论文理论包装，但完整诠释了记忆系统该有的四大生命周期阶段。

它回答一个朴素问题：**一个 AI Agent 如何记住过去、在合适时机压缩历史、并定期反思并写入长期记忆？**

答案是把记忆管理拆成三个时间维度上的独立流程：**在线压缩**（每轮对话触发）、**离线反思**（Cron 定时触发）、**闲置压缩**（空闲会话触发）。

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    NaiveMemory 记忆算法流水线                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  对话消息 ──► Consolidator ──► LLM 摘要 ──► history.jsonl          │
│    │              │                                               │
│    │      令牌预算检查 (token_budget)                               │
│    │      超过窗口则归档最旧消息                                     │
│    │                                                              │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  Cron 定时 ──► Dream (Phase 1) ──► LLM 分析 history.jsonl         │
│                        │                                          │
│                        ▼                                          │
│               Dream (Phase 2) ──► AgentRunner 编辑 MEMORY.md       │
│                                    SOUL.md / USER.md               │
│                                    skills/dreamed-*/SKILL.md      │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  空闲会话 ──► AutoCompact ──► 压缩旧消息 + 写入 history.jsonl       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 存储布局

```
workspace/
├── SOUL.md              # Agent 身份/个性/行为准则
├── USER.md              # 用户画像/偏好
└── memory/
    ├── MEMORY.md        # 长期记忆（供 LLM 上下文注入）
    ├── history.jsonl    # 对话历史（append-only JSONL）
    ├── .cursor          # 归档游标
    └── .dream_cursor    # Dream 处理游标
```

## 四大组件

### MemoryStore

文件 I/O 层——整个算法的唯一数据入口：

| 方法 | 功能 |
|------|------|
| `read_memory()` / `write_memory()` | 长期记忆文件 |
| `read_soul()` / `write_soul()` | Agent 身份文件 |
| `read_user()` / `write_user()` | 用户画像文件 |
| `append_history()` | 追加到 history.jsonl，生成游标 |
| `read_unprocessed_history()` | 按游标增量读取 |
| `compact_history()` | 按条目数上限裁剪 |
| `get_memory_context()` | 构建 LLM 上下文注入文本 |
| `git` (GitStore) | Git 行龄标注 + 自动提交 |

特点：
- 所有文件均为纯文本，人可直接阅读编辑
- 基于游标（cursor）实现增量处理，支持断点续传
- GitStore 集成：每次 MEMORY.md 修改可自动 git commit，并能按行标注修改时间

### Consolidator

在线令牌预算驱动的压缩器，核心职责是**在 LLM 会话还活跃时，避免上下文窗口溢出**。

**工作原理**：

1. **估算**：每轮对话前，计算当前会话消息+系统提示词+工具定义的 token 总数
2. **比较**：若预估 token 数 > 上下文窗口 - 最大输出 - 安全缓冲（1024 tokens），触发压缩
3. **归档**：将最旧的一批消息（默认 20 条）发送给 LLM 做摘要
4. **存储**：摘要写入 history.jsonl，原始消息从会话中移除

设计要点：`estimate_prompt_tokens_chain()` 支持两种估算策略——优先用 LLM provider 的本地 tokenizer（如果 `provider.estimate_tokens_chain` 可用），否则回退到简单的字符长度估算 `chars/4`。

### Dream

Cron 定时触发的深度记忆处理器，是**离线反思**的核心。两阶段流水线：

**Phase 1 — 分析（纯 LLM 推理）**：
- 读 history.jsonl 中未处理的条目（游标增量）
- 注入当前 MEMORY.md / SOUL.md / USER.md 完整内容
- **注入行龄标注**：每行末尾标注 `← Nd`，标记超过 14 天的陈旧内容
- LLM 分析："哪些历史值得写入长期记忆？MEMORY.md 该增删改什么？要不要创建 dreamed-* 技能？"

**Phase 2 — 执行（AgentRunner + 工具）**：
- 将 Phase 1 分析结果作为 prompt 发给 AgentRunner
- AgentRunner 获得 `read_file` / `edit_file` / `SkillPrefixWriteFileTool` 三个工具
- 自主决定如何编辑 MEMORY.md / SOUL.md / USER.md
- 可创建 `skills/dreamed-*/SKILL.md` 固化重复性工作流
- 处理完推进游标，自动裁剪 history.jsonl

### AutoCompact

空闲会话的主动压缩器——

- 定时扫描所有会话，找出超过 TTL（默认 60 分钟）且未活跃的非主会话
- 将此类会话的消息切片：保留最近 8 条作为上下文尾巴，其余全部交给 Consolidator 归档
- 写回压缩后的会话，释放内存和磁盘

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
   │       │              │
   │      No              │
   │       │              │
   │       ▼              │
   │  继续正常对话         │
   └─────────────────────┘
                              (Cron 定时)
                                   │
                                   ▼
   ┌──────────────────────────────────────────┐
   │  Dream                                    │
   │  Phase 1: read_unprocessed_history()      │
   │           + MEMORY.md (行龄标注)          │
   │           → LLM 分析                      │
   │                                           │
   │  Phase 2: AgentRunner(analysis)           │
   │           + read_file / edit_file          │
   │           → 编辑 MEMORY.md / SOUL.md 等     │
   │           → 创建 dreamed-* 技能             │
   │           → git commit                     │
   └──────────────────────────────────────────┘

   ┌──────────────────────────────────────────┐
   │  AutoCompact (定时)                       │
   │  扫描空闲会话 → 压缩旧消息 → history.jsonl  │
   └──────────────────────────────────────────┘
```

## 设计哲学

1. **简单即美**：所有存储都是纯文本文件，不会出现"数据库损坏"问题，用户随时可手动编辑
2. **增量处理**：游标机制保证 Dream 不会重复处理同一批历史
3. **令牌预算驱动**：Consolidator 是被动的——只有当真正需要时才压缩，不会过度介入
4. **两阶段 Dream**：分离分析（LLM 读）和执行（AgentRunner 写），让每个阶段职责单一

## 使用方式

```python
from nanobot.memory.registry import MemoryRegistry
from nanobot.memory.naive_memory import NaiveMemoryAlgorithm

registry = MemoryRegistry()
registry.register(NaiveMemoryAlgorithm())

algo = registry.get("naive_memory")
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
| `context_window_tokens` | — | LLM 上下文窗口大小 |
| `max_completion_tokens` | 4096 | 最大输出 token 数 |
| `session_ttl_minutes` | 60 | AutoCompact 空闲会话 TTL |
| `max_batch_size` | 20 | Dream 每次处理的最大历史条目数 |
| `max_iterations` | 10 | Dream Phase 2 AgentRunner 最大迭代次数 |
| `max_tool_result_chars` | 16,000 | 工具结果最大字符数 |
| `annotate_line_ages` | True | 是否在 MEMORY.md 中标注陈旧行 |

## 局限与适用场景

- **适用**：单 Agent、中小规模对话量、需要人可理解和编辑的记忆文件
- **不适用**：海量多用户场景（需要数据库后端）、需要复杂知识图谱关联、需要向量检索
- 作为基准算法，所有其他算法都在 NaiveMemory 的基础上叠加更高级的理论
