# MastraOM 记忆算法工作原理

> 基于 [Mastra Observational Memory](https://mastra.ai/research/observational-memory) 论文架构，为 summerclaw 完全重写，零外部依赖。在 LongMemEval 基准测试中以 gpt-5-mini 达到 **94.87%** SOTA 准确率。

---

## 1. 核心思想

传统记忆系统在 agent 运行中**动态检索**相关记忆——这导致每次对话轮次的上下文窗口都不同（不可缓存），且检索质量本身就是瓶颈。

MastraOM 提出了**观察记忆（Observational Memory）**范式：用两个后台 agent（Observer + Reflector）维护一份**密集观察日志**，逐步**替代**原始对话历史。核心洞察：

> **观察日志是稳定的、信息密度极高的上下文前缀——LLM prompt-cache 可以始终命中它，而原始消息的细节不会丢失——它们被 Observer 提炼成了结构化的观察。**

```
传统记忆:  每次检索 → 上下文变化 → 缓存失效 → 高延迟、高成本
MastraOM:  固定前缀（观察日志）+ 尾部未处理消息 → 上下文稳定 → 缓存始终命中
```

### 三代理架构

| 代理 | 角色 | 触发时机 | 类比 |
|------|------|----------|------|
| **Actor**（主 Agent） | 与用户对话、执行任务 | 每轮 | 意识 |
| **Observer** | 将原始消息转为结构化观察 | 未处理消息 tokens > 30,000 | 潜意识 |
| **Reflector** | 压缩凝练过长的观察日志 | 观察 tokens > 40,000 | 元记忆 |

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                  MastraOM 记忆算法流水线                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  对话消息 ──► Consolidator ──► Observer ──► OBSERVATIONS.md      │
│    │              │                  │                           │
│    │      令牌预算检查                ▼                           │
│    │    message_tokens>30k?    Reflector (tokens>40k)            │
│    │              │          渐进压缩 0→4 级                      │
│    │              ▼                                               │
│    │     ┌──────────────────┐                                     │
│    │     │ 原始消息          │→ history.jsonl  (Dream 分析)        │
│    │     │ Observer 摘要     │→ om-ops.jsonl  (运维追踪)           │
│    │     │ Observer 输出     │→ OBSERVATIONS.md                   │
│    │     └──────────────────┘                                     │
│    │                                                              │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  Cron 定时 ──► Dream (Phase 1) ──► LLM 分析                      │
│                        │           history.jsonl + OBSERVATIONS   │
│                        ▼                                         │
│               Dream (Phase 2) ──► AgentRunner 编辑 MEMORY.md      │
│                                    SOUL.md / USER.md              │
│                                    skills/dreamed-*/SKILL.md     │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  Hermes 触发 ──► extract_and_store() ──► Observer 提取事实         │
│                ┌─ 原始消息 → history.jsonl                       │
│                ├─ Observer 摘要 → om-ops.jsonl                   │
│                ├─ Observer 输出 → OBSERVATIONS.md                │
│                └─ SkillAutogen 读取 observations + 会话内存       │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  空闲会话 ──► AutoCompact ──► Observer 归档旧消息                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心机制

### 3.1 Observer — 消息→观察转换

Observer 负责将原始对话消息转换为密集的结构化观察。每次处理一批未归档的消息时：

1. **输入格式化**：`format_messages_for_observer()` 将消息转为时间分组的可读文本，标注角色（User/Assistant/Tool Call/Tool Result），截断过长内容
2. **去重指导**：传入已有观察日志，明确指示"不要重复已有的观察"
3. **LLM 提取**：Observer 系统提示词包含严格的观察提取指南

**Observer 输出的 XML 格式**：

```xml
<observations>
Date: May 9, 2025
* 🔴 (14:30) User prefers dark mode
* 🟡 (14:32) User might want notification support
  * -> ran git status, found 3 modified files
  * ✅ Auth feature completed
</observations>

<current-task>
- Primary: Implementing dark mode toggle
- Secondary: Notification settings (waiting for user)
</current-task>

<suggested-response>
Continue with dark mode implementation
</suggested-response>
```

**优先级体系**：

| 标记 | 优先级 | 含义 |
|------|--------|------|
| 🔴 | 高 | 用户显式事实、偏好、未解决目标、关键上下文 |
| 🟡 | 中 | 项目细节、学习到的信息、工具结果 |
| 🟢 | 低 | 次要细节、不确定的观察 |
| ✅ | 已完成 | 具体任务完成、问题已回答、问题已解决 |

**提取指南核心规则**：

- **区分断言与提问**：用户陈述（"我有两个孩子"）标记为断言；用户询问（"你能帮我做X吗？"）标记为问题
- **时序锚定**：每条观察有两个时间戳——对话时间和引用时间
- **用户断言优先**：当用户断言与后续提问冲突时，断言是答案
- **完成追踪**：使用 ✅ 回答"什么确切地完成了？"
- **避免重复**：不跨轮次重复相同观察，连续相似操作合并为父子观察

### 3.2 Reflector — 渐进压缩

当观察日志的 token 估算值超过 `observation_tokens_threshold`（默认 40,000）时，Reflector 触发凝练。

**渐进压缩 0→4 级**：

| 级别 | 指导 | 细节保留度 | 说明 |
|------|------|:---:|------|
| 0 | 无额外指导 | 100% | 首次尝试，正常压缩 |
| 1 | COMPRESSION REQUIRED | 8/10 | 轻度压缩，合并早期观察 |
| 2 | AGGRESSIVE COMPRESSION | 6/10 | 激进合并，移除冗余 |
| 3 | CRITICAL COMPRESSION | 4/10 | 最大化压缩，丢弃过程性细节 |
| 4 | EXTREME COMPRESSION | 2/10 | 极限压缩，仅保留关键决策和偏好 |

Reflector 的关键规则：
- 旧内容比新内容更激进地压缩（"recency bias"）
- 保留 ✅ 完成标记
- 用户断言优先于用户提问
- 必须是纯观察的凝练，不是总结

### 3.3 Token 预算驱动的触发

```
┌──────────────────────────────────────────────────────────┐
│               Consolidator 触发决策树                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  每轮对话 → estimate_session_prompt_tokens(session)        │
│       │                                                  │
│       ▼                                                  │
│   tokens < context_window - max_completion - 1024?        │
│       │                    │                             │
│      YES                  NO                             │
│       │                    │                             │
│       ▼                    ▼                             │
│   不做任何事         message_tokens > 30,000?              │
│                        │        │                        │
│                       YES      NO                        │
│                        │        │                        │
│                        ▼        ▼                        │
│                    Observer   跳过此轮                      │
│                   归档最旧消息                              │
│                        │                                 │
│                        ▼                                 │
│               observation_tokens > 40,000?                │
│                        │        │                        │
│                       YES      NO                        │
│                        │        │                        │
│                        ▼        ▼                        │
│                    Reflector   完成                        │
│                    渐进压缩                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 3.4 退化检测（Degenerate Repetition Detection）

Observer 和 Reflector 的输出都会经过退化检测——LLM 偶尔会陷入重复循环：

- **滑动窗口采样**：对输出采样 200 字符窗口，步长为 `len/50`
- **重复阈值**：若超过 40% 的窗口完全相同 → 标记为退化
- **超长行检测**：单行超过 50,000 字符 → 标记为退化
- 退化输出会触发回退：Observer 退化 → 原始消息转储；Reflector 退化 → 重试更高压缩级别

---

## 4. 数据格式

### 4.1 OBSERVATIONS.md（观察日志）

```
# Observational Memory

## Observation Cycle a1b2c3d4 — 2025-05-09 14:30
Date: May 9, 2025
* 🔴 (14:30) User prefers Python over Java for all projects
* 🟡 (14:32) User is building a REST API with FastAPI
  * -> read package.json, found fastapi==0.115.0
* ✅ (14:35) REST API scaffold created successfully

## Observation Cycle e5f6g7h8 — 2025-05-09 15:00
Date: May 9, 2025
* 🔴 (14:55) User wants PostgreSQL, not SQLite
* 🔴 (14:58) User explicitly stated: "no ORMs, raw SQL only"
```

### 4.2 history.jsonl（对话历史，append-only）

```jsonl
{"cursor": 1, "timestamp": "2025-05-09 14:30", "content": "User: I prefer Python"}
{"cursor": 2, "timestamp": "2025-05-09 14:35", "content": "[OM-OBSERVED] 5 messages → 342 chars of observations"}
```

---

## 5. 存储布局

```
workspace/
├── SOUL.md                    # Agent 身份/个性/行为准则
├── USER.md                    # 用户画像/偏好
└── memory/
    ├── OBSERVATIONS.md        # 观察日志（Observer 输出，Reflector 凝练）
    ├── MEMORY.md              # 长期记忆（Dream 输出，供 LLM 上下文注入）
    ├── history.jsonl          # 原始对话历史（Dream 分析输入）
    ├── om-ops.jsonl           # OM 管线操作日志（Observer/Buffer/Dream 摘要）
    ├── .cursor                # 历史归档游标
    ├── .dream_cursor          # Dream 处理游标
    ├── .obs_cursor            # 观察处理游标
    └── .om_generation         # Reflector 凝练代数计数器
```

---

## 6. 四大组件

### 6.1 MastraOMStore — 文件 I/O 层

整个算法的唯一数据入口：

| 方法 | 功能 |
|------|------|
| `read_observations()` / `write_observations()` | 观察日志读写 |
| `append_observations(content, cycle_id)` | 追加新观察周期（带时间戳和 UUID 头） |
| `replace_observations(content)` | Reflector 凝练后整体替换 |
| `read_memory()` / `write_memory()` | 长期记忆文件 |
| `read_soul()` / `write_soul()` | Agent 身份文件 |
| `read_user()` / `write_user()` | 用户画像文件 |
| `append_history(entry)` | 追加原始消息到 history.jsonl，返回游标 |
| `read_unprocessed_history(since_cursor)` | 按游标增量读取原始历史 |
| `append_om_ops(entry)` | 追加 OM 管线操作摘要到 om-ops.jsonl |
| `read_om_ops()` | 读取所有 OM 操作日志 |
| `compact_history()` | 超限裁剪（默认保留最近 1000 条） |
| `get_memory_context()` | 构建 LLM 上下文注入文本（OBSERVATIONS + MEMORY） |
| `get_generation_count()` / `increment_generation()` | Reflector 代数计数 |
| `get_last_dream_cursor()` / `set_last_dream_cursor()` | Dream 处理游标 |
| `get_last_obs_cursor()` / `set_last_obs_cursor()` | Observer 处理游标 |
| `raw_archive(messages)` | Observer 失败时的回退：原始消息转储 |
| `git` (GitStore) | Git 行龄标注 + 自动提交 |

**关键设计**：
- 所有文件均为纯文本，用户可直接阅读编辑
- 游标机制实现增量处理（断点续传）
- 观察日志以 `## Observation Cycle <uuid8>` 分段，每次 Observer 调用一个周期
- 支持从 legacy `HISTORY.md` 自动迁移到 `history.jsonl`

### 6.2 MastraOMConsolidator — Observer/Reflector 管道

在线 token 预算驱动的记忆处理器：

**令牌估算**：
- `estimate_session_prompt_tokens(session)` — 使用 LLM provider 的 tokenizer 链式估算（优先本地，回退 chars/4）
- 安全缓冲：`context_window - max_completion - 1024` 为实际预算

**边界选取**（同 naive Consolidator）：
- `pick_consolidation_boundary()` — 沿消息序列找到用户轮次边界
- `_cap_consolidation_boundary()` — 限制单次归档块大小（≤60 条）

**Observer/Reflector 调用**：
- `_observe_messages()` — LLM 调用 → Observer 提取观察
- `_reflect_observations()` — LLM 调用 → Reflector 凝练观察（支持渐进压缩重试）

**核心方法**：

| 方法 | 功能 |
|------|------|
| `observe_and_store(messages)` | Observer 提取 → 追加到 OBSERVATIONS.md + history.jsonl 摘要 |
| `reflect_and_condense()` | 检查 token 阈值 → Reflector 渐进压缩 0→4 级 |
| `maybe_consolidate_by_tokens(session)` | **主入口**：token 预算检查 → 逐轮 Observe → 最终 Reflect |
| `extract_and_store(messages)` | **Hermes 接口**：Observer 提取 → 以事实列表返回 |
| `build_context_system_messages()` | 构建注入 LLM 上下文的系统消息 |
| `archive(messages)` | AutoCompact 兼容委托 |

**回退策略**：
- Observer LLM 失败 → `raw_archive()` 原始消息转储
- Observer 退化输出 → `raw_archive()` 原始消息转储
- Reflector 退化输出 → 重试下一压缩级别
- Reflector 最终级别仍过大 → 接受最终输出（宁可大，不能丢）

### 6.3 MastraOMDream — 两阶段深度处理

cron 定时触发（默认每 2 小时）：

**Phase 1 — 分析（纯 LLM 推理）**：
1. 读取 history.jsonl 中未处理条目（游标增量）
2. 注入当前 OBSERVATIONS.md + MEMORY.md + SOUL.md + USER.md
3. **注入行龄标注**（可选）：MEMORY.md 每行末尾 `← Nd`（仅标记 >14 天的行）
4. LLM 分析建议：哪些应写入长期记忆、哪些应删除、应创建什么技能

**Phase 2 — 执行（AgentRunner + 工具）**：
1. 将 Phase 1 分析结果传给 AgentRunner
2. AgentRunner 获得 `read_file` / `edit_file` / `SkillPrefixWriteFileTool` 工具
3. 自主编辑 MEMORY.md / SOUL.md / USER.md
4. 可选创建 `skills/dreamed-*/SKILL.md`
5. 推进游标 + 裁剪 history.jsonl + Git 自动提交

**关键设计**：
- 行龄标注仅应用于 MEMORY.md，SOUL.md 和 USER.md 不标注（它们不是累积性的）
- `SkillPrefixWriteFileTool` 限制：只能在 `skills/` 下创建 `dreamed-*` 前缀技能
- 技能创建前会列出已有技能，防止重复

### 6.4 MastraOMAutoCompact — 空闲会话压缩

当 `session_ttl_minutes > 0` 时启用：

- 定时扫描所有会话，找出空闲超过 TTL（默认 60 分钟）的非活跃会话
- 将此类会话的消息切片：保留最近 8 条作为上下文尾巴，其余交给 Consolidator 的 Observer 归档
- 归档摘要写入 `_summaries` 内存字典 + session metadata，供 `prepare_session()` 恢复上下文

---

## 7. 接口兼容性

### 7.1 Hermes 模式（skill_autogen）

中途技能蒸馏触发 `consolidator.extract_and_store(messages)`：
- 调用 Observer 从最近的对话消息中提取事实
- 原始消息写入 history.jsonl，Observer 摘要写入 om-ops.jsonl
- Observer 输出追加到 OBSERVATIONS.md
- 返回 `list[str]` 格式的事实列表（以 `*` 开头的行）
- SkillAutogen 读取 OBSERVATIONS.md 作为累积观察，结合当前会话内存数据进行技能审查

### 7.2 Dream 模式

cron 服务定时调用 `dream.run()` → Phase 1 分析 → Phase 2 编辑。
- Phase 1 读取 **原始** history.jsonl 条目 + OBSERVATIONS.md 进行深度分析
- Phase 2 编辑 MEMORY.md / SOUL.md / USER.md

### 7.3 AgentLoop 集成

`loop.py` 调用 `consolidator.maybe_consolidate_by_tokens(session)` → token 预算检查 → Observer/Reflector 管道。

---

## 8. 上下文注入格式

AgentLoop 中，Consolidator 为 LLM 构建上下文系统消息：

**有观察日志时**：

```
The following observations block contains your memory of past conversations with this user.

<observations>
Date: May 9, 2025
* 🔴 User prefers dark mode
* ✅ Auth feature completed
</observations>

IMPORTANT: When responding, reference specific details from these observations...
KNOWLEDGE UPDATES: ...newer observation supersedes the older one.
MOST RECENT USER INPUT: Treat the most recent user message as the highest-priority signal...
```

**无观察日志时**（回退）：

```
<system-reminder>
Please continue naturally with the conversation so far and respond to the latest message...
</system-reminder>
```

---

## 9. 使用方式

```python
from summerclaw.memory.registry import MemoryRegistry
from summerclaw.memory.mastra_om_memory import MastraOMMemoryAlgorithm

registry = MemoryRegistry()
registry.register(MastraOMMemoryAlgorithm())

algo = registry.get("mastra_om_memory")
components = algo.build(
    workspace=Path("./agent_workspace"),
    provider=llm_provider,
    model="gpt-5-mini",
    sessions=session_manager,
    context_window_tokens=200_000,
    build_messages=build_messages_fn,
    get_tool_definitions=get_tool_definitions_fn,
    max_completion_tokens=8192,
    session_ttl_minutes=60,
    max_batch_size=20,
    max_iterations=10,
    max_tool_result_chars=16_000,
    annotate_line_ages=True,
)

# 组件
store = components.store               # MastraOMStore
consolidator = components.consolidator  # MastraOMConsolidator
dream = components.dream                # MastraOMDream
auto_compact = components.auto_compact  # MastraOMAutoCompact (None if ttl=0)
```

---

## 10. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `context_window_tokens` | — | LLM 上下文窗口大小 |
| `max_completion_tokens` | 4096 | 最大输出 token 数 |
| `message_tokens_threshold` | **30,000** | 触发 Observer 的未处理消息 token 阈值（Mastra OM 论文值） |
| `observation_tokens_threshold` | **40,000** | 触发 Reflector 的观察日志 token 阈值（Mastra OM 论文值） |
| `session_ttl_minutes` | 60 | AutoCompact 空闲会话 TTL（0=禁用） |
| `max_batch_size` | 20 | Dream 每次处理的最大历史条目数 |
| `max_iterations` | 10 | Dream Phase 2 AgentRunner 最大迭代次数 |
| `max_tool_result_chars` | 16,000 | 工具结果最大字符数 |
| `annotate_line_ages` | True | 是否在 MEMORY.md 中标注陈旧行 |

---

## 11. 与 naive_memory 的对比

| 特性 | naive_memory | mastra_om_memory |
|------|:---:|:---:|
| 核心机制 | LLM 直接摘要 | Observer + Reflector 双层提炼 |
| 记忆格式 | 纯文本摘要 | 结构化观察日志（XML + 优先级 emoji） |
| 上下文稳定性 | 每轮变化（NA） | **固定前缀**（prompt-cache 友好） |
| 信息密度 | 低（LLM 摘要有损） | 高（观察提炼 + Reflector 压缩） |
| 压缩策略 | 单次 LLM 调用 | 渐进 0→4 级（Reflector 重试） |
| 退化处理 | 无 | 滑动窗口 + 超长行检测 → 回退 |
| 时序锚定 | 仅对话时间 | documentDate + eventDate 双时间戳 |
| 任务追踪 | 依赖 LLM 推断 | `<current-task>` / `<suggested-response>` 显式标注 |
| 完成标记 | 隐式 | ✅ 显式标记 |
| 断言/提问区分 | 无 | 🔴 断言 vs 提问 明确区分 |
| 外部依赖 | 零 | 零 |
| LongMemEval | 基准线 | **94.87%**（gpt-5-mini） |

---

## 12. 参考文献

1. Mastra AI Research. Observational Memory: A New Architecture for Long-Term Agent Memory. https://mastra.ai/research/observational-memory
2. Wu et al. (2024). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. *arXiv:2410.10813*.
3. Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.
4. Anthropic. Prompt Caching with Claude. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
