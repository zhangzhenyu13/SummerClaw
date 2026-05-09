# ReMe 记忆算法

> 基于 [ReMeLight](https://github.com/reme-ai)（reme-ai）的记忆算法包装器——将 ReMe 的语义搜索和自动压缩能力接入 nanobot 的四大组件体系。

## 核心思想

ReMeMemory 是一个**适配器模式**的实现：它不实现自己的记忆理论，而是将外部的 ReMeLight 库包装成符合 nanobot `MemoryAlgorithm` 接口的组件。

ReMeLight 提供三个核心能力：
1. **语义记忆搜索**：基于向量嵌入的对话记忆检索
2. **自动压缩**：`compact_memory()` 将对话历史压缩为语义摘要
3. **推理前钩子**：`pre_reasoning_hook()` 在 LLM 推理前自动注入相关记忆

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    ReMe 记忆算法流水线                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────┐                                 │
│  │       ReMeLight 引擎         │  ← 外部依赖 (reme 包)            │
│  │  • compact_memory()         │                                 │
│  │  • pre_reasoning_hook()     │                                 │
│  │  • 内部语义索引              │                                 │
│  └──────────┬──────────────────┘                                 │
│             │                                                    │
│    ┌────────┴────────┐                                          │
│    │   ReMeStore     │  ← 适配器：包装 ReMeLight + 维护伴侣文件    │
│    │   ReMeConsolidator│  ← 适配器：代理 compact_memory            │
│    │   ReMeDream      │  ← 标准 Phase 1/2 流程                    │
│    │   ReMeAutoCompact │  ← 标准空闲压缩                           │
│    └─────────────────┘                                           │
│                                                                  │
│  对话消息 ──► ReMeConsolidator.archive()                           │
│                  │                                                │
│                  ├──► raw_archive() → remem_history.jsonl          │
│                  └──► reme_light.compact_memory() → LLM 摘要       │
│                                                                  │
│  会话压缩 ──► ReMeConsolidator.maybe_consolidate_by_tokens()       │
│                  └──► reme_light.pre_reasoning_hook()              │
│                                                                  │
│  Cron 定时 ──► ReMeDream.run()                                    │
│                  Phase 1 → Phase 2 → 编辑 MEMORY.md               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 关键设计决策

### 双轨存储

ReMeLight 内部管理自己的对话文件（dialog files），不直接暴露给 nanobot。为保证兼容性，ReMeStore 维护一个**伴侣 history.jsonl 文件**（`remem_history.jsonl`），使 nanobot 的游标增量处理和 Dream 管线正常工作。

```
ReMeLight 内部文件                  nanobot 伴侣文件
─────────────────                  ───────────────
(dialog files, 由 ReMe 管理)        memory/
                                     ├── remem_history.jsonl
                                     ├── .remem_cursor
                                     └── .remem_dream_cursor
```

### 日志冲突处理

ReMeLight 的 `start()` 方法会调用 `logger.remove()` 清除所有日志处理器（包括 nanobot 的），因此在启动后必须恢复 stderr 输出：

```python
try:
    asyncio.run(reme_light.start())
finally:
    logger.add(sys.stderr, level="INFO", colorize=True)
```

## 四大组件

### ReMeStore

适配器存储层——包装 ReMeLight 的同时提供 nanobot 期望的 `MemoryStore` 接口：

| 方法 | 功能 |
|------|------|
| `read_memory()` / `write_memory()` | 读写 MEMORY.md |
| `read_soul()` / `write_soul()` | 读写 SOUL.md |
| `read_user()` / `write_user()` | 读写 USER.md |
| `append_history()` | 追加到 `remem_history.jsonl` |
| `read_unprocessed_history()` | 增量游标读取 |
| `get_memory_context()` | LLM 上下文注入 |
| `git` (GitStore) | 行龄标注 + 自动提交 |
| `raw_archive()` | 降级归档（compact_memory 失败时） |

### ReMeConsolidator

在线压缩适配器——代理 `ReMeLight.compact_memory()` 和 `pre_reasoning_hook()`：

**`archive(messages)`** —
1. 调用 `store.raw_archive(messages)` 追加原始消息到伴侣历史
2. 将消息转换为 AgentScope `Msg` 对象
3. 调用 `reme_light.compact_memory(messages=msgs)` 生成 LLM 摘要
4. 摘要写入 `remem_history.jsonl`

**`maybe_consolidate_by_tokens(session)`** —
1. 估算当前会话 token 数
2. 若超过预算，调用 `reme_light.pre_reasoning_hook()`（ReMe 内部的推理前记忆注入）
3. 标记会话已压缩

### ReMeDream

标准的 Phase 1/2 流程——分析 `remem_history.jsonl` 增量条目，通过 AgentRunner 编辑 MEMORY.md。与 naive Dream 几乎一致，但从 ReMeStore 读取数据。

### ReMeAutoCompact

标准的 idle session 压缩器——与 naive AutoCompact 行为相同，但调用 ReMeConsolidator 的 `archive()` 方法（充分利用 ReMeLight 的语义压缩能力）。

## 存储布局

```
workspace/
├── SOUL.md
├── USER.md
├── MEMORY.md
└── memory/
    ├── remem_history.jsonl   # 伴侣历史文件
    ├── .remem_cursor         # 归档游标
    └── .remem_dream_cursor   # Dream 处理游标
```

## 使用方式

```python
from nanobot.memory.registry import MemoryRegistry
from nanobot.memory.remem_memory import ReMeMemoryAlgorithm

registry = MemoryRegistry()
registry.register(ReMeMemoryAlgorithm())

algo = registry.get("remem_memory")
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

## 前置依赖

```bash
pip install reme
```

ReMeLight 需要 LLM API 密钥——通过 `provider.api_key` 和 `provider.api_base` 自动传入。

## 降级策略

当 ReMeLight 的语义能力不可用时（API 错误、网络问题等），系统自动降级：

- `compact_memory()` 失败 → 回退到 `raw_archive()`，直接将原始消息写入历史
- `pre_reasoning_hook()` 失败 → 仅标记会话已压缩，不影响对话继续
- 每步都有 try/except 保护，确保永不阻塞主流程

## 与 naive_memory 的对比

| 特性 | naive_memory | remem_memory |
|------|:---:|:---:|
| 语义搜索 | 无 | ReMeLight 内部语义索引 |
| 压缩方式 | LLM 直接摘要 | ReMeLight.compact_memory() |
| 推理增强 | 无 | pre_reasoning_hook() 自动注入 |
| 外部依赖 | 无 | `pip install reme` |
| 存储模型 | 纯 nanobot 文件 | nanobot 伴侣文件 + ReMe 内部文件 |

## 参考文献

- ReMe (reme-ai). https://github.com/reme-ai
- ReMeLight: Lightweight conversational memory engine
