# Session 模块技术文档

## 概述

`summerclaw/session` 模块负责管理 Nanobot 的对话会话（Session），提供会话的创建、加载、保存、查询以及消息历史管理功能。该模块是整个对话系统的核心基础设施，为 Agent 循环、记忆系统、命令路由等组件提供持久化的会话状态支持。

**核心职责**：
- 管理多通道（Telegram、Discord、Slack 等）的对话会话
- 持久化会话消息历史（JSONL 格式）
- 支持统一会话模式（Unified Session）
- 提供消息历史的合法边界裁剪（避免中断工具调用）
- 支持旧版会话数据的自动迁移

---

## 文件结构

```
summerclaw/session/
├── __init__.py          # 模块导出：Session, SessionManager
├── manager.py           # 核心实现：Session 数据类 + SessionManager 管理器
└── __pycache__/         # Python 字节码缓存
```

**测试文件**：
```
tests/agent/
├── test_session_manager_history.py   # Session 历史管理测试（220 行）
└── test_unified_session.py           # 统一会话模式测试（502 行）
```

---

## 核心组件

### 1. Session 数据类

[Session](summerclaw/session/manager.py#L16-L93) 是一个 `@dataclass`,表示一个对话会话。

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | `str` | 会话唯一标识，格式通常为 `channel:chat_id`（如 `telegram:123456`） |
| `messages` | `list[dict]` | 消息列表，每个消息包含 `role`、`content`、`timestamp` 等 |
| `created_at` | `datetime` | 会话创建时间 |
| `updated_at` | `datetime` | 最后更新时间 |
| `metadata` | `dict` | 会话元数据（用于存储 `pending_user_turn`、`interrupted_task_id` 等） |
| `last_consolidated` | `int` | 已归档到记忆系统的消息数量（用于增量归档） |

#### 核心方法

**`add_message(role, content, **kwargs)`**
- 添加消息到会话
- 自动附加 `timestamp` 字段
- 更新 `updated_at` 时间戳

**`get_history(max_messages=500)`**
- 返回未归档的消息历史供 LLM 使用
- **关键特性**：自动裁剪到合法的工具调用边界
  - 从最近的 `user` 消息开始（避免半轮对话）
  - 移除开头的孤立 `tool` 结果（没有对应 `tool_calls` 的工具结果）
  - 保留 `reasoning_content`、`tool_calls` 等关键字段

**`retain_recent_legal_suffix(max_messages)`**
- 保留合法的消息后缀（用于会话压缩）
- 镜像 `get_history` 的边界规则
- 自动调整 `last_consolidated` 偏移量

**`clear()`**
- 清空所有消息
- 重置 `last_consolidated` 为 0

#### 消息历史裁剪算法

`get_history` 和 `retain_recent_legal_suffix` 使用 [find_legal_message_start](summerclaw/utils/helpers.py#L103) 工具函数确保消息历史从合法位置开始:

```python
# 裁剪流程
1. 提取未归档消息：messages[last_consolidated:]
2. 限制最大数量：[-max_messages:]
3. 向前对齐到 user 消息：避免 mid-turn 开始
4. 移除孤立 tool 结果：确保每个 tool 结果都有对应的 tool_call
5. 提取关键字段：role, content, tool_calls, tool_call_id, name, reasoning_content
```

**示例场景**：
```python
session.messages = [
    {"role": "user", "content": "old question"},
    {"role": "assistant", "tool_calls": [{"id": "tc1", ...}]},
    {"role": "tool", "tool_call_id": "tc1", ...},  # 如果窗口裁剪到这里，会被丢弃
    {"role": "user", "content": "new question"},
]

# max_messages=2 时，只返回：
# [{"role": "user", "content": "new question"}]
# 而不是孤立的 tool 结果
```

---

### 2. SessionManager 类

[SessionManager](summerclaw/session/manager.py#L96-L268) 是会话管理器,负责会话的生命周期管理。

#### 存储结构

**JSONL 文件格式**：
```jsonl
{"_type": "metadata", "key": "telegram:123", "created_at": "...", "updated_at": "...", "metadata": {}, "last_consolidated": 10}
{"role": "user", "content": "hello", "timestamp": "..."}
{"role": "assistant", "content": "hi there", "timestamp": "..."}
{"role": "user", "content": "how are you?", "timestamp": "..."}
```

- 第一行：元数据行（`_type: "metadata"`）
- 后续行：消息记录（每行一个 JSON 对象）
- 文件路径：`{workspace}/sessions/{safe_key}.jsonl`

#### 核心方法

**`__init__(workspace: Path)`**
- 初始化会话目录：`{workspace}/sessions/`
- 设置旧版会话目录：`~/.summerclaw/sessions/`（用于迁移）
- 创建内存缓存：`_cache: dict[str, Session]`

**`get_or_create(key: str) -> Session`**
- 获取或创建会话
- 优先从内存缓存返回
- 缓存未命中时从磁盘加载
- 不存在时创建新会话

**`_load(key: str) -> Session | None`**
- 从磁盘加载会话
- **自动迁移**：如果新路径不存在但旧路径存在，自动移动文件
- 解析 JSONL 文件，分离元数据和消息
- 加载失败时返回 `None`（记录警告日志）

**`save(session: Session) -> None`**
- 保存会话到磁盘
- 写入元数据行 + 所有消息
- 更新内存缓存

**`invalidate(key: str) -> None`**
- 从内存缓存移除会话
- 不删除磁盘文件
- 下次 `get_or_create` 会重新加载

**`list_sessions() -> list[dict]`**
- 列出所有会话（读取元数据行）
- 按 `updated_at` 降序排列
- 返回会话信息摘要（不加载完整消息）

**`list_all_sessions() -> list[Session]`**
- 加载所有会话（用于启动时扫描）
- 用于检测 `pending_user_turn` 标记的中断任务
- 跳过损坏的文件

#### 辅助方法

**`_get_session_path(key: str) -> Path`**
- 生成会话文件路径
- 使用 `safe_filename` 清理键名（替换 `:` 为 `_`）

**`_get_legacy_session_path(key: str) -> Path`**
- 生成旧版全局会话路径
- 用于向后兼容

---

## 关键特性

### 1. 统一会话模式（Unified Session）

统一会话模式允许多个通道共享同一个会话上下文。

**工作原理**：
- 配置项：`config.agents.defaults.unified_session`（默认 `False`）
- 启用后，所有消息使用固定键 `unified:default`
- 如果消息已有 `session_key_override`，则保留原键（尊重通道特定覆盖）

**在 AgentLoop 中的集成**：
```python
# summerclaw/agent/loop.py
def _effective_session_key(self, msg: InboundMessage) -> str:
    if self._unified_session and not msg.session_key_override:
        return UNIFIED_SESSION_KEY  # "unified:default"
    return msg.session_key
```

**应用场景**：
- 跨通道连续对话（Telegram → Discord → CLI）
- `/new` 命令清空共享会话
- `/stop` 命令跨通道取消任务

**测试覆盖**:[test_unified_session.py](tests/agent/test_unified_session.py)

### 2. 消息历史合法边界保护

Session 模块确保消息历史永远不会从非法位置开始（如孤立的工具结果）。

**保护规则**：
1. **User 消息对齐**：历史从 `user` 消息开始，避免 mid-turn
2. **工具调用完整性**：每个 `tool` 结果必须有对应的 `assistant.tool_calls`
3. **Orphan 检测**：开头的孤立 `tool` 结果被自动丢弃

**测试用例**：
- `test_get_history_drops_orphan_tool_results_when_window_cuts_tool_calls`
- `test_window_cuts_mid_tool_group`
- `test_all_orphan_prefix_stripped`
- `test_retain_recent_legal_suffix_keeps_legal_tool_boundary`

详见:[test_session_manager_history.py](tests/agent/test_session_manager_history.py)

### 3. 旧版会话自动迁移

当用户升级 Nanobot 时，旧版全局会话（`~/.summerclaw/sessions/`）会自动迁移到新的工作区路径。

**迁移逻辑**：
```python
if not path.exists():
    legacy_path = self._get_legacy_session_path(key)
    if legacy_path.exists():
        shutil.move(str(legacy_path), str(path))
        logger.info("Migrated session {} from legacy path", key)
```

**优势**：
- 无感知升级
- 保留历史对话
- 迁移失败时优雅降级（记录异常日志）

### 4. 会话元数据扩展

`metadata` 字段用于存储会话级别的运行时状态：

| 元数据键 | 用途 | 使用场景 |
|---------|------|---------|
| `pending_user_turn` | 标记等待用户响应的会话 | 服务重启后恢复中断任务 |
| `interrupted_task_id` | 中断任务的短 ID | `/tasks resume` 命令 |
| 自定义键 | 记忆算法、技能自动生成等 | 扩展功能 |

**示例**：
```python
session.metadata["pending_user_turn"] = True
session.metadata["interrupted_task_id"] = "a1b2c3"
self.sessions.save(session)
```

### 5. 增量归档支持

`last_consolidated` 字段支持记忆系统的增量归档：

```python
# Session 中的消息
messages = [msg0, msg1, msg2, ..., msg99]
last_consolidated = 50  # 前 50 条已归档

# get_history 只返回未归档部分
unconsolidated = messages[50:]  # msg50 ~ msg99
```

**与 Consolidator 的协作**：
```python
# summerclaw/memory/ 中的 Consolidator
await consolidator.maybe_consolidate_by_tokens(session)
# 归档后更新 session.last_consolidated
```

---

## 与其他模块的集成

### 1. AgentLoop 集成

[AgentLoop](summerclaw/agent/loop.py) 是 Session 的主要消费者:

```python
class AgentLoop:
    def __init__(self, ..., session_manager: SessionManager | None = None, ...):
        self.sessions = session_manager or SessionManager(workspace)
    
    async def _process_message(self, msg: InboundMessage):
        session = self.sessions.get_or_create(key)
        history = session.get_history(max_messages=0)
        messages = self.context.build_messages(history, ...)
        # ... 运行 LLM ...
        self.sessions.save(session)
```

**关键交互点**：
- `_dispatch`：消息分发时使用 `_effective_session_key` 确定会话键
- `_process_message`：加载会话历史、保存新消息
- `_run_agent_loop`：支持会话级别的 pending queue（中途消息注入）
- 任务管理：`_active_tasks` 使用会话键索引

### 2. 命令系统集成

[CommandRouter](summerclaw/command/router.py) 和内建命令使用 Session:

**`/new` 命令**：
```python
async def cmd_new(ctx: CommandContext):
    session = ctx.loop.sessions.get_or_create(ctx.key)
    session.clear()
    ctx.loop.sessions.save(session)
    return CommandResult(content="New session started")
```

**`/stop` 命令**：
```python
async def cmd_stop(ctx: CommandContext):
    tasks = ctx.loop._active_tasks.get(ctx.key, [])
    for task in tasks:
        task.cancel()
```

**`/tasks` 命令**：
- 扫描 `list_all_sessions()` 查找 `pending_user_turn` 标记
- 恢复中断任务

### 3. 记忆系统集成

所有记忆算法模块都依赖 SessionManager：

- [naive_memory](summerclaw/memory/naive_memory/)
- [mastra_om_memory](summerclaw/memory/mastra_om_memory/)
- [hindsight_memory](summerclaw/memory/hindsight_memory/)
- [mem0v3_memory](summerclaw/memory/mem0v3_memory/)

**典型用法**：
```python
class AutoCompact:
    def __init__(self, sessions: SessionManager, ...):
        self.sessions = sessions
    
    def check_expired(self, active_session_keys: Collection[str]):
        for info in self.sessions.list_sessions():
            if info["key"] not in active_session_keys:
                # 归档空闲会话
                self._archive(info["key"])
```

### 4. API 服务器集成

[API Server](summerclaw/api/server.py) 使用 Session 管理 SDK 会话:

```python
# 创建 SDK 会话
session = self.loop.sessions.get_or_create(f"sdk:{session_id}")
history = session.get_history(max_messages=0)
```

---

## 使用示例

### 基础用法

```python
from summerclaw.session.manager import SessionManager
from pathlib import Path

# 创建管理器
manager = SessionManager(Path("/workspace"))

# 获取或创建会话
session = manager.get_or_create("telegram:123456")

# 添加消息
session.add_message("user", "Hello!")
session.add_message("assistant", "Hi there!", tool_calls=[...])
session.add_message("tool", "Result", tool_call_id="tc1")

# 保存会话
manager.save(session)

# 获取历史
history = session.get_history(max_messages=50)
print(history)
# [{"role": "user", "content": "Hello!"}, ...]

# 清空会话
session.clear()
manager.save(session)
```

### 统一会话模式

```python
# 配置启用
config = {
    "agents": {
        "defaults": {
            "unifiedSession": True  # 或 unified_session
        }
    }
}

# AgentLoop 自动重写会话键
loop = AgentLoop(..., unified_session=True)

# 所有消息使用 unified:default
msg1 = InboundMessage(channel="telegram", chat_id="111", ...)
msg2 = InboundMessage(channel="discord", chat_id="222", ...)
# 两者共享同一会话
```

### 会话查询

```python
# 列出所有会话
sessions = manager.list_sessions()
for info in sessions:
    print(f"{info['key']}: updated at {info['updated_at']}")

# 加载所有会话（启动时扫描）
all_sessions = manager.list_all_sessions()
for session in all_sessions:
    if session.metadata.get("pending_user_turn"):
        print(f"Session {session.key} has pending user turn")
```

---

## 测试覆盖

### Session 历史管理测试

[test_session_manager_history.py](tests/agent/test_session_manager_history.py) 覆盖:

- ✅ 孤立工具结果裁剪
- ✅ 合法工具调用对保留
- ✅ `retain_recent_legal_suffix` 边界对齐
- ✅ `last_consolidated` 偏移调整
- ✅ 空会话处理
- ✅ `reasoning_content` 保留
- ✅ 窗口裁剪到工具调用中间场景
- ✅ 全孤儿前缀剥离

### 统一会话模式测试

[test_unified_session.py](tests/agent/test_unified_session.py) 覆盖:

- ✅ 会话键重写为 `unified:default`
- ✅ 不同通道共享会话
- ✅ 禁用时保留原始键
- ✅ 尊重现有 `session_key_override`
- ✅ 默认值为 `False`
- ✅ 配置序列化（camelCase `unifiedSession`）
- ✅ 配置解析（支持 camelCase 和 snake_case）
- ✅ onboard 生成的 config 包含 `unifiedSession`
- ✅ `/new` 命令清空共享会话
- ✅ `/new` 不是优先级命令（通过 `_dispatch`）
- ✅ Consolidation 不受统一会话影响
- ✅ `/stop` 命令在统一模式下找到任务
- ✅ 跨通道取消任务

---

## 设计原则

### 1. 数据完整性优先

- **合法边界保护**：永远不从非法位置（如孤立 tool 结果）开始消息历史
- **原子性保存**：完整写入元数据 + 所有消息
- **优雅降级**：加载失败返回 `None`，不抛出异常

### 2. 向后兼容

- **旧版迁移**：自动迁移 `~/.summerclaw/sessions/` 文件
- **配置兼容**：同时支持 camelCase 和 snake_case
- **API 稳定**：Session 和 SessionManager 接口保持稳定

### 3. 性能优化

- **内存缓存**：`_cache` 避免重复加载
- **惰性加载**：`list_sessions` 只读取元数据行
- **增量归档**：`last_consolidated` 支持只处理新消息

### 4. 扩展性

- **元数据扩展**：`metadata` 字段支持自定义键值对
- **算法无关**：不依赖特定记忆算法
- **通道无关**：支持任意通道（通过 `channel:chat_id` 键）

---

## 故障排查

### 常见问题

**Q: 会话文件在哪里？**
```
{workspace}/sessions/telegram_123.jsonl
{workspace}/sessions/discord_456.jsonl
{workspace}/sessions/unified_default.jsonl  # 统一会话模式
```

**Q: 如何查看会话内容？**
```bash
# 查看元数据
head -1 {workspace}/sessions/telegram_123.jsonl

# 查看所有消息
cat {workspace}/sessions/telegram_123.jsonl | python -m json.tool
```

**Q: 会话加载失败怎么办？**
- 检查 JSONL 文件格式（每行一个合法 JSON）
- 查看日志：`logger.warning("Failed to load session {}: {}", key, e)`
- 损坏的文件会被跳过（`_load` 返回 `None`）

**Q: 如何清空所有会话？**
```bash
rm -rf {workspace}/sessions/*.jsonl
# 或重启后使用 /new 命令
```

**Q: 统一会话模式下如何区分不同用户？**
- 统一会话模式共享上下文，不区分用户
- 如需区分，禁用 `unified_session` 配置

---

## 相关模块

| 模块 | 关系 |
|------|------|
| [summerclaw/agent/loop.py](summerclaw/agent/loop.py) | 主要消费者,管理会话生命周期 |
| [summerclaw/command/builtin.py](summerclaw/command/builtin.py) | `/new`、`/stop`、`/tasks` 命令 |
| [summerclaw/memory/](summerclaw/memory/) | 所有记忆算法使用 SessionManager |
| [summerclaw/api/server.py](summerclaw/api/server.py) | SDK API 会话管理 |
| [summerclaw/utils/helpers.py](summerclaw/utils/helpers.py) | `find_legal_message_start`、`safe_filename` |
| [summerclaw/config/schema.py](summerclaw/config/schema.py) | `unified_session`、`session_ttl_minutes` 配置 |

---

## 总结

`summerclaw/session` 模块是 Nanobot 对话系统的核心基础设施，提供：

1. **可靠的会话持久化**（JSONL 格式 + 内存缓存）
2. **智能的消息历史裁剪**（合法边界保护）
3. **灵活的会话模式**（独立会话 vs 统一会话）
4. **无缝的向后兼容**（旧版自动迁移）
5. **丰富的扩展点**（元数据、归档、任务恢复）

该模块经过全面测试（720+ 行测试代码），保证了在各种边界场景下的数据完整性和系统稳定性。
