# nanobot/command — 斜杠命令路由与内建处理器

## 概述

`command` 模块为 nanobot 提供了**斜杠命令（slash command）**的路由分发和全套内建命令处理能力。它是用户与 Agent 交互的控制面入口——所有以 `/` 开头的消息都会经过此模块的路由器进行匹配与分发。

## 文件结构

```
nanobot/command/
├── __init__.py    # 模块入口，导出公共 API
├── router.py      # 命令路由引擎（CommandRouter + CommandContext）
└── builtin.py     # 所有内建命令的处理器实现与注册
```

## 核心组件

### 1. CommandContext（`router.py`）

一个 `dataclass`，封装了命令处理器所需的一切上下文信息：

| 字段      | 类型               | 说明                          |
|-----------|--------------------|-------------------------------|
| `msg`     | `InboundMessage`   | 原始入站消息                   |
| `session` | `Session \| None`  | 当前会话对象                   |
| `key`     | `str`              | 会话键（session key）          |
| `raw`     | `str`              | 原始命令文本（含 `/` 前缀）     |
| `args`    | `str`              | 命令参数（前缀匹配后自动填充）   |
| `loop`    | `Any`              | 当前 AgentLoop 实例引用        |

### 2. CommandRouter（`router.py`）

纯字典驱动的命令分发器，采用**四级匹配策略**（按优先级从上到下）：

| 层级       | 注册方法              | 说明                                                                 |
|------------|-----------------------|----------------------------------------------------------------------|
| **Priority** | `priority()` / `priority_prefix()` | **绕过分发锁**，始终立即响应。即使 Agent 正忙于处理消息，优先命令也能即时应答 |
| **Exact**    | `exact()`             | 精确匹配，在分发锁内执行                                              |
| **Prefix**   | `prefix()`            | 最长前缀优先匹配（如 `/dream-log <sha>`）                             |
| **Interceptor** | `intercept()`     | 兜底拦截器，用于谓词匹配等降级场景（如 team-mode 激活检测）           |

关键方法：

- `is_priority(text)` — 判断给定文本是否为优先命令
- `dispatch_priority(ctx)` — 在无锁状态下分发优先命令
- `dispatch(ctx)` — 依次尝试 exact → prefix → interceptor，未命中返回 `None`

### 3. 内建命令处理器（`builtin.py`）

所有内建命令均为 `async` 函数，签名为：

```python
async def cmd_xxx(ctx: CommandContext) -> OutboundMessage
```

#### 命令清单

| 命令                   | 处理器函数              | 功能说明                                                               | 路由类型     |
|------------------------|-------------------------|------------------------------------------------------------------------|--------------|
| `/stop`                | `cmd_stop`              | 取消当前会话的所有活跃任务和子代理                                      | priority     |
| `/restart`             | `cmd_restart`           | 通过 `os.execv` 原地重启进程，重启前设置通知环境变量                    | priority     |
| `/status`              | `cmd_status`            | 展示 bot 运行状态（模型、Token 用量、运行时间、活跃任务数、搜索配额等） | priority     |
| `/tasks`               | `cmd_tasks`             | 列出当前运行的子代理任务和中断待决策的任务                              | priority     |
| `/tasks resume <id>`   | `cmd_tasks_resume`      | 按 ID 恢复中断的子代理或主会话任务                                     | priority_prefix |
| `/tasks discard <id>`  | `cmd_tasks_discard`     | 按 ID 丢弃中断的子代理或主会话任务                                     | priority_prefix |
| `/new`                 | `cmd_new`               | 清空当前会话，启动新对话                                               | exact        |
| `/dream`               | `cmd_dream`             | 手动触发 Dream 记忆整合，异步运行并返回结果                             | exact        |
| `/dream-log [sha]`     | `cmd_dream_log`         | 查看 Dream 记忆变更日志（默认最新一次，可选指定 SHA）                    | exact + prefix |
| `/dream-restore [sha]` | `cmd_dream_restore`     | 恢复 Dream 记忆到历史版本（无参数列出版本列表，带 SHA 执行回滚）         | exact + prefix |
| `/skill-autogen`       | `cmd_skill_autogen`     | 审查近期对话，提取可复用的工作流生成 SKILL.md                           | exact        |
| `/help`                | `cmd_help`              | 显示所有可用命令及其简短说明                                           | exact        |

#### 特别说明

- **Priority 命令**（`/stop`、`/restart`、`/status`、`/tasks`）注册了**双重路由**：既在 priority 表中，也在 exact 表中。这确保了即使通过不同调用路径（`dispatch_priority` vs `dispatch`）都能正确匹配。
- **Dream 系列命令**（`/dream`、`/dream-log`、`/dream-restore`）依赖 `consolidator.store.git`（内存版本控制系统）实现 Dream 记忆的版本回滚与 diff 展示。
- **`/skill-autogen`** 与 Dream 不同：Dream 整合记忆文件（MEMORY/SOUL/USER.md），Skill-Autogen 提取可复用技能（`skills/<name>/SKILL.md`）。

## 测试覆盖

对应测试文件位于：

```
tests/command/test_builtin_dream.py
```

覆盖了 Dream 系列命令（`cmd_dream_log`、`cmd_dream_restore`）的以下场景：

- 最新 Dream 日志展示（diff 内容、文件列表、undo 提示）
- 缺失 commit 时的用户引导提示
- Dream 首次运行前的友好提示
- `/dream-restore` 版本列表展示
- `/dream-restore` 成功恢复后的文件与后续操作提示

## 集成方式

`CommandRouter` 在 `AgentLoop.__init__()` 中被实例化并注册：

```python
# nanobot/agent/loop.py
self.commands = CommandRouter()
register_builtin_commands(self.commands)
```

消息处理流程（`AgentLoop._process_message()`）：

1. **模式前缀解析**（`/simple`、`/plan`、`/search-plan`、`/auto`）——这些不是命令，而是修改执行模式的前缀
2. **命令路由分发**：构造 `CommandContext`，调用 `self.commands.dispatch(ctx)`
3. 若命令命中，直接返回 `OutboundMessage`；否则进入正常的 Agent 推理流程

## 设计原则

- **优先级隔离**：`/stop`、`/restart` 等控制命令绕过分发锁，保证在 Agent 繁忙时也能立即响应
- **模块边界清晰**：`__init__.py` 仅暴露 `CommandRouter`、`CommandContext` 和 `register_builtin_commands` 三个公共 API，内部实现细节完全封装
- **处理器无状态**：每个命令处理器是纯异步函数，通过 `CommandContext` 获取所需状态，不依赖全局变量
- **路由表不可变**：命令注册完成后路由表结构稳定，dispatch 时均为字典/列表查找，性能优异