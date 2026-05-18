# summerclaw/cron 模块

## 概述

`summerclaw/cron` 是 SummerClaw/summerclaw 项目的**定时任务调度模块**，负责管理 Agent 定时任务的创建、调度、执行与持久化。该模块支持三种调度模式（一次性、周期、Cron 表达式），通过磁盘持久化 + 操作日志实现多实例协调，并提供受保护的系统任务（如 Dream 记忆固化）机制。

---

## 模块架构

```
summerclaw/cron/
├── __init__.py          # 模块公开接口
├── types.py             # 数据类型定义（Schedule、Job、Store 等）
└── service.py           # CronService：核心调度引擎（558 行）
```

---

## 核心组件详解

### 1. CronService（`service.py`）

CronService 是模块的**核心调度引擎**，负责管理所有定时任务的生命周期。

**核心职责：**
- 任务的增删改查与启用/禁用
- 根据调度规则计算下次执行时间
- 异步定时器驱动任务执行
- 任务执行状态追踪（运行历史，最多保留 20 条）
- 任务数据的磁盘持久化

**关键机制：**

| 机制 | 说明 |
|------|------|
| **异步定时器** | 通过 `asyncio.create_task` + `asyncio.sleep` 实现，动态计算下次唤醒时间（最大间隔 5 分钟） |
| **磁盘持久化** | 任务存储在 `store_path` 指定的 JSON 文件中，支持热加载 |
| **操作日志** | 多实例间通过 `action.jsonl` 记录增/删/改操作，配合 `FileLock` 实现无冲突协调 |
| **受保护系统任务** | `payload.kind == "system_event"` 的任务不可被用户删除或修改 |
| **一次性任务清理** | `kind="at"` 的任务执行后自动禁用（或通过 `delete_after_run` 删除） |

**公开 API：**

| 方法 | 说明 |
|------|------|
| `start()` / `stop()` | 启动/停止调度服务 |
| `list_jobs(include_disabled)` | 列出所有任务，按下次执行时间排序 |
| `add_job(name, schedule, message, ...)` | 添加定时任务 |
| `register_system_job(job)` | 注册系统级任务（幂等，重启时自动重建） |
| `remove_job(job_id)` | 删除任务（系统任务受保护） |
| `enable_job(job_id, enabled)` | 启用/禁用任务 |
| `update_job(job_id, ...)` | 更新任务的可变字段（名称、调度、消息、投递配置等） |
| `run_job(job_id, force)` | 手动立即执行一次任务 |
| `get_job(job_id)` | 按 ID 查询任务详情 |
| `status()` | 获取服务状态（运行中/已停止、任务数、下次唤醒时间） |

**回调机制：**
- 构造函数接收 `on_job: Callable[[CronJob], Coroutine]` 回调，任务触发时调用
- 回调返回 `str | None`，可用于返回执行结果

**内部流程：**

```
start()
  │
  ├── _load_store() → 加载磁盘数据 + 合并操作日志
  ├── _recompute_next_runs() → 计算所有启用任务的下次执行时间
  ├── _save_store() → 持久化当前状态
  └── _arm_timer() → 启动异步定时器
        │
        └── [定时触发] _on_timer()
              │
              ├── _load_store() → 重新加载（热更新）
              ├── 筛选到期任务 → _execute_job(job)
              │     ├── 调用 on_job 回调
              │     ├── 更新 run_history（最多 20 条）
              │     ├── 处理一次性任务（at 类型）
              │     └── 计算下次执行时间
              ├── _save_store()
              └── _arm_timer() → 重新编排定时器
```

---

### 2. 数据类型（`types.py`）

#### CronSchedule — 调度规则

| 字段 | 类型 | 说明 |
|------|------|------|
| `kind` | `"at" \| "every" \| "cron"` | 调度类型 |
| `at_ms` | `int \| None` | 一次性执行时间戳（毫秒，仅 `at` 模式） |
| `every_ms` | `int \| None` | 周期间隔（毫秒，仅 `every` 模式） |
| `expr` | `str \| None` | Cron 表达式（如 `"0 9 * * *"`，仅 `cron` 模式） |
| `tz` | `str \| None` | IANA 时区（仅 `cron` 模式），如 `"America/Vancouver"` |

**三种调度模式：**

| 模式 | 用途 | 示例 |
|------|------|------|
| `at` | 一次性定时执行 | 明天的 10:30 发一条提醒 |
| `every` | 固定间隔重复 | 每 5 分钟检查系统状态 |
| `cron` | Cron 表达式调度 | 每天 9:00 发送日报 |

#### CronPayload — 任务内容

| 字段 | 类型 | 说明 |
|------|------|------|
| `kind` | `"system_event" \| "agent_turn"` | 任务类型（系统/用户） |
| `message` | `str` | 触发后发送给 Agent 的指令文本 |
| `deliver` | `bool` | 执行结果是否投递给用户 |
| `channel` | `str \| None` | 投递通道（如 `"whatsapp"`） |
| `to` | `str \| None` | 投递目标（如手机号） |

#### CronJobState — 运行时状态

| 字段 | 类型 | 说明 |
|------|------|------|
| `next_run_at_ms` | `int \| None` | 下次执行时间戳 |
| `last_run_at_ms` | `int \| None` | 上次执行时间戳 |
| `last_status` | `"ok" \| "error" \| "skipped" \| None` | 上次执行状态 |
| `last_error` | `str \| None` | 上次错误信息 |
| `run_history` | `list[CronRunRecord]` | 运行历史（最多 20 条） |

#### CronJob — 任务实体

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | UUID 前 8 位，全局唯一 |
| `name` | `str` | 人类可读名称 |
| `enabled` | `bool` | 是否启用 |
| `schedule` | `CronSchedule` | 调度规则 |
| `payload` | `CronPayload` | 任务内容 |
| `state` | `CronJobState` | 运行时状态 |
| `created_at_ms` | `int` | 创建时间戳 |
| `updated_at_ms` | `int` | 更新时间戳 |
| `delete_after_run` | `bool` | 执行后自动删除（一次性任务标识） |

#### CronStore — 持久化存储

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | `int` | 存储格式版本号 |
| `jobs` | `list[CronJob]` | 所有任务列表 |

---

## 与系统其他模块的关系

### 集成架构

```
                    ┌─────────────────┐
                    │  CronService    │
                    │  (cron/service) │
                    └───────┬─────────┘
                            │
              ┌─────────────┼──────────────┐
              │             │              │
              ▼             ▼              ▼
    ┌─────────────┐  ┌───────────┐  ┌───────────┐
    │  CronTool   │  │ AgentLoop │  │    CLI    │
    │ (tools/cron)│  │ (loop.py) │  │(commands) │
    └─────────────┘  └───────────┘  └───────────┘
```

### 各模块使用方式

| 模块 | 使用方式 |
|------|----------|
| **`agent/tools/cron.py`** | 将 CronService 封装为 Agent 可调用的 `CronTool`，支持 `add` / `list` / `remove` 操作，自动感知会话上下文（channel、chat_id），防止 cron 任务内递归创建 cron 任务 |
| **`agent/loop.py`** | 通过 `on_job` 回调将 cron 任务消息注入 Agent 主循环，实现定时触发 Agent 执行 |
| **`cli/commands.py`** | 初始化 CronService 实例，管理生命周期（启动/停止），提供 CLI 命令接口 |
| **`config/schema.py`** | 引用 `CronSchedule` 类型用于配置解析 |

### 数据流（定时任务触发）

```
CronService 定时器触发
  │
  ├── _execute_job() → on_job 回调
  │     └── AgentLoop 接收 CronJob，注入为 Agent 消息
  │           └── Agent 处理消息 → 执行工具 → 生成响应
  │                 └── [deliver=True] → MessageBus → Channel → 用户
  │
  └── 更新 state（状态、历史、下次执行时间）
```

### 支持的任务类型

| 类型 | `payload.kind` | 说明 |
|------|----------------|------|
| **用户任务** | `agent_turn` | 通过 CronTool 或 CLI 创建，触发 Agent 处理 |
| **系统任务** | `system_event` | 框架级任务（如 Dream 记忆固化），受保护不可删除 |

---

## 公开接口（`__init__.py`）

```python
from summerclaw.cron import (
    CronService,    # 核心调度服务
    CronJob,        # 任务实体数据类
    CronSchedule,   # 调度规则数据类
)
```

---

## 关键设计原则

1. **三种调度模式统一抽象**：`at`（一次性）、`every`（周期）、`cron`（Cron 表达式）通过 `CronSchedule.kind` 统一建模，`_compute_next_run` 根据 `kind` 分发计算逻辑

2. **多实例协调**：通过 `action.jsonl` + `FileLock` 实现多实例间的操作合并，避免直接写 store 文件造成冲突

3. **优雅的运行时/离线模式切换**：
   - 服务运行时（`self._running=True`）：直接写 JSON store 并重新编排定时器
   - 服务停止时：通过 `_append_action` 写入操作日志，下次 `_load_store` 时自动合并

4. **系统任务保护**：`payload.kind == "system_event"` 的任务不允许用户删除（`remove_job` 返回 `"protected"`）或修改（`update_job` 返回 `"protected"`）

5. **一次性任务语义**：`kind="at"` 的任务执行后自动禁用 `enabled=False`；若设置了 `delete_after_run=True` 则从 store 中移除

6. **定时器 guard**：`_load_store` 在定时器回调执行期间（`_timer_active=True`）直接返回已有 store，防止并发覆盖

7. **运行历史修剪**：`run_history` 最多保留 20 条记录（`_MAX_RUN_HISTORY`），超出部分自动裁剪

---

## 测试覆盖

测试文件位于 `tests/cron/`：

| 文件 | 内容 |
|------|------|
| `test_cron_service.py`（566 行） | CronService 核心功能测试 |
| `test_cron_tool_list.py`（360 行） | CronTool 列表功能测试 |
| `test_cron_tool_schema_contract.py`（99 行） | CronTool Schema 契约测试 |