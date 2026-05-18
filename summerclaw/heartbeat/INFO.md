# heartbeat 模块 — 心跳唤醒与后台任务轮询服务

## 概述

`heartbeat` 模块为 summerclaw agent 提供**周期性后台唤醒**机制。它定时读取 workspace 下的 `HEARTBEAT.md` 任务清单，借助 LLM 判断是否有待执行的任务，并在有任务时自动触发完整 agent 循环执行，最终将结果投递到用户的活跃聊天频道。

模块遵循**三阶段流水线**设计（决策 → 执行 → 通知评估），确保只有真正有意义的结果才推送给用户，避免噪音打扰。

## 核心设计

### 三阶段流水线

```
HEARTBEAT.md ──→ Phase 1: LLM 决策 ──→ skip（无任务，静默）
                    │
                    └──→ run（有任务）
                           │
                           └──→ Phase 2: Agent 执行
                                    │
                                    └──→ Phase 3: 评估通知
                                              │
                                     ┌────────┴────────┐
                                     ▼                  ▼
                              通知用户              静默丢弃
                            (有实际意义)        (例行/无结果)
```

### Phase 1 — 虚拟工具调用决策

通过定制的 `heartbeat` 工具定义，让 LLM 以结构化方式返回决策，而非依赖自由文本解析或不稳定的 `HEARTBEAT_OK` token：

```python
_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "parameters": {
                "properties": {
                    "action": {"enum": ["skip", "run"]},
                    "tasks": {"description": "Natural-language summary of active tasks"},
                },
                "required": ["action"],
            },
        },
    }
]
```

- `skip`：无活跃任务，跳过本轮
- `run`：存在活跃任务，附带任务描述，进入 Phase 2

### Phase 2 — 完整 Agent 执行

当决策为 `run` 时，通过 `on_execute` 回调将任务描述送入完整 Agent 循环（`agent.process_direct`）。执行后保留最近的 heartbeat 会话历史（`keep_recent_messages`），确保短期上下文不丢失同时控制历史体量。

### Phase 3 — 通知评估（与 `summerclaw.utils.evaluator` 协同）

执行结果不会直接推送——先经过 `evaluate_response()` 做一次轻量 LLM 评估：
- 结果有意义/可操作 → 通知用户
- 结果为例行/空内容 → 静默丢弃

评估失败时**默认通知**（fail-open），确保重要信息不会因评估异常而丢失。

## 核心组件

### `HeartbeatService` (service.py)

| 属性/方法 | 说明 |
|---|---|
| `workspace` | 工作区路径，`HEARTBEAT.md` 位于 `workspace / "HEARTBEAT.md"` |
| `provider` | LLM 提供者，用于 Phase 1 决策 |
| `model` | 使用的模型标识 |
| `on_execute` | Phase 2 回调，接收任务描述，返回执行结果字符串 |
| `on_notify` | Phase 3 通知回调，接收评估后的响应文本 |
| `interval_s` | 轮询间隔（默认 1800s = 30 分钟） |
| `enabled` | 是否启用（默认 `True`） |
| `timezone` | 时区（传递给 LLM 的 `Current Time` 上下文） |
| `start()` | 启动心跳循环（创建 asyncio.Task） |
| `stop()` | 停止心跳循环（取消 Task） |
| `trigger_now()` | 手动立即触发一次心跳，不等待间隔 |

### 配置项 (`HeartbeatConfig`)

定义于 `summerclaw/config/schema.py`：

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | `bool` | `True` | 是否启用心跳服务 |
| `interval_s` | `int` | `1800` | 轮询间隔（秒） |
| `keep_recent_messages` | `int` | `8` | heartbeat 会话保留最近消息数 |

## 集成入口

心跳服务在 **Gateway 模式**下通过 CLI 命令 `summerclaw gateway` 启动（见 `summerclaw/cli/commands.py`）：

```python
heartbeat = HeartbeatService(
    workspace=config.workspace_path,
    provider=provider,
    model=agent.model,
    on_execute=on_heartbeat_execute,   # → agent.process_direct()
    on_notify=on_heartbeat_notify,     # → bus.publish_outbound()
    interval_s=hb_cfg.interval_s,
    enabled=hb_cfg.enabled,
    timezone=config.agents.defaults.timezone,
)
```

**通知目标路由策略**：优先使用最近活跃的非内部（`cli`/`system`）channel session，fallback 到 `cli:direct`（此时不投递外部消息）。

## 关键设计决策

1. **结构化工具调用替代文本解析**：Phase 1 用 LLM tool_call 替代 `HEARTBEAT_OK` 标记或自由文本匹配，消除了解析不可靠性。

2. **三阶段分离**：决策、执行、通知三个阶段的职责完全分离，便于独立测试和复用。

3. **Fail-open 通知评估**：`evaluate_response` 在任何异常时默认返回 `True`（通知），确保重要结果不会因评估问题丢失。

4. **会话历史裁剪**：heartbeat 保留最近 `keep_recent_messages` 条消息，平衡上下文连续性和内存/Token 开销。

5. **多通道路由**：通知自动选择用户最近活跃的聊天频道，实现无感的跨平台任务结果投递。

## 模块结构

```
summerclaw/heartbeat/
├── __init__.py        # 模块入口，导出 HeartbeatService
├── service.py         # HeartbeatService 核心实现（193 行）
└── INFO.md            # 本文件
```

### 相关模块

| 模块 | 关系 |
|---|---|
| `summerclaw/utils/evaluator.py` | Phase 3 通知评估，共享相同的 tool-call 模式 |
| `summerclaw/agent/loop.py` | Phase 2 通过 `agent.process_direct()` 执行任务 |
| `summerclaw/config/schema.py` | `HeartbeatConfig` 配置定义 |
| `summerclaw/cli/commands.py` | Gateway 入口，组装并启动 HeartbeatService |
| `summerclaw/bus/events.py` | `OutboundMessage` 事件，用于通知投递 |

### 测试

测试文件：`tests/agent/test_heartbeat_service.py`（289 行）

覆盖场景：
- 重复 `start()` 幂等性（不创建重复 Task）
- `trigger_now()` 手动触发完整流程
- Phase 1 `skip` → 不执行
- Phase 1 `run` → Phase 2 执行 → Phase 3 评估通知
- Phase 3 评估静默 → `on_notify` 不被调用
- `HEARTBEAT.md` 不存在 → 静默跳过
- tool_calls 被 finish_reason 抑制 → 回退到 skip