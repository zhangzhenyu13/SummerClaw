# bus — 消息总线模块

## 概述

`bus` 模块是 nanobot 的消息中枢，负责在**聊天通道（Channel）**与**Agent 核心**之间实现解耦的异步消息传递。它采用发布-订阅模型，Channel 与 Agent 互不直接依赖，全部通过总线进行通信。

## 架构

```
┌──────────┐   publish_inbound()   ┌──────────────┐   consume_inbound()   ┌───────────┐
│ Channel  │ ────────────────────► │              │ ────────────────────► │           │
│ (Telegram│                       │  MessageBus  │                       │ AgentLoop │
│  Discord │ ◄──────────────────── │              │ ◄──────────────────── │           │
│  ...)    │   send() / dispatch   │ inbound      │   publish_outbound() │           │
└──────────┘                       │ outbound     │                       └───────────┘
       ▲                           └──────────────┘
       │                                  │
       └──────────────────────────────────┘
           ChannelManager.dispatch()
```

- **Channel → Bus → AgentLoop**：Channel 收到用户消息后，通过 `bus.publish_inbound()` 推入入站队列；AgentLoop 通过 `bus.consume_inbound()` 阻塞等待并消费。
- **AgentLoop → Bus → Channel**：Agent 处理完成后，通过 `bus.publish_outbound()` 推入出站队列；`ChannelManager` 后台任务轮询出站队列并路由到对应 Channel 的 `send()` 方法。

## 文件说明

| 文件 | 职责 |
|------|------|
| `__init__.py` | 模块入口，导出核心公开符号 |
| `events.py` | 定义消息事件数据结构 |
| `queue.py` | `MessageBus` 异步队列实现 |

## 核心 API

### `InboundMessage` (events.py)

从聊天通道接收入站消息的数据类。

| 字段 | 类型 | 说明 |
|------|------|------|
| `channel` | `str` | 来源通道标识（如 `telegram`、`discord`、`slack`） |
| `sender_id` | `str` | 发送者唯一标识 |
| `chat_id` | `str` | 会话/频道标识 |
| `content` | `str` | 消息文本内容 |
| `timestamp` | `datetime` | 消息时间戳（默认当前时间） |
| `media` | `list[str]` | 媒体文件 URL 列表 |
| `metadata` | `dict[str, Any]` | 通道特有的附加数据 |
| `session_key_override` | `str \| None` | 会话键覆盖（用于话题/线程级别的会话隔离） |

**计算属性：**

- `session_key`：返回 `session_key_override`（若设置）或自动拼接 `"{channel}:{chat_id}"`，作为会话唯一标识。

### `OutboundMessage` (events.py)

向聊天通道发送出站消息的数据类。

| 字段 | 类型 | 说明 |
|------|------|------|
| `channel` | `str` | 目标通道标识 |
| `chat_id` | `str` | 目标会话标识 |
| `content` | `str` | 回复文本内容 |
| `reply_to` | `str \| None` | 被回复消息的 ID |
| `media` | `list[str]` | 携带的媒体文件 |
| `metadata` | `dict[str, Any]` | 附加元数据 |

### `MessageBus` (queue.py)

异步消息总线，内部包含两个 `asyncio.Queue`：

- `inbound: asyncio.Queue[InboundMessage]` — 入站消息队列
- `outbound: asyncio.Queue[OutboundMessage]` — 出站消息队列

| 方法 | 说明 |
|------|------|
| `publish_inbound(msg)` | 将入站消息推入队列（Channel 调用） |
| `consume_inbound()` | 阻塞等待并返回下一条入站消息（AgentLoop 调用） |
| `publish_outbound(msg)` | 将出站消息推入队列（AgentLoop 调用） |
| `consume_outbound()` | 阻塞等待并返回下一条出站消息（ChannelManager 调用） |
| `inbound_size` | 当前入站队列中待处理消息数 |
| `outbound_size` | 当前出站队列中待投递消息数 |

## 数据流

### 入站（用户 → Agent）

1. Channel（如 Telegram Bot）监听平台消息
2. 收到消息后构造 `InboundMessage`，调用 `bus.publish_inbound()`
3. `AgentLoop` 主循环中 `await bus.consume_inbound()` 阻塞等待新消息
4. Agent 执行推理 → 工具调用 → 生成回复

### 出站（Agent → 用户）

1. Agent 生成回复，构造 `OutboundMessage`，调用 `bus.publish_outbound()`
2. `ChannelManager._dispatch_loop()` 后台任务通过 `bus.consume_outbound()` 消费
3. 根据 `OutboundMessage.channel` 字段路由到对应的 Channel 实例
4. 调用 `channel.send(msg)` 将消息投递到目标平台

## 消费者与生产者

### 入站消息生产者（发布者）

| 模块 | 说明 |
|------|------|
| `channels/telegram.py` | Telegram 用户消息 |
| `channels/discord.py` | Discord 用户消息 |
| `channels/slack.py` | Slack 用户消息 |
| `channels/whatsapp.py` | WhatsApp 用户消息 |
| `channels/matrix.py` | Matrix 用户消息 |
| `channels/msteams.py` | Microsoft Teams 消息 |
| `channels/dingtalk.py` | 钉钉用户消息 |
| `channels/wecom.py` | 企业微信消息 |
| `channels/weixin.py` | 微信消息 |
| `channels/email.py` | 邮件通道消息 |
| `channels/websocket.py` | WebSocket 通道消息 |
| `channels/qq.py` | QQ 消息 |
| `cli/commands.py` | CLI 交互式对话 |
| `agent/loop.py` | Agent 内部消息注入（子任务触发等） |
| `agent/subagent.py` | 子 Agent 消息注入 |

### 入站消息消费者

| 模块 | 说明 |
|------|------|
| `agent/loop.py` | Agent 主循环，作为消息处理引擎 |

### 出站消息生产者

| 模块 | 说明 |
|------|------|
| `agent/loop.py` | Agent 处理完成后发布回复 |
| `cli/commands.py` | Cron 定时任务、Heartbeat 心跳通知 |

### 出站消息消费者与路由器

| 模块 | 说明 |
|------|------|
| `channels/manager.py` | `ChannelManager` 后台 dispatch 任务，按 `channel` 字段路由 |

## 设计原则

1. **完全解耦**：Channel 实现无需了解 Agent 内部细节，Agent 无需关心消息来自哪个平台。双方仅通过 `MessageBus` 交互。
2. **异步非阻塞**：基于 `asyncio.Queue` 实现，天然支持高并发消息处理。
3. **会话隔离**：通过 `session_key` 属性自动为不同通道/会话建立独立上下文，支持话题线程级别的隔离（`session_key_override`）。
4. **类型安全**：所有消息数据使用 `@dataclass` 强类型定义，避免字典传递带来的运行时错误。
5. **零外部依赖**：仅依赖 Python 标准库 `asyncio` 和 `dataclasses`，无第三方依赖。

## 与 nanobot 其他模块的关系

```
nanobot/
├── bus/              ← 本模块（消息中枢）
├── channels/         → Channel 实现，依赖 bus 进行消息收发
├── agent/loop.py     → AgentLoop 通过 consume_inbound() 获取消息，
│                       处理完成后通过 publish_outbound() 发送回复
├── cli/commands.py   → CLI 入口创建 MessageBus 实例并注入 ChannelManager 和 AgentLoop
├── agent/subagent.py → 子 Agent 也通过 bus.publish_inbound() 注入任务
└── cron/service.py   → 定时任务结果通过 bus 投递
```