# summerclaw/channels — 聊天通道模块

## 概述

`summerclaw/channels` 是 summerclaw 的**多平台聊天通道适配层**，采用插件架构为 Agent 提供与 14 种聊天平台的无缝集成能力。每个通道以统一的 `BaseChannel` 接口与消息总线（`MessageBus`）交互，使 Agent 核心完全无需感知消息来源平台的差异。

通道通过 `ChannelManager` 统一管理生命周期（启动/停止/状态监控）和出站消息路由，支持流式输出（Streaming）、指数退避重试、媒体文件上下传、语音转文字等高级特性。

---

## 模块结构

```
summerclaw/channels/
├── __init__.py      # 模块入口，导出 BaseChannel 和 ChannelManager
├── base.py          # 通道抽象基类（BaseChannel）
├── manager.py       # 通道管理器（ChannelManager）
├── registry.py      # 通道自动发现与外部插件加载
├── telegram.py      # Telegram Bot（long-polling）
├── discord.py       # Discord Bot（discord.py）
├── slack.py         # Slack Bot（Slack SDK）
├── whatsapp.py      # WhatsApp（Node.js Bridge + Baileys）
├── weixin.py        # 微信个人号（ilinkai HTTP long-poll）
├── feishu.py        # 飞书（Lark Open API）
├── dingtalk.py      # 钉钉（DingTalk Open API）
├── wecom.py         # 企业微信（WeCom API）
├── qq.py            # QQ Bot（官方 QQ Bot API）
├── mochat.py        # MoChat 聚合聊天平台
├── matrix.py        # Matrix 协议（matrix-nio）
├── msteams.py       # Microsoft Teams（Bot Framework）
├── email.py         # 邮件通道（IMAP 收件 + SMTP 发件）
└── websocket.py     # WebSocket Server（本地客户端接入）
```

每个通道文件遵循统一模式：
- **`XXXConfig(Base)`** — Pydantic 配置模型，声明 `enabled`、`token`、`allow_from` 等字段
- **`XXXChannel(BaseChannel)`** — 通道实现类，实现 `start()` / `stop()` / `send()` / `send_delta()` 等接口

---

## 核心组件

### 1. BaseChannel（`base.py`）

所有通道的抽象基类，定义了通道必须实现的契约接口：

| 方法 | 说明 |
|------|------|
| `start()` | 启动通道，进入 long-running 监听循环 |
| `stop()` | 停止通道，清理资源 |
| `send(msg: OutboundMessage)` | 发送完整消息到目标平台 |
| `send_delta(chat_id, delta, metadata)` | 流式增量发送（逐 chunk 编辑消息） |
| `login(force)` | 交互式登录（扫码等），默认返回 `True` |
| `transcribe_audio(file_path)` | 语音转文字（Whisper / Groq） |
| `is_allowed(sender_id)` | 权限校验（基于 `allow_from` 白名单） |
| `notify_startup(message)` | 系统启动通知钩子 |
| `default_config()` | 返回默认配置字典供 onboard 使用 |

**关键属性：**

| 属性 | 说明 |
|------|------|
| `name` | 通道标识（如 `"telegram"`、`"discord"`） |
| `display_name` | 人类可读名称 |
| `supports_streaming` | 是否支持流式输出（需同时启用 streaming 配置 + 覆写 `send_delta`） |
| `is_running` | 通道运行状态 |

**`_handle_message()` 内部流程**：
1. `is_allowed()` 权限校验
2. 注入 `_wants_stream` 元数据（若支持流式）
3. 构造 `InboundMessage` → 发布到 `bus.publish_inbound()`

### 2. ChannelManager（`manager.py`）

通道生命周期管理器，负责通道的发现、初始化、启动、停止和出站消息路由。

**初始化流程（`_init_channels()`）**：
1. 通过 `registry.discover_all()` 发现所有可用通道类
2. 遍历配置，仅实例化 `enabled: true` 的通道
3. 注入全局 transcription 配置（provider / api_key / api_base）
4. 校验 `allow_from` 白名单（空列表直接 `SystemExit`）

**出站消息分发（`_dispatch_outbound()`）**：
- 后台 asyncio 任务持续消费 `bus.outbound` 队列
- 按 `msg.channel` 字段路由到对应通道
- **流式合并**（`_coalesce_stream_deltas`）：对相同 `(channel, chat_id)` 的连续 `_stream_delta` 消息进行批次合并，减少 API 调用频率
- **指数退避重试**（`_send_with_retry`）：失败后按 1s → 2s → 4s 延迟重试，最多 `send_max_retries` 次
- **进度消息过滤**：按 `send_progress` / `send_tool_hints` 配置过滤 `_progress` 类型消息

**关键配置项（`config.channels`）**：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `send_progress` | `true` | 是否向通道推送 Agent 文本进度 |
| `send_tool_hints` | `false` | 是否推送工具调用提示 |
| `send_max_retries` | `3` | 消息发送最大重试次数 |
| `transcription_provider` | `"groq"` | 语音转写后端（`groq` / `openai`） |

### 3. Registry（`registry.py`）

通道自动发现与插件系统。

- **内置通道发现**（`discover_channel_names`）：使用 `pkgutil.iter_modules()` 扫描 `summerclaw/channels/` 包，排除 `base`、`manager`、`registry`
- **通道类加载**（`load_channel_class`）：动态导入模块，反射查找第一个 `BaseChannel` 子类
- **外部插件发现**（`discover_plugins`）：通过 `entry_points(group="summerclaw.channels")` 加载第三方通道插件
- **优先级**：内置通道优先 — 外部插件无法覆盖内置通道同名实现

```python
# 第三方通道插件注册示例 (pyproject.toml)
[project.entry-points."summerclaw.channels"]
my_channel = "my_package.channel:MyChannel"
```

---

## 数据流

```
                    publish_inbound()
┌──────────┐ ────────────────────────► ┌──────────────┐
│ Channel  │                           │              │
│ (任意平台) │                           │  MessageBus  │
│          │ ◄──────────────────────── │              │
└──────────┘   send() / send_delta()   └──────────────┘
       ▲                                      │
       │          ChannelManager              │
       └──────── _dispatch_outbound() ────────┘
```

1. **入站**：Channel 收到平台消息 → `_handle_message()` → `bus.publish_inbound()` → `AgentLoop` 消费
2. **出站**：`AgentLoop` 发布 `OutboundMessage` → `bus.outbound` 队列 → `ChannelManager._dispatch_outbound()` 消费 → 按 `msg.channel` 路由 → `channel.send()` 或 `channel.send_delta()`

---

## 通道详解

### Telegram（`telegram.py`）— 1082 行

使用 `python-telegram-bot` 库的 long-polling 模式，无需 webhook 或公网 IP。

| 特性 | 说明 |
|------|------|
| 协议 | Telegram Bot API（long-polling） |
| 认证 | Bot Token |
| 流式 | ✅ 渐进编辑（`edit_message_text`），可配 `stream_edit_interval` |
| 媒体 | 图片/语音/视频/文件/位置 全支持；群组 media_group 缓冲聚合 |
| 群组策略 | `open`（响应所有）/ `mention`（仅 @提及） |
| 交互反馈 | typing 指示器 + emoji 反应 + HTML 渲染 |

### Discord（`discord.py`）— 681 行

使用 `discord.py` 库，支持 Slash Commands 和流式消息编辑。

| 特性 | 说明 |
|------|------|
| 协议 | Discord Gateway（WebSocket） |
| 认证 | Bot Token + Intents |
| 流式 | ✅ 渐进编辑 |
| 媒体 | 附件下载（最大 25MB），Slash 命令支持 |
| 群组策略 | `open` / `mention` |
| 交互反馈 | typing 指示器 + 阅读回执 emoji + 工作中 emoji |

### Slack（`slack.py`）— 17.5KB

使用 Slack Bolt SDK 的 Socket Mode（无需公网 URL）。

| 特性 | 说明 |
|------|------|
| 协议 | Slack Socket Mode（WebSocket） |
| 认证 | Bot Token + App Token |
| 群组策略 | `open` / `mention` |
| 特殊能力 | DM 通道支持（`SlackDMConfig`）、线程回复 |

### WhatsApp（`whatsapp.py`）— 358 行

通过 Node.js Bridge（基于 `@whiskeysockets/baileys`）连接 WhatsApp Web 协议。

| 特性 | 说明 |
|------|------|
| 协议 | WhatsApp Web（Baileys） |
| 通信方式 | Python ↔ Node.js Bridge（WebSocket） |
| 认证 | QR 码扫码登录 + Bridge Token |
| 群组策略 | `open` / `mention` |
| 媒体 | 图片/文件 发送和接收 |

### 微信个人号（`weixin.py`）— 1417 行

基于逆向工程的 `ilinkai.weixin.qq.com` HTTP long-poll API，无需本地微信客户端。

| 特性 | 说明 |
|------|------|
| 协议 | ilinkai HTTP long-poll（逆向自 `@tencent-weixin/openclaw-weixin`） |
| 认证 | QR 码扫码登录 → bot_token |
| 媒体 | 图片/语音/视频/文件，AES-128-ECB 加解密 |
| 交互反馈 | typing 指示器（`sendtyping` API） |
| 状态持久化 | account.json 保存 token / context_tokens / typing_tickets |

### 飞书（`feishu.py`）— 64.6KB（最大通道文件）

功能最丰富的通道实现，支持事件订阅、卡片消息、多模态交互。

| 特性 | 说明 |
|------|------|
| 协议 | 飞书开放平台（Event + Webhook） |
| 认证 | App ID + App Secret |
| 特殊能力 | 卡片消息、图片/文件发送、多模态支持 |

### 钉钉（`dingtalk.py`）— 24.4KB

接入钉钉开放平台机器人。

| 特性 | 说明 |
|------|------|
| 协议 | 钉钉开放平台 API |
| 认证 | App Key + App Secret |
| 群组策略 | `open` / `mention` |

### 企业微信（`wecom.py`）— 20.8KB

接入企业微信 API。

| 特性 | 说明 |
|------|------|
| 协议 | 企业微信开放 API |
| 认证 | Corp ID + Secret |

### QQ（`qq.py`）— 24.0KB

接入 QQ 官方 Bot API。

| 特性 | 说明 |
|------|------|
| 协议 | QQ 开放平台 Bot API（WebSocket） |
| 认证 | App ID + Token |

### MoChat（`mochat.py`）— 37.0KB

接入 MoChat 聚合聊天平台。

### Matrix（`matrix.py`）— 36.3KB

接入 Matrix 去中心化通信协议。

| 特性 | 说明 |
|------|------|
| 协议 | Matrix Client-Server API（使用 `matrix-nio`） |
| 认证 | User ID + Password / Access Token |

### Microsoft Teams（`msteams.py`）— 20.8KB

接入 Microsoft Teams Bot Framework。

| 特性 | 说明 |
|------|------|
| 协议 | Bot Framework REST API |
| 认证 | App ID + App Password |

### Email（`email.py`）— 23.9KB

通过 IMAP 接收邮件和 SMTP 发送回复。

| 特性 | 说明 |
|------|------|
| 协议 | IMAP（收件）+ SMTP（发件） |
| 认证 | Email + Password / App Password |

### WebSocket Server（`websocket.py`）— 458 行

本地 WebSocket 服务器，供同一网络内的客户端直接连接。

| 特性 | 说明 |
|------|------|
| 协议 | WebSocket（`ws://` 或 `wss://`） |
| 认证 | Token（静态 token + 动态签发 token） |
| 流式 | ✅ `delta` / `stream_end` JSON 事件 |
| 安全 | TLS 支持（ssl_certfile / ssl_keyfile）、token 鉴权、单次 token 消耗 |
| 会话 | 每个 WebSocket 连接分配唯一 `chat_id` |

---

## 扩展：自定义通道插件

通过 `entry_points` 机制注册第三方通道，无需修改 summerclaw 源码：

```python
# my_package/my_channel.py
from summerclaw.channels.base import BaseChannel
from summerclaw.bus.queue import MessageBus

class MyChannel(BaseChannel):
    name = "my_channel"
    display_name = "My Platform"

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def send(self, msg) -> None:
        ...
```

```toml
# pyproject.toml
[project.entry-points."summerclaw.channels"]
my_channel = "my_package.my_channel:MyChannel"
```

---

## 设计原则

1. **统一接口，平台解耦**：Agent 核心只与 `MessageBus` 交互，完全不感知消息来源平台。所有通道遵循相同的 `start/stop/send/send_delta` 契约。
2. **插件化架构**：内置通道通过 pkgutil 自动发现，外部通道通过 entry_points 注册，扩展无需修改框架代码。
3. **流式优先**：支持 `send_delta` 的通道自动启用渐进式消息编辑，`ChannelManager` 负责 delta 合并优化，减少 API 调用。
4. **弹性投递**：`ChannelManager` 统一重试策略（指数退避），各通道只需在失败时抛出异常即可。
5. **安全默认**：空 `allow_from` 列表拒绝所有访问，非空白名单精确控制。`ChannelManager._validate_allow_from()` 启动时强校验。
6. **自包含状态持久化**：每个通道自行管理持久化状态（如微信的 `account.json`、WhatsApp 的 `bridge-token`），与 Agent 会话状态解耦。

---

## 相关模块

| 模块 | 关系 |
|------|------|
| `summerclaw/bus/` | 消息总线，Channel 通过 `MessageBus` 与 Agent 核心通信 |
| `summerclaw/config/schema.py` | `ChannelsConfig` 全局通道配置 + 各通道 `XXXConfig` 定义 |
| `summerclaw/config/paths.py` | 媒体文件目录、运行时子目录、Bridge 安装路径 |
| `summerclaw/providers/transcription.py` | 语音转写服务（Groq / OpenAI Whisper） |
| `summerclaw/utils/helpers.py` | `split_message` 长文本分块工具 |
| `summerclaw/security/network.py` | `validate_url_target` URL 安全性校验 |
| `summerclaw/cli/commands.py` | CLI 中 `channels login` / `channels status` 等管理命令 |
| `summerclaw/cli/onboard.py` | 初始化向导中生成各通道默认配置 |
| `bridge/` | WhatsApp 的 Node.js Bridge（Baileys 协议实现） |