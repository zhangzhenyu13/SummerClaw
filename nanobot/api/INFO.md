# nanobot/api — OpenAI-compatible HTTP API Module

## 模块概述

`nanobot/api` 模块为 nanobot 提供了一套 **OpenAI API 兼容的 HTTP 服务**，使得任何兼容 OpenAI SDK 的客户端（如 openai-python、LangChain、Continue.dev 等）都能直接调用 nanobot 的 Agent 能力。所有请求路由到同一个持久化的 `AgentLoop` 会话。

## 文件结构

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块入口，声明 `OpenAI-compatible HTTP API for nanobot` |
| `server.py` | 核心实现，包含 aiohttp 应用工厂和所有路由处理器 |

## 核心架构

```
外部客户端（OpenAI SDK / curl / ...）
        │
        ▼
┌─────────────────────────────┐
│  aiohttp Web Application     │
│  (create_app 工厂函数构建)    │
│                              │
│  路由:                       │
│  POST /v1/chat/completions  │──► handle_chat_completions()
│  GET  /v1/models            │──► handle_models()
│  GET  /health               │──► handle_health()
└──────────┬──────────────────┘
           │ 调用 agent_loop.process_direct()
           ▼
┌─────────────────────────────┐
│  AgentLoop                   │
│  (nanobot.agent.loop)       │
│  持久化会话，支持记忆/技能   │
└─────────────────────────────┘
```

## 支持的 API 端点

### 1. `POST /v1/chat/completions` — 对话补全

OpenAI Chat Completions API 的兼容端点，支持两种请求方式：

**JSON 模式** (`application/json`):
```json
{
  "model": "nanobot",
  "messages": [{"role": "user", "content": "你的问题"}],
  "stream": false,
  "session_id": "可选会话ID"
}
```

**Multipart 模式** (`multipart/form-data`):
| 字段 | 说明 |
|------|------|
| `message` | 文本消息 |
| `session_id` | 可选，会话标识 |
| `model` | 可选，模型名称 |
| `files` | 上传文件（支持多文件，单文件上限 10MB） |

**流式响应 (SSE)**：设置 `stream: true` 后返回 OpenAI 兼容的 Server-Sent Events 流。

**关键行为**：
- 只支持单条 `user` 消息（多轮对话通过 `session_id` 保持上下文）
- 会话锁（`asyncio.Lock`）按 `session_key` 隔离，保证同一会话的请求串行执行
- 空响应自动重试一次，仍为空则返回兜底消息
- 超时返回 504，内部错误返回 500
- 文件大小超过 10MB 返回 413

### 2. `GET /v1/models` — 模型列表

返回当前配置的模型名称：
```json
{
  "object": "list",
  "data": [{"id": "nanobot", "object": "model", "created": 0, "owned_by": "nanobot"}]
}
```

### 3. `GET /health` — 健康检查

```json
{"status": "ok"}
```

## 核心组件详解

### 请求解析

- `_parse_json_content(body)` — 解析 JSON 请求体，提取文本和 base64 data URL 图片
- `_parse_multipart(request)` — 解析 multipart/form-data 请求，提取文本、文件、session_id、model
- `_save_base64_data_url(data_url, media_dir)` — 解码 base64 data URL 并保存到磁盘

### SSE 流式输出

- `_sse_chunk(delta, model, chunk_id, finish_reason)` — 生成单个 OpenAI 兼容的 SSE chunk
- 流内通过 `asyncio.Queue` 传递 token，最终发送 `[DONE]` 标记

### 响应构建

- `_chat_completion_response(content, model)` — 构建非流式 JSON 响应
- `_response_text(value)` — 规范化 `process_direct` 输出为纯文本
- `_error_json(status, message, err_type)` — 构建标准错误响应

### 应用工厂

`create_app(agent_loop, model_name="nanobot", request_timeout=120.0)` 创建 aiohttp 应用实例：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `agent_loop` | *(必填)* | 已初始化的 `AgentLoop` 实例 |
| `model_name` | `"nanobot"` | API 响应中报告的模型名 |
| `request_timeout` | `120.0` | 单次请求超时（秒） |

应用级配置：
- `client_max_size`: 20MB（支持 base64 大图片）
- `session_locks`: 按 session_key 隔离的 `asyncio.Lock` 字典

## 配置方式

API 服务器的运行时配置来自 `config.json` 中的 `api` 段：

```json
{
  "api": {
    "host": "127.0.0.1",
    "port": 8900,
    "timeout": 120.0
  }
}
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `host` | `127.0.0.1` | 绑定地址（默认仅本地） |
| `port` | `8900` | 监听端口 |
| `timeout` | `120.0` | 单次请求超时（秒） |

## 启动方式

通过 CLI 命令启动：

```bash
nanobot serve --port 8900 --host 127.0.0.1 --timeout 120
```

启动流程：
1. 加载 `config.json`
2. 创建 `MessageBus` → `LLM Provider` → `SessionManager` → `AgentLoop`
3. 调用 `create_app(agent_loop, ...)` 构建 aiohttp 应用
4. 注册 `on_startup`（连接 MCP）和 `on_cleanup`（关闭 MCP）回调
5. `web.run_app()` 启动 HTTP 服务

## 关键约束与注意事项

1. **单消息限制** — 每次请求仅支持一条 `user` 消息，多轮对话通过 `session_id` 维持历史
2. **远程图片不支持** — `image_url` 仅接受 base64 data URL，远程 URL 会返回错误
3. **会话隔离** — 不同 `session_id` 获得独立对话历史，默认会话键为 `api:default`
4. **模型校验** — 请求中的 `model` 必须与配置的模型名一致，否则返回 400
5. **空响应保护** — 自动重试 + 兜底消息，防止客户端收到空内容
6. **文件安全** — 通过 `safe_filename()` 消毒文件名，防止路径遍历攻击
7. **依赖 aiohttp** — 需通过 `pip install 'nanobot-ai[api]'` 安装

## 与 OpenAI SDK 的兼容性

| 特性 | 支持状态 |
|------|----------|
| Chat Completions (JSON) | ✅ |
| Chat Completions (Streaming/SSE) | ✅ |
| multipart/form-data 上传 | ✅ |
| /v1/models | ✅ |
| 多轮对话 (session_id) | ✅ |
| 图片 (base64 data URL) | ✅ |
| 远程图片 URL | ❌ |
| 多 messages 数组 | ❌（仅支持单 user 消息） |
| tool_choice / function_call | ❌ |
| 真实 token 计数 | ❌（返回 0） |

## 相关模块

| 模块 | 关系 |
|------|------|
| `nanobot.agent.loop` | API 调用 `AgentLoop.process_direct()` 处理请求 |
| `nanobot.cli.commands` | `serve` 命令组装并启动 API 服务 |
| `nanobot.config.schema` | `ApiConfig` 定义 API 配置项 |
| `nanobot.utils.runtime` | 提供 `EMPTY_FINAL_RESPONSE_MESSAGE` 兜底消息 |
| `nanobot.utils.evaluator` | `evaluate_response()` 用于心跳执行后的通知评估 |
| `nanobot.utils.helpers` | `safe_filename()`、`current_time_str()` 辅助函数 |