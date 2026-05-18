# nanobot CLI 模块

## 概述

`nanobot/cli` 是 nanobot 的命令行交互层，基于 [Typer](https://typer.tiangolo.com/) 框架构建了完整的 CLI 应用入口。模块负责将所有底层能力（Agent 循环、Gateway、Channel 管理、配置管理等）以命令行形式暴露给用户，支持**单次对话**、**交互式聊天**、**初始化引导向导**、**API 服务管理**等多种使用模式。

## 文件结构

```
nanobot/cli/
├── __init__.py      # 模块入口
├── commands.py      # 核心 CLI 命令定义（主文件，~1580 行）
├── models.py        # 模型信息辅助（onboard 向导用）
├── onboard.py       # 交互式初始化引导向导（~1024 行）
└── stream.py        # CLI 流式输出渲染器
```

## 依赖关系

### 外部依赖
| 库 | 用途 |
|---|---|
| `typer` | CLI 应用框架，定义命令、参数、选项 |
| `prompt_toolkit` | 交互式输入（支持粘贴、历史、编辑） |
| `rich` | 终端富文本渲染（Markdown、Table、Panel） |
| `questionary` (可选) | onboard 向导的交互式问卷 |
| `loguru` | 日志输出 |

### 内部依赖
- `nanobot.agent.loop.AgentLoop` — Agent 核心循环
- `nanobot.bus.queue.MessageBus` — 消息总线
- `nanobot.config.loader` / `config.schema` — 配置加载与 Schema
- `nanobot.config.paths` — 路径管理（workspace、history）
- `nanobot.providers.registry` — LLM Provider 注册中心
- `nanobot.channels.registry` / `channels.manager` — Channel 注册与管理
- `nanobot.cron.service` — 定时任务服务
- `nanobot.heartbeat.service` — 心跳服务
- `nanobot.session.manager` — 会话管理
- `nanobot.api.server` — OpenAI 兼容 API Server

## 核心命令一览

### 1. `nanobot onboard` — 初始化配置

```bash
nanobot onboard [--workspace <path>] [--config <path>] [--wizard]
```

- 创建/刷新 `config.json`
- `--wizard` 启用交互式引导向导，支持配置：
  - LLM Provider（API Key、endpoint）
  - Chat Channel（Telegram、WhatsApp、微信等）
  - Agent Settings（模型、温度、上下文窗口等）
  - Gateway 设置
  - Tools 设置
  - 配置摘要查看

### 2. `nanobot agent` — Agent 对话

```bash
nanobot agent -m "Hello!"                   # 单次对话
nanobot agent                                # 交互式聊天（默认）
nanobot agent --session cli:my-session       # 指定会话 ID
nanobot agent --logs                         # 显示运行时日志
nanobot agent --no-markdown                  # 纯文本输出
```

- **单次模式**：`-m` 发送消息后立即退出
- **交互模式**：持续对话，支持 `exit`/`quit`/`Ctrl+C` 退出
- 使用 `prompt_toolkit` 提供历史记录（存储在 workspace CLI 历史文件中）

### 3. `nanobot serve` — OpenAI 兼容 API 服务

```bash
nanobot serve [--port <port>] [--host <host>] [--timeout <s>] [--verbose]
```

- 启动 `/v1/chat/completions` 端点
- 通过 `aiohttp` 提供 HTTP 服务
- 支持 MCP 连接自动管理

### 4. `nanobot gateway` — 全功能网关

```bash
nanobot gateway [--port <port>] [--verbose]
```

- 启动完整网关，包括：
  - Agent 消息循环
  - 全部 Channel 连接
  - Cron 定时任务（Dream 记忆整合、用户自定义定时任务）
  - Heartbeat 心跳服务
  - Health endpoint（`/health`）
- 启动时输出详细的运行状态摘要

### 5. `nanobot channels` — Channel 管理

```bash
nanobot channels status          # 查看 Channel 状态（启用/禁用）
nanobot channels login <name>    # 交互式登录（如 WhatsApp QR 码）
```

### 6. `nanobot plugins` — 插件管理

```bash
nanobot plugins list             # 列出所有发现的 Channel（内置 + 插件）
```

### 7. `nanobot status` — 状态查看

显示当前配置状态：配置文件位置、Workspace、模型、Provider API Key 状态。

### 8. `nanobot provider login` — OAuth 登录

```bash
nanobot provider login openai-codex
nanobot provider login github-copilot
```

- 支持 OpenAI Codex 和 GitHub Copilot 的 OAuth 设备流认证

## 关键组件详解

### SafeFileHistory（commands.py）

`prompt_toolkit` 的 `FileHistory` 子类，在写入前对 surrogate 字符做安全转义，避免 Windows 上特殊 Unicode 输入（emoji、混编字符）导致崩溃。

### StreamRenderer（stream.py）

基于 Rich Live 的流式 Markdown 渲染器，核心特性：

- **`ThinkingSpinner`**：在模型思考时显示旋转动画，支持暂停/恢复
- **流式渲染**：delta 到达时实时刷新 Live 面板，支持 Markdown 渲染
- **自动管理**：spinner → 首个可见 delta → 激活 Live 渲染 → `on_end` 停止
- **速率控制**：150ms 最小刷新间隔避免闪烁
- **input 安全**：`stop_for_input()` 在用户输入前停止 spinner

### Onboard 向导（onboard.py）

交互式配置引导系统，核心设计：

- **`OnboardResult`**：封装配置结果和是否保存标志
- **Pydantic 模型驱动**：自动识别字段类型（bool/int/float/list/dict/model），生成对应输入界面
- **敏感字段掩码**：自动检测 `api_key`/`token`/`secret` 等字段，显示时只展示后 4 位
- **`_select_with_back()`**：基于 `prompt_toolkit` 的自定义选择器，支持 Escape/Left 回退
- **模型自动补全**：使用 `DynamicModelCompleter` 提供模型名自动补全
- **上下文窗口自动填充**：选择模型后自动查询推荐 `context_window_tokens`
- **配置摘要**：以 Rich Table 展示所有配置项的当前值
- **未保存变更检测**：退出时检测是否有未保存的修改并提醒

### Provider 工厂（`_make_provider()`）

根据 `ProviderSpec.backend` 路由到对应 Provider 实现：
- `openai_compat` → `OpenAICompatProvider`
- `openai_codex` → `OpenAICodexProvider`
- `azure_openai` → `AzureOpenAIProvider`
- `github_copilot` → `GitHubCopilotProvider`
- `anthropic` → `AnthropicProvider`

## 测试

测试文件位于 `tests/cli/`：

| 测试文件 | 覆盖内容 |
|---|---|
| `test_commands.py` | CLI 命令功能测试（~46 KB） |
| `test_cli_input.py` | 交互式输入处理测试 |
| `test_restart_command.py` | 重启通知相关测试 |
| `test_safe_file_history.py` | SafeFileHistory 安全写入测试 |

## 技术要点

1. **终端兼容性**：Windows 上强制 UTF-8 编码并重新配置 stdout/stderr
2. **信号处理**：交互模式下注册 SIGINT/SIGTERM/SIGHUP 处理器，退出时恢复终端设置
3. **进程间通信**：交互模式通过 MessageBus 路由消息，支持流式 delta/progress/响应分离
4. **配置热加载**：`_load_runtime_config()` 支持通过 `--config` / `--workspace` 运行时覆盖
5. **废弃项提示**：自动检测 `config.json` 中已废弃的 `memoryWindow` 等字段并提示用户删除
6. **Bridge 构建**：WhatsApp 等 channel 需要 Node.js bridge，`_get_bridge_dir()` 自动检测并构建