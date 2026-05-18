# summerclaw/config — 配置模块

## 概述

`summerclaw/config` 是 SummerClaw/summerclaw 的配置管理核心模块，负责**配置模型的 schema 定义**、**配置文件加载与持久化**、**配置迁移**、**环境变量插值**以及**运行时路径解析**。

整个模块以 Pydantic v2 / pydantic-settings 为基础，支持 camelCase 与 snake_case 双写、环境变量覆盖（`NANOBOT_` 前缀）、以及配置的前向兼容迁移。

## 文件结构

| 文件 | 职责 |
|---|---|
| `__init__.py` | 模块入口，统一导出 `Config`、`load_config` 及所有路径工具函数 |
| `schema.py` | Pydantic 配置模型定义，是整个模块的核心 |
| `loader.py` | 配置文件的加载、保存、环境变量解析、旧格式迁移 |
| `paths.py` | 运行时目录路径推导（基于当前 config 实例的位置） |

## 核心数据流

```
config.json (磁盘)
     │
     ▼
loader.load_config() ◄── env(NANOBOT_*)
     │
     ├── json.load  →  dict
     ├── _migrate_config(dict)   ← 旧格式兼容
     ├── Config.model_validate(data)
     ├── _apply_ssrf_whitelist(config)
     │
     ▼
  Config 实例（运行时单例）
```

## 各模块详解

### 1. `schema.py` — 配置模型

定义根模型 `Config(BaseSettings)`，采用层级化结构：

```
Config  (root, BaseSettings, env_prefix="NANOBOT_")
├── agents: AgentsConfig
│   └── defaults: AgentDefaults
│       ├── 核心 Agent 参数（model, provider, max_tokens, temperature...）
│       ├── execution_mode: "simple" | "plan" | "search-plan" | "auto"
│       ├── memory_algorithm: 记忆算法名称
│       ├── max_subagent_depth / max_replan_iterations
│       ├── session_ttl_minutes: 空闲会话自动压缩阈值
│       ├── embedding: EmbeddingConfig
│       ├── dream: DreamConfig
│       ├── skill_autogen: SkillAutogenConfig
│       ├── search_enhanced_planning: SearchEnhancedPlanningConfig
│       └── injection: InjectionConfig
├── channels: ChannelsConfig (extra="allow"，支持插件通道)
├── providers: ProvidersConfig (28 个 LLM 提供商配置)
│   ├── anthropic, openai, deepseek, groq, zhipu, dashscope...
│   ├── vllm, ollama, lm_studio (本地推理)
│   ├── siliconflow, volcengine, byteplus (国内平台)
│   └── openai_codex, github_copilot (OAuth，exclude=True)
├── api: ApiConfig (OpenAI 兼容 API 服务，默认 127.0.0.1:8900)
├── gateway: GatewayConfig (网关服务，默认 127.0.0.1:18790)
│   └── heartbeat: HeartbeatConfig (每 30 分钟心跳)
├── proxy_pool: ProxyPoolConfig (IP 代理池，含健康检查与磁盘缓存)
└── tools: ToolsConfig
    ├── web: WebToolsConfig (含 search 子配置)
    ├── browser: BrowserToolsConfig (Playwright 无头浏览器)
    ├── exec: ExecToolConfig (Shell 执行)
    ├── my: MyToolConfig (自检工具)
    ├── mcp_servers: dict[str, MCPServerConfig] (MCP 服务连接)
    ├── restrict_to_workspace (工作区限制)
    └── ssrf_whitelist (SSRF 白名单)
```

**关键设计点：**

- **`Config._match_provider()`** — 根据模型名自动匹配提供商。通过前缀匹配 → 关键词匹配 → 本地 fallback（Ollama 等）→ 通用 fallback 的优先级链确定使用的 provider。
- **`Base` 模型** — 所有子模型继承此基类，统一启用 `alias_generator=to_camel` + `populate_by_name=True`，使 JSON 中的 `camelCase` 键和 Python 中的 `snake_case` 均可正确解析。
- **环境变量覆盖** — `BaseSettings` 配置了 `env_prefix="NANOBOT_"` 和 `env_nested_delimiter="__"`，支持如 `NANOBOT_AGENTS__DEFAULTS__MODEL=...` 的环境变量注入。
- **`EmbeddingConfig`** — 独立于 LLM provider 的嵌入模型配置，支持 `provider="auto"`（继承 LLM 凭证）、指定 provider、或 `provider="local"`（本地 Sentence-Transformers）。

### 2. `loader.py` — 加载与持久化

核心函数：

| 函数 | 功能 |
|---|---|
| `load_config(path)` | 读取 JSON 配置文件 → 迁移旧格式 → Pydantic 校验 → 返回 `Config`。失败时回退到默认配置 |
| `save_config(config, path)` | 将 Config 序列化为 JSON 写入文件 |
| `resolve_config_env_vars(config)` | 递归解析配置值中的 `${VAR}` 环境变量引用 |
| `set_config_path(path)` / `get_config_path()` | 多实例支持：设置/获取当前配置路径（默认 `~/.summerclaw/config.json`） |
| `_migrate_config(data)` | 前向兼容迁移（详见下文） |
| `_apply_ssrf_whitelist(config)` | 将 SSRF 白名单同步到网络安全模块 |

**配置迁移（`_migrate_config`）**：

1. `tools.exec.restrictToWorkspace` → `tools.restrictToWorkspace`（字段提升）
2. `tools.myEnabled` / `tools.mySet` → `tools.my.enable` / `tools.my.allowSet`（扁平键嵌套化）
3. `tools.proxyPool` → `proxyPool`（提升到顶层，因为代理池属于基础设施而非工具）

### 3. `paths.py` — 路径解析

提供基于当前 config 实例位置推导的运行时目录路径。所有路径函数通过 `get_config_path()` 获取配置位置来推导数据目录：

| 函数 | 返回路径（默认） |
|---|---|
| `get_data_dir()` | `~/.summerclaw/` |
| `get_runtime_subdir(name)` | `~/.summerclaw/{name}/`（自动创建） |
| `get_media_dir(channel?)` | `~/.summerclaw/media/`（支持按 channel 命名空间） |
| `get_cron_dir()` | `~/.summerclaw/cron/` |
| `get_logs_dir()` | `~/.summerclaw/logs/` |
| `get_workspace_path(workspace?)` | `~/.summerclaw/workspace/`（可自定义） |
| `get_cli_history_path()` | `~/.summerclaw/history/cli_history` |
| `get_bridge_install_dir()` | `~/.summerclaw/bridge/`（WhatsApp Bridge） |
| `get_legacy_sessions_dir()` | `~/.summerclaw/sessions/`（迁移 fallback） |
| `is_default_workspace(workspace)` | 判断工作区是否为默认路径 |

## 导出接口

模块通过 `__init__.py` 对外暴露以下公共 API：

```python
from summerclaw.config import (
    Config,                    # 配置模型类
    load_config,              # 加载配置
    get_config_path,          # 获取配置路径
    get_data_dir,             # 获取数据目录
    get_runtime_subdir,       # 获取运行时子目录
    get_media_dir,            # 获取媒体目录
    get_cron_dir,             # 获取 cron 目录
    get_logs_dir,             # 获取日志目录
    get_workspace_path,       # 获取工作区路径
    is_default_workspace,     # 判断是否默认工作区
    get_cli_history_path,     # 获取 CLI 历史路径
    get_bridge_install_dir,   # 获取 Bridge 安装路径
    get_legacy_sessions_dir,  # 获取旧版会话目录
)
```

## 关键特性

1. **多实例支持**：通过 `set_config_path()` 可切换不同配置实例，路径函数自动跟随
2. **前向兼容**：旧版配置键自动迁移，升级无需手动修改 `config.json`
3. **环境变量注入**：所有配置值支持 `${VAR}` 插值，`BaseSettings` 支持 `NANOBOT_*` 前缀覆盖
4. **camelCase / snake_case 双写**：JSON 使用 camelCase，Python 使用 snake_case，Pydantic alias 桥接
5. **容错回退**：配置加载失败时使用默认配置，不会阻断启动
6. **自动目录创建**：`paths.py` 中所有路径函数通过 `ensure_dir()` 自动创建缺失目录

## 相关测试

- `tests/config/test_config_migration.py` — 配置迁移逻辑测试
- `tests/config/test_config_paths.py` — 路径函数测试
- `tests/config/test_dream_config.py` — Dream 调度配置测试
- `tests/config/test_env_interpolation.py` — 环境变量插值测试