# Providers 模块

## 概述

`summerclaw/providers` 是 summerclaw 的 **LLM Provider 抽象层**，为上层 Agent 提供统一的 LLM 调用接口，屏蔽不同厂商 API 的差异。模块采用**策略模式**设计，通过 `LLMProvider` 抽象基类定义统一契约，各具体 Provider 实现对接不同 LLM 后端。

## 架构

```
providers/
├── __init__.py                  # 模块入口，懒加载导出
├── base.py                      # 抽象基类 & 核心数据模型
├── registry.py                  # Provider 注册表（30+ 厂商元数据）
├── anthropic_provider.py        # Anthropic 原生 SDK 集成（Claude）
├── openai_compat_provider.py    # OpenAI 兼容 API（覆盖大部分厂商）
├── openai_codex_provider.py     # OpenAI Codex（OAuth 认证）
├── github_copilot_provider.py   # GitHub Copilot（OAuth 设备流认证）
├── azure_openai_provider.py     # Azure OpenAI（Responses API）
├── transcription.py             # 语音转文字（Whisper）
└── openai_responses/            # Responses API 共享工具
    ├── __init__.py
    ├── converters.py            # 消息/工具格式转换
    └── parsing.py               # SSE 流解析 & SDK 响应解析
```

## 核心数据模型 (`base.py`)

| 类 | 说明 |
|---|---|
| `ToolCallRequest` | 工具调用请求，含 `id`、`name`、`arguments`，可序列化为 OpenAI 格式 |
| `LLMResponse` | LLM 响应，含 `content`、`tool_calls`、`finish_reason`、`usage`、`reasoning_content`、`thinking_blocks` 及结构化错误元数据 |
| `GenerationSettings` | 默认生成参数（`temperature=0.7`、`max_tokens=4096`） |
| `LLMProvider` | **抽象基类**，定义 `chat()` / `chat_stream()` / `embed()` / `get_default_model()` 接口 |

### LLMProvider 核心能力

- **重试机制**：内置标准重试（3 次，间隔 1/2/4s）和持久化重试（`retry_mode="persistent"`）两种模式，自动识别瞬时错误（429/5xx/timeout/connection）
- **429 智能处理**：区分可重试（`rate_limit_exceeded`）与不可重试（`insufficient_quota`）的 429 错误
- **消息清洗**：`_sanitize_empty_content()` 处理空 content 块，`_enforce_role_alternation()` 强制角色交替并处理尾部 assistant 消息
- **图片处理**：`_strip_image_content()` 可将 `image_url` 块替换为文本占位符（用于不支持图片的模型降级）
- **心跳等待**：重试等待期间通过 `on_retry_wait` 回调向用户反馈进度
- **Embeddings**：`embed()` 接口默认抛出 `NotImplementedError`，由各子类按需覆写

## Provider 注册表 (`registry.py`)

`ProviderSpec` 是单个 Provider 的元数据描述，`PROVIDERS` 元组按**优先级排序**定义所有已注册的 Provider。添加新 Provider 只需向 `PROVIDERS` 添加一条 `ProviderSpec`。

### ProviderSpec 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 配置字段名，如 `"dashscope"` |
| `keywords` | `tuple[str, ...]` | 模型名匹配关键字（小写） |
| `env_key` | `str` | API Key 环境变量名 |
| `display_name` | `str` | 状态展示名 |
| `backend` | `str` | 实现类：`openai_compat` / `anthropic` / `azure_openai` / `openai_codex` / `github_copilot` |
| `is_gateway` | `bool` | 是否为网关（可路由任意模型，如 OpenRouter、AiHubMix） |
| `is_local` | `bool` | 是否为本地部署（Ollama、vLLM、LM Studio） |
| `is_oauth` | `bool` | 是否使用 OAuth 认证（Codex、Copilot） |
| `is_direct` | `bool` | 是否跳过 API Key 校验 |
| `strip_model_prefix` | `bool` | 是否去掉 `provider/` 前缀 |
| `supports_max_completion_tokens` | `bool` | 是否使用 `max_completion_tokens` 而非 `max_tokens` |
| `supports_prompt_caching` | `bool` | 是否支持 Prompt Caching |
| `default_api_base` | `str` | 默认 API Base URL |
| `model_overrides` | `tuple` | 模型级参数覆盖（如 kimi-k2.5 强制 `temperature=1.0`） |

### 已注册 Provider 一览

#### 网关（Gateway）—— 优先级最高
| Provider | 关键字 | Backend | 特性 |
|---|---|---|---|
| OpenRouter | `openrouter` | `openai_compat` | 全局网关，Prompt Caching |
| AiHubMix | `aihubmix` | `openai_compat` | 全局网关，去前缀 |
| SiliconFlow | `siliconflow` | `openai_compat` | 硅基流动 |
| VolcEngine | `volcengine`, `ark` | `openai_compat` | 火山引擎 |
| BytePlus | `byteplus` | `openai_compat` | 火山引擎国际版 |

#### 标准 Provider
| Provider | 关键字 | Backend | 说明 |
|---|---|---|---|
| Anthropic | `anthropic`, `claude` | `anthropic` | 原生 Anthropic SDK，支持扩展思考 + Prompt Caching |
| OpenAI | `openai`, `gpt` | `openai_compat` | GPT 系列，`max_completion_tokens` |
| DeepSeek | `deepseek` | `openai_compat` | DeepSeek 官方 API |
| Gemini | `gemini` | `openai_compat` | Google Gemini |
| DashScope | `qwen`, `dashscope` | `openai_compat` | 阿里通义千问，支持 thinking + 多模态 Embedding |
| Zhipu AI | `zhipu`, `glm` | `openai_compat` | 智谱 GLM |
| Moonshot | `moonshot`, `kimi` | `openai_compat` | 月之暗面 Kimi，k2.5 支持 thinking |
| MiniMax | `minimax` | `openai_compat` | MiniMax 标准 API |
| MiniMax Anthropic | `minimax_anthropic` | `anthropic` | MiniMax Anthropic 兼容端点 |
| Mistral | `mistral` | `openai_compat` | Mistral AI |
| Step Fun | `stepfun`, `step` | `openai_compat` | 阶跃星辰 |
| Xiaomi MIMO | `mimo` | `openai_compat` | 小米 MIMO |
| Groq | `groq` | `openai_compat` | 主要用于 Whisper 转录 |
| Qianfan | `qianfan`, `ernie` | `openai_compat` | 百度千帆 |

#### OAuth Provider
| Provider | 关键字 | Backend | 认证方式 |
|---|---|---|---|
| OpenAI Codex | `openai-codex` | `openai_codex` | OAuth（`oauth-cli-kit`） |
| GitHub Copilot | `copilot` | `github_copilot` | GitHub Device Flow OAuth |

#### 本地部署
| Provider | 关键字 | Backend | 默认地址 |
|---|---|---|---|
| vLLM/Local | `vllm` | `openai_compat` | 用户配置 |
| Ollama | `ollama`, `nemotron` | `openai_compat` | `http://localhost:11434/v1` |
| LM Studio | `lm-studio`, `lmstudio` | `openai_compat` | `http://localhost:1234/v1` |
| OpenVINO | `openvino`, `ovms` | `openai_compat` | `http://localhost:8000/v3` |

#### 直接 Provider
| Provider | 关键字 | Backend | 说明 |
|---|---|---|---|
| Custom | — | `openai_compat` | 用户自配置的 OpenAI 兼容端点 |
| Azure OpenAI | `azure`, `azure-openai` | `azure_openai` | Azure Responses API |

## Provider 实现详解

### AnthropicProvider (`anthropic_provider.py`)

- **SDK**：`anthropic.AsyncAnthropic`（原生 Python SDK）
- **消息转换**：OpenAI Chat Completions 格式 → Anthropic Messages API 格式（`_convert_messages()`），支持 `thinking_blocks`、`tool_result`、多模态图片
- **Prompt Caching**：`_apply_cache_control()` 在 system 消息、倒数第二条消息、工具列表边界注入 `ephemeral` 缓存标记
- **扩展思考**：支持 `adaptive`（自适应思考）、`low` / `medium` / `high` 预算挡位，自动调整 `max_tokens` 和 `temperature`
- **流式**：通过 `stream.text_stream` 逐块推送，带 `idle_timeout_s` 超时保护
- **错误处理**：解析 Anthropic 错误体，提取 `x-should-retry` 头、`status_code`、结构化错误类型/代码
- **默认模型**：`claude-sonnet-4-20250514`

### OpenAICompatProvider (`openai_compat_provider.py`)

- **SDK**：`openai.AsyncOpenAI`（可选 Langfuse 追踪包装）
- **适用范围**：覆盖注册表中所有 `backend="openai_compat"` 的 Provider（~20+ 厂商）
- **消息清洗**：标准化 tool_call ID（SHA1 截断 9 字符）、强制标准化 `function.arguments` 为合法 JSON、角色交替合并
- **Prompt Caching**：Anthropic 模型经过网关时，在 OpenAI 格式消息上注入 `cache_control` 标记
- **Responses API**：对 GPT-5 / o1 / o3 / o4 等推理模型自动路由到 `client.responses.create()`，失败时自动回退到 Chat Completions
- **Thinking 参数注入**：DashScope → `extra_body.enable_thinking`；火山引擎/BytePlus → `extra_body.thinking.type`；Kimi → `extra_body.thinking.type`
- **温度处理**：推理模型（GPT-5/o1/o3/o4）在 `reasoning_effort` 激活时自动跳过 `temperature` 参数
- **流式**：支持 Chat Completions 流式（`stream_options.include_usage`）和 Responses API 流式
- **Embeddings**：同步客户端调用 `/embeddings` 端点；DashScope 多模态 Embedding 模型（`tongyi-embedding-vision-*`、`qwen*-vl-embedding`）走原生 DashScope API
- **错误处理**：本地 Provider 连接失败时附加排查提示

### OpenAICodexProvider (`openai_codex_provider.py`)

- **认证**：OAuth（`oauth_cli_kit.get_token`）获取 Codex Token
- **API**：`POST https://chatgpt.com/backend-api/codex/responses`（SSE 流）
- **特性**：Prompt Cache Key（SHA256）、SSL 自动降级、`text.verbosity` 控制
- **默认模型**：`openai-codex/gpt-5.1-codex`

### GitHubCopilotProvider (`github_copilot_provider.py`)

- **继承**：`OpenAICompatProvider`
- **认证**：GitHub Device Flow OAuth → 换取 Copilot Access Token（带过期管理和自动刷新）
- **登录**：`login_github_copilot()` 通过设备码流程认证，token 持久化到本地文件
- **默认模型**：`github-copilot/gpt-4.1`

### AzureOpenAIProvider (`azure_openai_provider.py`)

- **SDK**：`openai.AsyncOpenAI`，`base_url = {endpoint}/openai/v1/`
- **API**：Azure Responses API（`client.responses.create()`）
- **要求**：必须提供 `api_key` 和 `api_base`
- **Embeddings**：同步客户端调用 Azure Embeddings 端点
- **默认模型**：`gpt-5.2-chat`

### 语音转录 (`transcription.py`)

| Provider | API | 模型 |
|---|---|---|
| `OpenAITranscriptionProvider` | `https://api.openai.com/v1/audio/transcriptions` | `whisper-1` |
| `GroqTranscriptionProvider` | `https://api.groq.com/openai/v1/audio/transcriptions` | `whisper-large-v3` |

### openai_responses 子模块

为使用 OpenAI Responses API 的 Provider（Codex、Azure OpenAI、OpenAICompat 推理模式）提供共享工具：

- **converters.py**：`convert_messages()` / `convert_tools()` / `convert_user_message()` 将 Chat Completions 格式转为 Responses API 格式
- **parsing.py**：`iter_sse()` / `consume_sse()` 解析 HTTP SSE 流；`consume_sdk_stream()` / `parse_response_output()` 解析 SDK 响应对象；`map_finish_reason()` 映射状态码

## 模块入口 (`__init__.py`)

采用**懒加载**模式——模块级别只导入 `LLMProvider` 和 `LLMResponse`，各 Provider 类通过 `__getattr__` 按需导入，避免初始化时加载所有后端 SDK。

```python
from summerclaw.providers import LLMProvider, LLMResponse, AnthropicProvider, OpenAICompatProvider, ...
```

## 关键设计决策

1. **重试集中化**：所有重试逻辑在 `LLMProvider._run_with_retry()` 中统一管理，各 Provider 的 `chat()` 只负责单次调用
2. **错误结构化**：`LLMResponse` 携带 `error_status_code`、`error_kind`、`error_type`、`error_code`、`error_should_retry` 等字段，支持精确的重试决策
3. **消息兼容性**：通过 `_sanitize_empty_content()`、`_sanitize_request_messages()`、`_enforce_role_alternation()` 等多层清洗，确保消息格式兼容各种 Provider 的严格校验
4. **Responses API 渐进式采用**：推理模型优先使用 Responses API，失败自动回退 Chat Completions，保证兼容性
5. **Embeddings 与 LLM 解耦**：`embed()` 是独立接口，各 Provider 按需覆写（OpenAI 兼容走同步 `/embeddings` 端点，DashScope 多模态走原生 API，Anthropic 不支持）