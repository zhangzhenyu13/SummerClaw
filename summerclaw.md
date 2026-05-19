## 📦 Install

**Install from source** (latest features, experimental changes may land here first; recommended for development)

```bash
git clone https://github.com/HKUDS/summerclaw.git
cd summerclaw
pip install -e .
```

**Install with [uv](https://github.com/astral-sh/uv)** (stable release, fast)

```bash
uv tool install summerclaw-ai
```

**Install from PyPI** (stable release)

```bash
pip install summerclaw-ai
```

### Update to latest version

**PyPI / pip**

```bash
pip install -U summerclaw-ai
summerclaw --version
```

**uv**

```bash
uv tool upgrade summerclaw-ai
summerclaw --version
```

**Using WhatsApp?** Rebuild the local bridge after upgrading:

```bash
rm -rf ~/.summerclaw/bridge
summerclaw channels login whatsapp
```

## 🚀 Quick Start

> [!TIP]
> Set your API key in `~/.summerclaw/config.json`.
> Get API keys: [OpenRouter](https://openrouter.ai/keys) (Global)
>
> For other LLM providers, please see the [Providers](#providers) section.
>
> For web search capability setup, please see [Web Search](#web-search).

**1. Initialize**

```bash
summerclaw onboard
```

Use `summerclaw onboard --wizard` if you want the interactive setup wizard.

**2. Configure** (`~/.summerclaw/config.json`)

Configure these **two parts** in your config (other options have defaults).

*Set your API key* (e.g. OpenRouter, recommended for global users):
```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  }
}
```

*Set your model* (optionally pin a provider — defaults to auto-detection):
```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "provider": "openrouter"
    }
  }
}
```

**3. Chat**

```bash
summerclaw agent
```

That's it! You have a working AI agent in 2 minutes.

## 💬 Chat Apps

Connect SummerClaw to your favorite chat platform. Want to build your own? see the [Channel Plugin Guide](./docs/CHANNEL_PLUGIN_GUIDE.md).

| Channel | What you need |
|---------|---------------|
| **Telegram** | Bot token from @BotFather |
| **Discord** | Bot token + Message Content intent |
| **WhatsApp** | QR code scan (`summerclaw channels login whatsapp`) |
| **WeChat (Weixin)** | QR code scan (`summerclaw channels login weixin`) |
| **Feishu** | App ID + App Secret |
| **Slack** | Bot token + App token |
| **QQ** | App ID + Secret |
| **WeCom** | Bot ID + Secret |
| **DingTalk** | Client ID + Client Secret |
| **Microsoft Teams** | App ID + App Password + Tenant ID |
| **Matrix** | Homeserver URL + Username + Password |
| **Email** | IMAP + SMTP credentials |
| **WebSocket** | Optional token for auth |

### Channel Configuration

Enable channels in `~/.summerclaw/config.json`:

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "your-bot-token"
    },
    "discord": {
      "enabled": true,
      "token": "your-bot-token"
    }
  }
}
```

### Channel Login (QR Code)

For channels requiring authentication:

```bash
summerclaw channels login whatsapp
summerclaw channels login weixin
```

### Channel Status

```bash
summerclaw channels status
```

## 🌐 Agent Social Network

SummerClaw can connect to various platforms to interact with users across different channels. Each channel plugin is independently maintained and follows a unified interface.

## ⚙️ Configuration

Copy `config.example.json` to `~/.summerclaw/config.json` and edit as needed.
All keys below live under `agents.defaults` unless noted otherwise.

### Core Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | string | `""` | LLM model identifier |
| `provider` | string | `"auto"` | Provider name or `"auto"` for detection |
| `workspace` | string | `~/.summerclaw/workspace` | Agent workspace directory |
| `maxToolIterations` | int | `200` | Maximum tool execution iterations |
| `maxToolResultChars` | int | `16000` | Max characters in tool result |
| `contextWindowTokens` | int | `1000000` | Context window size |
| `temperature` | float | `0.1` | Sampling temperature |
| `maxTokens` | int | `65536` | Max output tokens |
| `timezone` | string | `"UTC"` | System timezone |

### Execution Mode

| Key | Type | Default | Description |
|---|---|---|---|
| `executionMode` | string | `"auto"` | `"simple"` / `"plan"` / `"search-plan"` / `"auto"` |
| `max_subagent_depth` | int | `2` | Max DAG subagent nesting depth (0 = flat) |
| `max_replan_iterations` | int | `2` | Maximum replanning cycles |

### Memory Algorithm

| Key | Type | Default | Description |
|---|---|---|---|
| `memoryAlgorithm` | string | `"naive_memory"` | Memory backend: `naive_memory`, `layerga_memory`, `emem_memory`, `nemori_memory`, `mem0v3_memory`, `supermemory_memory`, `hindsight_memory`, or `mastra_om_memory` |

### Dream (Memory Consolidation)

| Key | Type | Default | Description |
|---|---|---|---|
| `dream.intervalH` | float | `2` | Hours between automatic Dream runs |
| `dream.modelOverride` | string\|null | `null` | Override model for Dream |
| `dream.maxBatchSize` | int | `20` | Max conversation turns per Dream run |
| `dream.maxIterations` | int | `15` | Max LLM iterations in Dream |

### Skill Distillation (Hermes-Autogen)

| Key | Type | Default | Description |
|---|---|---|---|
| `skill_autogen.enable` | bool | `false` | Enable auto skill generation |
| `skill_autogen.nudge_interval` | int | `10` | Tool call count to trigger distillation |
| `skill_autogen.max_iterations` | int | `8` | Max LLM iterations per run |

### Embedding Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `embedding.model` | string | `"text-embedding-3-small"` | Embedding model |
| `embedding.provider` | string | `"auto"` | `"auto"` or `"local"` |
| `embedding.apiKey` | string\|null | `null` | Optional API key override |
| `embedding.apiBase` | string\|null | `null` | Optional API base URL |
| `embedding.batchSize` | int | `16` | Batch size for embedding requests |
| `embedding.normalize` | bool | `true` | Normalize embedding vectors |

## 🧠 Memory

SummerClaw features a **pluggable memory algorithm** system. Choose from eight backends via the `memoryAlgorithm` config key:

| Algorithm | Strategy | Best For |
|-----------|----------|----------|
| `naive_memory` | File-based (MEMORY.md + history.jsonl) | Simple setups, zero extra deps |
| `emem_memory` | EDU extraction + embedding vectors | Structured fact & entity tracking |
| `layerga_memory` | L0-L4 layered (constitution → insight → facts → SOP → archives) | Self-organising hierarchical knowledge |
| `nemori_memory` | Episode + semantic self-organising | Long-term knowledge evolution |
| `mem0v3_memory` | ADD-only single-pass extraction + entity linking + multi-signal fusion | Token-efficient LLM-native memory |
| `supermemory_memory` | Chunk-based + relational versioning + temporal grounding + hybrid search | SOTA agent memory with version tracking |
| `hindsight_memory` | Built-in local TEMPR engine (RRF fusion + fact types + graph expansion) | Zero-dependency multi-strategy retrieval |
| `mastra_om_memory` | Observer/Reflector pipeline + async buffering + observation groups | High-density observational memory |

See the **[Memory Documentation](docs/MEMORY.md)** for storage structures, configuration, and algorithm details.

## 💬 In-Chat Commands

| Command | Description |
|---------|-------------|
| `/dream` | Run Dream immediately |
| `/dream-log` | Show the latest Dream memory change |
| `/dream-log <sha>` | Show a specific Dream change |
| `/dream-restore` | List recent Dream memory versions |
| `/dream-restore <sha>` | Restore memory to a previous state |
| `/tasks` | List running subagents and interrupted tasks |
| `/tasks resume <id>` | Resume a task |
| `/tasks discard <id>` | Discard a task |
| `/skill-autogen` | Trigger skill distillation |
| `/status` | Show agent status |
| `/restart` | Restart the agent |

## Periodic Tasks

SummerClaw supports scheduled tasks via cron expressions or interval-based scheduling.

### Commands

```bash
# List tasks
summerclaw cron list

# Add a task
summerclaw cron add --name "reminder" --message "Check email" --every 1h

# Remove a task
summerclaw cron remove <id>

# Enable/disable
summerclaw cron enable <id>
summerclaw cron disable <id>
```

## 🐍 Python SDK

```python
from summerclaw import SummerClaw

# Create from config
bot = SummerClaw.from_config()

# Run a message
result = await bot.run("Summarize this repo")
print(result.content)

# Custom config path
bot = SummerClaw.from_config(config_path="/path/to/config.json")

# Custom workspace
bot = SummerClaw.from_config(workspace="/path/to/workspace")
```

## 🔌 OpenAI-Compatible API

SummerClaw provides an OpenAI-compatible API server for integration with existing tooling.

### Start API Server

```bash
summerclaw serve --host 0.0.0.0 --port 8900
```

### Example Usage

```bash
curl http://localhost:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "summerclaw",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8900/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="summerclaw",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## 🐳 Docker

### Quick Start

```bash
docker compose up -d summerclaw-gateway
```

### Build from Source

```bash
docker build -t summerclaw .
docker run -d -p 18790:18790 \
  -v ~/.summerclaw:/home/summerclaw/.summerclaw \
  summerclaw gateway
```

### Services

- `summerclaw-gateway`: Main gateway service (port 18790)
- `summerclaw-api`: OpenAI-compatible API (port 8900)
- `summerclaw-cli`: Interactive CLI (profile: cli)

## 🧩 Multiple Instances

Run multiple SummerClaw instances with separate configs:

```bash
# Instance 1
summerclaw gateway --config ~/.summerclaw/config1.json

# Instance 2
summerclaw gateway --config ~/.summerclaw/config2.json
```

Each instance should have:
- Separate workspace directory
- Separate config file
- Different gateway ports (if running on same machine)

## 🔧 Providers

SummerClaw supports multiple LLM providers through a unified interface.

### Supported Providers

| Provider | Backend | OAuth | Local | Notes |
|----------|---------|-------|-------|-------|
| `openai` | openai_compat | ❌ | ❌ | OpenAI API |
| `openrouter` | openai_compat | ❌ | ❌ | Multi-model gateway |
| `anthropic` | anthropic | ❌ | ❌ | Claude models |
| `azure_openai` | azure_openai | ❌ | ❌ | Requires api_base |
| `github_copilot` | github_copilot | ✅ | ❌ | Device flow auth |
| `openai_codex` | openai_codex | ✅ | ❌ | OAuth login required |
| `deepseek` | openai_compat | ❌ | ❌ | DeepSeek models |
| `groq` | openai_compat | ❌ | ❌ | Fast inference |
| `zhipu` | openai_compat | ❌ | ❌ | ChatGLM models |
| `dashscope` | openai_compat | ❌ | ❌ | Alibaba Cloud |
| `vllm` | openai_compat | ❌ | ✅ | Self-hosted |
| `ollama` | openai_compat | ❌ | ✅ | Local models |
| `lmstudio` | openai_compat | ❌ | ✅ | Local GUI |
| `gemini` | openai_compat | ❌ | ❌ | Google models |
| `moonshot` | openai_compat | ❌ | ❌ | Kimi models |
| `minimax` | openai_compat | ❌ | ❌ | MiniMax models |
| `mistral` | openai_compat | ❌ | ❌ | Mistral models |
| `stepfun` | openai_compat | ❌ | ❌ | StepFun models |
| `volcengine` | openai_compat | ❌ | ❌ | VolcEngine |
| `qianfan` | openai_compat | ❌ | ❌ | Baidu Cloud |

### Provider Configuration

```json
{
  "providers": {
    "openai": {
      "apiKey": "sk-xxx",
      "apiBase": null
    },
    "anthropic": {
      "apiKey": "sk-ant-xxx",
      "apiBase": null
    }
  }
}
```

### OAuth Login

For providers supporting OAuth:

```bash
summerclaw provider login openai-codex
summerclaw provider login github-copilot
```

## 🔍 Web Search

SummerClaw supports multiple search backends for web-enhanced planning.

### Configuration

```json
{
  "tools": {
    "web": {
      "enable": true,
      "search": {
        "provider": "duckduckgo",
        "apiKey": "",
        "maxResults": 5,
        "timeout": 30
      }
    }
  }
}
```

### Supported Backends

| Provider | API Key Required | Notes |
|----------|-----------------|-------|
| `duckduckgo` | ❌ | Default, no setup needed |
| `bing` | ✅ | Requires Bing Search API |
| `google` | ✅ | Requires Custom Search API |
| `searxng` | ❌ | Self-hosted instance |
| `kagi` | ✅ | Requires Kagi API |

## 📁 Project Structure

```
summerclaw/
├── agent/          # Agent core (loop, planner, tools)
├── api/            # OpenAI-compatible API server
├── bus/            # Message bus (events, queue)
├── channels/       # Chat platform integrations
├── cli/            # CLI commands and streaming
├── command/        # Built-in commands and router
├── config/         # Configuration loading and schema
├── cron/           # Scheduled task service
├── heartbeat/      # Heartbeat service
├── memory/         # Pluggable memory algorithms
│   ├── naive_memory/
│   ├── layerga_memory/
│   ├── emem_memory/
│   ├── nemori_memory/
│   ├── mem0v3_memory/
│   ├── supermemory_memory/
│   ├── hindsight_memory/
│   └── mastra_om_memory/
├── providers/      # LLM provider implementations
├── proxy/          # Proxy pool management
├── security/       # Security and sandboxing
├── session/        # Session management
├── skills/         # Built-in skills
├── templates/      # Jinja2 templates
└── utils/          # Utility functions
```

## 🤝 Contribute & Roadmap

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/HKUDS/summerclaw.git
cd summerclaw
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Linting

```bash
ruff check summerclaw/
ruff format summerclaw/
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
