"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict

from summerclaw.cron.types import CronSchedule


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    send_progress: bool = True  # stream agent's text progress to the channel
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    send_max_retries: int = Field(default=3, ge=0, le=10)  # Max delivery attempts (initial send included)
    transcription_provider: str = "groq"  # Voice transcription backend: "groq" or "openai"


class DreamConfig(Base):
    """Dream memory consolidation configuration."""

    _HOUR_MS = 3_600_000

    interval_h: int = Field(default=2, ge=1)  # Every 2 hours by default
    cron: str | None = Field(default=None, exclude=True)  # Legacy compatibility override
    model_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modelOverride", "model", "model_override"),
    )  # Optional Dream-specific model override
    max_batch_size: int = Field(default=20, ge=1)  # Max history entries per run
    # Bumped from 10 to 15 in #3212 (exp002: +30% dedup, no accuracy loss; >15 plateaus).
    max_iterations: int = Field(default=15, ge=1)  # Max tool calls per Phase 2
    # Per-line git-blame age annotation in Phase 1 prompt (see #3212). Default
    # on — set to False to feed MEMORY.md raw if a specific LLM reacts poorly
    # to the `← Nd` suffix or you want deterministic, git-independent prompts.
    annotate_line_ages: bool = True

    def build_schedule(self, timezone: str) -> CronSchedule:
        """Build the runtime schedule, preferring the legacy cron override if present."""
        if self.cron:
            return CronSchedule(kind="cron", expr=self.cron, tz=timezone)
        return CronSchedule(kind="every", every_ms=self.interval_h * self._HOUR_MS)

    def describe_schedule(self) -> str:
        """Return a human-readable summary for logs and startup output."""
        if self.cron:
            return f"cron {self.cron} (legacy)"
        hours = self.interval_h
        return f"every {hours}h"


class SkillAutogenConfig(Base):
    """Skill auto-generation configuration.

    When enabled, after every ``nudge_interval`` cumulative tool calls in a
    turn the agent spawns a background review pass that inspects the recent
    conversation and creates reusable SKILL.md files for non-trivial patterns.
    This is distinct from Dream (which consolidates memory files on a timer).
    """

    enable: bool = False                           # Disabled by default — must be opted in
    nudge_interval: int = Field(default=10, ge=1)  # Trigger after this many cumulative tool calls
    max_iterations: int = Field(default=8, ge=1)   # Max tool calls in the review agent
    model_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modelOverride", "model", "model_override"),
    )  # Optional model override for the review agent


class SearchEnhancedPlanningConfig(Base):
    """Configuration for search-enhanced planning (Module 8 pre-planning web retrieval)."""

    enable: bool = False             # Enable pre-planning web search augmentation
    max_results: int = Field(default=5, ge=1, le=20)   # Max search results per query
    timeout: int = Field(default=15, ge=1, le=120)       # Search timeout in seconds (circuit breaker)
    search_on_replan: bool = False   # Re-search on GLOBAL_REPLAN (refresh stale info)
    max_purified_chars: int = Field(default=2000, ge=200, le=10000)  # Cap on purified info injected into planner
    web_search_backend: str = Field(
        default="auto",
        pattern="^(auto|browser|web)$",
        description=(
            "Which web-search tool set the SearchAgent may use. "
            "'browser' — headless-browser tools only (browser_search / browser_fetch, no API key needed). "
            "'web'     — HTTP-API tools only (web_search / web_fetch). "
            "'auto'    — prefer browser tools when registered, fall back to web tools."
        ),
    )


class InjectionConfig(Base):
    """Mid-turn message injection configuration.

    Controls how follow-up messages sent by the user while the agent is
    executing a task are drained from the pending queue and injected into
    the conversation.  Higher values allow more messages per turn but
    increase the risk of overwhelming the LLM with interleaved context.
    """

    max_per_turn: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Max number of pending messages drained in one injection cycle.",
    )
    max_cycles: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max number of injection cycles allowed per agent run.",
    )


class EmbeddingConfig(Base):
    """Embedding model configuration.

    Controls which embedding model and API endpoint are used for generating
    embeddings in memory algorithms (e.g. EMem) and other features that
    require vector embeddings.

    When ``provider`` is ``"auto"`` (default), the embedding API credentials
    are resolved from the LLM provider (same api_key/api_base).  Set
    ``provider`` to a specific provider name (e.g. ``"openai"``,
    ``"siliconflow"``) to use that provider's credentials instead.  Use
    ``provider="local"`` to run a Sentence-Transformers model locally
    (requires ``pip install summerclaw-ai[emem]``).
    """

    model: str = "text-embedding-3-small"  # Embedding model name (OpenAI-compatible or HuggingFace for local)
    provider: str = "auto"  # Provider name or "auto" (inherit LLM provider) or "local" (Sentence-Transformers)
    api_key: str | None = None  # Optional override for embedding API key
    api_base: str | None = None  # Optional override for embedding API base URL
    batch_size: int = Field(default=16, ge=1)  # Batch size for embedding model calls
    normalize: bool = True  # L2-normalize output vectors


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.summerclaw/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = (
        "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    )
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    context_block_limit: int | None = None
    temperature: float = 0.1
    max_tool_iterations: int = 200
    max_tool_result_chars: int = 16_000
    provider_retry_mode: Literal["standard", "persistent"] = "standard"
    reasoning_effort: str | None = None  # low / medium / high / adaptive - enables LLM thinking mode
    timezone: str = "UTC"  # IANA timezone, e.g. "Asia/Shanghai", "America/New_York"
    unified_session: bool = False  # Share one session across all channels (single-user multi-device)
    disabled_skills: list[str] = Field(default_factory=list)  # Skill names to exclude from loading (e.g. ["summarize", "skill-creator"])
    memory_algorithm: str = Field(
        default="naive_memory",
        pattern=r"^[a-z][a-z0-9_]*$",
    )  # Memory algorithm name (must be registered in MemoryRegistry, e.g. "naive_memory")
    plan_and_solve: bool = False  # DEPRECATED: Use execution_mode instead. If True (legacy), maps to "auto".
    execution_mode: Literal["simple", "plan", "search-plan", "auto"] = Field(
        default="auto",
        description=(
            "Task execution mode:\n"
            "  'simple'      — direct ReAct execution, no planning\n"
            "  'plan'        — always generate and follow an execution plan (no web search)\n"
            "  'search-plan' — always perform web search + planning before execution\n"
            "  'auto'        — auto-detect via LLM complexity evaluator (default)"
        ),
    )
    max_subagent_depth: int = Field(
        default=0,
        ge=0,
        le=5,
    )  # Max recursive subagent depth (0=no recursion, recommended max=5)
    max_replan_iterations: int = Field(
        default=2,
        ge=0,
        le=10,
    )  # Max replan iterations in plan-and-solve closed-loop (0=no replan, single-pass only)
    session_ttl_minutes: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("idleCompactAfterMinutes", "sessionTtlMinutes"),
        serialization_alias="idleCompactAfterMinutes",
    )  # Auto-compact idle threshold in minutes (0 = disabled)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    skill_autogen: SkillAutogenConfig = Field(default_factory=SkillAutogenConfig)  # Skill auto-generation (disabled by default)
    search_enhanced_planning: SearchEnhancedPlanningConfig = Field(
        default_factory=SearchEnhancedPlanningConfig
    )  # Search-enhanced planning: pre-planning web search augmentation
    injection: InjectionConfig = Field(
        default_factory=InjectionConfig
    )  # Mid-turn message injection control


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str | None = None
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI (model = deployment name)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama local models
    lm_studio: ProviderConfig = Field(default_factory=ProviderConfig)  # LM Studio local models
    ovms: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenVINO Model Server (OVMS)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax_anthropic: ProviderConfig = Field(default_factory=ProviderConfig)  # MiniMax Anthropic endpoint (thinking)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)
    stepfun: ProviderConfig = Field(default_factory=ProviderConfig)  # Step Fun (阶跃星辰)
    xiaomi_mimo: ProviderConfig = Field(default_factory=ProviderConfig)  # Xiaomi MIMO (小米)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # SiliconFlow (硅基流动)
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine (火山引擎)
    volcengine_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine Coding Plan
    byteplus: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus (VolcEngine international)
    byteplus_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus Coding Plan
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # OpenAI Codex (OAuth)
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # Github Copilot (OAuth)
    qianfan: ProviderConfig = Field(default_factory=ProviderConfig)  # Qianfan (百度千帆)


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes
    keep_recent_messages: int = 8


class ApiConfig(Base):
    """OpenAI-compatible API server configuration."""

    host: str = "127.0.0.1"  # Safer default: local-only bind.
    port: int = 8900
    timeout: float = 120.0  # Per-request timeout in seconds.


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "127.0.0.1"  # Safer default: local-only bind.
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "duckduckgo"  # brave, tavily, duckduckgo, searxng, jina, kagi
    api_key: str = ""
    base_url: str = ""  # SearXNG base URL
    max_results: int = 5
    timeout: int = 30  # Wall-clock timeout (seconds) for search operations


class BrowserToolsConfig(Base):
    """Headless-browser tool configuration (browser_search / browser_fetch).

    These tools use Playwright / crawl4ai and do not require any API key.
    Playwright must be installed: ``pip install playwright && playwright install chromium``.
    """

    enable: bool = False
    enable_control: bool = Field(
        default=False,
        description=(
            "Enable stateful browser control tools (browser_navigate / browser_snapshot / "
            "browser_execute_js). These maintain a persistent headless Chromium session "
            "across calls, allowing multi-step page interaction, JS injection, and "
            "accessibility-tree snapshots. Requires ``enable: true`` as well."
        ),
    )
    proxy: str | None = None  # HTTP/SOCKS5 proxy URL for Playwright (ignored when proxy_pool is enabled)
    timeout: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Playwright navigation timeout in milliseconds (default 30 000).",
    )


class ProxyPoolConfig(Base):
    """IP proxy pool configuration.

    When enabled, maintains a pool of usable proxy servers with periodic
    health checks and automatic collection from public sources.  Each
    web/browser request picks a random proxy from the pool to avoid
    rate-limiting and IP bans.
    """

    enabled: bool = False
    min_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Minimum number of available proxies before triggering collection",
    )
    max_pool_size: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Maximum number of proxies to keep in the pool",
    )
    health_check_interval: int = Field(
        default=300,
        ge=30,
        description="Seconds between periodic health checks of all proxies (0 = disabled)",
    )
    health_check_url: str = Field(
        default="https://httpbin.org/ip",
        description="URL used to validate whether a proxy is working",
    )
    proxy_test_timeout: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Timeout in seconds for proxy validation requests",
    )
    max_fail_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive failures before marking a proxy as dead",
    )
    collect_interval: int = Field(
        default=600,
        ge=60,
        description="Seconds between pool size checks and auto-collection triggers",
    )
    initial_proxies: list[str] = Field(
        default_factory=list,
        description="Proxy URLs to load at startup (e.g. ['http://1.2.3.4:8080', 'socks5://...'])",
    )
    proxy_cache_enabled: bool = Field(
        default=True,
        description="Persist valid proxies to disk so the pool survives restarts",
    )
    proxy_cache_path: str = Field(
        default="",
        description="Path to the proxy cache JSON file (default: ~/.summerclaw/proxy_cache.json)",
    )


class WebToolsConfig(Base):
    """Web tools configuration."""

    enable: bool = True
    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""
    sandbox: str = ""  # sandbox backend: "" (none) or "bwrap"
    allowed_env_keys: list[str] = Field(default_factory=list)  # Env var names to pass through to subprocess (e.g. ["GOPATH", "JAVA_HOME"])

class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all tools; [] = no tools

class MyToolConfig(Base):
    """Self-inspection tool configuration."""

    enable: bool = True  # register the `my` tool (agent runtime state inspection)
    allow_set: bool = False  # let `my` modify loop state (read-only if False)


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    browser: BrowserToolsConfig = Field(default_factory=BrowserToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    my: MyToolConfig = Field(default_factory=MyToolConfig)
    restrict_to_workspace: bool = False  # restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    ssrf_whitelist: list[str] = Field(default_factory=list)  # CIDR ranges to exempt from SSRF blocking (e.g. ["100.64.0.0/10"] for Tailscale)


class Config(BaseSettings):
    """Root configuration for summerclaw."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    proxy_pool: ProxyPoolConfig = Field(default_factory=ProxyPoolConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self, model: str | None = None
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from summerclaw.providers.registry import PROVIDERS, find_by_name

        forced = self.agents.defaults.provider
        if forced != "auto":
            spec = find_by_name(forced)
            if spec:
                p = getattr(self.providers, spec.name, None)
                return (p, spec.name) if p else (None, None)
            return None, None

        model_lower = (model or self.agents.defaults.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins — prevents `github-copilot/...codex` matching openai_codex.
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # Fallback: configured local providers can route models without
        # provider-specific keywords (for example plain "llama3.2" on Ollama).
        # Prefer providers whose detect_by_base_keyword matches the configured api_base
        # (e.g. Ollama's "11434" in "http://localhost:11434") over plain registry order.
        local_fallback: tuple[ProviderConfig, str] | None = None
        for spec in PROVIDERS:
            if not spec.is_local:
                continue
            p = getattr(self.providers, spec.name, None)
            if not (p and p.api_base):
                continue
            if spec.detect_by_base_keyword and spec.detect_by_base_keyword in p.api_base:
                return p, spec.name
            if local_fallback is None:
                local_fallback = (p, spec.name)
        if local_fallback:
            return local_fallback

        # Fallback: gateways first, then others (follows registry order)
        # OAuth providers are NOT valid fallbacks — they require explicit model selection
        for spec in PROVIDERS:
            if spec.is_oauth:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for gateway/local providers."""
        from summerclaw.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # resolve their base URL from the registry in the provider constructor.
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    model_config = SettingsConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="allow",
        env_prefix="NANOBOT_",
        env_nested_delimiter="__",
    )
