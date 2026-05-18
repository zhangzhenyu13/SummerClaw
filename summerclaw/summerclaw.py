"""High-level programmatic interface to summerclaw."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from summerclaw.agent.hook import AgentHook
from summerclaw.agent.loop import AgentLoop
from summerclaw.bus.queue import MessageBus


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]


class SummerClaw:
    """Programmatic facade for running the summerclaw agent.

    Usage::

        bot = SummerClaw.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)
    """

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> SummerClaw:
        """Create a SummerClaw instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``~/.summerclaw/config.json``.
            workspace: Override the workspace directory from config.
        """
        from summerclaw.config.loader import load_config, resolve_config_env_vars
        from summerclaw.config.schema import Config

        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(load_config(resolved))
        if workspace is not None:
            config.agents.defaults.workspace = str(
                Path(workspace).expanduser().resolve()
            )

        provider = _make_provider(config)
        bus = MessageBus()
        defaults = config.agents.defaults

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=defaults.model,
            max_iterations=defaults.max_tool_iterations,
            context_window_tokens=defaults.context_window_tokens,
            context_block_limit=defaults.context_block_limit,
            max_tool_result_chars=defaults.max_tool_result_chars,
            provider_retry_mode=defaults.provider_retry_mode,
            web_config=config.tools.web,
            exec_config=config.tools.exec,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            execution_mode=defaults.execution_mode,
            max_subagent_depth=defaults.max_subagent_depth,
            max_replan_iterations=defaults.max_replan_iterations,
            search_enhanced_planning_config=defaults.search_enhanced_planning,
            tools_config=config.tools,
            skill_autogen_config=defaults.skill_autogen,
            memory_algorithm_name=defaults.memory_algorithm,
            embedding_config=defaults.embedding,
            max_injections_per_turn=defaults.injection.max_per_turn,
            max_injection_cycles=defaults.injection.max_cycles,
            proxy_pool_config=config.proxy_pool,
        )
        return cls(loop)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        hooks: list[AgentHook] | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            hooks: Optional lifecycle hooks for this run.
        """
        prev = self._loop._extra_hooks
        if hooks is not None:
            self._loop._extra_hooks = list(hooks)
        try:
            response = await self._loop.process_direct(
                message, session_key=session_key,
            )
        finally:
            self._loop._extra_hooks = prev

        content = (response.content if response else None) or ""
        return RunResult(content=content, tools_used=[], messages=[])


def _make_provider(config: Any) -> Any:
    """Create the LLM provider from config (extracted from CLI)."""
    from summerclaw.providers.base import GenerationSettings
    from summerclaw.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise ValueError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from summerclaw.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "github_copilot":
        from summerclaw.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "azure_openai":
        from summerclaw.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key, api_base=p.api_base, default_model=model
        )
    elif backend == "anthropic":
        from summerclaw.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from summerclaw.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider
