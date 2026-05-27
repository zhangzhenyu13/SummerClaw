"""Channel command handler — ``/train``.

Registers the ``/train`` command with the CommandRouter.
The dashboard is started automatically by the gateway at boot.
``/train`` simply returns the dashboard URL and status info.

Sub-commands:
  - ``/train status``  — show active training sessions
  - ``/train stop``    — cancel a running training
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from summerclaw.bus.events import OutboundMessage
from summerclaw.command.router import CommandContext

if TYPE_CHECKING:
    from summerclaw.agent.loop import AgentLoop


# ── Active training sessions ──────────────────────────────────────────────

_ACTIVE_TRAININGS: dict[str, dict] = {}
"""Tracks active training sessions: {session_key: {engine, dashboard, task}}"""


# ── Notification helpers ──────────────────────────────────────────────────

def _reply(ctx: CommandContext, content: str) -> OutboundMessage:
    """Build an outbound reply."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata=dict(ctx.msg.metadata or {}),
    )


# ── Command handler ───────────────────────────────────────────────────────

# Sub-command table for ``/train <sub>``
_SUB_COMMANDS: dict[str, Any] = {}


def _get_gateway_dashboard_info() -> dict[str, str | None] | None:
    """Return dashboard URL info from the gateway, or None."""
    for key, info in _ACTIVE_TRAININGS.items():
        dashboard = info.get("dashboard")
        if dashboard and dashboard.url:
            return {
                "local_url": dashboard.local_url,
                "share_url": dashboard.share_url,
                "url": dashboard.url,
            }
    return None


async def cmd_train(ctx: CommandContext) -> OutboundMessage:
    """Handle ``/train`` command — return dashboard URL and info.

    The dashboard is started automatically by the gateway at boot.
    This command simply reports the URL so users can open it.

    Sub-commands:
      - ``/train status``  — show active training sessions
      - ``/train stop``    — cancel a running training
    """
    args = ctx.args.strip()
    first_word = args.split(" ", 1)[0].lower() if args else ""

    # Dispatch to sub-command if matched
    if first_word in _SUB_COMMANDS:
        ctx.args = args[len(first_word):].strip() if first_word else args
        return await _SUB_COMMANDS[first_word](ctx)

    # Return dashboard URL
    info = _get_gateway_dashboard_info()
    if info:
        from summerclaw.agent_trainer.registry import list_algorithms
        available = list_algorithms()
        alg_info = ", ".join(available) if available else "none"

        url_lines = []
        if info.get("local_url"):
            url_lines.append(f"Local (LAN): {info['local_url']}")
        if info.get("share_url"):
            url_lines.append(f"Public URL: {info['share_url']}")
        if not url_lines:
            url_lines.append(f"Dashboard: {info['url']}")
        url_text = "\n".join(url_lines)

        return _reply(
            ctx,
            f"**Agent Trainer Dashboard**\n\n"
            f"{url_text}\n\n"
            f"Available algorithms: {alg_info}\n\n"
            f"Sub-commands:\n"
            f"- `/train status` — show active training sessions\n"
            f"- `/train stop <task-id>` — cancel a running training (e.g. `/train stop skillopt`)",
        )
    return _reply(
        ctx,
        "Dashboard is not running. "
        "Make sure the gateway was started correctly.",
    )


async def cmd_train_status(ctx: CommandContext) -> OutboundMessage:
    """Handle ``/train status`` command — show training sessions."""
    if not _ACTIVE_TRAININGS:
        return _reply(ctx, "No training sessions.")

    lines = ["**Training Sessions:**\n"]
    for key, info in _ACTIVE_TRAININGS.items():
        engine = info["engine"]
        dashboard = info.get("dashboard")
        local_url = dashboard.local_url if dashboard else "N/A"
        share_url = dashboard.share_url if dashboard else None
        url_parts = [f"local={local_url}"]
        if share_url:
            url_parts.append(f"public={share_url}")
        url_text = ", ".join(url_parts)
        task_id = Path(engine.out_dir).name
        status = "running" if engine.is_running else "stopped"
        lines.append(
            f"- **{task_id}**: {status}, "
            f"best={engine.best_score:.4f}, "
            f"steps={engine.history.total_steps}, "
            f"dashboard={url_text}"
        )
    return _reply(ctx, "\n".join(lines))


async def cmd_train_stop(ctx: CommandContext) -> OutboundMessage:
    """Handle ``/train stop <task-id>`` — cancel a running training by task ID.

    Task ID is the directory name, e.g. ``skillopt-20250527-143022``.
    Partial match (prefix) is also supported.
    """
    task_id = ctx.args.strip()
    if not task_id:
        return _reply(ctx, "Usage: `/train stop <task-id>`")

    # Search active sessions by task_id (exact or prefix match on out_dir name)
    matched_key = None
    matched_info = None
    for key, info in _ACTIVE_TRAININGS.items():
        engine = info["engine"]
        out_dir_name = Path(engine.out_dir).name  # e.g. skillopt-20250527-143022
        if out_dir_name == task_id or out_dir_name.startswith(task_id):
            matched_key = key
            matched_info = info
            break

    if not matched_info:
        return _reply(ctx, f"No active training matching task **{task_id}**.")

    engine = matched_info["engine"]
    out_dir_name = Path(engine.out_dir).name
    engine.request_cancel()

    # Push notification to all channels (best-effort)
    notify_fn = matched_info.get("notify_fn")
    if notify_fn:
        try:
            import asyncio
            msg = f"\u23f9 Training stop requested via channel \u2014 task: {out_dir_name}"
            asyncio.create_task(notify_fn(msg))
        except Exception:
            pass

    return _reply(ctx, f"Cancellation requested for task **{out_dir_name}**.")


# Populate sub-command table
_SUB_COMMANDS.update({
    "status": cmd_train_status,
    "stop": cmd_train_stop,
})


# ── Standalone dashboard launcher (used by gateway boot) ─────────────────

def start_dashboard_from_agent(
    agent: AgentLoop,
    algorithm_name: str = "skillopt",
    dashboard_port: int = 7860,
    share: bool = True,
) -> str | None:
    """Start a training dashboard without a CommandContext.

    Called by the gateway at boot to start the Gradio dashboard
    in a background thread.

    Returns the dashboard URL string, or *None* on failure.
    """
    from summerclaw.agent_trainer.config import build_trainer_config
    from summerclaw.agent_trainer.dashboard.app import DashboardServer
    from summerclaw.agent_trainer.engine.trainer import TrainerEngine
    from summerclaw.agent_trainer.env.summerclaw_env import SummerClawEnvAdapter
    from summerclaw.agent_trainer.registry import get_algorithm

    try:
        algo_cls = get_algorithm(algorithm_name)
    except KeyError as exc:
        logger.error("start_dashboard_from_agent: {}", exc)
        return None

    trainer_cfg = build_trainer_config(agent, algorithm_name)

    provider = getattr(agent, "provider", None)
    if not provider:
        logger.error("start_dashboard_from_agent: LLM provider not available")
        return None

    model = getattr(agent, "model", "")
    workspace = Path(agent.workspace).expanduser() if agent.workspace else Path.cwd()

    train_root = Path.home() / ".summerclaw" / "train-algs"
    train_root.mkdir(parents=True, exist_ok=True)
    # NOTE: Task directory (e.g. skillopt-20250527-143022) is NOT created here.
    # It is created lazily by TrainerEngine._ensure_out_dir() when training
    # actually starts or data is uploaded — avoiding empty dirs on every boot.

    memory_algorithm_name = getattr(agent, "memory_algorithm_name", "naive")

    env = SummerClawEnvAdapter(
        provider=provider,
        model=model,
        workspace=workspace,
        train_out_dir=train_root,  # task subdir created lazily by engine
        memory_algorithm_name=memory_algorithm_name,
        context_window_tokens=getattr(agent, "context_window_tokens", 8000),
        max_tool_iterations=getattr(agent, "max_iterations", 10),
        max_tool_result_chars=getattr(agent, "max_tool_result_chars", 4000),
        temperature=0.7,
        max_tokens=8192,
        workers=trainer_cfg.get("workers", 4),
        tool_registry=getattr(agent, "tools", None),  # use main agent's tools
    )

    skill_init_path = trainer_cfg.get("skill_init", "")
    if skill_init_path and Path(skill_init_path).exists():
        skill_init = Path(skill_init_path).read_text(encoding="utf-8")
    else:
        skill_init = ""

    algo = algo_cls(
        provider=provider,
        model=model,
        minibatch_size=trainer_cfg.get("minibatch_size", 5),
        edit_budget=trainer_cfg.get("edit_budget", 4),
        workers=trainer_cfg.get("workers", 4),
        optimizer_model=trainer_cfg.get("optimizer_model"),
        update_mode=trainer_cfg.get("update_mode", "patch"),
        lr_mode=trainer_cfg.get("lr_mode", "constant"),
        min_lr=trainer_cfg.get("min_lr", 2),
        reasoning_effort=trainer_cfg.get("reasoning_effort", "high"),
        env=trainer_cfg.get("env"),
        merge_batch_size=trainer_cfg.get("merge_batch_size", 8),
        max_analyst_rounds=trainer_cfg.get("max_analyst_rounds", 3),
        use_slow_update=trainer_cfg.get("use_slow_update", True),
        use_meta_skill=trainer_cfg.get("use_meta_skill", True),
        longitudinal_pair_policy=trainer_cfg.get("longitudinal_pair_policy", "mixed"),
        rewrite_reasoning_effort=trainer_cfg.get("rewrite_reasoning_effort"),
        rewrite_max_completion_tokens=trainer_cfg.get("rewrite_max_completion_tokens", 64000),
    )

    engine = TrainerEngine(
        algorithm=algo,
        env=env,
        data_loader=None,
        out_dir=train_root,  # task subdir created lazily on first use
        skill_init=skill_init,
        skill_init_path=skill_init_path,
        num_epochs=trainer_cfg.get("num_epochs", 3),
        batch_size=trainer_cfg.get("batch_size", 5),
        edit_budget=trainer_cfg.get("edit_budget", 4),
        seed=trainer_cfg.get("seed", 42),
    )

    # Try pre-loading data
    data_dir = trainer_cfg.get("data_dir", "")
    if data_dir and Path(data_dir).exists():
        from summerclaw.agent_trainer.datasets.loader import DataLoader
        loader = DataLoader(data_dir)
        if loader.split_names:
            engine.set_data_loader(loader)
            logger.info("Pre-loaded data from {}: {}", data_dir, loader.summary())

    port = dashboard_port or trainer_cfg.get("dashboard_port", 7860)
    dashboard = DashboardServer(
        engine=engine,
        port=port,
        share=share,
        train_root=train_root,
        active_sessions=_ACTIVE_TRAININGS,
    )
    dashboard_url = dashboard.start()

    import asyncio as _aio
    session_key = f"train:{algorithm_name}:gateway"
    _ACTIVE_TRAININGS[session_key] = {
        "engine": engine,
        "dashboard": dashboard,
        "task": None,
        "notify_fn": None,
        "main_loop": _aio.get_event_loop(),
    }
    engine._trainer_cfg = trainer_cfg

    local_url = dashboard.local_url or ""
    share_url = dashboard.share_url or ""
    if share_url:
        logger.info("Dashboard started — local: {}, public: {}", local_url, share_url)
    else:
        logger.info("Dashboard started — local: {}", local_url)
    return dashboard_url


# ── Registration ──────────────────────────────────────────────────────────

def register_commands(router: Any) -> None:
    """Register training commands with the command router.

    Called from the main application startup.
    Uses priority routing so commands work even when the agent is busy.
    """
    # Import algorithm modules to trigger @algorithm decorator registration
    try:
        import summerclaw.agent_trainer.algorithms.skillopt.algorithm  # noqa: F401
    except ImportError:
        logger.warning("SkillOpt algorithm not available")

    # Register as priority commands (bypass dispatch lock)
    router.priority("/train", cmd_train)
    router.priority_prefix("/train ", cmd_train)

    # Also register in normal dispatch for completeness
    router.exact("/train", cmd_train)
    router.prefix("/train ", cmd_train)
