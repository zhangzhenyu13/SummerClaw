"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from summerclaw import __version__
from summerclaw.bus.events import OutboundMessage
from summerclaw.command.router import CommandContext, CommandRouter
from summerclaw.utils.helpers import build_status_content
from summerclaw.utils.restart import set_restart_notice_to_env


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content,
        metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg
    set_restart_notice_to_env(channel=msg.channel, chat_id=msg.chat_id)

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "summerclaw"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        metadata=dict(msg.metadata or {})
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    
    # Fetch web search provider usage (best-effort, never blocks the response)
    search_usage_text: str | None = None
    try:
        from summerclaw.utils.searchusage import fetch_search_usage
        web_cfg = getattr(loop, "web_config", None)
        search_cfg = getattr(web_cfg, "search", None) if web_cfg else None
        if search_cfg is not None:
            provider = getattr(search_cfg, "provider", "duckduckgo")
            api_key = getattr(search_cfg, "api_key", "") or None
            usage = await fetch_search_usage(provider=provider, api_key=api_key)
            search_usage_text = usage.format()
    except Exception:
        pass  # Never let usage fetch break /status
    active_tasks = loop._active_tasks.get(ctx.key, [])
    task_count = sum(1 for t in active_tasks if not t.done())
    try:
        task_count += loop.subagents.get_running_count_by_session(ctx.key)
    except Exception:
        pass
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            search_usage_text=search_usage_text,
            active_task_count=task_count,
            max_completion_tokens=getattr(
                getattr(loop.provider, "generation", None), "max_tokens", 8192
            ),
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.consolidator.archive(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
        metadata=dict(ctx.msg.metadata or {})
    )


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Dream consolidation run."""
    import time

    loop = ctx.loop
    msg = ctx.msg

    async def _run_dream():
        t0 = time.monotonic()
        try:
            did_work = await loop.dream.run()
            elapsed = time.monotonic() - t0
            if did_work:
                content = f"Dream completed in {elapsed:.1f}s."
            else:
                content = "Dream: nothing to process."
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Dream failed after {elapsed:.1f}s: {e}"
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_dream())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Dreaming...",
    )


def _extract_changed_files(diff: str) -> list[str]:
    """Extract changed file paths from a unified diff."""
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "No tracked memory files changed."
    return ", ".join(f"`{path}`" for path in files)


def _format_dream_log_content(commit, diff: str, *, requested_sha: str | None = None) -> str:
    files_line = _format_changed_files(diff)
    lines = [
        "## Dream Update",
        "",
        "Here is the selected Dream memory change." if requested_sha else "Here is the latest Dream memory change.",
        "",
        f"- Commit: `{commit.sha}`",
        f"- Time: {commit.timestamp}",
        f"- Changed files: {files_line}",
    ]
    if diff:
        lines.extend([
            "",
            f"Use `/dream-restore {commit.sha}` to undo this change.",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend([
            "",
            "Dream recorded this version, but there is no file diff to display.",
        ])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list) -> str:
    lines = [
        "## Dream Restore",
        "",
        "Choose a Dream memory version to restore. Latest first:",
        "",
    ]
    for c in commits:
        lines.append(f"- `{c.sha}` {c.timestamp} - {c.message.splitlines()[0]}")
    lines.extend([
        "",
        "Preview a version with `/dream-log <sha>` before restoring it.",
        "Restore a version with `/dream-restore <sha>`.",
    ])
    return "\n".join(lines)


async def cmd_dream_log(ctx: CommandContext) -> OutboundMessage:
    """Show what the last Dream changed.

    Default: diff of the latest commit (HEAD~1 vs HEAD).
    With /dream-log <sha>: diff of that specific commit.
    """
    store = ctx.loop.consolidator.store
    git = store.git

    if not git.is_initialized():
        if store.get_last_dream_cursor() == 0:
            msg = "Dream has not run yet. Run `/dream`, or wait for the next scheduled Dream cycle."
        else:
            msg = "Dream history is not available because memory versioning is not initialized."
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=msg, metadata={"render_as": "text"},
        )

    args = ctx.args.strip()

    if args:
        # Show diff of a specific commit
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        if not result:
            content = (
                f"Couldn't find Dream change `{sha}`.\n\n"
                "Use `/dream-restore` to list recent versions, "
                "or `/dream-log` to inspect the latest one."
            )
        else:
            commit, diff = result
            content = _format_dream_log_content(commit, diff, requested_sha=sha)
    else:
        # Default: show the latest commit's diff
        commits = git.log(max_entries=1)
        result = git.show_commit_diff(commits[0].sha) if commits else None
        if result:
            commit, diff = result
            content = _format_dream_log_content(commit, diff)
        else:
            content = "Dream memory has no saved versions yet."

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_dream_restore(ctx: CommandContext) -> OutboundMessage:
    """Restore memory files from a previous dream commit.

    Usage:
        /dream-restore          — list recent commits
        /dream-restore <sha>    — revert a specific commit
    """
    store = ctx.loop.consolidator.store
    git = store.git
    if not git.is_initialized():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="Dream history is not available because memory versioning is not initialized.",
        )

    args = ctx.args.strip()
    if not args:
        # Show recent commits for the user to pick
        commits = git.log(max_entries=10)
        if not commits:
            content = "Dream memory has no saved versions to restore yet."
        else:
            content = _format_dream_restore_list(commits)
    else:
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        changed_files = _format_changed_files(result[1]) if result else "the tracked memory files"
        new_sha = git.revert(sha)
        if new_sha:
            content = (
                f"Restored Dream memory to the state before `{sha}`.\n\n"
                f"- New safety commit: `{new_sha}`\n"
                f"- Restored files: {changed_files}\n\n"
                f"Use `/dream-log {new_sha}` to inspect the restore diff."
            )
        else:
            content = (
                f"Couldn't restore Dream change `{sha}`.\n\n"
                "It may not exist, or it may be the first saved version with no earlier state to restore."
            )
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_skill_autogen(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Skill-Autogen review pass.

    Reviews the recent session history and creates a SKILL.md for any
    non-trivial reusable workflow found.  Distinct from Dream — Dream
    consolidates memory files (MEMORY/SOUL/USER.md); Skill-Autogen extracts
    reusable skills (skills/<name>/SKILL.md).

    Usage:
        /skill-autogen
    """
    import time

    loop = ctx.loop
    msg = ctx.msg

    if loop.skill_autogen is None:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=(
                "Skill-Autogen is not enabled. "
                "Set `skill_autogen.enable = true` in your config to activate it."
            ),
        )

    # Build a recent history snapshot from the current session
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    history = session.get_history(max_messages=0)
    # Convert session history entries to message dicts the reviewer can parse
    recent_msgs: list[dict] = [
        {"role": m.role, "content": m.content}
        for m in history[-60:]  # last 60 messages (generous window)
        if hasattr(m, "role") and hasattr(m, "content")
    ]

    async def _run_skill_autogen():
        t0 = time.monotonic()
        try:
            created = await loop.skill_autogen.run(recent_msgs)
            elapsed = time.monotonic() - t0
            if created:
                content = f"Skill-Autogen completed in {elapsed:.1f}s: a new skill was created in workspace/skills/."
            else:
                content = f"Skill-Autogen completed in {elapsed:.1f}s: no reusable pattern found."
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Skill-Autogen failed after {elapsed:.1f}s: {e}"
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_skill_autogen())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content="Reviewing conversation for reusable skills...",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed-seconds value as a human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m {s % 60}s"
    h = m // 60
    return f"{h}h {m % 60}m"


async def cmd_tasks(ctx: CommandContext) -> OutboundMessage:
    """Show currently running subagent tasks and interrupted tasks pending decision."""
    import time

    loop = ctx.loop
    statuses = loop.subagents.get_all_statuses()
    lines: list[str] = []

    # --- Active main-session processing ----------------------------------
    active_processing = getattr(loop, "_active_processing", {})
    if active_processing:
        lines.append(f"**Active sessions ({len(active_processing)}):**\n")
        for sess_key, (msg_preview, started_at) in active_processing.items():
            elapsed = _format_elapsed(time.monotonic() - started_at)
            suffix = "..." if len(msg_preview) == 80 else ""
            lines.append(
                f"\u2022 session `{sess_key}` \u2014 {elapsed}"
                f"\n  > {msg_preview}{suffix}"
            )

    # --- Running subagents -----------------------------------------------
    if statuses:
        lines.append(f"**Running subagents ({len(statuses)}):**\n")
        for s in statuses:
            elapsed = _format_elapsed(time.monotonic() - s.started_at)
            lines.append(
                f"\u2022 `[{s.task_id}]` **{s.label}**"
                f" \u2014 phase: {s.phase}, {s.iteration} iter, {elapsed}"
            )
    elif not active_processing:
        lines.append("No tasks currently running.")

    # --- Interrupted tasks from last restart -----------------------------
    interrupted = loop._task_registry.list_all()
    session_interrupted: list[str] = []
    try:
        for session in loop.sessions.list_all_sessions():
            tid = session.metadata.get(loop._INTERRUPTED_TASK_ID_KEY)
            if tid and session.metadata.get(loop._PENDING_USER_TURN_KEY):
                # Find the last user message for preview
                preview = ""
                for msg in reversed(session.messages):
                    if msg.get("role") == "user":
                        raw = msg.get("content", "")
                        if isinstance(raw, str):
                            preview = raw.strip()[:80]
                        break
                session_interrupted.append(
                    f"\u2022 `[{tid}]` session `{session.key}`: \"{preview}...\""
                    f"\n  `/tasks resume {tid}` or `/tasks discard {tid}`"
                )
    except Exception:
        pass

    if interrupted or session_interrupted:
        lines.append("")
        lines.append("**Interrupted tasks (awaiting decision):**\n")
        for pt in interrupted:
            lines.append(
                f"\u2022 `[{pt.task_id}]` **{pt.label}** ({pt.elapsed_description()})"
                f"\n  `/tasks resume {pt.task_id}` or `/tasks discard {pt.task_id}`"
            )
        lines.extend(session_interrupted)

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_tasks_resume(ctx: CommandContext) -> OutboundMessage:
    """Resume an interrupted task by ID (/tasks resume <id>)."""
    task_id = ctx.args.strip()
    if not task_id:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Usage: /tasks resume <id>",
        )

    loop = ctx.loop

    # Check subagent registry first
    persisted = loop._task_registry.get(task_id)
    if persisted is not None:
        loop._task_registry.remove(task_id)
        try:
            result_msg = await loop.subagents.spawn(
                task=persisted.description,
                label=persisted.label,
                origin_channel=persisted.origin_channel,
                origin_chat_id=persisted.origin_chat_id,
                session_key=persisted.session_key,
                current_depth=persisted.depth,
            )
            content = f"\u25b6️ Resuming subagent task `[{task_id}]`:\n> {persisted.label}\n\n{result_msg}"
        except Exception as e:
            content = f"Failed to resume subagent task `[{task_id}]`: {e}"
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=content,
        )

    # Check main-session tasks
    result = await loop._resume_session_task(task_id)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=result,
    )


async def cmd_tasks_discard(ctx: CommandContext) -> OutboundMessage:
    """Discard an interrupted task by ID (/tasks discard <id>)."""
    task_id = ctx.args.strip()
    if not task_id:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Usage: /tasks discard <id>",
        )

    loop = ctx.loop

    # Check subagent registry first
    persisted = loop._task_registry.get(task_id)
    if persisted is not None:
        loop._task_registry.remove(task_id)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"\u274c Task `[{task_id}]` ({persisted.label}) discarded.",
        )

    # Check main-session tasks
    result = await loop._discard_session_task(task_id)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=result,
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "🐈 summerclaw commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/tasks — Show running & interrupted tasks",
        "/tasks resume <id> — Resume an interrupted task",
        "/tasks discard <id> — Discard an interrupted task",
        "/dream — Manually trigger Dream consolidation",
        "/dream-log — Show what the last Dream changed",
        "/dream-restore — Revert memory to a previous state",
        "/skill-autogen — Review recent conversation for reusable skills",
        "/help — Show available commands",
    ]
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.priority("/tasks", cmd_tasks)
    router.priority_prefix("/tasks resume ", cmd_tasks_resume)
    router.priority_prefix("/tasks discard ", cmd_tasks_discard)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/tasks", cmd_tasks)
    router.prefix("/tasks resume ", cmd_tasks_resume)
    router.prefix("/tasks discard ", cmd_tasks_discard)
    router.exact("/dream", cmd_dream)
    router.exact("/dream-log", cmd_dream_log)
    router.prefix("/dream-log ", cmd_dream_log)
    router.exact("/dream-restore", cmd_dream_restore)
    router.prefix("/dream-restore ", cmd_dream_restore)
    router.exact("/skill-autogen", cmd_skill_autogen)
    router.exact("/help", cmd_help)
