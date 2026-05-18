"""Minimal command routing table for slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from summerclaw.bus.events import InboundMessage, OutboundMessage
    from summerclaw.session.manager import Session

Handler = Callable[["CommandContext"], Awaitable["OutboundMessage | None"]]


@dataclass
class CommandContext:
    """Everything a command handler needs to produce a response."""

    msg: InboundMessage
    session: Session | None
    key: str
    raw: str
    args: str = ""
    loop: Any = None


class CommandRouter:
    """Pure dict-based command dispatch.

    Three tiers checked in order:
      1. *priority* — exact-match or prefix commands handled before the dispatch
         lock (e.g. /stop, /restart, /tasks).  These bypass pending-queue
         routing so they are always answered immediately, even when a session
         is busy processing another message.
      2. *exact* — exact-match commands handled inside the dispatch lock.
      3. *prefix* — longest-prefix-first match (e.g. "/team ").
      4. *interceptors* — fallback predicates (e.g. team-mode active check).
    """

    def __init__(self) -> None:
        self._priority: dict[str, Handler] = {}
        self._priority_prefix: list[tuple[str, Handler]] = []
        self._exact: dict[str, Handler] = {}
        self._prefix: list[tuple[str, Handler]] = []
        self._interceptors: list[Handler] = []

    def priority(self, cmd: str, handler: Handler) -> None:
        self._priority[cmd] = handler

    def priority_prefix(self, pfx: str, handler: Handler) -> None:
        """Register a prefix-matched priority command (dispatched without the lock)."""
        self._priority_prefix.append((pfx, handler))
        self._priority_prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def exact(self, cmd: str, handler: Handler) -> None:
        self._exact[cmd] = handler

    def prefix(self, pfx: str, handler: Handler) -> None:
        self._prefix.append((pfx, handler))
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def intercept(self, handler: Handler) -> None:
        self._interceptors.append(handler)

    def is_priority(self, text: str) -> bool:
        t = text.strip().lower()
        if t in self._priority:
            return True
        for pfx, _ in self._priority_prefix:
            if t.startswith(pfx):
                return True
        return False

    async def dispatch_priority(self, ctx: CommandContext) -> OutboundMessage | None:
        """Dispatch a priority command. Called from run() without the lock."""
        handler = self._priority.get(ctx.raw.lower())
        if handler:
            return await handler(ctx)
        cmd = ctx.raw.lower()
        for pfx, handler in self._priority_prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> OutboundMessage | None:
        """Try exact, prefix, then interceptors. Returns None if unhandled."""
        cmd = ctx.raw.lower()

        if handler := self._exact.get(cmd):
            return await handler(ctx)

        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        for interceptor in self._interceptors:
            result = await interceptor(ctx)
            if result is not None:
                return result

        return None
