"""AskUser tool — blocking user interaction for the agent loop.

When the agent needs clarification, confirmation, or a choice from the user,
it calls this tool to pause execution, send a question via the message channel,
and wait for the user's reply.  The runner detects the special
``ASK_USER_PENDING`` marker and triggers an injection drain cycle so the
user's answer becomes the next turn's prompt.
"""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import ArraySchema, IntegerSchema, StringSchema, tool_parameters_schema

# ---------------------------------------------------------------------------
# Special marker returned by ask_user to signal the runner to pause and wait
# for the user's reply via the injection (pending_queue) mechanism.
# ---------------------------------------------------------------------------
ASK_USER_PENDING = "__ASK_USER_PENDING__"


@tool_parameters(
    tool_parameters_schema(
        question=StringSchema("The question to ask the user"),
        candidates=ArraySchema(
            StringSchema(""),
            description="Optional list of candidate answers to present as options",
        ),
        timeout=IntegerSchema(
            300,
            description="Seconds to wait for user reply (default 300, max 600)",
            minimum=10,
            maximum=600,
        ),
        required=["question"],
    )
)
class AskUserTool(Tool):
    """Pause and ask the user a question, then inject the answer into the next turn.

    The question is delivered via the configured message channel.  The agent
    loop blocks (stops making LLM calls) until the user replies or the timeout
    expires.  The user's reply text becomes a user message in the conversation.
    """

    def __init__(
        self,
        send_callback: Any = None,
        *,
        default_channel: str = "",
        default_chat_id: str = "",
    ) -> None:
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    # -- context wiring (called by _set_tool_context) ------------------------

    def set_context(self, channel: str, chat_id: str) -> None:
        """Update routing info from the active session."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    # -- Tool interface ------------------------------------------------------

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Pause execution and ask the user a question. "
            "The agent will wait for the user's reply before continuing. "
            "Use when you need clarification, confirmation, or a choice. "
            "Provide 'candidates' to give the user options to pick from. "
            "The user's answer appears in your next prompt as a user message."
        )

    @property
    def exclusive(self) -> bool:
        """Must run alone — cannot be batched with other tools."""
        return True

    async def execute(
        self,
        question: str,
        candidates: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> str:
        candidates = candidates or []

        # Deliver the question to the user via the message channel
        if self._send_callback:
            from nanobot.bus.events import OutboundMessage

            lines = [f"❓ {question}"]
            if candidates:
                lines.append("\nOptions:")
                for c in candidates:
                    lines.append(f"  • {c}")
            await self._send_callback(OutboundMessage(
                channel=self._default_channel,
                chat_id=self._default_chat_id,
                content="\n".join(lines),
                media=[],
            ))

        # Return the special marker — the runner interprets this to pause
        # the LLM loop and drain the pending injection queue for a user reply.
        return ASK_USER_PENDING
