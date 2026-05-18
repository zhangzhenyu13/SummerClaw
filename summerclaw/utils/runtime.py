"""Runtime-specific helper functions and constants."""

from __future__ import annotations

from typing import Any

from loguru import logger

from summerclaw.utils.helpers import stringify_text_blocks

_MAX_REPEAT_EXTERNAL_LOOKUPS = 2
_MAX_BLOCK_BEFORE_FATAL = 3  # additional blocked attempts before forcing the loop to stop

EMPTY_FINAL_RESPONSE_MESSAGE = (
    "I completed the tool steps but couldn't produce a final answer. "
    "Please try again or narrow the task."
)

FINALIZATION_RETRY_PROMPT = (
    "Please provide your response to the user based on the conversation above."
)

LENGTH_RECOVERY_PROMPT = (
    "Output limit reached. Continue exactly where you left off "
    "— no recap, no apology. Break remaining work into smaller steps if needed."
)


def empty_tool_result_message(tool_name: str) -> str:
    """Short prompt-safe marker for tools that completed without visible output."""
    return f"({tool_name} completed with no output)"


def ensure_nonempty_tool_result(tool_name: str, content: Any) -> Any:
    """Replace semantically empty tool results with a short marker string."""
    if content is None:
        return empty_tool_result_message(tool_name)
    if isinstance(content, str) and not content.strip():
        return empty_tool_result_message(tool_name)
    if isinstance(content, list):
        if not content:
            return empty_tool_result_message(tool_name)
        text_payload = stringify_text_blocks(content)
        if text_payload is not None and not text_payload.strip():
            return empty_tool_result_message(tool_name)
    return content


def is_blank_text(content: str | None) -> bool:
    """True when *content* is missing or only whitespace."""
    return content is None or not content.strip()


def build_finalization_retry_message() -> dict[str, str]:
    """A short no-tools-allowed prompt for final answer recovery."""
    return {"role": "user", "content": FINALIZATION_RETRY_PROMPT}


def build_length_recovery_message() -> dict[str, str]:
    """Prompt the model to continue after hitting output token limit."""
    return {"role": "user", "content": LENGTH_RECOVERY_PROMPT}


def external_lookup_signature(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Stable signature for repeated external lookups we want to throttle."""
    if tool_name == "web_fetch":
        url = str(arguments.get("url") or "").strip()
        if url:
            return f"web_fetch:{url.lower()}"
    if tool_name == "web_search":
        query = str(arguments.get("query") or arguments.get("search_term") or "").strip()
        if query:
            return f"web_search:{query.lower()}"
    if tool_name == "exec":
        command = str(arguments.get("command") or "").strip()
        if command:
            return f"exec:{command}"
    return None


def repeated_external_lookup_error(
    tool_name: str,
    arguments: dict[str, Any],
    seen_counts: dict[str, int],
) -> tuple[str, bool] | None:
    """Block repeated external lookups after a small retry budget.

    Returns ``(error_message, is_fatal)`` when the lookup should be blocked,
    or ``None`` when it is still allowed.  *is_fatal* is ``True`` when the
    same call has been blocked too many times and the agent loop should stop.
    """
    signature = external_lookup_signature(tool_name, arguments)
    if signature is None:
        return None
    count = seen_counts.get(signature, 0) + 1
    seen_counts[signature] = count
    if count <= _MAX_REPEAT_EXTERNAL_LOOKUPS:
        return None
    fatal = count > _MAX_REPEAT_EXTERNAL_LOOKUPS + _MAX_BLOCK_BEFORE_FATAL
    logger.warning(
        "Blocking repeated external lookup {} on attempt {} (fatal={})",
        signature[:160],
        count,
        fatal,
    )
    if fatal:
        message = (
            "CRITICAL: This exact external lookup has been blocked after "
            f"{count} repeated attempts. You MUST stop calling this command "
            "immediately. Provide your best answer with whatever information "
            "you already have, or inform the user that this lookup could not "
            "be completed."
        )
    else:
        message = (
            "Error: repeated external lookup blocked. "
            "Use the results you already have to answer, or try a meaningfully different source."
        )
    return message, fatal
