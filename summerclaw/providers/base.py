"""Base LLM provider interface."""

import asyncio
import itertools
import json
import os
import re
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from loguru import logger

from summerclaw.utils.helpers import image_placeholder_text


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to an OpenAI-style tool_call payload."""
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        if self.extra_content:
            tool_call["extra_content"] = self.extra_content
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = self.function_provider_specific_fields
        return tool_call


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    retry_after: float | None = None  # Provider supplied retry wait in seconds.
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1, MiMo etc.
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking
    # Structured error metadata used by retry policy when finish_reason == "error".
    error_status_code: int | None = None
    error_kind: str | None = None  # e.g. "timeout", "connection"
    error_type: str | None = None  # Provider/type semantic, e.g. insufficient_quota.
    error_code: str | None = None  # Provider/code semantic, e.g. rate_limit_exceeded.
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def should_execute_tools(self) -> bool:
        """Tools execute only when has_tool_calls AND finish_reason is ``tool_calls`` / ``stop``.
        Blocks gateway-injected calls under ``refusal`` / ``content_filter`` / ``error`` (#3220)."""
        if not self.has_tool_calls:
            return False
        return self.finish_reason in ("tool_calls", "stop")


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation settings."""

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


_SYNTHETIC_USER_CONTENT = "(conversation continued)"


class _CrossLoopSemaphore:
    """Async context manager that limits concurrency across event loops.

    ``asyncio.Semaphore`` binds to a single event loop, so sharing one
    ``LLMProvider`` across threads (each with its own loop) fails with
    *"bound to a different event loop"*.

    This implementation uses a ``threading.Lock`` (nanosecond-scale) for
    the shared counter and per-loop ``asyncio.Queue`` instances for async
    notification — no thread-pool workers are consumed while waiting.
    """

    __slots__ = ("_value", "_lock", "_queues")

    def __init__(self, value: int) -> None:
        self._value = value
        self._lock = threading.Lock()
        self._queues: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Queue
        ] = weakref.WeakKeyDictionary()

    async def __aenter__(self) -> "_CrossLoopSemaphore":
        loop = asyncio.get_running_loop()
        queue = self._queues.get(loop)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[loop] = queue
        while True:
            with self._lock:
                if self._value > 0:
                    self._value -= 1
                    return self
            await queue.get()

    async def __aexit__(self, *exc: object) -> None:
        with self._lock:
            self._value += 1
        for lp, q in list(self._queues.items()):
            if not lp.is_closed():
                lp.call_soon_threadsafe(q.put_nowait, None)


class LLMProvider(ABC):
    """Base class for LLM providers."""

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _PERSISTENT_MAX_DELAY = 60
    _PERSISTENT_IDENTICAL_ERROR_LIMIT = 10
    _RETRY_HEARTBEAT_CHUNK = 30
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "temporarily unavailable",
    )
    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
    _TRANSIENT_ERROR_KINDS = frozenset({"timeout", "connection"})
    _NON_RETRYABLE_429_ERROR_TOKENS = frozenset({
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit_reached",
        "insufficient_balance",
        "credit_balance_too_low",
        "billing_not_active",
        "payment_required",
    })
    _RETRYABLE_429_ERROR_TOKENS = frozenset({
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
        "request_limit_exceeded",
        "requests_limit_exceeded",
        "overloaded_error",
    })
    _NON_RETRYABLE_429_TEXT_MARKERS = (
        "insufficient_quota",
        "insufficient quota",
        "quota exceeded",
        "quota exhausted",
        "billing hard limit",
        "billing_hard_limit_reached",
        "billing not active",
        "insufficient balance",
        "insufficient_balance",
        "credit balance too low",
        "payment required",
        "out of credits",
        "out of quota",
        "exceeded your current quota",
    )
    _RETRYABLE_429_TEXT_MARKERS = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "retry after",
        "try again in",
        "temporarily unavailable",
        "overloaded",
        "concurrency limit",
    )

    _SENTINEL = object()

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()
        self.max_concurrency: int = 20
        self._concurrency_semaphore: _CrossLoopSemaphore | None = None
        self._semaphore_max: int = -1

    def _get_concurrency_semaphore(self) -> _CrossLoopSemaphore | None:
        """Return a shared semaphore for limiting concurrent LLM API calls.

        Returns ``None`` when ``max_concurrency <= 0`` (unlimited).
        The semaphore is created lazily on first access so that
        ``max_concurrency`` can be changed after construction.

        Uses ``threading.Semaphore`` internally so the same instance
        works correctly across multiple asyncio event loops (e.g. when
        training runs in a background thread).

        When ``SUMMERCLAW_DEBUG_LLM=1``, forces concurrency to 1 so that
        requests are serialized and easier to trace in logs.
        """
        if os.environ.get("SUMMERCLAW_DEBUG_LLM"):
            if self._concurrency_semaphore is None or self._semaphore_max != 1:
                self._semaphore_max = 1
                self._concurrency_semaphore = _CrossLoopSemaphore(1)
            return self._concurrency_semaphore
        if self.max_concurrency <= 0:
            return None
        if self._concurrency_semaphore is None or self._semaphore_max != self.max_concurrency:
            self._semaphore_max = self.max_concurrency
            self._concurrency_semaphore = _CrossLoopSemaphore(self.max_concurrency)
        return self._concurrency_semaphore

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize message content: fix empty blocks, strip internal _meta fields."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                new_items: list[Any] = []
                changed = False
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    ):
                        changed = True
                        continue
                    if isinstance(item, dict) and "_meta" in item:
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(item)
                if changed:
                    clean = dict(msg)
                    if new_items:
                        clean["content"] = new_items
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        return result

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        """Extract tool name from either OpenAI or Anthropic-style tool schemas."""
        name = tool.get("name")
        if isinstance(name, str):
            return name
        fn = tool.get("function")
        if isinstance(fn, dict):
            fname = fn.get("name")
            if isinstance(fname, str):
                return fname
        return ""

    @classmethod
    def _tool_cache_marker_indices(cls, tools: list[dict[str, Any]]) -> list[int]:
        """Return cache marker indices: builtin/MCP boundary and tail index."""
        if not tools:
            return []

        tail_idx = len(tools) - 1
        last_builtin_idx: int | None = None
        for i in range(tail_idx, -1, -1):
            if not cls._tool_name(tools[i]).startswith("mcp_"):
                last_builtin_idx = i
                break

        ordered_unique: list[int] = []
        for idx in (last_builtin_idx, tail_idx):
            if idx is not None and idx not in ordered_unique:
                ordered_unique.append(idx)
        return ordered_unique

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Keep only provider-safe message keys and normalize assistant content."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            tool_choice: Tool selection strategy ("auto", "required", or specific tool dict).

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def _is_transient_response(cls, response: LLMResponse) -> bool:
        """Prefer structured error metadata, fallback to text markers for legacy providers."""
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)

        if response.error_status_code is not None:
            status = int(response.error_status_code)
            if status == 429:
                return cls._is_retryable_429_response(response)
            if status in cls._RETRYABLE_STATUS_CODES or status >= 500:
                return True

        kind = (response.error_kind or "").strip().lower()
        if kind in cls._TRANSIENT_ERROR_KINDS:
            return True

        return cls._is_transient_error(response.content)

    @staticmethod
    def _normalize_error_token(value: Any) -> str | None:
        if value is None:
            return None
        token = str(value).strip().lower()
        return token or None

    @classmethod
    def _extract_error_type_code(cls, payload: Any) -> tuple[str | None, str | None]:
        data: dict[str, Any] | None = None
        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            text = payload.strip()
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    data = parsed
        if not isinstance(data, dict):
            return None, None

        error_obj = data.get("error")
        type_value = data.get("type")
        code_value = data.get("code")
        if isinstance(error_obj, dict):
            type_value = error_obj.get("type") or type_value
            code_value = error_obj.get("code") or code_value

        return cls._normalize_error_token(type_value), cls._normalize_error_token(code_value)

    @classmethod
    def _is_retryable_429_response(cls, response: LLMResponse) -> bool:
        type_token = cls._normalize_error_token(response.error_type)
        code_token = cls._normalize_error_token(response.error_code)
        semantic_tokens = {
            token for token in (type_token, code_token)
            if token is not None
        }
        if any(token in cls._NON_RETRYABLE_429_ERROR_TOKENS for token in semantic_tokens):
            return False

        content = (response.content or "").lower()
        if any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS):
            return False

        if any(token in cls._RETRYABLE_429_ERROR_TOKENS for token in semantic_tokens):
            return True
        if any(marker in content for marker in cls._RETRYABLE_429_TEXT_MARKERS):
            return True
        # Unknown 429 defaults to WAIT+retry.
        return True

    @staticmethod
    def _enforce_role_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages and drop trailing assistant messages.

        Some providers (OpenAI-compat, Azure, vLLM, Ollama, etc.) reject requests
        where the last message is 'assistant' (prefill not supported) or two
        consecutive non-system messages share the same role.
        """
        if not messages:
            return messages

        merged: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if (
                merged
                and role != "system"
                and role not in ("tool",)
                and merged[-1].get("role") == role
                and role in ("user", "assistant")
            ):
                prev = merged[-1]
                if role == "assistant":
                    prev_has_tools = bool(prev.get("tool_calls"))
                    curr_has_tools = bool(msg.get("tool_calls"))
                    if curr_has_tools:
                        merged[-1] = dict(msg)
                        continue
                    if prev_has_tools:
                        continue
                prev_content = prev.get("content") or ""
                curr_content = msg.get("content") or ""
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    prev["content"] = (prev_content + "\n\n" + curr_content).strip()
                else:
                    merged[-1] = dict(msg)
            else:
                merged.append(dict(msg))

        last_popped = None
        while merged and merged[-1].get("role") == "assistant":
            last_popped = merged.pop()

        # If removing trailing assistant messages left only system messages,
        # the request would be invalid for most providers (e.g. Zhipu/GLM
        # error 1214).  Recover by converting the last popped assistant
        # message to a user message so the LLM can still see the content.
        if (
            merged
            and last_popped is not None
            and not any(m.get("role") in ("user", "tool") for m in merged)
        ):
            recovered = dict(last_popped)
            recovered["role"] = "user"
            merged.append(recovered)

        # Safety net: ensure the first non-system message is not a bare
        # ``assistant`` message.  Providers like GLM reject system→assistant
        # with error 1214.  This can happen when upstream truncation (e.g.
        # _snip_history) drops the only user message.  Insert a synthetic
        # user message to keep the sequence valid.
        for i, msg in enumerate(merged):
            if msg.get("role") != "system":
                if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    merged.insert(i, {"role": "user", "content": _SYNTHETIC_USER_CONTENT})
                break

        return merged

    @staticmethod
    def _strip_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Replace image_url blocks with text placeholder. Returns None if no images found."""
        found = False
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        path = (b.get("_meta") or {}).get("path", "")
                        placeholder = image_placeholder_text(path, empty="[image omitted]")
                        new_content.append({"type": "text", "text": placeholder})
                        found = True
                    else:
                        new_content.append(b)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    @staticmethod
    def _strip_image_content_inplace(messages: list[dict[str, Any]]) -> bool:
        """Replace image_url blocks with text placeholder *in-place*.

        Mutates the content lists of the original message dicts so that
        callers holding references to those dicts also see the stripped
        version.
        """
        found = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for i, b in enumerate(content):
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        path = (b.get("_meta") or {}).get("path", "")
                        placeholder = image_placeholder_text(path, empty="[image omitted]")
                        content[i] = {"type": "text", "text": placeholder}
                        found = True
        return found

    # ── Debug logging helpers ──────────────────────────────────────────

    _llm_debug_counter = itertools.count(1)

    @staticmethod
    def _debug_log_request(req_num: int, kwargs: dict[str, Any]) -> None:
        """Log outgoing LLM request messages (SUMMERCLAW_DEBUG_LLM=1)."""
        model = kwargs.get("model", "?")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        max_tokens = kwargs.get("max_tokens", "?")
        temperature = kwargs.get("temperature", "?")
        reasoning_effort = kwargs.get("reasoning_effort", None)
        tool_choice = kwargs.get("tool_choice", None)
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"[DEBUG LLM #{req_num}] REQUEST  | model={model} | {len(messages)} messages")
        print(f"[DEBUG LLM #{req_num}]   max_tokens={max_tokens} | temperature={temperature} | "
              f"reasoning_effort={reasoning_effort} | tool_choice={tool_choice}")
        print(sep)
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                print(f"  [{i}] {role} ({len(content)} chars):")
                print(f"      {content[:2000]}")
            elif isinstance(content, list):
                text = "; ".join(
                    p.get("text", "")[:200] for p in content if isinstance(p, dict) and p.get("text")
                )
                print(f"  [{i}] {role} (list, {len(content)} parts): {text[:2000]}")
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    print(f"      tool_call: {fn.get('name', '?')}({fn.get('arguments', '')})")
        if tools:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools[:10]]
            print(f"  tools ({len(tools)}): {tool_names}")
        print(sep)
        # Flush immediately so logs appear even if the process hangs
        import sys
        sys.stdout.flush()

    @staticmethod
    def _debug_log_response(req_num: int, response: "LLMResponse", elapsed: float) -> None:
        """Log LLM response content (SUMMERCLAW_DEBUG_LLM=1)."""
        sep = "-" * 80
        print(f"[DEBUG LLM #{req_num}] RESPONSE | finish={response.finish_reason} | total {elapsed:.1f}s")
        print(sep)
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  tool_call: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:500]})")
        if response.usage:
            print(f"  usage: {response.usage}")
        # Full content already streamed in real-time, just show final length
        if response.content:
            print(f"  final_content ({len(response.content)} chars): {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
        print(f"{'=' * 80}\n")
        import sys
        sys.stdout.flush()

    @staticmethod
    def _debug_make_stream_printer(req_num: int) -> Callable[[str], Awaitable[None]]:
        """Create an ``on_content_delta`` callback that prints chunks live.

        Also prints a 'still streaming' heartbeat every 10 seconds so you can
        tell the request is alive even during long reasoning/thinking phases.
        """
        state: dict[str, Any] = {
            "chars": 0,
            "first_chunk_time": None,
            "last_heartbeat": None,
        }

        async def _printer(delta: str) -> None:
            import sys
            now = time.monotonic()
            # First-chunk marker
            if state["first_chunk_time"] is None:
                state["first_chunk_time"] = now
                state["last_heartbeat"] = now
                print(f"[DEBUG LLM #{req_num}] STREAM >>> (first chunk)", flush=True)
            # Print the chunk inline
            print(delta, end="", flush=True)
            state["chars"] += len(delta)
            state["last_heartbeat"] = now

        return _printer

    @staticmethod
    async def _debug_heartbeat(req_num: int, t0: float) -> None:
        """Print a heartbeat every 15s while waiting for LLM response."""
        import sys
        while True:
            await asyncio.sleep(15)
            elapsed = time.monotonic() - t0
            print(
                f"\n[DEBUG LLM #{req_num}] ⏳ still waiting... ({elapsed:.0f}s elapsed)",
                flush=True,
            )
            sys.stdout.flush()

    # Keys from _build_request_kwargs that are NOT accepted by chat()/chat_stream()
    _RETRY_ONLY_KEYS = frozenset({"retry_mode", "on_retry_wait"})

    # Default per-request timeout for non-streaming chat() calls.
    # Prevents indefinite hangs when the provider accepts the connection
    # but never responds (common with thinking models on DashScope).
    _CHAT_REQUEST_TIMEOUT_S = int(os.environ.get(
        "SUMMERCLAW_CHAT_TIMEOUT_S", "300",
    ))

    async def _safe_chat(self, **kwargs: Any) -> LLMResponse:
        """Call chat() and convert unexpected exceptions to error responses.

        Every ``chat()`` call is wrapped in ``asyncio.wait_for`` with a
        configurable timeout (default 300s, override via
        ``SUMMERCLAW_CHAT_TIMEOUT_S``) so a single hung request cannot
        block the entire agent loop indefinitely.

        In debug mode (``SUMMERCLAW_DEBUG_LLM=1``), logs request/response
        details and runs a background heartbeat every 15s.
        """
        sem = self._get_concurrency_semaphore()
        async with (sem or nullcontext()):
            debug = os.environ.get("SUMMERCLAW_DEBUG_LLM")
            req_num = 0
            t0 = time.monotonic()
            heartbeat_task: asyncio.Task | None = None
            timeout_s = self._CHAT_REQUEST_TIMEOUT_S
            if debug:
                req_num = next(LLMProvider._llm_debug_counter)
                self._debug_log_request(req_num, kwargs)
                print(f"[DEBUG LLM #{req_num}] chat() timeout={timeout_s}s", flush=True)
                # ── Background heartbeat ─────────────────────────────
                heartbeat_task = asyncio.create_task(
                    self._debug_heartbeat(req_num, t0)
                )
            try:
                # Filter out retry-only keys — chat() does not accept them
                # (they are consumed by chat_with_retry / _run_with_retry)
                chat_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in self._RETRY_ONLY_KEYS
                }
                result = await asyncio.wait_for(
                    self.chat(**chat_kwargs),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                err_msg = (
                    f"chat() timed out after {elapsed:.0f}s "
                    f"(limit={timeout_s}s, model={kwargs.get('model', '?')})"
                )
                if debug:
                    print(f"\n[DEBUG LLM #{req_num}] TIMEOUT: {err_msg}", flush=True)
                result = LLMResponse(content=err_msg, finish_reason="error")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
            if debug:
                self._debug_log_response(req_num, result, time.monotonic() - t0)
            return result

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream a chat completion, calling *on_content_delta* for each text chunk.

        Returns the same ``LLMResponse`` as :meth:`chat`.  The default
        implementation falls back to a non-streaming call and delivers the
        full content as a single delta.  Providers that support native
        streaming should override this method.
        """
        response = await self.chat(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    async def _safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        """Call chat_stream() and convert unexpected exceptions to error responses.

        In debug mode, wraps ``on_content_delta`` so each chunk is also printed
        to stdout with timing info.  A background heartbeat prints every 15s.
        """
        sem = self._get_concurrency_semaphore()
        async with (sem or nullcontext()):
            debug = os.environ.get("SUMMERCLAW_DEBUG_LLM")
            req_num = 0
            t0 = time.monotonic()
            heartbeat_task: asyncio.Task | None = None
            # ── Filter retry-only keys (same as _safe_chat) ─────────
            # chat_stream() does not accept retry_mode / on_retry_wait;
            # they are consumed by chat_stream_with_retry / _run_with_retry.
            stream_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in self._RETRY_ONLY_KEYS
            }
            if debug:
                req_num = next(LLMProvider._llm_debug_counter)
                self._debug_log_request(req_num, stream_kwargs)
                # Wrap on_content_delta to also print chunks live
                original_delta = stream_kwargs.get("on_content_delta")
                debug_printer = self._debug_make_stream_printer(req_num)

                async def _combined_delta(delta: str) -> None:
                    await debug_printer(delta)
                    if original_delta:
                        await original_delta(delta)

                stream_kwargs = dict(stream_kwargs)
                stream_kwargs["on_content_delta"] = _combined_delta
                # ── Background heartbeat ─────────────────────────────
                heartbeat_task = asyncio.create_task(
                    self._debug_heartbeat(req_num, t0)
                )
            try:
                result = await self.chat_stream(**stream_kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
            if debug:
                print()  # newline after streamed output
                self._debug_log_response(req_num, result, time.monotonic() - t0)
            return result

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient provider failures."""
        if max_tokens is self._SENTINEL or max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL or temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )
        return await self._run_with_retry(
            self._safe_chat_stream,
            kw,
            messages,
            retry_mode=retry_mode,
            on_retry_wait=on_retry_wait,
        )

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient provider failures.

        Parameters default to ``self.generation`` when not explicitly passed,
        so callers no longer need to thread temperature / max_tokens /
        reasoning_effort through every layer. Explicit ``None`` is also
        normalized to the provider's generation defaults so that downstream
        ``_build_kwargs`` never sees ``None`` for ``max_tokens`` / ``temperature``
        (which would crash ``max(1, max_tokens)``).
        """
        if max_tokens is self._SENTINEL or max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL or temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        return await self._run_with_retry(
            self._safe_chat,
            kw,
            messages,
            retry_mode=retry_mode,
            on_retry_wait=on_retry_wait,
        )

    @classmethod
    def _extract_retry_after(cls, content: str | None) -> float | None:
        text = (content or "").lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)",
            r"wait\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)\s*before retry",
            r"retry[_-]?after[\"'\s:=]+(\d+(?:\.\d+)?)",
        )
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2) if idx < 3 else "s"
            return cls._to_retry_seconds(value, unit)
        return None

    @classmethod
    def _to_retry_seconds(cls, value: float, unit: str | None = None) -> float:
        normalized_unit = (unit or "s").lower()
        if normalized_unit in {"ms", "milliseconds"}:
            return max(0.1, value / 1000.0)
        if normalized_unit in {"m", "min", "minutes"}:
            return max(0.1, value * 60.0)
        return max(0.1, value)

    @classmethod
    def _extract_retry_after_from_headers(cls, headers: Any) -> float | None:
        if not headers:
            return None

        def _header_value(name: str) -> Any:
            if hasattr(headers, "get"):
                value = headers.get(name) or headers.get(name.title())
                if value is not None:
                    return value
            if isinstance(headers, dict):
                for key, value in headers.items():
                    if isinstance(key, str) and key.lower() == name.lower():
                        return value
            return None

        try:
            retry_ms = _header_value("retry-after-ms")
            if retry_ms is not None:
                value = float(retry_ms) / 1000.0
                if value > 0:
                    return value
        except (TypeError, ValueError):
            pass

        retry_after = _header_value("retry-after")
        if retry_after is None:
            return None
        retry_after_text = str(retry_after).strip()
        if not retry_after_text:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", retry_after_text):
            return cls._to_retry_seconds(float(retry_after_text), "s")
        try:
            retry_at = parsedate_to_datetime(retry_after_text)
        except Exception:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        remaining = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
        return max(0.1, remaining)

    @classmethod
    def _extract_retry_after_from_response(cls, response: LLMResponse) -> float | None:
        if response.error_retry_after_s is not None and response.error_retry_after_s > 0:
            return response.error_retry_after_s
        if response.retry_after is not None and response.retry_after > 0:
            return response.retry_after
        return cls._extract_retry_after(response.content)

    async def _sleep_with_heartbeat(
        self,
        delay: float,
        *,
        attempt: int,
        persistent: bool,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        remaining = max(0.0, delay)
        while remaining > 0:
            if on_retry_wait:
                kind = "persistent retry" if persistent else "retry"
                await on_retry_wait(
                    f"Model request failed, {kind} in {max(1, int(round(remaining)))}s "
                    f"(attempt {attempt})."
                )
            chunk = min(remaining, self._RETRY_HEARTBEAT_CHUNK)
            await asyncio.sleep(chunk)
            remaining -= chunk

    async def _run_with_retry(
        self,
        call: Callable[..., Awaitable[LLMResponse]],
        kw: dict[str, Any],
        original_messages: list[dict[str, Any]],
        *,
        retry_mode: str,
        on_retry_wait: Callable[[str], Awaitable[None]] | None,
    ) -> LLMResponse:
        attempt = 0
        delays = list(self._CHAT_RETRY_DELAYS)
        persistent = retry_mode == "persistent"
        last_response: LLMResponse | None = None
        last_error_key: str | None = None
        identical_error_count = 0
        while True:
            attempt += 1
            response = await call(**kw)
            if response.finish_reason != "error":
                return response
            last_response = response
            error_key = ((response.content or "").strip().lower() or None)
            if error_key and error_key == last_error_key:
                identical_error_count += 1
            else:
                last_error_key = error_key
                identical_error_count = 1 if error_key else 0

            if not self._is_transient_response(response):
                stripped = self._strip_image_content(original_messages)
                if stripped is not None and stripped != kw["messages"]:
                    logger.warning(
                        "Non-transient LLM error with image content, retrying without images"
                    )
                    retry_kw = dict(kw)
                    retry_kw["messages"] = stripped
                    result = await call(**retry_kw)
                    # Permanently strip images from the original messages so
                    # subsequent iterations do not repeat the error-retry cycle.
                    if result.finish_reason != "error":
                        self._strip_image_content_inplace(original_messages)
                    return result
                return response

            if persistent and identical_error_count >= self._PERSISTENT_IDENTICAL_ERROR_LIMIT:
                logger.warning(
                    "Stopping persistent retry after {} identical transient errors: {}",
                    identical_error_count,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_wait:
                    await on_retry_wait(
                        f"Persistent retry stopped after {identical_error_count} identical errors."
                    )
                return response

            if not persistent and attempt > len(delays):
                logger.warning(
                    "LLM request failed after {} retries, giving up: {}",
                    attempt,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_wait:
                    await on_retry_wait(
                        f"Model request failed after {attempt} retries, giving up."
                    )
                break

            base_delay = delays[min(attempt - 1, len(delays) - 1)]
            delay = self._extract_retry_after_from_response(response) or base_delay
            if persistent:
                delay = min(delay, self._PERSISTENT_MAX_DELAY)

            logger.warning(
                "LLM transient error (attempt {}{}), retrying in {}s: {}",
                attempt,
                "+" if persistent and attempt > len(delays) else f"/{len(delays)}",
                int(round(delay)),
                (response.content or "")[:120].lower(),
            )
            await self._sleep_with_heartbeat(
                delay,
                attempt=attempt,
                persistent=persistent,
                on_retry_wait=on_retry_wait,
            )

        return last_response if last_response is not None else await call(**kw)

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings for the given texts.

        OpenAI-compatible providers override this to call their
        ``/embeddings`` endpoint using the same credentials as chat.
        Providers that do not support embeddings (e.g. Anthropic)
        raise :class:`NotImplementedError`.

        Args:
            texts: List of input strings to embed.
            model: The embedding model name (e.g. ``"text-embedding-3-small"``).

        Returns:
            List of embedding vectors, each as ``list[float]``.
        """
        raise NotImplementedError(
            f"Embeddings are not supported by the '{type(self).__name__}' provider."
        )
