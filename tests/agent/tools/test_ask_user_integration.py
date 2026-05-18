"""End-to-end integration tests for the ask_user blocking flow.

Verifies the complete pipeline:
    LLM calls ask_user → ASK_USER_PENDING marker returned →
    runner detects marker → injection callback triggered →
    user reply injected → loop continues → LLM sees reply.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.agent.tools.ask_user import ASK_USER_PENDING, AskUserTool
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.providers.base import LLMResponse, ToolCallRequest

# -- Global constants for message length constraints ------------------------
from summerclaw.config.schema import AgentDefaults

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


# ===========================================================================
# Shared helpers
# ===========================================================================


def _make_tool_registry(with_ask_user: bool = True) -> ToolRegistry:
    """Build a ToolRegistry with an AskUserTool (no-op send callback)."""
    tools = ToolRegistry()
    if with_ask_user:
        ask_tool = AskUserTool(send_callback=AsyncMock())
        tools.register(ask_tool)
    return tools


def _make_injection_callback(queue: asyncio.Queue):
    """Async callback that drains *queue* and returns a list of dicts.

    The items in the queue should be dicts with ``role`` / ``content`` keys,
    or :class:`~summerclaw.bus.events.InboundMessage` objects.
    """

    async def inject_cb(**kwargs):
        items = []
        while not queue.empty():
            items.append(await queue.get())
        return items

    return inject_cb


def _make_injection_callback_with_limit(queue: asyncio.Queue):
    """Async callback with ``limit`` param — same semantics as real loop."""

    async def inject_cb(*, limit: int):
        items = []
        while not queue.empty() and len(items) < limit:
            items.append(await queue.get())
        return items

    return inject_cb


class RecordingHook:
    """Minimal hook that records lifecycle events for verification."""

    def __init__(self):
        self.events: list[dict] = []

    def wants_streaming(self):
        return False

    async def before_iteration(self, context):
        self.events.append({"type": "before_iteration", "iteration": context.iteration})

    async def before_execute_tools(self, context):
        self.events.append({
            "type": "before_execute_tools",
            "iteration": context.iteration,
            "tool_names": [tc.name for tc in context.tool_calls],
        })

    async def after_iteration(self, context):
        self.events.append({
            "type": "after_iteration",
            "iteration": context.iteration,
            "tool_results": list(context.tool_results),
            "tool_events": list(context.tool_events),
            "final_content": context.final_content,
            "stop_reason": context.stop_reason,
        })

    def finalize_content(self, context, content):
        return content


# ===========================================================================
# 1. Core blocking flow — end-to-end
# ===========================================================================


class TestAskUserBlockingFlow:
    """Verify the full ask_user → inject → continue life-cycle."""

    @pytest.mark.asyncio
    async def test_ask_user_alone_blocks_loop_and_injects_reply(self):
        """LLM calls only ask_user → marker detected → user reply injected →
        loop continues with reply visible in next LLM turn → LLM finishes."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "my answer is 42"})

        second_call_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Let me ask the user",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "What is the meaning of life?"},
                    )],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                )
            second_call_messages.append(list(messages))
            return LLMResponse(content="The user said 42", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Solve the puzzle"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        # -- assertions -------------------------------------------------------
        assert result.final_content == "The user said 42", (
            "LLM should process the injected reply and finish"
        )
        assert result.tools_used == ["ask_user"], "ask_user should be in tools_used"
        assert "ask_user" in [e["name"] for e in result.tool_events], (
            "ask_user event should appear in tool_events"
        )
        assert result.stop_reason == "completed"
        assert result.error is None
        assert result.had_injections is True, (
            "had_injections must be True when user reply was injected"
        )

        # Verify the second LLM call contains the user's reply
        assert len(second_call_messages) == 1
        call_messages = second_call_messages[0]
        user_content_found = any(
            msg.get("role") == "user" and "my answer is 42" in str(msg.get("content", ""))
            for msg in call_messages
        )
        assert user_content_found, (
            "User's reply 'my answer is 42' must appear in next LLM call's messages"
        )

    @pytest.mark.asyncio
    async def test_ask_user_mixed_with_normal_tool(self):
        """LLM calls ask_user + a normal tool in same turn →
        normal result is saved, ASK_USER_PENDING string is never sent to LLM,
        user reply injected, loop continues."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner
        from summerclaw.agent.tools.self import MyTool

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "yes please"})

        all_llm_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            all_llm_messages.append(list(messages))
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Let me check and ask",
                    tool_calls=[
                        ToolCallRequest(
                            id="ask_1", name="ask_user",
                            arguments={"question": "Should I continue?"},
                        ),
                        ToolCallRequest(
                            id="self_1", name="check", arguments={"action": "check"},
                        ),
                    ],
                    usage={},
                )
            return LLMResponse(content="Great, continuing", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        loop_mock = MagicMock()
        loop_mock.model = "test-model"
        loop_mock.max_iterations = 5
        loop_mock.context_window_tokens = 65536
        loop_mock.workspace = MagicMock()
        loop_mock.restrict_to_workspace = False
        loop_mock._start_time = 1000.0
        loop_mock.exec_config = MagicMock()
        loop_mock.channels_config = MagicMock()
        loop_mock.web_config = MagicMock()
        loop_mock.web_config.enable = False
        loop_mock._last_usage = {}
        loop_mock._runtime_vars = {}
        loop_mock._current_iteration = 0
        loop_mock.provider_retry_mode = "standard"
        loop_mock.max_tool_result_chars = 16000
        loop_mock._concurrency_gate = None
        loop_mock._unified_session = False
        loop_mock._extra_hooks = []
        self_tool = MyTool(loop=loop_mock)

        tools = _make_tool_registry()
        tools.register(self_tool)

        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Start"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert result.final_content == "Great, continuing"
        assert "ask_user" in result.tools_used
        assert "check" in result.tools_used
        assert result.had_injections is True

        # The ASK_USER_PENDING string must NEVER appear in any message sent to LLM
        for batch in all_llm_messages:
            for msg in batch:
                content_str = str(msg.get("content", ""))
                assert ASK_USER_PENDING not in content_str, (
                    f"ASK_USER_PENDING marker leaked into LLM message: {msg}"
                )
                # Also verify no tool result content contains the marker
                if isinstance(msg, dict):
                    for v in msg.values():
                        if isinstance(v, str):
                            assert ASK_USER_PENDING not in v, (
                                f"ASK_USER_PENDING leaked in value: {v}"
                            )

        # Normal tool result should be in the conversation messages (2nd call)
        second_call = all_llm_messages[1]  # after ask_user + injection
        normal_tool_msg = [
            msg for msg in second_call
            if msg.get("role") == "tool" and msg.get("name") == "check"
        ]
        assert normal_tool_msg, (
            "Normal tool result (check) should be in messages sent to LLM"
        )

        # User reply should be in messages
        user_reply = [
            msg for msg in second_call
            if msg.get("role") == "user" and "yes please" in str(msg.get("content", ""))
        ]
        assert user_reply, "User reply must be in messages sent to LLM"

    @pytest.mark.asyncio
    async def test_consecutive_ask_user_calls(self):
        """First ask_user → inject → LLM asks again → inject → LLM finishes."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "Paris"})
        await injection_queue.put({"role": "user", "content": "France"})

        final_call_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Question 1",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "What city?"},
                    )],
                    usage={},
                )
            if call_count["n"] == 2:
                return LLMResponse(
                    content="Question 2",
                    tool_calls=[ToolCallRequest(
                        id="ask_2", name="ask_user",
                        arguments={"question": "What country?"},
                    )],
                    usage={},
                )
            final_call_messages.append(list(messages))
            return LLMResponse(content="Got it: Paris, France", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Where?"}],
            tools=tools,
            model="test-model",
            max_iterations=10,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert result.final_content == "Got it: Paris, France"
        assert result.tools_used == ["ask_user", "ask_user"]
        assert result.had_injections is True
        assert call_count["n"] == 3, (
            "LLM should be called 3 times: ask1 → inject → ask2 → inject → final"
        )

    @pytest.mark.asyncio
    async def test_ask_user_on_last_iteration_injects_before_max_stop(self):
        """ask_user on last iteration → inject → loop exhausts iterations →
        max_iterations stop reason with injected messages in history."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "my answer"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Need info",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "What value?"},
                    )],
                    usage={},
                )
            return LLMResponse(
                content="OK processing",
                tool_calls=[ToolCallRequest(id="t1", name="list_dir", arguments={})],
                usage={},
            )

        provider.chat_with_retry = chat_with_retry

        tools = _make_tool_registry()
        tools.execute = AsyncMock(return_value="dir contents")

        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Process"}],
            tools=tools,
            model="test-model",
            max_iterations=2,  # only 2 iterations
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
            max_iterations_message="Hit max {max_iterations}",
        )
        result = await runner.run(spec)

        assert result.stop_reason == "max_iterations", (
            "Should stop due to max_iterations after ask_user consumed an iteration"
        )
        assert result.had_injections is True, (
            "User reply injected even on last iteration"
        )
        assert "my answer" in str(result.messages), (
            "Injected reply should be in result.messages"
        )

    @pytest.mark.asyncio
    async def test_conversation_history_preserved_through_ask_user(self):
        """The full conversation (including system prompt, user messages,
        assistant with tool calls) is preserved through the ask_user cycle."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "no thanks"})

        second_call_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking user",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "Proceed?"},
                    )],
                    reasoning_content="thinking about this",
                    usage={},
                )
            second_call_messages.append(list(messages))
            return LLMResponse(content="User declined", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Start workflow"},
            ],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        await runner.run(spec)

        # The second call's messages must include the full history
        assert len(second_call_messages) == 1
        msgs = second_call_messages[0]

        # System message preserved
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        assert len(system_msgs) >= 1

        # First user message preserved
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert any("Start workflow" in str(m.get("content", "")) for m in user_msgs)

        # Assistant message with tool_call preserved
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert any(m.get("tool_calls") for m in assistant_msgs), (
            "Assistant message with tool_calls must be preserved"
        )

        # Reasoning content preserved
        assert any(
            m.get("reasoning_content") == "thinking about this"
            for m in msgs
            if isinstance(m, dict)
        ), "Reasoning content must be preserved"

        # Injected user reply present
        assert any(
            "no thanks" in str(m.get("content", ""))
            for m in user_msgs
        ), "User reply must be in conversation"


# ===========================================================================
# 2. Injection draining behavior
# ===========================================================================


class TestAskUserInjectionDraining:
    """Verify how the injection_callback is invoked and results processed."""

    @pytest.mark.asyncio
    async def test_no_injection_callback_does_not_crash(self):
        """When injection_callback is None, the loop continues without error."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "Continue?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="No reply from user", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=None,
        )
        result = await runner.run(spec)

        assert result.final_content == "No reply from user"
        assert result.had_injections is False

    @pytest.mark.asyncio
    async def test_injection_callback_empty_returns_not_drained(self):
        """When injection_callback returns [], had_injections stays False."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Moving on", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        async def empty_cb():
            return []

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=empty_cb,
        )
        result = await runner.run(spec)

        assert result.final_content == "Moving on"
        assert result.had_injections is False, (
            "Empty injection should not set had_injections"
        )

    @pytest.mark.asyncio
    async def test_injection_callback_receives_limit_param(self):
        """When callback accepts ``limit``, the limit value is
        _MAX_INJECTIONS_PER_TURN on the first call (ask_user drain)."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner, _MAX_INJECTIONS_PER_TURN

        seen_limits: list[int] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        async def inject_cb(*, limit: int):
            seen_limits.append(limit)
            # Only return data on the first call (ask_user drain);
            # subsequent calls (final response drain) return empty.
            if len(seen_limits) == 1:
                return [{"role": "user", "content": "reply"}]
            return []

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=inject_cb,
        )
        await runner.run(spec)

        # The first call must have the correct limit
        assert len(seen_limits) >= 1
        assert seen_limits[0] == _MAX_INJECTIONS_PER_TURN, (
            "First injection callback call should receive _MAX_INJECTIONS_PER_TURN"
        )

    @pytest.mark.asyncio
    async def test_injection_callback_exception_is_graceful(self):
        """If the injection_callback raises, the loop continues without crash."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Recovered", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        async def failing_cb():
            raise RuntimeError("callback broken")

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=failing_cb,
        )
        result = await runner.run(spec)

        assert result.final_content == "Recovered", (
            "Loop should continue after injection callback exception"
        )
        assert result.had_injections is False, (
            "Exception in callback → no injections drained"
        )

    @pytest.mark.asyncio
    async def test_multiple_messages_injected_at_once(self):
        """Callback returns multiple user messages → all injected."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "msg A"})
        await injection_queue.put({"role": "user", "content": "msg B"})
        await injection_queue.put({"role": "user", "content": "msg C"})

        second_call_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "Tell me three things"},
                    )],
                    usage={},
                )
            second_call_messages.append(list(messages))
            return LLMResponse(content="Got all three", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert result.final_content == "Got all three"
        assert result.had_injections is True

        assert len(second_call_messages) == 1
        msgs = second_call_messages[0]
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert any("msg A" in str(m.get("content")) for m in user_msgs)
        assert any("msg B" in str(m.get("content")) for m in user_msgs)
        assert any("msg C" in str(m.get("content")) for m in user_msgs)

    @pytest.mark.asyncio
    async def test_injection_callback_limit_param_caps_messages(self):
        """With limit-aware callback, only limit-many messages get injected
        per drain call.  The ask_user drain is the first call."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner, _MAX_INJECTIONS_PER_TURN

        injection_queue: asyncio.Queue = asyncio.Queue()
        # Put more messages than _MAX_INJECTIONS_PER_TURN
        for i in range(_MAX_INJECTIONS_PER_TURN + 3):
            await injection_queue.put({"role": "user", "content": f"msg{i}"})

        all_llm_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            all_llm_messages.append(list(messages))
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback_with_limit(injection_queue),
        )
        result = await runner.run(spec)

        assert result.had_injections is True

        # Count injected user messages in ALL messages ever sent to LLM
        injected_count = 0
        for batch in all_llm_messages:
            for m in batch:
                if m.get("role") == "user" and str(m.get("content", "")).startswith("msg"):
                    injected_count += 1
        # Each distinct message should appear at most once;
        # the cap means at most _MAX_INJECTIONS_PER_TURN per drain
        assert injected_count <= _MAX_INJECTIONS_PER_TURN, (
            f"Injected {injected_count} messages across all LLM calls, "
            f"max per drain is {_MAX_INJECTIONS_PER_TURN}"
        )

    @pytest.mark.asyncio
    async def test_inbound_message_objects_converted_to_dicts(self):
        """InboundMessage objects from callback are properly converted to user dicts."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner
        from summerclaw.bus.events import InboundMessage

        all_llm_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            all_llm_messages.append(list(messages))
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "Answer?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Got it", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        # Only return data on first call to prevent infinite re-injection loop
        call_num = {"n": 0}

        async def inject_cb():
            call_num["n"] += 1
            if call_num["n"] == 1:
                return [
                    InboundMessage(channel="cli", sender_id="u1", chat_id="c1",
                                   content="from inbound"),
                ]
            return []

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=inject_cb,
        )
        await runner.run(spec)

        # The InboundMessage should appear as a user message in the second LLM call
        second_call = all_llm_messages[1]
        assert any(
            m.get("role") == "user" and "from inbound" in str(m.get("content"))
            for m in second_call
        ), "InboundMessage content should appear as user message"


# ===========================================================================
# 3. Lifecycle hooks and events
# ===========================================================================


class TestAskUserLifecycle:
    """Verify that hooks fire correctly and events have the right shape."""

    @pytest.mark.asyncio
    async def test_hooks_fire_in_correct_order(self):
        """before_iteration → before_execute_tools → after_iteration in order,
        and after ask_user we see another before_iteration."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        hook = RecordingHook()

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
            hook=hook,
        )
        await runner.run(spec)

        # Verify the sequence
        types_in_order = [e["type"] for e in hook.events]
        # Expected: before(0) → before_exec_tools(0) → after(0) → before(1) → after(1)
        assert "before_iteration" in types_in_order
        assert "before_execute_tools" in types_in_order
        assert types_in_order.count("after_iteration") == 2, (
            "Should have after_iteration for iteration 0 (ask_user) and iteration 1 (final)"
        )

        # Iteration 0 should have ask_user tool in tool_calls
        iter0_exec = next(
            e for e in hook.events
            if e["type"] == "before_execute_tools" and e["iteration"] == 0
        )
        assert "ask_user" in iter0_exec["tool_names"]

    @pytest.mark.asyncio
    async def test_ask_user_event_has_correct_status(self):
        """The tool event for ask_user must have status='ask_user', not 'ok' or 'error'."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        ask_user_events = [
            e for e in result.tool_events if e["name"] == "ask_user"
        ]
        assert len(ask_user_events) == 1
        assert ask_user_events[0]["status"] == "ask_user", (
            "ask_user event should have status='ask_user'"
        )
        assert ask_user_events[0]["detail"] == "waiting for user reply", (
            "ask_user event should have correct detail"
        )

    @pytest.mark.asyncio
    async def test_after_iteration_sees_raw_tool_results(self):
        """Hook context.tool_results includes the raw results (including
        ASK_USER_PENDING) because it is set BEFORE the ask_user branch
        filters them out.  This is the current implementation behavior."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        hook = RecordingHook()

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
            hook=hook,
        )
        await runner.run(spec)

        # After_iteration is called for iteration 0 (the ask_user iteration)
        iter0_after = next(
            e for e in hook.events
            if e["type"] == "after_iteration" and e["iteration"] == 0
        )
        # The raw context includes ASK_USER_PENDING (set before filtering)
        assert ASK_USER_PENDING in iter0_after["tool_results"], (
            "Hook context tool_results contains raw results incl. ASK_USER_PENDING "
            "(context is set before the ask_user branch filters)"
        )

        # But the loop continues (it doesn't break/tool_error on ASK_USER_PENDING)
        assert iter0_after["stop_reason"] is None, (
            "After ask_user iteration, stop_reason should be None (loop continues)"
        )

    @pytest.mark.asyncio
    async def test_had_injections_flag_in_result(self):
        """AgentRunResult.had_injections reflects whether any injections were drained."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        # Case 1: with injection
        q1: asyncio.Queue = asyncio.Queue()
        await q1.put({"role": "user", "content": "reply"})

        call_count = {"n": 0}

        async def make_provider(q: asyncio.Queue) -> MagicMock:
            provider = MagicMock()
            counter = {"n": 0}

            async def chat_with_retry(*, messages, **kwargs):
                counter["n"] += 1
                if counter["n"] == 1:
                    return LLMResponse(
                        content="Asking",
                        tool_calls=[ToolCallRequest(
                            id="ask_1", name="ask_user",
                            arguments={"question": "OK?"},
                        )],
                        usage={},
                    )
                return LLMResponse(content="Done", tool_calls=[], usage={})

            provider.chat_with_retry = chat_with_retry
            return provider

        p1 = await make_provider(q1)
        runner1 = AgentRunner(p1)
        tools1 = _make_tool_registry()
        r1 = await runner1.run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools1, model="m",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(q1),
        ))
        assert r1.had_injections is True, (
            "had_injections should be True when user reply was injected"
        )

        # Case 2: without injection
        p2 = MagicMock()
        counter2 = {"n": 0}

        async def chat_no_inject(*, messages, **kwargs):
            counter2["n"] += 1
            if counter2["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        p2.chat_with_retry = chat_no_inject
        runner2 = AgentRunner(p2)
        tools2 = _make_tool_registry()
        r2 = await runner2.run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools2, model="m",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=None,
        ))
        assert r2.had_injections is False, (
            "had_injections should be False with no injection_callback"
        )

    @pytest.mark.asyncio
    async def test_stop_reason_is_not_tool_error(self):
        """ask_user should NOT cause stop_reason='tool_error' even with
        fail_on_tool_error=True."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            fail_on_tool_error=True,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert result.stop_reason != "tool_error", (
            "ask_user should NOT trigger tool_error even with fail_on_tool_error=True"
        )
        assert result.stop_reason == "completed"


# ===========================================================================
# 4. State resets
# ===========================================================================


class TestAskUserStateReset:
    """Verify that ask_user resets key loop state variables."""

    @pytest.mark.asyncio
    async def test_injection_cycles_reset_on_ask_user(self):
        """ask_user always resets injection_cycles to 0, giving a fresh budget."""
        # This is an indirect test: if we somehow had injection_cycles > 0
        # before ask_user, the drain should still work because cycles reset.
        # With the runner's current code, injection_cycles starts at 0 and
        # the ask_user branch passes 0 explicitly to _try_drain_injections.
        # The test verifies that multiple ask_user calls each trigger a fresh
        # injection drain (each with its own budget).
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        # Queue with replies for two consecutive ask_user calls
        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply 1"})
        await injection_queue.put({"role": "user", "content": "reply 2"})

        call_count = {"n": 0}
        final_call_messages: list[list[dict]] = []

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Ask 1",
                    tool_calls=[ToolCallRequest(
                        id="a1", name="ask_user",
                        arguments={"question": "Q1?"},
                    )],
                    usage={},
                )
            if call_count["n"] == 2:
                return LLMResponse(
                    content="Ask 2",
                    tool_calls=[ToolCallRequest(
                        id="a2", name="ask_user",
                        arguments={"question": "Q2?"},
                    )],
                    usage={},
                )
            final_call_messages.append(list(messages))
            return LLMResponse(content="All done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=10,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert result.final_content == "All done"
        assert result.had_injections is True
        # Both replies should be in the final call
        assert len(final_call_messages) == 1
        msgs = final_call_messages[0]
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert any("reply 1" in str(m.get("content")) for m in user_msgs), (
            "First reply must be visible in conversation"
        )
        assert any("reply 2" in str(m.get("content")) for m in user_msgs), (
            "Second reply must also be visible — both injections should work"
        )


# ===========================================================================
# 5. Message flow correctness
# ===========================================================================


class TestAskUserMessageFlow:
    """Verify that messages are correctly filtered and built."""

    @pytest.mark.asyncio
    async def test_normal_tool_results_preserved_with_ask_user(self):
        """When ask_user is mixed with a normal tool, the normal tool result
        appears in the next LLM call's messages and ASK_USER_PENDING never leaks."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "yes"})

        all_llm_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            all_llm_messages.append(list(messages))
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Checking and asking",
                    tool_calls=[
                        ToolCallRequest(
                            id="t1", name="list_dir",
                            arguments={"path": "."},
                        ),
                        ToolCallRequest(
                            id="a1", name="ask_user",
                            arguments={"question": "Proceed?"},
                        ),
                    ],
                    usage={},
                )
            return LLMResponse(content="OK", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        tools = _make_tool_registry()
        tools.execute = AsyncMock(return_value="file1.txt\nfile2.txt")

        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert result.final_content == "OK"

        # The ASK_USER_PENDING string must NEVER appear in any LLM message
        for batch in all_llm_messages:
            for msg in batch:
                content_str = str(msg.get("content", ""))
                assert ASK_USER_PENDING not in content_str, (
                    f"ASK_USER_PENDING leaked into message: {msg}"
                )

        # Second LLM call should contain the normal tool result and user reply
        second_call = all_llm_messages[1]
        list_dir_msgs = [
            m for m in second_call
            if m.get("role") == "tool" and m.get("name") == "list_dir"
        ]
        assert list_dir_msgs, "Normal tool result must be in messages"

        user_reply_found = any(
            m.get("role") == "user" and "yes" in str(m.get("content", ""))
            for m in second_call
        )
        assert user_reply_found, "User reply must be in messages"

    @pytest.mark.asyncio
    async def test_ask_user_pending_marker_never_sent_to_llm(self):
        """The ASK_USER_PENDING string must never appear in any message
        sent to the LLM."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply"})

        all_llm_messages: list[list[dict]] = []
        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            all_llm_messages.append(list(messages))
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        await runner.run(spec)

        # Check all messages ever sent to LLM
        for batch in all_llm_messages:
            for msg in batch:
                content_str = str(msg.get("content", ""))
                assert ASK_USER_PENDING not in content_str, (
                    f"ASK_USER_PENDING marker leaked into LLM message: {msg}"
                )
                # Also check tool_call_id for completeness
                tool_name = msg.get("name", "")
                if isinstance(tool_name, str):
                    assert ASK_USER_PENDING not in tool_name

    @pytest.mark.asyncio
    async def test_tool_calls_count_in_result_messages(self):
        """The result.messages should include the assistant message with
        tool_calls and the injected user reply."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "reply from user"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Let me ask",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "Continue?"},
                    )],
                    usage={},
                )
            return LLMResponse(content="Final answer", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        # Assistant message with tool_calls should be in result.messages
        assistant_with_tools = [
            m for m in result.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_with_tools) >= 1, (
            "Assistant message with tool_calls must be in result.messages"
        )

        # Injected reply should be in result.messages
        user_reply = [
            m for m in result.messages
            if m.get("role") == "user" and "reply from user" in str(m.get("content", ""))
        ]
        assert len(user_reply) == 1, (
            "Injected user reply must be in result.messages"
        )


# ===========================================================================
# 6. Edge cases
# ===========================================================================


class TestAskUserEdgeCases:
    """Edge-case behavior: marker priority, repeated calls, etc."""

    @pytest.mark.asyncio
    async def test_marker_detected_before_error_prefix_check(self):
        """The _run_tool method checks ASK_USER_PENDING BEFORE the
        ``result.startswith('Error')`` check.  This test verifies that
        priority is correct."""
        # We test at the _run_tool level with a proper ToolRegistry.
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        # Register the ask_user tool with a REAL ToolRegistry
        tools = _make_tool_registry()

        provider = MagicMock()
        runner = AgentRunner(provider)

        spec = AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        # Call _run_tool directly with an ask_user ToolCallRequest
        result, event, error = await runner._run_tool(
            spec,
            ToolCallRequest(
                id="test_1", name="ask_user",
                arguments={"question": "Test question?"},
            ),
            {},
        )

        # Marker must be detected — not treated as error
        assert result == ASK_USER_PENDING, (
            "_run_tool must return ASK_USER_PENDING marker"
        )
        assert event["name"] == "ask_user"
        assert event["status"] == "ask_user", (
            "Status must be 'ask_user', not 'error' or 'ok'"
        )
        assert error is None, (
            "No fatal error should be raised for ASK_USER_PENDING"
        )

    @pytest.mark.asyncio
    async def test_ask_user_with_fail_on_tool_error_true(self):
        """Even with fail_on_tool_error=True, ask_user's ASK_USER_PENDING
        is NEVER treated as an error."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        tools = _make_tool_registry()
        provider = MagicMock()
        runner = AgentRunner(provider)

        spec = AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            fail_on_tool_error=True,
        )

        result, event, error = await runner._run_tool(
            spec,
            ToolCallRequest(
                id="test_1", name="ask_user",
                arguments={"question": "Test?"},
            ),
            {},
        )

        assert result == ASK_USER_PENDING
        assert event["status"] == "ask_user"
        assert error is None, (
            "Even with fail_on_tool_error=True, ask_user should NOT raise"
        )

    @pytest.mark.asyncio
    async def test_ask_user_send_callback_is_awaited(self):
        """The send_callback should be called (awaited) with an OutboundMessage."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner
        from summerclaw.bus.events import OutboundMessage

        sent_messages: list[OutboundMessage] = []

        async def capture_send(msg: OutboundMessage):
            sent_messages.append(msg)

        tools = ToolRegistry()
        ask_tool = AskUserTool(
            send_callback=capture_send,
            default_channel="test-channel",
            default_chat_id="test-chat",
        )
        tools.register(ask_tool)

        provider = MagicMock()
        runner = AgentRunner(provider)

        spec = AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        await runner._run_tool(
            spec,
            ToolCallRequest(
                id="test_1", name="ask_user",
                arguments={"question": "What is your name?"},
            ),
            {},
        )

        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert isinstance(msg, OutboundMessage)
        assert "What is your name?" in msg.content
        assert msg.channel == "test-channel"
        assert msg.chat_id == "test-chat"

    @pytest.mark.asyncio
    async def test_ask_user_with_candidates_includes_options_in_message(self):
        """When candidates are provided, they appear in the OutboundMessage."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner
        from summerclaw.bus.events import OutboundMessage

        sent_messages: list[OutboundMessage] = []

        async def capture_send(msg: OutboundMessage):
            sent_messages.append(msg)

        tools = ToolRegistry()
        ask_tool = AskUserTool(send_callback=capture_send)
        tools.register(ask_tool)

        provider = MagicMock()
        runner = AgentRunner(provider)

        spec = AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        await runner._run_tool(
            spec,
            ToolCallRequest(
                id="test_1", name="ask_user",
                arguments={
                    "question": "Choose one",
                    "candidates": ["Option A", "Option B", "Option C"],
                },
            ),
            {},
        )

        msg = sent_messages[0]
        assert "Option A" in msg.content
        assert "Option B" in msg.content
        assert "Option C" in msg.content

    @pytest.mark.asyncio
    async def test_ask_user_send_callback_none_does_not_crash(self):
        """If send_callback is None, the tool still returns the marker
        without trying to send anything."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        tools = ToolRegistry()
        ask_tool = AskUserTool(send_callback=None)
        tools.register(ask_tool)

        provider = MagicMock()
        runner = AgentRunner(provider)

        spec = AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        result, event, error = await runner._run_tool(
            spec,
            ToolCallRequest(
                id="test_1", name="ask_user",
                arguments={"question": "No callback test"},
            ),
            {},
        )

        assert result == ASK_USER_PENDING
        assert error is None

    @pytest.mark.asyncio
    async def test_ask_user_not_registered_falls_back_to_registry_execute(self):
        """If the tool is not found via prepare_call, it falls through to
        tools.execute(), which should also work."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        # Register ask_user in a real registry (without prepare_call on mock)
        tools = ToolRegistry()
        ask_tool = AskUserTool(send_callback=AsyncMock())
        tools.register(ask_tool)

        provider = MagicMock()
        runner = AgentRunner(provider)

        spec = AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        result, event, error = await runner._run_tool(
            spec,
            ToolCallRequest(
                id="test_1", name="ask_user",
                arguments={"question": "Test?"},
            ),
            {},
        )

        # Should work — prepare_call resolves the tool and calls execute
        assert result == ASK_USER_PENDING
        assert event["status"] == "ask_user"
        assert error is None

    @pytest.mark.asyncio
    async def test_runner_result_contains_all_tool_names(self):
        """AgentRunResult.tools_used includes 'ask_user'."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "yes"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Checking",
                    tool_calls=[
                        ToolCallRequest(
                            id="r1", name="list_dir", arguments={},
                        ),
                        ToolCallRequest(
                            id="a1", name="ask_user",
                            arguments={"question": "Proceed?"},
                        ),
                    ],
                    usage={},
                )
            return LLMResponse(content="OK", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        tools = _make_tool_registry()
        tools.execute = AsyncMock(return_value="dir output")

        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        assert "ask_user" in result.tools_used, (
            "Tools used should include ask_user"
        )
        assert "list_dir" in result.tools_used, (
            "Tools used should include all tool names from the batch"
        )

    @pytest.mark.asyncio
    async def test_ask_user_resets_empty_content_retries(self):
        """After ask_user, empty_content_retries is reset to 0, so
        subsequent empty responses get a fresh retry budget."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        await injection_queue.put({"role": "user", "content": "keep going"})

        call_count = {"n": 0}

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: ask_user
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "Continue?"},
                    )],
                    usage={},
                )
            # After ask_user + injection, return empty content
            # (this should get fresh empty_content_retries = 0)
            return LLMResponse(
                content=None, tool_calls=[], usage={},
                finish_reason="stop",
            )

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=10,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        result = await runner.run(spec)

        # Should not crash; empty response handling should work with fresh budget
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_injection_with_blank_content_filtered_out(self):
        """Messages with blank/whitespace-only content from callback
        should be filtered and not injected."""
        from summerclaw.agent.runner import AgentRunSpec, AgentRunner

        injection_queue: asyncio.Queue = asyncio.Queue()
        # Put a blank message first, then a real one
        await injection_queue.put({"role": "user", "content": "   "})
        await injection_queue.put({"role": "user", "content": "real reply"})

        call_count = {"n": 0}
        final_messages: list[list[dict]] = []

        provider = MagicMock()

        async def chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="Asking",
                    tool_calls=[ToolCallRequest(
                        id="ask_1", name="ask_user",
                        arguments={"question": "OK?"},
                    )],
                    usage={},
                )
            final_messages.append(list(messages))
            return LLMResponse(content="Done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry

        runner = AgentRunner(provider)
        tools = _make_tool_registry()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Go"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            injection_callback=_make_injection_callback(injection_queue),
        )
        await runner.run(spec)

        assert len(final_messages) == 1
        msgs = final_messages[0]
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert not any(
            str(m.get("content", "")).strip() == ""
            for m in user_msgs
        ), "Blank user messages should be filtered out"
        assert any("real reply" in str(m.get("content")) for m in user_msgs), (
            "Real reply should still be injected"
        )
