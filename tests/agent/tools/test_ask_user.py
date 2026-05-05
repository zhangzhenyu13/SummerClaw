"""Comprehensive tests for AskUserTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.ask_user import ASK_USER_PENDING, AskUserTool
from nanobot.bus.events import OutboundMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(**overrides):
    defaults = {"send_callback": None, "default_channel": "", "default_chat_id": ""}
    defaults.update(overrides)
    return AskUserTool(**defaults)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestAskUserMetadata:

    def test_name_is_ask_user(self):
        tool = _make_tool()
        assert tool.name == "ask_user"

    def test_description_mentions_pause_and_reply(self):
        tool = _make_tool()
        desc = tool.description
        assert "Pause execution" in desc or "pause" in desc.lower()
        assert "user" in desc.lower()
        assert "reply" in desc.lower() or "answer" in desc.lower()

    def test_description_mentions_candidates(self):
        tool = _make_tool()
        assert "candidates" in tool.description

    def test_exclusive_is_true(self):
        tool = _make_tool()
        assert tool.exclusive is True

    def test_concurrency_safe_is_false(self):
        tool = _make_tool()
        assert tool.concurrency_safe is False

    def test_read_only_is_false(self):
        tool = _make_tool()
        assert tool.read_only is False


# ---------------------------------------------------------------------------
# Parameters schema
# ---------------------------------------------------------------------------

class TestAskUserParameters:

    def test_parameters_is_object_type(self):
        tool = _make_tool()
        params = tool.parameters
        assert params["type"] == "object"

    def test_question_is_required(self):
        tool = _make_tool()
        required = tool.parameters.get("required", [])
        assert "question" in required

    def test_question_property_is_string_type(self):
        tool = _make_tool()
        props = tool.parameters["properties"]
        assert props["question"]["type"] == "string"

    def test_candidates_is_array_of_strings(self):
        tool = _make_tool()
        props = tool.parameters["properties"]
        assert props["candidates"]["type"] == "array"
        assert props["candidates"]["items"]["type"] == "string"

    def test_timeout_is_integer_with_bounds(self):
        tool = _make_tool()
        props = tool.parameters["properties"]
        tp = props["timeout"]
        assert tp["type"] == "integer"
        assert tp["minimum"] == 10
        assert tp["maximum"] == 600

    def test_candidates_not_required(self):
        tool = _make_tool()
        required = tool.parameters.get("required", [])
        assert "candidates" not in required

    def test_timeout_not_required(self):
        tool = _make_tool()
        required = tool.parameters.get("required", [])
        assert "timeout" not in required

    def test_no_extra_required_fields(self):
        tool = _make_tool()
        required = tool.parameters.get("required", [])
        assert required == ["question"]

    def test_no_extra_properties(self):
        tool = _make_tool()
        props = tool.parameters["properties"]
        assert set(props.keys()) == {"question", "candidates", "timeout"}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestAskUserValidation:

    def test_valid_minimal_params_passes(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "What now?"})
        assert errors == []

    def test_valid_full_params_passes(self):
        tool = _make_tool()
        errors = tool.validate_params({
            "question": "Pick one",
            "candidates": ["A", "B", "C"],
            "timeout": 120,
        })
        assert errors == []

    def test_missing_question_fails(self):
        tool = _make_tool()
        errors = tool.validate_params({})
        assert any("required" in e or "question" in e for e in errors)

    def test_question_wrong_type_fails(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": 123})
        assert any("string" in e for e in errors)

    def test_timeout_below_minimum_fails(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "timeout": 5})
        assert any(">=" in e or "minimum" in e for e in errors)

    def test_timeout_above_maximum_fails(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "timeout": 999})
        assert any("<=" in e or "maximum" in e for e in errors)

    def test_timeout_at_minimum_passes(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "timeout": 10})
        assert errors == []

    def test_timeout_at_maximum_passes(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "timeout": 600})
        assert errors == []

    def test_candidates_wrong_type_fails(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "candidates": "not_a_list"})
        assert len(errors) > 0

    def test_candidates_elements_wrong_type_fails(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "candidates": [1, 2, 3]})
        assert len(errors) > 0

    def test_empty_candidates_list_passes(self):
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "candidates": []})
        assert errors == []

    def test_unknown_extra_param_passes(self):
        """Extra params should be tolerated (typical OpenAI behavior)."""
        tool = _make_tool()
        errors = tool.validate_params({"question": "Hi", "extra_field": "ignored"})
        assert errors == []

    def test_type_coercion_string_to_int_for_timeout(self):
        tool = _make_tool()
        coerced = tool.cast_params({"question": "Hi", "timeout": "300"})
        assert coerced["timeout"] == 300


# ---------------------------------------------------------------------------
# set_context
# ---------------------------------------------------------------------------

class TestAskUserSetContext:

    def test_set_context_stores_channel(self):
        tool = _make_tool()
        tool.set_context("telegram", "chat_99")
        assert tool._default_channel == "telegram"

    def test_set_context_stores_chat_id(self):
        tool = _make_tool()
        tool.set_context("discord", "channel_42")
        assert tool._default_chat_id == "channel_42"

    def test_set_context_overwrites_previous(self):
        tool = _make_tool(default_channel="old_chan", default_chat_id="old_chat")
        tool.set_context("new_chan", "new_chat")
        assert tool._default_channel == "new_chan"
        assert tool._default_chat_id == "new_chat"


# ---------------------------------------------------------------------------
# execute — core behavior
# ---------------------------------------------------------------------------

class TestAskUserExecute:

    @pytest.mark.asyncio
    async def test_execute_returns_ask_user_pending(self):
        tool = _make_tool()
        result = await tool.execute(question="Continue?")
        assert result == ASK_USER_PENDING

    @pytest.mark.asyncio
    async def test_execute_returns_marker_not_string(self):
        tool = _make_tool()
        result = await tool.execute(question="???")
        assert result == "__ASK_USER_PENDING__"

    @pytest.mark.asyncio
    async def test_execute_without_callback_still_returns_marker(self):
        tool = _make_tool(send_callback=None)
        result = await tool.execute(question="OK?")
        assert result == ASK_USER_PENDING

    @pytest.mark.asyncio
    async def test_execute_with_candidates_returns_marker(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        result = await tool.execute(question="Choose", candidates=["X", "Y"])
        assert result == ASK_USER_PENDING

    @pytest.mark.asyncio
    async def test_execute_with_timeout_returns_marker(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        result = await tool.execute(question="Wait", timeout=60)
        assert result == ASK_USER_PENDING


# ---------------------------------------------------------------------------
# execute — message delivery via callback
# ---------------------------------------------------------------------------

class TestAskUserMessageDelivery:

    @pytest.mark.asyncio
    async def test_sends_outbound_message_with_question(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        await tool.execute(question="Are you sure?")
        cb.assert_awaited_once()
        msg = cb.call_args[0][0]
        assert isinstance(msg, OutboundMessage)
        assert "Are you sure?" in msg.content

    @pytest.mark.asyncio
    async def test_sends_to_default_channel(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb, default_channel="slack", default_chat_id="C123")
        await tool.execute(question="Hello")
        msg = cb.call_args[0][0]
        assert msg.channel == "slack"
        assert msg.chat_id == "C123"

    @pytest.mark.asyncio
    async def test_sends_with_set_context_routing(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        tool.set_context("matrix", "room_abc")
        await tool.execute(question="Hi")
        msg = cb.call_args[0][0]
        assert msg.channel == "matrix"
        assert msg.chat_id == "room_abc"

    @pytest.mark.asyncio
    async def test_includes_candidates_in_message(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        await tool.execute(question="Pick:", candidates=["Apples", "Oranges"])
        msg = cb.call_args[0][0]
        assert "Oranges" in msg.content

    @pytest.mark.asyncio
    async def test_candidates_formatted_as_options(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        await tool.execute(question="Choose:", candidates=["A", "B", "C"])
        msg = cb.call_args[0][0]
        assert "Options:" in msg.content
        assert "• A" in msg.content
        assert "• B" in msg.content
        assert "• C" in msg.content

    @pytest.mark.asyncio
    async def test_no_candidates_no_options_section(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        await tool.execute(question="Just a question")
        msg = cb.call_args[0][0]
        assert "Options:" not in msg.content

    @pytest.mark.asyncio
    async def test_no_media_attached(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        await tool.execute(question="?")
        msg = cb.call_args[0][0]
        assert msg.media == []

    @pytest.mark.asyncio
    async def test_does_not_call_callback_when_none(self):
        tool = _make_tool(send_callback=None)
        result = await tool.execute(question="Silent?")
        assert result == ASK_USER_PENDING

    @pytest.mark.asyncio
    async def test_callback_exception_is_not_caught(self):
        cb = AsyncMock(side_effect=RuntimeError("channel down"))
        tool = _make_tool(send_callback=cb)
        with pytest.raises(RuntimeError, match="channel down"):
            await tool.execute(question="Boom")

    @pytest.mark.asyncio
    async def test_none_candidates_treated_as_empty(self):
        cb = AsyncMock()
        tool = _make_tool(send_callback=cb)
        result = await tool.execute(question="OK?", candidates=None)
        assert result == ASK_USER_PENDING
        cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# execute — integration contract (runner expectations)
# ---------------------------------------------------------------------------

class TestAskUserRunnerContract:

    def test_marker_is_hashable_for_set_membership(self):
        """Runner uses 'r in results' checks; marker must be hashable."""
        r = {ASK_USER_PENDING, "normal_result"}
        assert ASK_USER_PENDING in r

    def test_marker_is_a_string(self):
        """Runner's isinstance(result, str) checks must not reject it."""
        assert isinstance(ASK_USER_PENDING, str)

    def test_marker_distinct_from_real_result(self):
        """Marker must not collide with any real tool output."""
        assert ASK_USER_PENDING != "normal output"
        assert ASK_USER_PENDING != "Error: something"
        assert ASK_USER_PENDING != ""

    def test_marker_does_not_start_with_error(self):
        """Runner checks result.startswith('Error'); marker must not match."""
        assert not ASK_USER_PENDING.startswith("Error")


# ---------------------------------------------------------------------------
# to_schema (OpenAI function schema)
# ---------------------------------------------------------------------------

class TestAskUserToSchema:

    def test_to_schema_has_correct_type(self):
        tool = _make_tool()
        schema = tool.to_schema()
        assert schema["type"] == "function"

    def test_to_schema_includes_name(self):
        tool = _make_tool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "ask_user"

    def test_to_schema_includes_description(self):
        tool = _make_tool()
        schema = tool.to_schema()
        assert len(schema["function"]["description"]) > 10

    def test_to_schema_includes_parameters(self):
        tool = _make_tool()
        schema = tool.to_schema()
        assert schema["function"]["parameters"]["type"] == "object"
