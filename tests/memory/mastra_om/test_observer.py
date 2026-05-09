"""Tests for MastraOM Observer — message formatting, XML parsing, etc."""

import pytest

from nanobot.memory.mastra_om_memory.observer import (
    build_observer_system_prompt,
    build_observer_prompt,
    build_observer_task_prompt,
    parse_observer_output,
    format_messages_for_observer,
    detect_degenerate_repetition,
    sanitize_observation_lines,
    optimize_observations_for_context,
    has_current_task_section,
    extract_current_task,
    OBSERVER_SYSTEM_PROMPT,
    OBSERVATION_CONTINUATION_HINT,
    OBSERVATION_CONTEXT_PROMPT,
    OBSERVATION_CONTEXT_INSTRUCTIONS,
)


class TestObserverSystemPrompt:
    """Tests for Observer system prompt generation."""

    def test_default_prompt_contains_key_sections(self):
        prompt = build_observer_system_prompt()
        assert "memory consciousness" in prompt
        assert "observations" in prompt.lower()
        assert "OUTPUT FORMAT" in prompt
        assert "GUIDELINES" in prompt
        assert "CRITICAL: DISTINGUISH" in prompt

    def test_prompt_excludes_multithread_by_default(self):
        prompt = build_observer_system_prompt()
        assert "MULTI-THREAD" not in prompt
        assert "<thread id=" not in prompt

    def test_prompt_includes_multithread_when_enabled(self):
        prompt = build_observer_system_prompt(multi_thread=True)
        assert "MULTI-THREAD" in prompt
        assert "<thread id=" in prompt

    def test_prompt_includes_custom_instruction(self):
        prompt = build_observer_system_prompt(instruction="Custom behavior rule")
        assert "Custom behavior rule" in prompt

    def test_prompt_includes_thread_title_when_enabled(self):
        prompt = build_observer_system_prompt(include_thread_title=True)
        assert "<thread-title>" in prompt

    def test_default_constant_matches_built(self):
        assert "memory consciousness" in OBSERVER_SYSTEM_PROMPT


class TestObserverMessageFormatting:
    """Tests for format_messages_for_observer."""

    def test_empty_messages(self):
        result = format_messages_for_observer([])
        assert result == "(no messages)"

    def test_basic_formatting(self):
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2025-05-09 10:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2025-05-09 10:01"},
        ]
        result = format_messages_for_observer(messages)
        assert "User" in result
        assert "Hello" in result
        assert "Assistant" in result
        assert "Hi there!" in result

    def test_date_grouping(self):
        messages = [
            {"role": "user", "content": "Morning", "timestamp": "2025-05-09 09:00"},
            {"role": "user", "content": "Evening", "timestamp": "2025-05-09 21:00"},
        ]
        result = format_messages_for_observer(messages)
        # Both same date, no second date header
        assert result.count("2025-05-09") <= 2  # date header + timestamp

    def test_tool_call_formatting(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [{"function": {"name": "read_file"}}],
                "timestamp": "2025-05-09 10:00",
            },
        ]
        result = format_messages_for_observer(messages)
        assert "Tool Call" in result
        assert "read_file" in result

    def test_tool_result_formatting(self):
        messages = [
            {
                "role": "tool",
                "content": "File contents here",
                "name": "read_file",
                "timestamp": "2025-05-09 10:01",
            },
        ]
        result = format_messages_for_observer(messages)
        assert "Tool Result" in result
        assert "read_file" in result

    def test_max_part_length_truncation(self):
        long_content = "x" * 500
        messages = [{"role": "user", "content": long_content, "timestamp": "2025-05-09 10:00"}]
        result = format_messages_for_observer(messages, max_part_length=100)
        assert "[truncated" in result
        assert len(long_content) > len(result.split(": ")[-1])

    def test_skip_empty_content(self):
        messages = [
            {"role": "user", "content": "", "timestamp": "2025-05-09 10:00"},
            {"role": "assistant", "content": "Valid", "timestamp": "2025-05-09 10:01"},
        ]
        result = format_messages_for_observer(messages)
        assert "Valid" in result
        # Empty content should not appear
        assert result.count("User") == 0


class TestObserverTaskPrompt:
    """Tests for build_observer_task_prompt."""

    def test_no_existing_observations(self):
        prompt = build_observer_task_prompt(existing_observations=None)
        assert "Your Task" in prompt
        assert "Previous Observations" not in prompt

    def test_with_existing_observations(self):
        prompt = build_observer_task_prompt(existing_observations="Old observation")
        assert "Previous Observations" in prompt
        assert "Old observation" in prompt
        assert "Do not repeat" in prompt

    def test_with_prior_metadata(self):
        prompt = build_observer_task_prompt(
            existing_observations=None,
            prior_current_task="Working on feature X",
            prior_suggested_response="Continue",
        )
        assert "Prior Thread Metadata" in prompt
        assert "Working on feature X" in prompt
        assert "Continue" in prompt

    def test_with_truncated_flag(self):
        prompt = build_observer_task_prompt(
            existing_observations="Old",
            prior_current_task="Working on feature X",
            was_truncated=True,
        )
        assert "truncated" in prompt.lower()


class TestObserverFullPrompt:
    """Tests for build_observer_prompt."""

    def test_full_prompt_structure(self):
        messages = [{"role": "user", "content": "Test", "timestamp": "2025-05-09 10:00"}]
        prompt = build_observer_prompt(
            existing_observations=None,
            messages_to_observe=messages,
        )
        assert "New Message History to Observe" in prompt
        assert "Test" in prompt
        assert "Your Task" in prompt

    def test_full_prompt_with_existing(self):
        messages = [{"role": "user", "content": "New info", "timestamp": "2025-05-09 10:00"}]
        prompt = build_observer_prompt(
            existing_observations="Old observation",
            messages_to_observe=messages,
        )
        assert "Old observation" in prompt
        assert "New info" in prompt


class TestObserverOutputParsing:
    """Tests for parse_observer_output."""

    def test_parse_empty_output(self):
        result = parse_observer_output("")
        assert result["observations"] == ""
        assert result["current_task"] is None
        assert result["degenerate"] is False

    def test_parse_valid_xml(self):
        output = """<observations>
Date: May 9, 2025
* 🔴 User prefers dark mode
* 🟡 User might want notifications
</observations>

<current-task>
Working on settings page
</current-task>

<suggested-response>
Continue with dark mode toggle
</suggested-response>"""
        result = parse_observer_output(output)
        assert "dark mode" in result["observations"]
        assert result["current_task"] == "Working on settings page"
        assert "dark mode toggle" in result["suggested_continuation"]
        assert result["degenerate"] is False

    def test_parse_xml_with_thread_title(self):
        output = """<observations>
Date: May 9
* 🔴 User likes Python
</observations>
<thread-title>Python setup</thread-title>"""
        result = parse_observer_output(output)
        assert result["thread_title"] == "Python setup"

    def test_parse_without_xml_tags(self):
        """Fallback: extract list items when no XML tags."""
        output = """Here are my observations:
* First observation
* Second observation
And some extra text."""
        result = parse_observer_output(output)
        assert "First observation" in result["observations"]
        assert "Second observation" in result["observations"]
        assert "extra text" not in result["observations"]

    def test_parse_multiple_observations_blocks(self):
        output = """<observations>
Date: May 9
* 🔴 First
</observations>

<observations>
Date: May 10
* 🟡 Second
</observations>"""
        result = parse_observer_output(output)
        assert "First" in result["observations"]
        assert "Second" in result["observations"]


class TestObserverDegenerateDetection:
    """Tests for detect_degenerate_repetition."""

    def test_small_text_not_degenerate(self):
        assert detect_degenerate_repetition("short") is False

    def test_empty_text(self):
        assert detect_degenerate_repetition("") is False
        assert detect_degenerate_repetition("") is False

    def test_normal_text_not_degenerate(self):
        # Build varied text by concatenating different phrases
        parts = [
            "The weather is nice today and everyone is outside.",
            "I wonder what we should have for dinner tonight.",
            "The project deadline has been moved to next Friday.",
            "She decided to learn a new programming language.",
            "The meeting was postponed due to scheduling conflicts.",
            "He bought a new laptop with better specifications.",
            "They visited the museum and saw ancient artifacts.",
            "Reading books is one of the best ways to learn.",
            "The cat jumped onto the table and knocked over a vase.",
            "We need to refactor this module for better performance.",
        ] * 20
        text = "\n".join(parts)
        assert detect_degenerate_repetition(text) is False

    def test_repetitive_text_detected(self):
        # Create text with highly repetitive windows
        block = "x" * 200
        text = block * 100
        assert detect_degenerate_repetition(text) is True

    def test_very_long_line_detected(self):
        text = "a" * 60000 + "\n"
        assert detect_degenerate_repetition(text) is True

    def test_varied_text_below_threshold(self):
        # Text just at the boundary
        text = "a" * 1999
        assert detect_degenerate_repetition(text) is False


class TestObserverSanitization:
    """Tests for sanitize_observation_lines."""

    def test_sanitize_normal_lines(self):
        obs = "Line 1\nLine 2\nLine 3"
        result = sanitize_observation_lines(obs)
        assert result == obs

    def test_sanitize_long_line(self):
        obs = "Normal line\n" + ("x" * 15000) + "\nNormal line"
        result = sanitize_observation_lines(obs)
        assert "[truncated]" in result
        assert "Normal line" in result
        assert len(result.split("\n")[1]) < 12000

    def test_sanitize_empty(self):
        assert sanitize_observation_lines("") == ""


class TestObserverOptimization:
    """Tests for optimize_observations_for_context."""

    def test_optimize_empty(self):
        assert optimize_observations_for_context("") == ""

    def test_remove_medium_low_priority_emojis(self):
        obs = "* 🟡 Medium priority\n* 🟢 Low priority\n* 🔴 High priority"
        result = optimize_observations_for_context(obs)
        assert "🟡" not in result
        assert "🟢" not in result
        assert "🔴" in result

    def test_remove_arrows(self):
        obs = "* 🔴 Agent -> did something -> found result"
        result = optimize_observations_for_context(obs)
        assert "->" not in result

    def test_cleanup_whitespace(self):
        obs = "Line 1\n\n\n\nLine 2"
        result = optimize_observations_for_context(obs)
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestObserverCurrentTask:
    """Tests for current task extraction."""

    def test_has_current_task_section_xml(self):
        assert has_current_task_section("<current-task>Something</current-task>") is True

    def test_has_current_task_section_markdown(self):
        assert has_current_task_section("**Current Task:** Something") is True
        assert has_current_task_section("Current Task: Something") is True

    def test_no_current_task(self):
        assert has_current_task_section("Just some observations") is False

    def test_extract_current_task(self):
        obs = "<observations>\n*\n</observations>\n<current-task>Working on X</current-task>"
        result = extract_current_task(obs)
        assert result == "Working on X"

    def test_extract_no_current_task(self):
        assert extract_current_task("Just observations") is None


class TestObserverContinuationHints:
    """Tests for continuation hint constants."""

    def test_continuation_hint_not_empty(self):
        assert len(OBSERVATION_CONTINUATION_HINT) > 50
        assert "conversation" in OBSERVATION_CONTINUATION_HINT.lower()

    def test_context_prompt_not_empty(self):
        assert len(OBSERVATION_CONTEXT_PROMPT) > 10
        assert "observations" in OBSERVATION_CONTEXT_PROMPT.lower()

    def test_context_instructions_not_empty(self):
        assert len(OBSERVATION_CONTEXT_INSTRUCTIONS) > 50
        assert "IMPORTANT" in OBSERVATION_CONTEXT_INSTRUCTIONS
