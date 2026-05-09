"""Tests for MastraOM Reflector — observation condensation, compression levels."""

import pytest

from nanobot.memory.mastra_om_memory.reflector import (
    build_reflector_system_prompt,
    build_reflector_prompt,
    parse_reflector_output,
    validate_compression,
    COMPRESSION_GUIDANCE,
    REFLECTOR_SYSTEM_PROMPT,
)


class TestReflectorSystemPrompt:
    """Tests for Reflector system prompt generation."""

    def test_default_prompt_contains_key_sections(self):
        prompt = build_reflector_system_prompt()
        assert "memory consciousness" in prompt
        assert "observation reflector" in prompt.lower()
        assert "OUTPUT FORMAT" in prompt
        assert "CRITICAL: USER ASSERTIONS" in prompt

    def test_prompt_includes_observer_instructions(self):
        prompt = build_reflector_system_prompt()
        assert "observational-memory-instruction" in prompt
        assert "DISTINGUISH USER ASSERTIONS" in prompt

    def test_prompt_includes_custom_instruction(self):
        prompt = build_reflector_system_prompt(instruction="Focus on security issues")
        assert "Focus on security issues" in prompt

    def test_default_constant_matches_built(self):
        assert "memory consciousness" in REFLECTOR_SYSTEM_PROMPT


class TestCompressionGuidance:
    """Tests for compression levels."""

    def test_level_0_is_empty(self):
        assert COMPRESSION_GUIDANCE[0] == ""

    def test_level_1_contains_compression_keyword(self):
        assert "COMPRESSION REQUIRED" in COMPRESSION_GUIDANCE[1]
        assert "8/10" in COMPRESSION_GUIDANCE[1]

    def test_level_2_contains_aggressive(self):
        assert "AGGRESSIVE" in COMPRESSION_GUIDANCE[2]
        assert "6/10" in COMPRESSION_GUIDANCE[2]

    def test_level_3_contains_critical(self):
        assert "CRITICAL" in COMPRESSION_GUIDANCE[3]
        assert "4/10" in COMPRESSION_GUIDANCE[3]

    def test_level_4_contains_extreme(self):
        assert "EXTREME" in COMPRESSION_GUIDANCE[4]
        assert "2/10" in COMPRESSION_GUIDANCE[4]

    def test_all_levels_have_completion_marker_guidance(self):
        for level in range(1, 5):
            assert "✅" in COMPRESSION_GUIDANCE[level], f"Level {level} missing ✅ guidance"


class TestReflectorPrompt:
    """Tests for build_reflector_prompt."""

    def test_basic_prompt(self):
        prompt = build_reflector_prompt(observations="Some observations")
        assert "OBSERVATIONS TO REFLECT ON" in prompt
        assert "Some observations" in prompt
        assert "produce a refined, condensed version" in prompt

    def test_prompt_with_manual_guidance(self):
        prompt = build_reflector_prompt(
            observations="Obs",
            manual_prompt="Keep only security-related",
        )
        assert "SPECIFIC GUIDANCE" in prompt
        assert "Keep only security-related" in prompt

    def test_prompt_with_compression_level(self):
        prompt = build_reflector_prompt(
            observations="Obs",
            compression_level=2,
        )
        assert "AGGRESSIVE" in prompt

    def test_prompt_with_skip_continuation(self):
        prompt = build_reflector_prompt(
            observations="Obs",
            skip_continuation_hints=True,
        )
        assert "Do NOT include" in prompt
        assert "current-task" in prompt

    def test_prompt_level_0_no_guidance(self):
        prompt = build_reflector_prompt(
            observations="Obs",
            compression_level=0,
        )
        assert "COMPRESSION" not in prompt


class TestReflectorOutputParsing:
    """Tests for parse_reflector_output."""

    def test_parse_empty_output(self):
        result = parse_reflector_output("")
        assert result["observations"] == ""
        assert result["degenerate"] is False

    def test_parse_valid_xml(self):
        output = """<observations>
Date: May 9, 2025
* 🔴 User prefers TypeScript
* ✅ Auth feature completed
</observations>

<suggested-response>
Continue with deployment pipeline
</suggested-response>"""
        result = parse_reflector_output(output)
        assert "TypeScript" in result["observations"]
        assert "Auth feature completed" in result["observations"]
        assert result["suggested_continuation"] == "Continue with deployment pipeline"
        assert result["degenerate"] is False

    def test_parse_degenerate_output(self):
        # Create repetitive output that triggers degenerate detection
        block = "x" * 200
        output = block * 100
        result = parse_reflector_output(output)
        assert result["degenerate"] is True

    def test_parse_without_xml_tags(self):
        """Fallback: extract list items or use full content."""
        output = """Here's the reflection:
* Consolidated item 1
* Consolidated item 2"""
        result = parse_reflector_output(output)
        assert "Consolidated item 1" in result["observations"]
        assert "Consolidated item 2" in result["observations"]

    def test_parse_with_source_observations(self):
        output = """<observations>
Date: May 9
* 🔴 Condensed observation
</observations>"""
        result = parse_reflector_output(output, source_observations="Original observations")
        assert "Condensed observation" in result["observations"]


class TestCompressionValidation:
    """Tests for validate_compression."""

    def test_compression_successful(self):
        assert validate_compression(reflected_tokens=100, target_threshold=1000) is True

    def test_compression_at_threshold(self):
        # Must be strictly less than threshold
        assert validate_compression(reflected_tokens=1000, target_threshold=1000) is False

    def test_compression_failed(self):
        assert validate_compression(reflected_tokens=2000, target_threshold=1000) is False

    def test_compression_with_small_numbers(self):
        assert validate_compression(reflected_tokens=0, target_threshold=1) is True
        assert validate_compression(reflected_tokens=1, target_threshold=1) is False
