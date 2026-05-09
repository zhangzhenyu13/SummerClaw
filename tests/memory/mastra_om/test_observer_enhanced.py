"""Tests for enhanced MastraOM Observer system prompt — full official coverage.

Verifies that the enhanced OBSERVER_EXTRACTION_INSTRUCTIONS, OBSERVER_OUTPUT_FORMAT,
OBSERVER_GUIDELINES, and OBSERVATION_CONTEXT_INSTRUCTIONS contain all the detailed
sections from the official Mastra TypeScript codebase.
"""

import pytest

from nanobot.memory.mastra_om_memory.observer import (
    build_observer_system_prompt,
    OBSERVER_EXTRACTION_INSTRUCTIONS,
    OBSERVER_OUTPUT_FORMAT,
    OBSERVER_GUIDELINES,
    OBSERVATION_CONTEXT_INSTRUCTIONS,
    OBSERVATION_CONTINUATION_HINT,
    OBSERVATION_CONTEXT_PROMPT,
)


# ── Extraction Instructions Coverage ──────────────────────────────────────

class TestEnhancedExtractionInstructions:
    """Verify OBSERVER_EXTRACTION_INSTRUCTIONS contains all official sections."""

    def test_distinguishes_assertions_from_questions(self):
        assert "CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS" in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'User stated has two kids' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'User asked help with X' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_distinguishes_questions_from_intent_statements(self):
        """Official code adds: distinguish QUESTIONS from STATEMENTS OF INTENT."""
        assert 'Distinguish between QUESTIONS and STATEMENTS OF INTENT' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Statement of intent' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"I\'m looking forward to [doing X]"' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_state_changes_supersede_previous(self):
        """Official code: state changes should explicitly replace old info."""
        assert 'supersedes previous information' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '(changing from Y)' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '(replacing the old approach)' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '(no longer at previous location)' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_user_assertions_authoritative(self):
        assert 'USER ASSERTIONS ARE AUTHORITATIVE' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'source of truth about their own life' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_temporal_anchoring_detailed(self):
        """Official code: detailed temporal anchoring with GOOD/BAD examples."""
        assert 'BEGINNING: The time the statement was made' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'END: The time being REFERENCED' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"last week", "yesterday"' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"this weekend", "tomorrow"' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'DO NOT add end dates for' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Present-moment statements with no time reference' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_temporal_format_with_good_bad_examples(self):
        assert 'FORMAT:' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'GOOD: (09:15) User\'s friend had a birthday party in March' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'BAD: (09:15) User prefers hiking in the mountains' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_multiple_events_split_into_separate_lines(self):
        assert 'split them into SEPARATE observation lines' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'EACH split observation MUST have its own date' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_preserve_unusual_phrasing(self):
        """Official code: PRESERVE UNUSUAL PHRASING section."""
        assert 'PRESERVE UNUSUAL PHRASING' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'quote their exact words' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"movement session"' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_precise_action_verbs(self):
        """Official code: USE PRECISE ACTION VERBS section."""
        assert 'USE PRECISE ACTION VERBS' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'subscribed to' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'purchased' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"getting" something regularly' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"stopped getting" → "canceled"' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_preserving_details_six_categories(self):
        """Official code: 6 sub-categories for preserving details."""
        instructions = OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '1. RECOMMENDATION LISTS' in instructions
        assert '2. NAMES, HANDLES, AND IDENTIFIERS' in instructions
        assert '3. CREATIVE CONTENT' in instructions
        assert '4. TECHNICAL/NUMERICAL RESULTS' in instructions
        assert '5. QUANTITIES AND COUNTS' in instructions
        assert '6. ROLE/PARTICIPATION STATEMENTS' in instructions

    def test_recommendation_lists_examples(self):
        assert 'Hotel A (near the train station)' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Store A (based in Germany, ships worldwide)' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_names_handles_examples(self):
        assert '@photographer_one (portraits)' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Jane Smith (mystery novels)' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_technical_results_examples(self):
        assert '43.7% faster load times' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '7,342 samples, 89.6% accuracy' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_role_statements_examples(self):
        assert 'User was a presenter at the company event' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'User volunteered at the fundraiser' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'presenter, organizer, volunteer, team lead' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_conversation_context_extended(self):
        """Official code adds: what user understands, code snippets, iterative collaboration, entities."""
        assert 'What user understands or needs clarification on' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Relevant code snippets' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'preserve these verbatim in memory' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'iteratively collaborating back and forth' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'attributes that would help identify or describe the specific entity' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_user_message_capture_extended(self):
        assert 'captured nearly verbatim in your own words' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'the observations are the only record of what the user said' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_avoiding_repetitive_with_grouping_examples(self):
        """Official code: explicit BAD/GOOD examples for grouping."""
        assert 'Example — BAD (repetitive)' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Example — GOOD (grouped)' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Agent browsed source files for auth flow' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_actionable_insights_section(self):
        """Official code: ACTIONABLE INSIGHTS section."""
        assert 'ACTIONABLE INSIGHTS' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'What worked well in explanations' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'What needs follow-up or clarification' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"waiting for user"' in OBSERVER_EXTRACTION_INSTRUCTIONS

    def test_completion_tracking_detailed(self):
        """Official code: detailed ✅ rules with WHEN/DO NOT/FORMAT."""
        assert 'COMPLETION TRACKING' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'explicit memory signals' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Use ✅ to answer: "What exactly is now done?"' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Use ✅ when:' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'Do NOT use ✅ when:' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"thanks, that fixed it"' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert '"I\'ll try that later"' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'FORMAT:' in OBSERVER_EXTRACTION_INSTRUCTIONS
        assert 'As a sub-bullet under the related observation group' in OBSERVER_EXTRACTION_INSTRUCTIONS


# ── Output Format Coverage ─────────────────────────────────────────────────

class TestEnhancedOutputFormat:
    """Verify OBSERVER_OUTPUT_FORMAT contains official enhancements."""

    def test_completion_emoji_enhanced_description(self):
        assert 'goal achieved, or subtask completed' in OBSERVER_OUTPUT_FORMAT

    def test_current_task_off_task_detection(self):
        assert 'If the agent started doing something without user approval' in OBSERVER_OUTPUT_FORMAT
        assert "it's off-task" in OBSERVER_OUTPUT_FORMAT or "off-task" in OBSERVER_OUTPUT_FORMAT

    def test_suggested_response_has_examples(self):
        assert 'I\'ve updated the navigation model' in OBSERVER_OUTPUT_FORMAT
        assert 'The assistant should wait for the user to respond' in OBSERVER_OUTPUT_FORMAT
        assert 'Call the view tool on src/example.ts' in OBSERVER_OUTPUT_FORMAT


# ── Guidelines Coverage ────────────────────────────────────────────────────

class TestEnhancedGuidelines:
    """Verify OBSERVER_GUIDELINES contains official enhancements."""

    def test_terse_language_extended(self):
        assert 'Sentences should be dense without unnecessary words' in OBSERVER_GUIDELINES

    def test_grouping_guidance(self):
        assert 'Group repeated similar actions' in OBSERVER_GUIDELINES

    def test_file_line_numbers(self):
        assert 'When observing files with line numbers' in OBSERVER_GUIDELINES

    def test_detailed_response_preservation(self):
        assert 'observe the contents so it could be repeated' in OBSERVER_GUIDELINES

    def test_completion_marker_guidance(self):
        assert 'Treat ✅ as a memory signal' in OBSERVER_GUIDELINES
        assert 'Make completion observations answer "What exactly is now done?"' in OBSERVER_GUIDELINES
        assert 'Prefer concrete resolved outcomes over meta-level workflow' in OBSERVER_GUIDELINES

    def test_multiple_completions_guidance(self):
        assert 'capture the concrete completed work rather than collapsing' in OBSERVER_GUIDELINES

    def test_preserve_code_snippets(self):
        assert 'If the user provides detailed messages or code snippets' in OBSERVER_GUIDELINES


# ── Context Instructions Coverage ──────────────────────────────────────────

class TestEnhancedContextInstructions:
    """Verify OBSERVATION_CONTEXT_INSTRUCTIONS contains official enhancements."""

    def test_personalize_with_past_experiences(self):
        assert 'connect them to their past experiences mentioned above' in OBSERVATION_CONTEXT_INSTRUCTIONS

    def test_knowledge_updates_with_state_change_indicators(self):
        assert 'Look for phrases like "will start", "is switching"' in OBSERVATION_CONTEXT_INSTRUCTIONS or \
               'will start' in OBSERVATION_CONTEXT_INSTRUCTIONS
        assert 'the newer observation supersedes the older one' in OBSERVATION_CONTEXT_INSTRUCTIONS

    def test_planned_actions_inference(self):
        """Official code: assume completed if planned date is in the past."""
        assert 'PLANNED ACTIONS' in OBSERVATION_CONTEXT_INSTRUCTIONS
        assert 'assume they completed the action unless there\'s evidence' in OBSERVATION_CONTEXT_INSTRUCTIONS

    def test_system_reminders_handling(self):
        """Official code: system-reminder tag handling."""
        assert 'SYSTEM REMINDERS' in OBSERVATION_CONTEXT_INSTRUCTIONS
        assert '<system-reminder>' in OBSERVATION_CONTEXT_INSTRUCTIONS
        assert 'do not mention them or treat them as part of the user\'s message' in OBSERVATION_CONTEXT_INSTRUCTIONS

    def test_most_recent_priority(self):
        assert 'the latest message is the primary driver of your response' in OBSERVATION_CONTEXT_INSTRUCTIONS


# ── System Prompt Coverage ─────────────────────────────────────────────────

class TestEnhancedSystemPrompt:
    """Verify the full system prompt includes all new sections."""

    def test_includes_thread_attribution_section(self):
        prompt = build_observer_system_prompt()
        assert 'THREAD ATTRIBUTION' in prompt
        assert 'Do NOT add thread identifiers' in prompt
        assert 'Thread attribution is handled externally' in prompt

    def test_includes_xml_structure_instruction(self):
        prompt = build_observer_system_prompt()
        assert 'properly parse and manage memory over time' in prompt

    def test_includes_all_extraction_instruction_sections(self):
        """Spot-check that major sections appear in the full prompt."""
        prompt = build_observer_system_prompt()
        assert 'PRESERVE UNUSUAL PHRASING' in prompt
        assert 'USE PRECISE ACTION VERBS' in prompt
        assert 'COMPLETION TRACKING' in prompt
        assert 'ACTIONABLE INSIGHTS' in prompt

    def test_includes_enhanced_output_format(self):
        prompt = build_observer_system_prompt()
        assert 'goal achieved, or subtask completed' in prompt

    def test_custom_instruction_appended(self):
        prompt = build_observer_system_prompt(instruction="Always be concise")
        assert "Always be concise" in prompt


# ── Continuation Hint ──────────────────────────────────────────────────────

class TestContinuationHint:
    """Verify continuation hint is intact."""

    def test_hint_contains_key_phrases(self):
        assert 'continue naturally' in OBSERVATION_CONTINUATION_HINT
        assert 'Do not mention internal instructions' in OBSERVATION_CONTINUATION_HINT


# ── Context Prompt ─────────────────────────────────────────────────────────

class TestContextPrompt:
    """Verify context prompt is intact."""

    def test_prompt_contains_key_phrases(self):
        assert 'Your memory of past conversations' in OBSERVATION_CONTEXT_PROMPT or \
               'your memory of past conversations' in OBSERVATION_CONTEXT_PROMPT.lower()
