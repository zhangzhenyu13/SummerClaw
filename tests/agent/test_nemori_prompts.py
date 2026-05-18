"""Tests for Nemori PromptTemplates — formatting helpers and template methods."""

import json

from summerclaw.memory.nemori_memory.prompts import PromptTemplates


# ────────────────────────────────────────────────────────────────────────────
# Template formatting
# ────────────────────────────────────────────────────────────────────────────


class TestPromptTemplates:
    """Verify all prompt templates format correctly."""

    def test_episode_generation_prompt(self):
        result = PromptTemplates.get_episode_generation_prompt(
            conversation="Hello world", boundary_reason="topic shift",
        )
        assert "Hello world" in result
        assert "topic shift" in result
        assert "episodic memory" in result.lower()
        # Contains JSON schema
        assert '"title"' in result
        assert '"content"' in result
        assert '"timestamp"' in result

    def test_semantic_generation_prompt(self):
        result = PromptTemplates.get_semantic_generation_prompt("Episode 1: test")
        assert "Episode 1: test" in result
        assert "HIGH-VALUE" in result
        assert '"statements"' in result

    def test_prediction_prompt(self):
        result = PromptTemplates.get_prediction_prompt(
            "Hiking Plan", ["User likes hiking", "User lives in Seattle"],
        )
        assert "Hiking Plan" in result
        assert "User likes hiking" in result
        assert "User lives in Seattle" in result
        assert "Relevant Knowledge Statements" in result

    def test_batch_segmentation_prompt(self):
        result = PromptTemplates.get_batch_segmentation_prompt(
            count=5, messages="1. user: hello\n2. assistant: hi",
        )
        assert "5 messages" in result
        assert "1. user: hello" in result
        assert '"episodes"' in result
        assert '"indices"' in result

    def test_merge_decision_prompt(self):
        result = PromptTemplates.get_merge_decision_prompt(
            new_time_range="2025-01-01", new_content="test content",
            candidates="1. cand",
        )
        assert "2025-01-01" in result
        assert "test content" in result
        assert "1. cand" in result
        assert '"decision"' in result

    def test_merge_content_prompt(self):
        result = PromptTemplates.get_merge_content_prompt(
            original_time_range="t1", original_title="T1", original_content="C1",
            new_time_range="t2", new_title="T2", new_content="C2",
            combined_events="CE",
        )
        assert "T1" in result
        assert "T2" in result
        assert "C1" in result
        assert "C2" in result
        assert "CE" in result

    def test_multimodal_guidance(self):
        guidance = PromptTemplates.get_multimodal_guidance()
        assert "image" in guidance.lower()
        assert "visual" in guidance.lower()

    # ── Formatting helpers ─────────────────────────────────────────────

    def test_format_conversation_basic(self):
        msgs = [
            {"role": "user", "content": "hello", "timestamp": "2025-01-01T12:00:00"},
            {"role": "assistant", "content": "hi", "timestamp": "2025-01-01T12:00:01"},
        ]
        result = PromptTemplates.format_conversation(msgs)
        assert "user: hello" in result
        assert "assistant: hi" in result
        assert "2025-01-01" in result

    def test_format_conversation_multimodal(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
                "timestamp": "2025-01-01T12:00:00",
            },
        ]
        result = PromptTemplates.format_conversation(msgs)
        assert "Look at this" in result
        assert "[Image attached]" in result

    def test_format_conversation_no_timestamp(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = PromptTemplates.format_conversation(msgs)
        assert "user: hello" in result

    def test_format_conversation_datetime_timestamp(self):
        from datetime import datetime, timezone
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        msgs = [{"role": "user", "content": "hi", "timestamp": ts}]
        result = PromptTemplates.format_conversation(msgs)
        assert "2025-01-01" in result

    def test_format_episodes_for_semantic(self):
        eps = [
            {"title": "Ep1", "content": "C1", "created_at": "2025-01-01"},
            {"title": "Ep2", "content": "C2", "created_at": "2025-01-02"},
        ]
        result = PromptTemplates.format_episodes_for_semantic(eps)
        assert "Episode 1:" in result
        assert "Ep1" in result
        assert "C2" in result

    def test_format_episodes_for_semantic_handles_missing_fields(self):
        eps = [{"title": "T", "content": "C"}]
        result = PromptTemplates.format_episodes_for_semantic(eps)
        assert "T" in result
        assert "C" in result

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_batch_segmentation_single_message(self):
        result = PromptTemplates.get_batch_segmentation_prompt(1, "1. user: hi")
        assert "1 messages" in result

    def test_prediction_empty_knowledge(self):
        result = PromptTemplates.get_prediction_prompt("Title", [])
        assert "Title" in result
        # Empty knowledge => just empty formatted string
