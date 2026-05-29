"""Unit tests for MOSCOP Skill Pool module (pool.py).

Covers:
- SkillPool data structure and ensure_state
- Serialization roundtrip (serialize_pool / parse_pool)
- Gate selection logic (parse_gate_output, fallback_top_k, format_summary_table)
- Credit assignment (update_q_scores, update_cooccurrence, rank_skills_by_failure_contribution)
- Collective evolution utilities (select_lowest_scored, select_top_parents, reassign_skill_ids, compute_diversity)
- Distillation (distill_top_skill, distill_merged_skills)
- Prompt building (build_agent_prompt, build_gate_prompt)
- Async LLM functions with mock provider (generate_diverse_pool, mutate_skill, inject_foreign_gene)
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.agent_trainer.algorithms.moscopt.pool import (
    DEFAULT_GATE_PROMPT,
    SkillPool,
    build_agent_prompt,
    build_gate_prompt,
    call_gate_llm,
    compute_diversity,
    distill_merged_skills,
    distill_top_skill,
    fallback_top_k,
    format_summary_table,
    generate_diverse_pool,
    generate_gate_prompt,
    get_activated_skill_ids,
    get_top_cooccurrence_pair,
    inject_foreign_gene,
    mutate_skill,
    parse_gate_output,
    parse_pool,
    rank_skills_by_failure_contribution,
    reassign_skill_ids,
    select_lowest_scored,
    select_top_parents,
    serialize_pool,
    update_cooccurrence,
    update_q_scores,
    update_summaries,
)
from summerclaw.agent_trainer.types import RolloutResult


MockResponse = namedtuple("MockResponse", ["content"])


def _make_mock_provider(response_text: str = "mock response"):
    """Create a mock provider that returns the given text."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MockResponse(content=response_text))
    provider.max_concurrency = 4
    return provider


def _make_pool(n: int = 3, k: int = 2) -> SkillPool:
    """Create a test pool with N skills."""
    pool = SkillPool(n=n, k=k, epoch=1)
    for i in range(1, n + 1):
        sid = str(i)
        pool.skills[sid] = f"# Skill {sid}\nThis is skill {sid} content."
    pool.gate = DEFAULT_GATE_PROMPT
    pool.ensure_state()
    return pool


# ═══════════════════════════════════════════════════════════════
# 1.1 SkillPool data structure
# ═══════════════════════════════════════════════════════════════


class TestSkillPool:
    def test_default_fields(self):
        pool = SkillPool()
        assert pool.skills == {}
        assert pool.gate == ""
        assert pool.n == 5
        assert pool.k == 2
        assert pool.epoch == 0
        assert pool.q_scores == {}
        assert pool.activation_counts == {}
        assert pool.cooccurrence == {}
        assert pool.summaries == {}

    def test_size_and_skill_ids(self):
        pool = _make_pool(n=3)
        assert pool.size == 3
        assert set(pool.skill_ids()) == {"1", "2", "3"}

    def test_get_skill(self):
        pool = _make_pool(n=2)
        assert pool.get_skill("1") is not None
        assert "Skill 1" in pool.get_skill("1")
        assert pool.get_skill("99") is None

    def test_ensure_state(self):
        pool = _make_pool(n=3)
        for sid in ["1", "2", "3"]:
            assert sid in pool.q_scores
            assert pool.q_scores[sid] == 0.0
            assert sid in pool.activation_counts
            assert pool.activation_counts[sid] == 0
            assert sid in pool.summaries
            assert sid in pool.cooccurrence


# ═══════════════════════════════════════════════════════════════
# 1.2 Serialization roundtrip
# ═══════════════════════════════════════════════════════════════


class TestSerialization:
    def test_serialize_parse_roundtrip(self):
        pool = _make_pool(n=5, k=2)
        pool.epoch = 3
        pool.q_scores["1"] = 0.7
        pool.summaries["1"]["label"] = "Expert Planner"
        text = serialize_pool(pool)
        assert "MOSCOPT Pool Start" in text
        assert "N=5, K=2, epoch=3" in text

        parsed = parse_pool(text)
        assert parsed.n == 5
        assert parsed.k == 2
        assert parsed.epoch == 3
        assert parsed.size == 5
        assert "Expert Planner" in parsed.summaries.get("1", {}).get("label", "")

    def test_parse_plain_text_fallback(self):
        plain = "This is just a plain skill document with no MOSCOPT markers."
        pool = parse_pool(plain)
        assert pool.n == 1
        assert pool.k == 1
        assert pool.size == 1
        assert pool.skills["1"] == plain

    def test_parse_skills_without_labels(self):
        text = (
            "<!-- MOSCOPT Pool Start -->\n"
            "<!-- N=2, K=1, epoch=0 -->\n"
            "## Gate\nSome gate text\n\n"
            "## Skill 1\nSkill one content\n\n"
            "## Skill 2\nSkill two content\n\n"
            "<!-- MOSCOPT Pool End -->"
        )
        pool = parse_pool(text)
        assert pool.size == 2
        assert "Skill one content" in pool.skills.get("1", "")


# ═══════════════════════════════════════════════════════════════
# 1.3 Gate selection logic
# ═══════════════════════════════════════════════════════════════


class TestParseGateOutput:
    def test_normal_parse(self):
        result = parse_gate_output("ACTIVATE: 1, 3", {"1", "2", "3", "4"}, 2)
        assert result == ["1", "3"]

    def test_case_insensitive(self):
        result = parse_gate_output("activate: 2, 4", {"1", "2", "3", "4"}, 2)
        assert result == ["2", "4"]

    def test_too_many_ids(self):
        result = parse_gate_output("ACTIVATE: 1, 2, 3", {"1", "2", "3"}, 2)
        assert result is None

    def test_too_few_ids(self):
        result = parse_gate_output("ACTIVATE: 1", {"1", "2", "3"}, 2)
        assert result is None

    def test_invalid_ids(self):
        result = parse_gate_output("ACTIVATE: 1, 99", {"1", "2", "3"}, 2)
        assert result is None

    def test_duplicate_ids(self):
        result = parse_gate_output("ACTIVATE: 1, 1", {"1", "2", "3"}, 2)
        assert result is None

    def test_fallback_digits_only(self):
        result = parse_gate_output("1 3", {"1", "2", "3"}, 2)
        assert result == ["1", "3"]


class TestFallbackTopK:
    def test_basic_top_k(self):
        q = {"1": 0.9, "2": 0.3, "3": 0.6}
        act = {"1": 10, "2": 10, "3": 10}
        result = fallback_top_k(q, 2, act, c_min=5)
        assert result == ["1", "3"]

    def test_c_min_filter(self):
        q = {"1": 0.9, "2": 0.3, "3": 0.6}
        act = {"1": 10, "2": 2, "3": 1}  # only "1" above c_min
        result = fallback_top_k(q, 2, act, c_min=5)
        assert "1" in result
        assert len(result) == 2

    def test_exclude_set(self):
        q = {"1": 0.9, "2": 0.8, "3": 0.7}
        act = {"1": 10, "2": 10, "3": 10}
        result = fallback_top_k(q, 2, act, c_min=5, exclude={"1"})
        assert "1" not in result
        assert result == ["2", "3"]

    def test_insufficient_skills(self):
        q = {"1": 0.9}
        act = {"1": 10}
        result = fallback_top_k(q, 3, act, c_min=5)
        assert "1" in result


class TestFormatSummaryTable:
    def test_early_epoch_minimal_info(self):
        pool = _make_pool(n=2)
        table = format_summary_table(pool, epoch=1, enrichment_epochs=(2, 4))
        assert "—" in table  # No score or expertise shown

    def test_mid_epoch_shows_q_score(self):
        pool = _make_pool(n=2)
        pool.q_scores["1"] = 0.75
        table = format_summary_table(pool, epoch=3, enrichment_epochs=(2, 4))
        assert "0.75" in table

    def test_late_epoch_shows_expertise(self):
        pool = _make_pool(n=2)
        pool.q_scores["1"] = 0.8
        pool.activation_counts["1"] = 15
        pool.summaries["1"]["expertise"] = "Math Expert"
        table = format_summary_table(pool, epoch=5, enrichment_epochs=(2, 4))
        assert "Math Expert" in table
        assert "act=15" in table


@pytest.mark.asyncio
class TestCallGateLLM:
    async def test_success(self):
        provider = _make_mock_provider("ACTIVATE: 1, 3")
        result, parse_failed = await call_gate_llm(
            provider, "model", "gate text", "summary", "state", "history",
            2, {"1", "2", "3"},
        )
        assert result == ["1", "3"]
        assert parse_failed is False

    async def test_parse_failure(self):
        provider = _make_mock_provider("ACTIVATE: 1, 2, 3")  # too many
        result, parse_failed = await call_gate_llm(
            provider, "model", "gate text", "summary", "state", "history",
            2, {"1", "2", "3"},
        )
        assert result is None
        assert parse_failed is True

    async def test_llm_failure(self):
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=Exception("API error"))
        provider.max_concurrency = 4
        result, parse_failed = await call_gate_llm(
            provider, "model", "gate text", "summary", "state", "history",
            2, {"1", "2", "3"},
        )
        assert result is None
        assert parse_failed is False


# ═══════════════════════════════════════════════════════════════
# 1.4 Credit Assignment
# ═══════════════════════════════════════════════════════════════


class TestCreditAssignment:
    def test_get_activated_skill_ids(self):
        r = RolloutResult(id="t1", hard=1, soft=0.9)
        r.extras["moscopt_activated_skills"] = ["1", "3"]
        assert get_activated_skill_ids(r) == ["1", "3"]

    def test_get_activated_skill_ids_missing(self):
        r = RolloutResult(id="t1", hard=1, soft=0.9)
        assert get_activated_skill_ids(r) == []

    def test_get_activated_skill_ids_non_list(self):
        r = RolloutResult(id="t1", hard=1, soft=0.9)
        r.extras["moscopt_activated_skills"] = "not_a_list"
        assert get_activated_skill_ids(r) == []

    def test_update_q_scores_ema(self):
        pool = _make_pool(n=3)
        pool.q_scores["1"] = 0.5
        r1 = RolloutResult(id="t1", hard=1, soft=0.9)
        r1.extras["moscopt_activated_skills"] = ["1"]
        r2 = RolloutResult(id="t2", hard=0, soft=0.2)
        r2.extras["moscopt_activated_skills"] = ["1"]

        update_q_scores(pool, [r1, r2], ema_beta=0.3)
        # new_obs = (1 + 0) / 2 = 0.5; Q_new = 0.7*0.5 + 0.3*0.5 = 0.5
        assert abs(pool.q_scores["1"] - 0.5) < 0.01
        # Unactivated skills unchanged
        assert pool.q_scores["2"] == 0.0
        # Activation counts incremented
        assert pool.activation_counts["1"] == 2

    def test_update_cooccurrence_success_only(self):
        pool = _make_pool(n=3)
        r_success = RolloutResult(id="t1", hard=1, soft=0.9)
        r_success.extras["moscopt_activated_skills"] = ["1", "2"]
        r_fail = RolloutResult(id="t2", hard=0, soft=0.2)
        r_fail.extras["moscopt_activated_skills"] = ["1", "3"]

        update_cooccurrence(pool, [r_success, r_fail])
        # Only success counted
        assert pool.cooccurrence.get("1", {}).get("2", 0) == 1
        assert pool.cooccurrence.get("1", {}).get("3", 0) == 0

    def test_get_top_cooccurrence_pair(self):
        pool = _make_pool(n=3)
        pool.cooccurrence["1"] = {"2": 5, "3": 2}
        pool.cooccurrence["2"] = {"3": 3}
        result = get_top_cooccurrence_pair(pool)
        assert result is not None
        assert result[0] == ("1", "2")
        assert result[1] == 5

    def test_get_top_cooccurrence_pair_empty(self):
        pool = _make_pool(n=2)
        assert get_top_cooccurrence_pair(pool) is None

    def test_rank_skills_by_failure_contribution(self):
        pool = _make_pool(n=3)
        pool.q_scores["1"] = 0.9  # high Q, low failure contribution
        pool.q_scores["2"] = 0.1  # low Q, high failure contribution
        pool.q_scores["3"] = 0.5

        r1 = RolloutResult(id="t1", hard=0, soft=0.1)
        r1.extras["moscopt_activated_skills"] = ["2"]
        r2 = RolloutResult(id="t2", hard=1, soft=0.9)
        r2.extras["moscopt_activated_skills"] = ["1"]
        r3 = RolloutResult(id="t3", hard=1, soft=0.8)
        r3.extras["moscopt_activated_skills"] = ["3"]

        ranked = rank_skills_by_failure_contribution(pool, [r1, r2, r3])
        # Skill 2 should be first (highest failure contribution: 1.0 * 0.9 = 0.9)
        assert ranked[0] == "2"

    def test_update_summaries(self):
        pool = _make_pool(n=2)
        pool.q_scores["1"] = 0.8
        pool.activation_counts["1"] = 10
        pool.cooccurrence["1"] = {"2": 5}

        update_summaries(pool, epoch=5, enrichment_epochs=(2, 4))
        assert pool.summaries["1"]["q_score"] == 0.8
        assert pool.summaries["1"]["activation_count"] == 10
        assert "top_cooccurrence" in pool.summaries["1"]


# ═══════════════════════════════════════════════════════════════
# 1.5 Collective evolution utilities
# ═══════════════════════════════════════════════════════════════


class TestCollectiveEvolutionUtils:
    def test_select_lowest_scored(self):
        pool = _make_pool(n=4)
        pool.q_scores = {"1": 0.9, "2": 0.2, "3": 0.5, "4": 0.1}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10, "4": 10}
        result = select_lowest_scored(pool, 2, min_activations=5)
        assert result == ["4", "2"]

    def test_select_lowest_scored_protected(self):
        pool = _make_pool(n=3)
        pool.q_scores = {"1": 0.9, "2": 0.1, "3": 0.5}
        pool.activation_counts = {"1": 10, "2": 10, "3": 10}
        result = select_lowest_scored(pool, 2, min_activations=5, protected={"2"})
        assert "2" not in result

    def test_select_lowest_scored_below_min_activations(self):
        pool = _make_pool(n=3)
        pool.q_scores = {"1": 0.9, "2": 0.1, "3": 0.5}
        pool.activation_counts = {"1": 10, "2": 2, "3": 10}  # "2" below c_min
        result = select_lowest_scored(pool, 2, min_activations=5)
        assert "2" not in result

    def test_select_top_parents(self):
        pool = _make_pool(n=4)
        pool.q_scores = {"1": 0.3, "2": 0.9, "3": 0.7, "4": 0.5}
        result = select_top_parents(pool, 2)
        assert result == ["2", "3"]

    def test_reassign_skill_ids(self):
        pool = SkillPool(n=3, k=2)
        pool.skills = {"2": "a", "5": "b", "8": "c"}
        pool.q_scores = {"2": 0.1, "5": 0.2, "8": 0.3}
        pool.activation_counts = {"2": 1, "5": 2, "8": 3}
        pool.summaries = {"2": {"id": "2"}, "5": {"id": "5"}, "8": {"id": "8"}}
        pool.cooccurrence = {"2": {"5": 3}, "5": {}, "8": {}}

        reassign_skill_ids(pool)
        assert set(pool.skills.keys()) == {"1", "2", "3"}
        # Cooccurrence remapped: old "2"->new "1", old "5"->new "2"
        assert pool.cooccurrence.get("1", {}).get("2", 0) == 3

    def test_reassign_already_consecutive(self):
        pool = _make_pool(n=3)
        old_keys = list(pool.skills.keys())
        reassign_skill_ids(pool)
        assert list(pool.skills.keys()) == old_keys

    def test_compute_diversity_identical(self):
        pool = SkillPool()
        pool.skills = {"1": "same text", "2": "same text"}
        assert compute_diversity(pool) == 1.0

    def test_compute_diversity_different(self):
        pool = SkillPool()
        pool.skills = {"1": "completely different text A", "2": "entirely unique text B"}
        d = compute_diversity(pool)
        assert d < 1.0

    def test_compute_diversity_single_skill(self):
        pool = SkillPool()
        pool.skills = {"1": "only one"}
        assert compute_diversity(pool) == 1.0

    def test_distill_top_skill(self):
        pool = _make_pool(n=3)
        pool.q_scores = {"1": 0.3, "2": 0.8, "3": 0.5}
        result = distill_top_skill(pool, min_q=0.5)
        assert "Skill 2" in result

    def test_distill_top_skill_below_threshold(self):
        pool = _make_pool(n=2)
        pool.q_scores = {"1": 0.2, "2": 0.3}
        result = distill_top_skill(pool, min_q=0.5)
        assert result == ""

    def test_distill_top_skill_empty_pool(self):
        pool = SkillPool()
        assert distill_top_skill(pool) == ""

    def test_build_agent_prompt(self):
        skills = {"2": "Skill 2 text", "5": "Skill 5 text"}
        prompt = build_agent_prompt(skills, state="Task: solve math")
        assert "[Activated Skills]" in prompt
        assert "Skill 2:" in prompt
        assert "Skill 5:" in prompt
        assert "[Current State]" in prompt
        assert "Task: solve math" in prompt

    def test_build_gate_prompt(self):
        prompt = build_gate_prompt("| table |", "state", "history", 3)
        assert "exactly 3 skills" in prompt
        assert "| table |" in prompt
        assert "ACTIVATE:" in prompt


# ═══════════════════════════════════════════════════════════════
# 1.6 Async LLM functions (mock provider)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAsyncLLMFunctions:
    async def test_generate_diverse_pool(self):
        call_count = 0
        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MockResponse(content=f"# Unique Skill {call_count}\nDistinct content for skill {call_count} with enough text to pass the length filter.")
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=mock_chat)
        provider.max_concurrency = 4

        result = await generate_diverse_pool(provider, "model", "seed skill text", n=3)
        assert len(result) == 3
        for label, text in result:
            assert isinstance(label, str)
            assert len(text) > 50

    async def test_generate_diverse_pool_fallback_on_failure(self):
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=Exception("API error"))
        provider.max_concurrency = 4

        result = await generate_diverse_pool(provider, "model", "seed skill text", n=2)
        assert len(result) == 2
        # Fallback variants contain role prompt text
        for label, text in result:
            assert len(text) > 0

    async def test_mutate_skill(self):
        provider = _make_mock_provider("# Mutated Skill\nThis is a heavily mutated version of the skill with enough content to pass the filter.")
        result = await mutate_skill(provider, "model", "# Original\nOriginal skill content")
        assert "Mutated Skill" in result

    async def test_mutate_skill_force(self):
        provider = _make_mock_provider("# Radically Mutated Skill\nCompletely different approach with enough text to pass the 100 char filter easily and more content here.")
        result = await mutate_skill(provider, "model", "# Original\nOriginal skill content", force=True)
        # force=True should still produce a valid result (uses different system prompt internally)
        assert len(result) > 50

    async def test_inject_foreign_gene_success(self):
        provider = _make_mock_provider("# Novel Strategy\nThis is a completely novel approach to problem solving with sufficient length content.")
        pool = _make_pool(n=3)
        label, text = await inject_foreign_gene(provider, "model", pool)
        assert "Foreign" in label or len(text) > 50

    async def test_inject_foreign_gene_fallback(self):
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=Exception("API error"))
        provider.max_concurrency = 4
        pool = _make_pool(n=2)
        label, text = await inject_foreign_gene(provider, "model", pool)
        assert len(text) > 20  # Fallback still produces something

    async def test_distill_merged_skills(self):
        provider = _make_mock_provider("# Merged Skill\nThis is a unified merged skill document combining the best of all inputs with sufficient content.")
        pool = _make_pool(n=3)
        pool.q_scores = {"1": 0.9, "2": 0.8, "3": 0.7}

        result = await distill_merged_skills(pool, provider, "model", top_k=2, min_q=0.5)
        assert len(result) > 50

    async def test_distill_merged_skills_fallback_single(self):
        """When fewer than 2 skills qualify, falls back to distill_top_skill."""
        pool = _make_pool(n=3)
        pool.q_scores = {"1": 0.9, "2": 0.2, "3": 0.1}  # only "1" above 0.5

        provider = _make_mock_provider("")
        result = await distill_merged_skills(pool, provider, "model", top_k=3, min_q=0.5)
        assert "Skill 1" in result  # Returns top skill text

    async def test_generate_gate_prompt(self):
        provider = _make_mock_provider("You are a scheduler. Select exactly K skills. ACTIVATE: id1, id2. This is a custom gate prompt with enough length.")
        result = await generate_gate_prompt(
            provider, "model", "math tasks",
            pool_summaries=[("1", "Planner"), ("2", "Executor")], k=2,
        )
        assert len(result) > 50

    async def test_generate_gate_prompt_fallback(self):
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=Exception("error"))
        provider.max_concurrency = 4
        result = await generate_gate_prompt(provider, "model", "task")
        assert result == DEFAULT_GATE_PROMPT
