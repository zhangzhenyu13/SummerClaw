"""MOSCOPT Skill Pool — multi-skill pool and gating management.

Implements the MOSCOPT skill-pool data structure, compound document
serialization, gate-based skill selection, credit assignment, and
collective evolution utilities.

Compound document format (compatible with TrainerEngine's ``skill: str``
single-text interface)::

    <!-- MOSCOPT Pool Start -->
    <!-- N=5, K=2, epoch=3 -->

    ## Gate
    <gate G full text>

    ## Skill 1: Label
    <skill_1 full text, may include Slow Update protected region>

    ## Skill 2: Label
    <skill_2 full text>

    <!-- MOSCOPT Pool End -->

When N=1, K=1 the pool degrades to plain SkillOpt (no gate LLM call).
"""
from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.types import RolloutResult


# ── Pool data structure ─────────────────────────────────────────────────


@dataclass
class SkillPool:
    """MOSCOPT skill pool with gating and scoring state."""

    skills: dict[str, str] = field(default_factory=dict)
    gate: str = ""
    n: int = 5
    k: int = 2
    epoch: int = 0
    q_scores: dict[str, float] = field(default_factory=dict)
    activation_counts: dict[str, int] = field(default_factory=dict)
    cooccurrence: dict[str, dict[str, int]] = field(default_factory=dict)
    summaries: dict[str, dict] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.skills)

    def skill_ids(self) -> list[str]:
        return list(self.skills.keys())

    def get_skill(self, sid: str) -> str | None:
        return self.skills.get(sid)

    def ensure_state(self) -> None:
        """Ensure every skill has entries in q_scores / activation_counts."""
        for sid in self.skills:
            self.q_scores.setdefault(sid, 0.0)
            self.activation_counts.setdefault(sid, 0)
            self.summaries.setdefault(sid, {"id": sid, "label": f"Skill {sid}"})
            self.cooccurrence.setdefault(sid, {})


# ── Default gate prompt ─────────────────────────────────────────────────

DEFAULT_GATE_PROMPT = """\
You are a skill scheduler. Given the current task state and a summary table \
of available skills, select exactly K skills to activate for the current step.

Rules:
- Always output exactly K skill IDs, no more and no less.
- Prefer skills whose expertise matches the current task requirements.
- In early stages, prefer planning-oriented skills.
- If the task involves computation, include at least one math-oriented skill.

Output format (strict):
ACTIVATE: id1, id2, ...

Example (K=2):
ACTIVATE: 2, 5"""


# ── Serialization ────────────────────────────────────────────────────────

_POOL_START = "<!-- MOSCOPT Pool Start -->"
_POOL_END = "<!-- MOSCOPT Pool End -->"
_INFO_RE = re.compile(r"<!--\s*N=(\d+),\s*K=(\d+),\s*epoch=(\d+)\s*-->")


def serialize_pool(pool: SkillPool) -> str:
    """Serialize a SkillPool into the compound document format."""
    lines: list[str] = [
        _POOL_START,
        f"<!-- N={pool.n}, K={pool.k}, epoch={pool.epoch} -->",
        "",
        "## Gate",
        pool.gate or DEFAULT_GATE_PROMPT,
        "",
    ]
    for sid in sorted(pool.skills.keys(), key=lambda s: int(s)):
        label = pool.summaries.get(sid, {}).get("label", f"Skill {sid}")
        lines.append(f"## Skill {sid}: {label}")
        lines.append(pool.skills[sid])
        lines.append("")
    lines.append(_POOL_END)
    return "\n".join(lines)


def parse_pool(text: str) -> SkillPool:
    """Parse a compound document into a SkillPool.

    Falls back to wrapping the whole text as a single skill when the
    document does not use the MOSCOPT format.
    """
    pool = SkillPool()

    if _POOL_START not in text:
        # Backward compat: plain text → single skill
        pool.skills["1"] = text
        pool.n = 1
        pool.k = 1
        pool.gate = DEFAULT_GATE_PROMPT
        pool.ensure_state()
        return pool

    start_idx = text.index(_POOL_START) + len(_POOL_START)
    end_idx = text.index(_POOL_END) if _POOL_END in text else len(text)
    body = text[start_idx:end_idx].strip()

    # Parse metadata line
    info_m = _INFO_RE.search(body)
    if info_m:
        pool.n = int(info_m.group(1))
        pool.k = int(info_m.group(2))
        pool.epoch = int(info_m.group(3))

    # Extract gate section
    gate_re = re.compile(
        r"## Gate\s*\n(.*?)(?=\n## Skill \d+|\Z)", re.DOTALL,
    )
    gate_m = gate_re.search(body)
    if gate_m:
        pool.gate = gate_m.group(1).strip()
    else:
        pool.gate = DEFAULT_GATE_PROMPT

    # Extract skill sections
    skill_re = re.compile(
        r"## Skill (\d+):\s*(.*?)\n(.*?)(?=\n## Skill \d+|\Z)", re.DOTALL,
    )
    for m in skill_re.finditer(body):
        sid = m.group(1)
        label = m.group(2).strip()
        content = m.group(3).strip()
        pool.skills[sid] = content
        pool.summaries[sid] = {"id": sid, "label": label}

    # If no skills found, try simpler pattern (no label)
    if not pool.skills:
        simple_re = re.compile(r"## Skill (\d+)\s*\n(.*?)(?=\n## |\Z)", re.DOTALL)
        for m in simple_re.finditer(body):
            sid = m.group(1)
            content = m.group(2).strip()
            pool.skills[sid] = content
            pool.summaries[sid] = {"id": sid, "label": f"Skill {sid}"}

    pool.ensure_state()
    return pool


# ── Gate selection logic ─────────────────────────────────────────────────


def format_summary_table(
    pool: SkillPool,
    epoch: int,
    enrichment_epochs: tuple[int, int] = (2, 4),
) -> str:
    """Format skill summaries for the gate LLM (progressive disclosure).

    Epoch 1–early: ID + label only
    Early–late:    + recent Q-score
    Late+:         + co-occurrence, expertise, activation frequency

    Parameters
    ----------
    enrichment_epochs : tuple[int, int]
        (early, late) epoch thresholds for progressive enrichment.
        Defaults to (2, 4) matching the spec Section 3.4.1.
    """
    early, late = enrichment_epochs
    rows: list[str] = ["| ID | Label | Recent Score | Expertise |", "|----|-------|-------------|-----------|"]

    for sid in sorted(pool.skills.keys(), key=lambda s: int(s)):
        summary = pool.summaries.get(sid, {})
        label = summary.get("label", f"Skill {sid}")

        if epoch <= early:
            rows.append(f"| {sid} | {label} | — | — |")
        elif epoch <= late:
            q = pool.q_scores.get(sid, 0.0)
            rows.append(f"| {sid} | {label} | {q:.2f} | — |")
        else:
            q = pool.q_scores.get(sid, 0.0)
            act = pool.activation_counts.get(sid, 0)
            expertise = summary.get("expertise", label)
            rows.append(f"| {sid} | {label} | {q:.2f} | {expertise} (act={act}) |")

    return "\n".join(rows)


def parse_gate_output(
    text: str,
    valid_ids: set[str],
    k: int,
) -> list[str] | None:
    """Parse ``ACTIVATE: id1, id2, ...`` from gate LLM output.

    Returns a list of exactly K valid skill IDs, or None on failure.
    """
    text = text.strip()

    activate_m = re.search(r"ACTIVATE:\s*(.+)", text, re.IGNORECASE)
    if activate_m:
        ids_part = activate_m.group(1).strip()
        ids = [t.strip() for t in re.split(r"[,\s]+", ids_part) if t.strip()]
    else:
        ids = [t.strip() for t in re.split(r"[,\s]+", text) if t.strip() and t.strip().isdigit()]

    # Validate
    if len(ids) != k:
        return None

    valid_selected: list[str] = []
    seen: set[str] = set()
    for sid in ids:
        if sid in valid_ids and sid not in seen:
            valid_selected.append(sid)
            seen.add(sid)

    if len(valid_selected) == k:
        return valid_selected
    return None


def fallback_top_k(
    q_scores: dict[str, float],
    k: int,
    activation_counts: dict[str, int] | None = None,
    c_min: int = 5,
    exclude: set[str] | None = None,
) -> list[str]:
    """Fallback: select top-K skills by Q-score.

    Skills with activation_count >= c_min are preferred.  If fewer than K
    such skills exist, below-threshold skills are appended.
    """
    activation_counts = activation_counts or {}
    exclude = exclude or set()

    above = [
        (sid, score)
        for sid, score in q_scores.items()
        if sid not in exclude and activation_counts.get(sid, 0) >= c_min
    ]
    above.sort(key=lambda x: x[1], reverse=True)

    selected = [sid for sid, _ in above[:k]]

    if len(selected) < k:
        below = [
            (sid, score)
            for sid, score in q_scores.items()
            if sid not in exclude
            and sid not in set(selected)
            and activation_counts.get(sid, 0) < c_min
        ]
        below.sort(key=lambda x: x[1], reverse=True)
        selected.extend(sid for sid, _ in below[: k - len(selected)])

    if len(selected) < k:
        remaining = [sid for sid in q_scores if sid not in exclude and sid not in set(selected)]
        random.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])

    return selected


# ── Credit assignment (Section 3.5) ──────────────────────────────────────


def get_activated_skill_ids(result: RolloutResult) -> list[str]:
    """Extract activated skill IDs from a rollout result's extras."""
    raw = result.extras.get("moscopt_activated_skills", [])
    if isinstance(raw, list):
        return [str(sid) for sid in raw]
    return []


def update_q_scores(
    pool: SkillPool,
    results: list[RolloutResult],
    ema_beta: float = 0.3,
) -> None:
    """EMA-update Q-scores based on rollout results.

    Q(s_i) = EMA( sum_tau[c_i(tau)*R(tau)] / sum_tau[c_i(tau)] )
    """
    skill_reward_sum: dict[str, float] = {}
    skill_count: dict[str, int] = {}

    for result in results:
        activated = get_activated_skill_ids(result)
        reward = float(result.hard)
        for sid in activated:
            skill_reward_sum[sid] = skill_reward_sum.get(sid, 0.0) + reward
            skill_count[sid] = skill_count.get(sid, 0) + 1
            pool.activation_counts[sid] = pool.activation_counts.get(sid, 0) + 1

    for sid in pool.skills:
        if skill_count.get(sid, 0) > 0:
            new_obs = skill_reward_sum[sid] / skill_count[sid]
            old_q = pool.q_scores.get(sid, 0.0)
            # EMA: Q_new = (1-beta)*Q_old + beta*obs
            pool.q_scores[sid] = (1 - ema_beta) * old_q + ema_beta * new_obs


def update_cooccurrence(
    pool: SkillPool,
    results: list[RolloutResult],
) -> None:
    """Increment co-occurrence counts for skill pairs in successful trajectories."""
    for result in results:
        if result.hard != 1:
            continue
        activated = sorted(set(get_activated_skill_ids(result)))
        for i in range(len(activated)):
            for j in range(i + 1, len(activated)):
                si, sj = activated[i], activated[j]
                pool.cooccurrence.setdefault(si, {})
                pool.cooccurrence[si][sj] = pool.cooccurrence[si].get(sj, 0) + 1


def get_top_cooccurrence_pair(
    pool: SkillPool,
) -> tuple[tuple[str, str], int] | None:
    """Return the skill pair with the highest co-occurrence count."""
    best: tuple[tuple[str, str], int] | None = None
    for si, partners in pool.cooccurrence.items():
        for sj, count in partners.items():
            if best is None or count > best[1]:
                best = ((si, sj), count)
    return best


# ── Progressive summary enrichment (Section 3.4.1) ──────────────────────


def update_summaries(
    pool: SkillPool,
    epoch: int,
    enrichment_epochs: tuple[int, int] = (2, 4),
) -> None:
    """Refresh per-skill summaries with current Q-scores and stats.

    Parameters
    ----------
    enrichment_epochs : tuple[int, int]
        (early, late) epoch thresholds.  Co-occurrence info is added
        once ``epoch > late`` (Section 3.4.1).
    """
    _, late = enrichment_epochs
    for sid in pool.skills:
        summary = pool.summaries.setdefault(sid, {"id": sid, "label": f"Skill {sid}"})
        summary["q_score"] = pool.q_scores.get(sid, 0.0)
        summary["activation_count"] = pool.activation_counts.get(sid, 0)

        if epoch > late:
            # Add co-occurrence info
            cooc = pool.cooccurrence.get(sid, {})
            if cooc:
                top_partner = max(cooc.items(), key=lambda x: x[1])
                summary["top_cooccurrence"] = f"Skill {top_partner[0]} ({top_partner[1]}x)"


# ── Agent prompt building ────────────────────────────────────────────────


def build_agent_prompt(
    activated_skills: dict[str, str],
    state: str = "",
) -> str:
    """Build agent prompt from activated skill texts.

    Follows the format from the MOSCOP paper Section 3.4::

        [Activated Skills]
        Skill 2: <full text>
        Skill 5: <full text>

        [Current State]
        <environment observations and history>

        Please decide the next action using the activated strategies.
    """
    parts: list[str] = ["[Activated Skills]"]
    for sid, text in activated_skills.items():
        parts.append(f"Skill {sid}:\n{text}")
    if state:
        parts.extend(["", "[Current State]", state])
    parts.extend([
        "",
        "Please decide the next action using the activated strategies.",
    ])
    return "\n".join(parts)


def build_gate_prompt(
    summary_table: str,
    state: str,
    history: str,
    k: int,
) -> str:
    """Build the LLM prompt for the gate selector."""
    return (
        f"You are a skill scheduler. Select exactly {k} skills to activate.\n\n"
        f"## Skill Summary Table\n{summary_table}\n\n"
        f"## Current Task State\n{state or '(initial state)'}\n\n"
        f"## Recent History\n{history or '(no history yet)'}\n\n"
        f"Output exactly {k} skill IDs in the format: ACTIVATE: id1, id2, ...\n"
        f"Do not output anything else."
    )


async def call_gate_llm(
    provider: Any,
    model: str,
    gate_text: str,
    summary_table: str,
    state: str,
    history: str,
    k: int,
    valid_ids: set[str],
) -> tuple[list[str] | None, bool]:
    """Call the LLM with the gate prompt and parse the output.

    Returns
    -------
    tuple[list[str] | None, bool]
        (activated_ids, parse_failed)
        - activated_ids: list of K skill IDs, or None on failure.
        - parse_failed: True if the LLM responded but output could not be parsed
          (as opposed to an LLM call failure).  Used by the failure-rate monitor.
    """
    from .reflect import _call_llm

    user_prompt = build_gate_prompt(summary_table, state, history, k)
    system_prompt = gate_text or DEFAULT_GATE_PROMPT

    try:
        response = await _call_llm(
            provider=provider,
            model=model,
            system=system_prompt,
            user=user_prompt,
            max_tokens=128,
            retries=1,
            stage="gate_select",
        )
        if response:
            result = parse_gate_output(response, valid_ids, k)
            if result is not None:
                return result, False  # success
            return None, True  # LLM responded but parse failed
    except Exception as exc:
        logger.warning("[GATE] LLM call failed: {}", exc)

    return None, False  # LLM call failure (not a parse failure)


# ── LLM-based gate initialization (Section 3.3) ──────────────────────

_GATE_GEN_SYSTEM = """\
You are a skill scheduling expert. Your task is to write a concise but effective \
gating prompt that instructs an LLM to select skills for task execution.

The gating prompt must include:
1. **Input specification**: Describe the skill summary table format (ID, label, score, expertise).
2. **Output specification**: The scheduler must output exactly K skill IDs in the format "ACTIVATE: id1, id2, ...".
3. **Fallback rule**: If the output cannot be parsed, the system falls back to top-K by score.
4. **Selection rules**: Provide 3-5 heuristic rules for choosing skills based on task type, stage, and skill expertise.

Rules:
- Keep the gating prompt under 500 words.
- Be specific about the output format.
- Include example output.
- Output ONLY the gating prompt text, no commentary."""


async def generate_gate_prompt(
    provider: Any,
    model: str,
    task_description: str = "",
    pool_summaries: list[tuple[str, str]] | None = None,
    k: int = 2,
) -> str:
    """Generate a customized gating prompt via LLM based on task context.

    Parameters
    ----------
    provider : Any
        LLM provider instance.
    model : str
        Model name.
    task_description : str
        Description of the task domain.
    pool_summaries : list of (id, label) tuples, optional
        The initial skill labels for context.
    k : int
        Number of skills to activate per step.

    Returns
    -------
    str
        A customized gating prompt, or ``DEFAULT_GATE_PROMPT`` on failure.
    """
    from .reflect import _call_llm

    skill_info = ""
    if pool_summaries:
        skill_info = "\n".join(
            f"- Skill {sid}: {label}" for sid, label in pool_summaries
        )

    user = f"## Task Domain\n{task_description or '(general problem-solving)'}\n\n"
    user += f"## Number of Skills to Activate (K)\n{k}\n\n"
    if skill_info:
        user += f"## Available Skills\n{skill_info}\n\n"
    user += "Write a gating prompt for this configuration."

    try:
        result = await _call_llm(
            provider=provider,
            model=model,
            system=_GATE_GEN_SYSTEM,
            user=user,
            max_tokens=2048,
            retries=1,
            stage="generate_gate",
        )
        if result and len(result.strip()) > 50:
            logger.info(
                "[GATE INIT] LLM-generated gate prompt ({} chars)", len(result.strip()),
            )
            return result.strip()
    except Exception as exc:
        logger.warning("[GATE INIT] LLM generation failed: {}; using default", exc)

    return DEFAULT_GATE_PROMPT


# ── Collective evolution utilities (Section 3.9 Phase 3) ────────────────


_MUTATE_SYSTEM = """\
You are a skill mutation engine. Given a parent skill document and stable \
rules inherited from the meta-skill pool, produce a **mutated variant** \
that explores a different strategy angle while retaining the core strengths.

Rules:
- Modify approximately 30% of the text (add/remove/rewrite rules).
- Introduce a new strategic perspective or role.
- Preserve the document structure (YAML frontmatter, sections).
- Keep the total length within 20% of the original.
- Inherit and preserve the rules listed in the <Inherited Rules> section.
- Output only the mutated skill document, no commentary."""

_MUTATE_FORCE_SYSTEM = """\
You are a radical skill mutation engine. Given a parent skill document, \
produce a **heavily mutated variant** that takes a fundamentally different \
approach while retaining the domain.

Rules:
- Modify approximately 50% of the text (add/remove/rewrite rules).
- Introduce completely new role instructions and strategic perspectives.
- Preserve the document structure but change the core approach.
- Keep the total length within 30% of the original.
- Inherit and preserve the rules listed in the <Inherited Rules> section.
- Output only the mutated skill document, no commentary."""


async def mutate_skill(
    provider: Any,
    model: str,
    parent_text: str,
    inherited_rules: str = "",
    force: bool = False,
) -> str:
    """Create a mutated variant of a parent skill via LLM.

    Parameters
    ----------
    force : bool
        If True, apply aggressive mutation (~50% text change) instead of
        the default moderate mutation (~30%).  Used when diversity is low.
    """
    from .reflect import _call_llm

    user = f"## Parent Skill\n{parent_text}\n\n"
    if inherited_rules:
        user += f"## Inherited Rules (preserve these)\n{inherited_rules}\n\n"
    user += "Produce a mutated variant of this skill."

    system = _MUTATE_FORCE_SYSTEM if force else _MUTATE_SYSTEM
    try:
        result = await _call_llm(
            provider=provider,
            model=model,
            system=system,
            user=user,
            max_tokens=8192,
            retries=2,
            stage="mutate_skill",
            temperature=0.9 if force else None,
        )
        if result and len(result.strip()) > 100:
            return result.strip()
    except Exception as exc:
        logger.warning("[EVOLUTION] mutation failed: {}", exc)

    return parent_text


def compute_diversity(pool: SkillPool) -> float:
    """Compute average pairwise text similarity (0=diverse, 1=identical)."""
    skills = list(pool.skills.values())
    if len(skills) < 2:
        return 1.0
    total_sim = 0.0
    count = 0
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            total_sim += SequenceMatcher(None, skills[i], skills[j]).ratio()
            count += 1
    return total_sim / max(count, 1)


# ── Diverse pool generation (Section 3.2) ─────────────────────────────

_ROLE_PROMPTS = [
    ("Cautious Planner", "You are a cautious, methodical planner. You carefully analyze problems step by step, verify assumptions before acting, and prefer structured reasoning over quick guesses. You always outline a plan before executing."),
    ("Efficient Executor", "You are an efficient, action-oriented executor. You prioritize getting things done quickly, break tasks into small actionable steps, and avoid over-analysis. You prefer trial-and-error over lengthy deliberation."),
    ("Math Expert", "You are a mathematical and computational expert. You approach problems with rigorous formal reasoning, use precise calculations, and verify numerical results. You prefer analytical solutions over heuristic approaches."),
    ("Creative Thinker", "You are a creative, lateral thinker. You explore unconventional solutions, draw analogies from diverse domains, and challenge assumptions. You prefer brainstorming multiple alternatives before committing."),
    ("Detail Verifier", "You are a meticulous verifier and quality checker. You scrutinize every detail, look for edge cases and potential errors, and validate outputs against known constraints. You prefer thoroughness over speed."),
    ("Domain Synthesizer", "You are a cross-domain synthesizer. You combine knowledge from multiple fields to find novel solutions. You look for patterns that connect seemingly unrelated concepts."),
    ("Pragmatic Solver", "You are a pragmatic problem solver. You use practical heuristics, rules of thumb, and real-world experience to find good-enough solutions efficiently. You balance quality and speed."),
    ("Systematic Decomposer", "You are a systematic problem decomposer. You break complex problems into smaller sub-problems, solve each independently, and combine results. You prefer divide-and-conquer strategies."),
]


async def generate_diverse_pool(
    provider: Any,
    model: str,
    seed_skill: str,
    n: int,
    task_description: str = "",
    quality_filter_fn: Any | None = None,
) -> list[tuple[str, str]]:
    """Generate N diverse skills via LLM with different role prompts.

    Parameters
    ----------
    provider : Any
        LLM provider instance.
    model : str
        Model name to use for generation.
    seed_skill : str
        The initial seed skill text to use as a reference.
    n : int
        Number of diverse skills to generate.
    task_description : str
        Optional task description to guide generation.
    quality_filter_fn : Callable, optional
        An async callable ``(skill_text) -> float`` returning a quality score
        in [0, 1].  Candidates scoring below 0.1 are replaced with fallback
        variants.  If None, quality filtering is skipped (Section 3.2).

    Returns
    -------
    list[tuple[str, str]]
        List of (label, skill_text) tuples.  Length is exactly N
        (fallback variants fill any LLM-generation gaps).
    """
    from .reflect import _call_llm

    roles = _ROLE_PROMPTS[:n]
    # Pad with generic roles if n > len(_ROLE_PROMPTS)
    while len(roles) < n:
        idx = len(roles) % len(_ROLE_PROMPTS)
        base_label, base_prompt = _ROLE_PROMPTS[idx]
        roles.append((f"{base_label} v{len(roles) + 1}", base_prompt))

    candidates: list[tuple[str, str]] = []
    seen_texts: list[str] = []

    for label, role_prompt in roles:
        system = (
            "You are a skill document generator. Given a seed skill and a specific role, "
            "generate a new skill document that embodies the given role while addressing "
            "the same general task domain as the seed skill.\n\n"
            "Rules:\n"
            "- The output should be a complete, standalone skill document.\n"
            "- Maintain the structural format of the seed skill.\n"
            "- Emphasize the specific strengths and approach of the assigned role.\n"
            "- Keep length within 20% of the seed skill.\n"
            "- Output only the skill document, no commentary."
        )
        user_msg = (
            f"## Seed Skill\n{seed_skill[:2000]}\n\n"
            f"## Assigned Role: {label}\n{role_prompt}\n\n"
        )
        if task_description:
            user_msg += f"## Task Context\n{task_description}\n\n"
        user_msg += f"Generate a skill document for the '{label}' role."

        try:
            result = await _call_llm(
                provider=provider,
                model=model,
                system=system,
                user=user_msg,
                max_tokens=8192,
                retries=1,
                stage="generate_diverse_pool",
                temperature=0.8,
            )
            if result and len(result.strip()) > 100:
                text = result.strip()
                # Deduplicate by text similarity
                is_dup = any(
                    SequenceMatcher(None, text, seen).ratio() > 0.7
                    for seen in seen_texts
                )
                if not is_dup:
                    candidates.append((label, text))
                    seen_texts.append(text)
                    logger.info("[POOL GEN] generated skill '{}' ({} chars)", label, len(text))
                    continue
        except Exception as exc:
            logger.warning("[POOL GEN] LLM generation for '{}' failed: {}", label, exc)

        # Fallback: minor variant of seed skill with role header
        variant = f"# {label}\n\n{role_prompt}\n\n---\n\n{seed_skill[:3000]}"
        candidates.append((label, variant))
        seen_texts.append(variant)
        logger.info("[POOL GEN] fallback variant for '{}' ({} chars)", label, len(variant))

    # Trim to exactly N
    candidates = candidates[:n]

    # ── Quality filtering (Section 3.2) ──────────────────────────
    if quality_filter_fn is not None:
        filtered: list[tuple[str, str]] = []
        for label, text in candidates:
            try:
                score = await quality_filter_fn(text)
                if score >= 0.1:
                    filtered.append((label, text))
                    logger.info("[POOL GEN] quality filter: '{}' passed (score={:.2f})", label, score)
                else:
                    logger.info("[POOL GEN] quality filter: '{}' failed (score={:.2f} < 0.1)", label, score)
            except Exception as exc:
                logger.warning("[POOL GEN] quality filter for '{}' raised: {}; keeping", label, exc)
                filtered.append((label, text))

        # Replace filtered-out candidates with fallback variants
        while len(filtered) < n:
            idx = len(filtered)
            label, prompt = _ROLE_PROMPTS[idx % len(_ROLE_PROMPTS)]
            variant = f"# {label}\n\n{prompt}\n\n---\n\n{seed_skill[:3000]}"
            filtered.append((f"{label} (fallback)", variant))
            logger.info("[POOL GEN] quality filter: added fallback for slot {}", idx)

        candidates = filtered[:n]

    return candidates


# ── Foreign gene injection (Section 5.2) ──────────────────────────────

_FOREIGN_GENE_SYSTEM = """\
You are a creative skill generator. Your task is to produce a completely novel \
problem-solving strategy that is **fundamentally different** from the existing \
skills in the pool.

Rules:
- Do NOT mimic any existing skill's approach or structure.
- Introduce an unconventional perspective or methodology.
- The skill should be self-contained and actionable.
- Keep the document concise (500-1500 words).
- Output only the skill document, no commentary."""


async def inject_foreign_gene(
    provider: Any,
    model: str,
    pool: SkillPool,
    task_description: str = "",
) -> tuple[str, str]:
    """Generate a novel 'foreign gene' skill to inject diversity.

    Parameters
    ----------
    provider : Any
        LLM provider instance.
    model : str
        Model name.
    pool : SkillPool
        Current pool (used to inform the LLM about existing strategies).
    task_description : str
        Optional task context.

    Returns
    -------
    tuple[str, str]
        (label, skill_text) for the new foreign skill.
    """
    from .reflect import _call_llm

    # Summarize existing strategies for the LLM
    existing_summary = "\n".join(
        f"- Skill {sid}: {pool.summaries.get(sid, {}).get('label', 'Unknown')}"
        for sid in pool.skills
    )

    user = (
        f"## Existing Skills in Pool\n{existing_summary}\n\n"
        "Generate a completely novel problem-solving skill that takes a fundamentally "
        "different approach from all existing skills listed above."
    )
    if task_description:
        user += f"\n\n## Task Context\n{task_description}"

    try:
        result = await _call_llm(
            provider=provider,
            model=model,
            system=_FOREIGN_GENE_SYSTEM,
            user=user,
            max_tokens=4096,
            retries=1,
            stage="inject_foreign_gene",
        )
        if result and len(result.strip()) > 100:
            return "Foreign Explorer", result.strip()
    except Exception as exc:
        logger.warning("[FOREIGN GENE] LLM generation failed: {}", exc)

    # Fallback: pick a random diverse role template
    import random as _rng
    label, prompt = _rng.choice(_ROLE_PROMPTS)
    fallback_text = (
        f"# Diverse Explorer ({label})\n\n{prompt}\n\n"
        "You approach problems from an unconventional angle, exploring "
        "multiple alternative strategies before committing to a solution.\n"
    )
    return f"Foreign {label}", fallback_text


def rank_skills_by_failure_contribution(
    pool: SkillPool,
    results: list[RolloutResult],
) -> list[str]:
    """Rank skill IDs by failure contribution (worst first)."""
    failure_counts: dict[str, int] = {sid: 0 for sid in pool.skills}
    total_activations: dict[str, int] = {sid: 0 for sid in pool.skills}

    for result in results:
        activated = get_activated_skill_ids(result)
        for sid in activated:
            if sid in total_activations:
                total_activations[sid] += 1
                if result.hard == 0:
                    failure_counts[sid] = failure_counts.get(sid, 0) + 1

    def sort_key(sid: str) -> float:
        total = total_activations.get(sid, 0)
        if total == 0:
            return 1.0
        failure_rate = failure_counts.get(sid, 0) / total
        q = pool.q_scores.get(sid, 0.0)
        return failure_rate * (1.0 - q)

    return sorted(pool.skills.keys(), key=sort_key, reverse=True)


def select_lowest_scored(
    pool: SkillPool,
    count: int,
    min_activations: int = 5,
    protected: set[str] | None = None,
) -> list[str]:
    """Select the *count* lowest-Q-score skills for culling.

    Skills with activation_count < min_activations are exempt, as are
    skills in the *protected* set (e.g. high co-occurrence pairs).
    """
    protected = protected or set()
    candidates = [
        sid
        for sid in pool.skills
        if (
            pool.activation_counts.get(sid, 0) >= min_activations
            and sid not in protected
        )
    ]
    candidates.sort(key=lambda sid: pool.q_scores.get(sid, 0.0))
    return candidates[:count]


def select_top_parents(
    pool: SkillPool,
    count: int,
) -> list[str]:
    """Select the *count* highest-Q-score skills as reproduction parents."""
    ranked = sorted(
        pool.skills.keys(),
        key=lambda sid: pool.q_scores.get(sid, 0.0),
        reverse=True,
    )
    return ranked[:count]


def reassign_skill_ids(pool: SkillPool) -> None:
    """Re-index skill IDs to be consecutive 1..N after culling/breeding."""
    old_ids = sorted(pool.skills.keys(), key=lambda s: int(s))
    if old_ids == [str(i) for i in range(1, len(old_ids) + 1)]:
        return  # already consecutive

    new_skills: dict[str, str] = {}
    new_q: dict[str, float] = {}
    new_act: dict[str, int] = {}
    new_sum: dict[str, dict] = {}
    new_cooc: dict[str, dict[str, int]] = {}

    id_map: dict[str, str] = {}
    for new_idx, old_id in enumerate(old_ids, 1):
        new_id = str(new_idx)
        id_map[old_id] = new_id
        new_skills[new_id] = pool.skills[old_id]
        new_q[new_id] = pool.q_scores.get(old_id, 0.0)
        new_act[new_id] = pool.activation_counts.get(old_id, 0)
        new_sum[new_id] = pool.summaries.get(old_id, {"id": new_id, "label": f"Skill {new_id}"})
        new_sum[new_id]["id"] = new_id

    # Remap co-occurrence
    for old_si, partners in pool.cooccurrence.items():
        new_si = id_map.get(old_si, old_si)
        new_cooc[new_si] = {}
        for old_sj, count in partners.items():
            new_sj = id_map.get(old_sj, old_sj)
            new_cooc[new_si][new_sj] = count

    pool.skills = new_skills
    pool.q_scores = new_q
    pool.activation_counts = new_act
    pool.summaries = new_sum
    pool.cooccurrence = new_cooc


# ── Output distillation (Section 3.10) ──────────────────────────────────────


def distill_top_skill(pool: SkillPool, min_q: float = 0.5) -> str:
    """Extract the single best skill from the pool for SkillOpt-compatible deployment.

    Parameters
    ----------
    pool : SkillPool
        The trained skill pool.
    min_q : float
        Minimum Q-score threshold.  If the top skill's Q-score is below
        this, return an empty string (pool has not converged to a reliable skill).

    Returns
    -------
    str
        The full text of the highest-Q-score skill, or "" if not yet reliable.
    """
    if pool.size == 0:
        return ""

    top_sid = max(pool.skills, key=lambda sid: pool.q_scores.get(sid, 0.0))
    top_q = pool.q_scores.get(top_sid, 0.0)

    if top_q < min_q:
        logger.info(
            "[DISTILL] top skill {} Q={:.3f} < {:.2f}; pool not yet reliable",
            top_sid, top_q, min_q,
        )
        return ""

    logger.info(
        "[DISTILL] selected skill {} (Q={:.3f}, label={})",
        top_sid, top_q, pool.summaries.get(top_sid, {}).get("label", ""),
    )
    return pool.skills[top_sid]


# ── LLM-merged distillation (Section 3.10) ──────────────────────────────

_MERGE_DISTILL_SYSTEM = """\
You are a skill document merger. Given multiple high-quality skill documents, \
produce a single unified skill document that combines the strengths of all inputs.

Rules:
- Preserve the best rules and strategies from each input skill.
- Resolve any conflicts by choosing the more effective approach.
- The output should be a complete, standalone skill document.
- Maintain a clear, structured format.
- Keep the total length within 20% of the longest input.
- Output only the merged skill document, no commentary."""


async def distill_merged_skills(
    pool: SkillPool,
    provider: Any,
    model: str,
    top_k: int = 3,
    min_q: float = 0.5,
) -> str:
    """Merge the top-K highest-Q-score skills into a single unified skill via LLM.

    This implements Section 3.10's "LLM-merged output" for deployment in
    environments that do not support runtime gating.

    Parameters
    ----------
    pool : SkillPool
        The trained skill pool.
    provider : Any
        LLM provider instance.
    model : str
        Model name for the merge call.
    top_k : int
        Number of top skills to merge.
    min_q : float
        Minimum Q-score threshold.  Skills below this are excluded.

    Returns
    -------
    str
        The merged skill text, or "" if not enough qualifying skills.
    """
    from .reflect import _call_llm

    if pool.size == 0:
        return ""

    # Select top-K skills above min_q
    qualified = [
        (sid, pool.q_scores.get(sid, 0.0))
        for sid in pool.skills
        if pool.q_scores.get(sid, 0.0) >= min_q
    ]
    qualified.sort(key=lambda x: x[1], reverse=True)
    top_skills = qualified[:top_k]

    if len(top_skills) < 2:
        # Not enough skills to merge; fall back to single best
        return distill_top_skill(pool, min_q=min_q)

    # Build merge prompt
    skill_sections = []
    for sid, q in top_skills:
        label = pool.summaries.get(sid, {}).get("label", sid)
        skill_sections.append(
            f"## Skill {sid}: {label} (Q={q:.2f})\n{pool.skills[sid]}"
        )
    user = "\n\n---\n\n".join(skill_sections)
    user += (
        "\n\nMerge the above skills into a single unified skill document "
        "that combines the strengths of all inputs."
    )

    try:
        result = await _call_llm(
            provider=provider,
            model=model,
            system=_MERGE_DISTILL_SYSTEM,
            user=user,
            max_tokens=8192,
            retries=2,
            stage="distill_merged",
        )
        if result and len(result.strip()) > 100:
            logger.info(
                "[DISTILL] merged {} skills (top Q={:.2f}) into {} chars",
                len(top_skills), top_skills[0][1], len(result.strip()),
            )
            return result.strip()
    except Exception as exc:
        logger.warning("[DISTILL] LLM merge failed: {}; falling back to top skill", exc)

    return distill_top_skill(pool, min_q=min_q)


# ── Static routing table extraction (Section 3.10) ─────────────────────

_ROUTING_TABLE_SYSTEM = """\
You are a skill routing analyst. Given a gating prompt and a set of skill summaries, \
extract the decision logic from the gating prompt and convert it into a static \
if-then routing table.

Rules:
- Each rule should be a clear IF-THEN statement.
- Reference skill IDs and labels from the summary table.
- Cover all decision patterns found in the gating prompt.
- If a pattern is ambiguous, note it as a fallback rule.
- Output ONLY the routing table, no commentary.

Format:
```
IF <condition> THEN ACTIVATE: <skill_ids>
IF <condition> THEN ACTIVATE: <skill_ids>
...
DEFAULT: ACTIVATE: <skill_ids>
```"""


async def extract_routing_table(
    pool: SkillPool,
    provider: Any,
    model: str,
) -> str:
    """Extract a static if-then routing table from the gate prompt.

    This implements Section 3.10's "static skill subset + routing table"
    output format, which reduces inference overhead by replacing the
    LLM-based gate with deterministic rules.

    Parameters
    ----------
    pool : SkillPool
        The trained skill pool with an optimized gate prompt.
    provider : Any
        LLM provider instance.
    model : str
        Model name.

    Returns
    -------
    str
        A static routing table as text, or "" on failure.
    """
    from .reflect import _call_llm

    if pool.size == 0 or not pool.gate:
        return ""

    summary_table = format_summary_table(pool, pool.epoch)
    user = (
        f"## Gate Prompt\n{pool.gate}\n\n"
        f"## Skill Summary Table\n{summary_table}\n\n"
        "Extract the routing logic from the gate prompt into a static if-then routing table."
    )

    try:
        result = await _call_llm(
            provider=provider,
            model=model,
            system=_ROUTING_TABLE_SYSTEM,
            user=user,
            max_tokens=4096,
            retries=2,
            stage="extract_routing_table",
        )
        if result and len(result.strip()) > 30:
            logger.info(
                "[ROUTING] extracted routing table ({} chars)", len(result.strip()),
            )
            return result.strip()
    except Exception as exc:
        logger.warning("[ROUTING] LLM extraction failed: {}", exc)

    return ""
