"""SkillOpt Meta Skill — cross-epoch optimizer memory.

Ported from SkillOpt's ``optimizer/meta_skill.py`` with SummerClaw LLM
provider adaptation.

This module maintains a compact optimizer-facing memory distilled from
adjacent-epoch skill comparisons.  Unlike ``slow_update``, it does **not**
modify the target skill document.  Instead, it produces guidance meant to
improve future optimizer behavior when proposing, merging, and ranking edits.

The meta_skill_content is injected into all subsequent LLM calls as context
via :func:`format_meta_skill_context`.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from summerclaw.agent_trainer.algorithms.skillopt.reflect import _call_llm, _extract_json

from .prompts_loader import load_prompt
from .slow_update import format_comparison_text


def format_meta_skill_context(meta_skill_content: str) -> str:
    """Render optimizer memory into a prompt-ready context block.

    Parameters
    ----------
    meta_skill_content : str
        The distilled meta skill guidance text.

    Returns
    -------
    str
        A formatted block ready to prepend to user prompts, or empty
        string if *meta_skill_content* is blank.
    """
    content = (meta_skill_content or "").strip()
    if not content:
        return ""
    return (
        "## Optimizer Meta Skill\n"
        "This is optimizer-side memory distilled from prior epoch transitions in "
        "this environment. Use it to improve how you propose, merge, and rank "
        "skill edits. Prefer it when the current evidence is ambiguous, but do "
        "not force it if the current trajectories clearly contradict it.\n\n"
        f"{content}"
    )


async def run_meta_skill(
    provider: Any,
    model: str,
    prev_skill: str,
    curr_skill: str,
    comparison_pairs: list[dict],
    *,
    prev_meta_skill_content: str = "",
    system_prompt: str | None = None,
) -> dict | None:
    """Produce updated optimizer-side meta skill from adjacent epochs.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider.
    model : str
        Model name for the optimizer call.
    prev_skill : str
        The previous epoch's last-step skill document.
    curr_skill : str
        The current epoch's last-step skill document.
    comparison_pairs : list[dict]
        Per-sample comparison dicts (from slow_update.build_comparison_pairs).
    prev_meta_skill_content : str
        The meta skill content from the previous epoch (if any).
    system_prompt : str | None
        Custom override; if None, loads ``meta_skill.txt`` from file.

    Returns
    -------
    dict | None
        ``{"reasoning": "...", "meta_skill_content": "..."}`` on success,
        or None on failure.
    """
    actual_system = system_prompt
    if actual_system is None:
        try:
            actual_system = load_prompt("meta_skill")
        except FileNotFoundError:
            logger.warning("meta_skill prompt not found; skipping meta skill update")
            return None

    # Truncate skill displays to avoid token overflow
    prev_skill_display = prev_skill
    if len(prev_skill_display) > 6000:
        prev_skill_display = prev_skill_display[:6000] + "\n...[truncated]..."

    curr_skill_display = curr_skill
    if len(curr_skill_display) > 6000:
        curr_skill_display = curr_skill_display[:6000] + "\n...[truncated]..."

    prev_meta_section = (
        prev_meta_skill_content.strip()
        if prev_meta_skill_content and prev_meta_skill_content.strip()
        else "(No previous optimizer meta skill — this is the first update.)"
    )

    # Build comparison summary — reuse slow_update.format_comparison_text
    # (mirrors official: ``from skillopt.optimizer.slow_update import format_comparison_text``)
    comparison_text = format_comparison_text(comparison_pairs)

    user = (
        f"## Previous Epoch Last-Step Skill\n{prev_skill_display}\n\n"
        f"## Current Epoch Last-Step Skill\n{curr_skill_display}\n\n"
        f"## Previous Optimizer Meta Skill\n"
        f"The following optimizer memory was available during the current epoch. "
        f"Reflect on whether it improved or harmed the quality of edits.\n\n"
        f"{prev_meta_section}\n\n"
        f"## Longitudinal Comparison (same tasks, two last-step skills)\n"
        f"{comparison_text}"
    )

    try:
        response = await _call_llm(
            provider, model, actual_system, user,
            max_tokens=3072, stage="meta_skill",
        )
        if response:
            result = _extract_json(response)
            if result and result.get("meta_skill_content"):
                return {
                    "reasoning": str(result.get("reasoning", "")).strip(),
                    "meta_skill_content": str(result["meta_skill_content"]).strip(),
                }
    except Exception as exc:
        logger.error("Meta skill LLM call failed: {}", exc)

    return None


# NOTE: _format_comparison_summary was removed — we now reuse
# ``slow_update.format_comparison_text`` to stay aligned with the official
# implementation (``from skillopt.optimizer.slow_update import format_comparison_text``).
