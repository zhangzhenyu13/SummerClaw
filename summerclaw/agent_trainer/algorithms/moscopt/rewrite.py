"""Optimizer-driven full skill rewrite from selected revise_suggestions.

Ported from SkillOpt's ``optimizer/rewrite.py`` with SummerClaw LLM
provider adaptation.

In ``rewrite_from_suggestions`` mode, after the select stage produces
ranked suggestions, this module calls the LLM to generate a complete
new skill document that integrates the selected suggestions.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.algorithms.skillopt.reflect import _call_llm, _extract_json
from summerclaw.agent_trainer.algorithms.skillopt.update_modes import get_payload_items

from .prompts_loader import load_prompt


async def rewrite_skill_from_suggestions(
    provider: Any,
    model: str,
    skill_content: str,
    patch: dict,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    env: str | None = None,
    reasoning_effort: str = "high",
    max_completion_tokens: int = 64000,
) -> dict | None:
    """Rewrite the full skill document by integrating selected suggestions.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider.
    model : str
        Model name.
    skill_content : str
        Current skill document.
    patch : dict
        Patch dict containing ``revise_suggestions`` payload.
    system_prompt : str | None
        Custom override; if None, loads ``rewrite_skill.txt``.
    step_buffer_context : str
        Previous steps context within this epoch.
    env : str | None
        Optional environment label for prompt variant selection
        (mirrors official ``load_prompt("rewrite_skill", env=env)``).
    reasoning_effort : str
        Reasoning effort hint for the LLM call (e.g. ``"high"``).
    max_completion_tokens : int
        Max tokens for the rewrite response (needs to be large).

    Returns
    -------
    dict | None
        ``{"new_skill": "...", "reasoning": "...", "change_summary": [...]}``
        on success, or None on failure.
    """
    suggestions = get_payload_items(patch, "rewrite_from_suggestions")
    if not suggestions:
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Selected Revise Suggestions ({len(suggestions)} total)\n"
        f"{json.dumps(suggestions, ensure_ascii=False, indent=2)}\n\n"
    )
    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"
    user += (
        "Rewrite the full skill document so it integrates the selected suggestions. "
        "Return the complete new skill in `new_skill`."
    )

    actual_system = system_prompt
    if actual_system is None:
        try:
            actual_system = load_prompt("rewrite_skill", env=env)
        except FileNotFoundError:
            logger.warning("rewrite_skill prompt not found")
            return None

    try:
        response = await _call_llm(
            provider, model, actual_system, user,
            max_tokens=max_completion_tokens,
            stage="rewrite",
            reasoning_effort=reasoning_effort,
        )
        if response:
            result = _extract_json(response)
            if result and str(result.get("new_skill", "")).strip():
                result["new_skill"] = str(result["new_skill"]).rstrip() + "\n"
                if "change_summary" not in result or not isinstance(result["change_summary"], list):
                    result["change_summary"] = []
                return result
    except Exception as exc:
        logger.error("Rewrite LLM call failed: {}", exc)

    return None
