"""Optimizer-driven autonomous update-size decisions.

Ported from SkillOpt's ``optimizer/lr_autonomous.py`` with SummerClaw LLM
provider adaptation.

When the LR scheduler is in ``autonomous`` mode, instead of applying all
proposed edits, the optimizer LLM is asked to decide how many edits
should actually be applied at each step.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.algorithms.skillopt.reflect import _call_llm, _extract_json
from summerclaw.agent_trainer.algorithms.skillopt.update_modes import (
    describe_item,
    get_payload_items,
    payload_label,
)

from .prompts_loader import load_prompt


def _coerce_nonnegative_int(value: Any) -> int | None:
    """Safely coerce a value to a non-negative integer.

    Mirrors the official implementation: accepts bool, int, and
    float-with-integer-value; for other types falls through to
    regex-based text extraction.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return max(0, int(match.group(0)))


async def decide_autonomous_learning_rate(
    provider: Any,
    model: str,
    *,
    skill_content: str,
    merged_patch: dict,
    update_mode: str,
    rollout_hard: float,
    rollout_soft: float,
    rollout_n: int,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    system_prompt: str | None = None,
) -> dict:
    """Ask the optimizer to choose the number of update items for this step.

    The prompt intentionally avoids default budgets, candidate budget lists,
    or scheduler history.  The only hard post-processing is validity: the
    returned integer is clamped to the available item count.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider.
    model : str
        Model name.
    skill_content : str
        Current skill document.
    merged_patch : dict
        The aggregated patch dict with payload items.
    update_mode : str
        One of "patch", "rewrite_from_suggestions", "full_rewrite_minibatch".
    rollout_hard : float
        Hard accuracy from the current rollout.
    rollout_soft : float
        Soft accuracy from the current rollout.
    rollout_n : int
        Number of rollout samples.
    step_buffer_context : str
        Previous steps context within this epoch.
    meta_skill_context : str
        Cross-epoch optimizer memory context.
    system_prompt : str | None
        Custom override; if None, loads ``lr_autonomous.txt``.

    Returns
    -------
    dict
        ``{learning_rate, raw_learning_rate, available_update_items,
        clamped, fallback, reasoning, confidence, risk_notes, raw_response}``
    """
    items = get_payload_items(merged_patch, update_mode)
    available = len(items)
    item_lines = [
        f"[{idx}] {describe_item(item, update_mode, max_chars=700)}"
        for idx, item in enumerate(items)
    ]

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Current Step Evidence\n"
        f"rollout_n={rollout_n}\n"
        f"rollout_hard={rollout_hard:.6f}\n"
        f"rollout_soft={rollout_soft:.6f}\n"
        f"proposed_update_items={available}\n"
        f"update_item_type={payload_label(update_mode)}\n\n"
        f"## Proposed Update Items\n"
        + "\n".join(item_lines)
        + f"\n\nDecide how many proposed update items should be applied now."
    )
    if step_buffer_context.strip():
        user += f"\n\n## Previous Steps in This Epoch\n{step_buffer_context}"
    if meta_skill_context.strip():
        user = f"{meta_skill_context}\n\n{user}"

    actual_system = system_prompt
    if actual_system is None:
        try:
            actual_system = load_prompt("lr_autonomous")
        except FileNotFoundError:
            logger.warning("lr_autonomous prompt not found; using all items")
            return {
                "learning_rate": available,
                "raw_learning_rate": available,
                "available_update_items": available,
                "clamped": False,
                "fallback": True,
                "reasoning": "prompt file missing, using all items",
                "confidence": "",
                "risk_notes": [],
                "raw_response": "",
            }

    response = ""
    parsed: dict | None = None
    decision: int | None = None
    try:
        response = await _call_llm(
            provider, model, actual_system, user,
            max_tokens=2048, stage="lr_autonomous",
        ) or ""
        if response:
            parsed = _extract_json(response)
            if parsed:
                decision = _coerce_nonnegative_int(parsed.get("learning_rate"))
    except Exception as exc:
        logger.error("Autonomous LR LLM call failed: {}", exc)
        parsed = {"error": str(exc)}

    fallback = False
    if decision is None:
        decision = 0
        fallback = True

    chosen = min(decision, available)
    record = {
        "learning_rate": chosen,
        "raw_learning_rate": decision,
        "available_update_items": available,
        "clamped": chosen != decision,
        "fallback": fallback,
        "reasoning": (parsed or {}).get("reasoning", ""),
        "confidence": (parsed or {}).get("confidence", ""),
        "risk_notes": (parsed or {}).get("risk_notes", []),
        "raw_response": response,
    }
    if parsed and "error" in parsed:
        record["error"] = parsed["error"]
    return record
