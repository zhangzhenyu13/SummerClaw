"""SkillOpt gradient clipping — LLM-driven edit ranking and selection.

Ported from SkillOpt's ``optimizer/clip.py`` with SummerClaw LLM
provider adaptation.  Supports all three update modes and injects
meta_skill_context for cross-epoch optimizer memory.

Ranks candidate edits by importance and selects the top-L to apply,
controlling the effective step size.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.algorithms.skillopt.reflect import _call_llm, _extract_json
from summerclaw.agent_trainer.types import Edit, Patch

from .prompts_loader import resolve_prompt
from .update_modes import (
    describe_item,
    get_payload_items,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    set_payload_items,
)


# ── Public API ───────────────────────────────────────────────────────────

async def rank_and_select(
    provider: Any,
    model: str,
    skill_content: str,
    patch: Patch,
    max_edits: int,
    *,
    meta_skill_context: str = "",
    update_mode: str = "patch",
) -> Patch:
    """Rank edits by importance via LLM, then keep top-L.

    If the edit pool is within budget, returns the patch unchanged.
    Otherwise calls the LLM to rank and select the most impactful edits.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider.
    model : str
        Model name.
    skill_content : str
        Current skill document.
    patch : Patch
        Aggregated patch with edits.
    max_edits : int
        Maximum number of edits to keep (the "edit budget").
    meta_skill_context : str
        Cross-epoch optimizer memory context.
    update_mode : str
        One of "patch", "rewrite_from_suggestions", "full_rewrite_minibatch".

    Returns
    -------
    Patch
        Selected patch with ranking details.
    """
    mode = normalize_update_mode(update_mode)
    edits = patch.edits
    if len(edits) <= max_edits:
        return patch

    # Build the edit pool description
    edits_desc: list[str] = []
    for i, edit in enumerate(edits):
        d = edit.to_dict() if isinstance(edit, Edit) else edit
        edits_desc.append(f"[{i}] {describe_item(d, mode, max_chars=500)}")

    lbl = payload_label(mode, title=True)
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## {lbl} Pool ({len(edits)} {payload_label(mode)}, budget={max_edits})\n"
        + "\n".join(edits_desc)
        + f"\n\nSelect the {max_edits} most important {payload_label(mode)}. "
        f"Return their 0-based indices in priority order."
    )

    if meta_skill_context.strip():
        user = f"## Optimizer Meta Skill\n{meta_skill_context}\n\n{user}"

    # Resolve ranking prompt (rewrite variant if applicable)
    prompt_name = "ranking_rewrite" if is_rewrite_mode(mode) else "ranking"
    ranking_prompt = resolve_prompt(None, prompt_name, mode)

    response = await _call_llm(
        provider, model, ranking_prompt, user,
        max_tokens=2048, stage="ranking",
    )
    if response:
        result = _extract_json(response)
        if result and "selected_indices" in result:
            indices = result["selected_indices"]
            selected: list[Edit] = []
            seen: set[int] = set()
            for idx in indices:
                if (
                    isinstance(idx, int)
                    and 0 <= idx < len(edits)
                    and idx not in seen
                ):
                    selected.append(edits[idx])
                    seen.add(idx)
                if len(selected) >= max_edits:
                    break
            if selected:
                logger.info(
                    "[SELECT] ranked {}/{} → selected {} {}",
                    len(edits), len(edits), len(selected), payload_label(mode),
                )
                return Patch(
                    edits=selected,
                    reasoning=patch.reasoning
                    + f" [optimizer-ranked: selected {len(selected)}/{len(edits)} {payload_label(mode)}]",
                    ranking_details=result,
                )

    # Fallback: simple truncation
    logger.info(
        "[SELECT] fallback truncated {}→{} {}",
        len(edits), max_edits, payload_label(mode),
    )
    return Patch(
        edits=edits[:max_edits],
        reasoning=patch.reasoning + f" [fallback truncated {len(edits)}->{max_edits} {payload_label(mode)}]",
    )
