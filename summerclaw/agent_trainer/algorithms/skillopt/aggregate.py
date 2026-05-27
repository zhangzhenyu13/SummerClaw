"""SkillOpt Aggregate stage — hierarchical patch merging.

Ported from SkillOpt's ``gradient/aggregate.py`` with SummerClaw LLM
provider adaptation.  Supports all three update modes and injects
meta_skill_context for cross-epoch optimizer memory.

Failure-driven patches take priority over success-driven ones.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.algorithms.skillopt.reflect import _call_llm, _extract_json, _normalize_response
from summerclaw.agent_trainer.types import Edit, Patch, RawPatch

from .prompts_loader import resolve_prompt
from .update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    set_payload_items,
)


# ── Internal helpers ─────────────────────────────────────────────────────

async def _merge_batch(
    provider: Any,
    model: str,
    skill_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str = "patch",
    meta_skill_context: str = "",
    level: int = 1,
) -> dict:
    """Call LLM to merge a batch of patches into one."""
    mode = normalize_update_mode(update_mode)
    pkey = payload_key(mode)

    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n{patches_text}"
    )
    if meta_skill_context.strip():
        user = f"## Optimizer Meta Skill\n{meta_skill_context}\n\n{user}"

    max_tokens = 64000 if is_full_rewrite_minibatch_mode(mode) else 4096
    response = await _call_llm(
        provider, model, system_prompt, user,
        max_tokens=max_tokens, stage="merge",
    )
    if response:
        merged = _extract_json(response)
        if merged:
            merged = _normalize_response(merged, pkey)
        if merged and pkey in merged:
            for e in merged.get(pkey, []):
                e["merge_level"] = level
            return merged

    # Fallback: concatenate all payload items
    all_items: list[dict] = []
    for p in patches:
        for item in get_payload_items(p, mode):
            item.setdefault("merge_level", level)
            all_items.append(item)
    result: dict[str, Any] = {"reasoning": "fallback concatenation"}
    set_payload_items(result, all_items, mode)
    return result


async def _hierarchical_merge(
    provider: Any,
    model: str,
    skill_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str = "patch",
    batch_size: int = 8,
    label: str = "",
    meta_skill_context: str = "",
) -> dict:
    """Hierarchically merge N patches using the given system prompt."""
    mode = normalize_update_mode(update_mode)
    pkey = payload_key(mode)

    if not patches:
        result: dict[str, Any] = {"reasoning": "no patches"}
        set_payload_items(result, [], mode)
        return result
    if len(patches) == 1:
        return patches[0]

    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        batches = [current[i : i + batch_size] for i in range(0, len(current), batch_size)]

        logger.info(
            "[aggregate {}] level={} {} patches → {} batches",
            label, level, len(current), len(batches),
        )

        tasks = []
        for batch in batches:
            if len(batch) == 1:
                # Pass through single-patch batches
                async def _passthrough(b=batch):
                    return b[0]
                tasks.append(_passthrough())
            else:
                tasks.append(_merge_batch(
                    provider, model, skill_content, batch, system_prompt,
                    update_mode=mode,
                    meta_skill_context=meta_skill_context,
                    level=level,
                ))

        current = await asyncio.gather(*tasks)

    return current[0]


# ── Public API ───────────────────────────────────────────────────────────

async def merge_patches(
    provider: Any,
    model: str,
    skill_content: str,
    failure_patches: list[dict],
    success_patches: list[dict],
    batch_size: int = 8,
    *,
    update_mode: str = "patch",
    meta_skill_context: str = "",
) -> Patch:
    """Failure-first hierarchical merge with support count tracking.

    1. Merge failure patches independently
    2. Merge success patches independently
    3. Final merge: combine both groups with failure priority

    Parameters
    ----------
    update_mode : str
        One of "patch", "rewrite_from_suggestions", "full_rewrite_minibatch".
    meta_skill_context : str
        Cross-epoch optimizer memory context.

    Returns a merged Patch.
    """
    mode = normalize_update_mode(update_mode)
    pkey = payload_key(mode)

    logger.info(
        "[AGGREGATE] failure={} success={} (mode={})",
        len(failure_patches), len(success_patches), mode,
    )

    # Resolve merge prompts per mode
    merge_failure_prompt = resolve_prompt(None, "merge_failure", mode)
    merge_success_prompt = resolve_prompt(None, "merge_success", mode)
    merge_final_prompt = resolve_prompt(None, "merge_final", mode)

    failure_merged = await _hierarchical_merge(
        provider, model, skill_content,
        failure_patches, merge_failure_prompt,
        update_mode=mode,
        batch_size=batch_size, label="failure",
        meta_skill_context=meta_skill_context,
    )

    success_merged = await _hierarchical_merge(
        provider, model, skill_content,
        success_patches, merge_success_prompt,
        update_mode=mode,
        batch_size=batch_size, label="success",
        meta_skill_context=meta_skill_context,
    )

    f_items = get_payload_items(failure_merged, mode)
    s_items = get_payload_items(success_merged, mode)

    if not f_items and not s_items:
        return Patch(reasoning="no updates from either group")
    if not s_items:
        return Patch.from_dict({
            "reasoning": failure_merged.get("reasoning", ""),
            "edits": f_items,
        })
    if not f_items:
        return Patch.from_dict({
            "reasoning": success_merged.get("reasoning", ""),
            "edits": s_items,
        })

    # Final merge: failure + success with priority
    combined = [failure_merged, success_merged]
    combined_text = json.dumps(combined, ensure_ascii=False, indent=2)
    lbl = payload_label(mode)
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Two pre-merged patch groups to combine\n"
        f"Group 1 (failure-driven, HIGH priority): {len(f_items)} {lbl}\n"
        f"Group 2 (success-driven, lower priority): {len(s_items)} {lbl}\n\n"
        f"{combined_text}"
    )
    if meta_skill_context.strip():
        user = f"## Optimizer Meta Skill\n{meta_skill_context}\n\n{user}"

    max_tokens = 64000 if is_full_rewrite_minibatch_mode(mode) else 4096
    response = await _call_llm(
        provider, model, merge_final_prompt, user,
        max_tokens=max_tokens, stage="merge",
    )
    if response:
        final = _extract_json(response)
        if final:
            final = _normalize_response(final, pkey)
        if final and pkey in final:
            logger.info(
                "[aggregate final] {}+{} → {} {}",
                len(f_items), len(s_items), len(final[pkey]), lbl,
            )
            return Patch.from_dict({
                "reasoning": final.get("reasoning", ""),
                "edits": final[pkey],
            })

    # Fallback: failure first, then success
    return Patch(
        edits=[Edit.from_dict(e) for e in f_items + s_items],
        reasoning="fallback: failure first, then success",
    )
