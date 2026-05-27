"""SkillOpt Reflect engine — minibatch trajectory analysis.

Ported from SkillOpt's ``gradient/reflect.py`` with SummerClaw LLM
provider adaptation.  Uses the same two-level prompt priority system
and supports all three update modes (patch / rewrite / full_rewrite).

Two-level prompt priority system:

1. **Custom prompt** (caller passes non-None) — used as-is.
2. **Built-in default prompt** — loaded from ``templates/trainer/skillopt/``
   with mode-specific variants selected automatically.

Public API
----------
- :func:`fmt_trajectory`               -- format one conversation into text
- :func:`fmt_minibatch_trajectories`   -- format multiple trajectories
- :func:`run_error_analyst_minibatch`   -- one LLM call for failures
- :func:`run_success_analyst_minibatch` -- one LLM call for successes
- :func:`run_minibatch_reflect`         -- full reflect stage dispatcher
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.types import Patch, RawPatch, RolloutResult

from .prompts_loader import resolve_prompt
from .update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_key,
)

_MAX_TRAJ_CHARS = 12_000


def _normalize_response(result: dict, pkey: str) -> dict:
    """Normalize an LLM JSON response that may wrap the payload inside a
    ``patch`` object (official prompt schema) or return it flat.

    Returns a flat dict with ``reasoning`` and *pkey* at the top level.
    Extra fields (``batch_size``, ``failure_summary``, ``success_patterns``)
    are preserved.
    """
    if not isinstance(result, dict):
        return result
    # Official format: {"patch": {"reasoning": ..., "edits": [...]}, ...}
    if "patch" in result and isinstance(result["patch"], dict):
        inner = result["patch"]
        normalized = {k: v for k, v in result.items() if k != "patch"}
        normalized["reasoning"] = inner.get("reasoning", result.get("reasoning", ""))
        if pkey in inner:
            normalized[pkey] = inner[pkey]
        return normalized
    return result


# ── Trajectory formatting ────────────────────────────────────────────────

def _clip_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value)[:limit]


def fmt_trajectory(
    conversation: list[dict],
    max_chars: int = _MAX_TRAJ_CHARS,
) -> str:
    """Format a conversation list into analyst-readable text.

    Handles OpenAI-style messages with role=assistant/tool/user/system,
    as well as tool_call and step-action records.
    """
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {_clip_text(item, 500)}")
            continue
        if item.get("type") == "tool_call":
            cmd = _clip_text(item.get("cmd"), 500)
            obs = _clip_text(item.get("obs"), 800)
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = _clip_text(item.get("reasoning"), 300)
            action = _clip_text(item.get("action"), 200)
            feedback = _clip_text(item.get("env_feedback"), 500)
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif item.get("role") == "system":
            # Post-execution verification / enrichment info
            msg = _clip_text(item.get("content"), 2000)
            lines.append(f"[verification] {msg}")
        elif item.get("role") == "tool":
            content = _clip_text(item.get("content"), 800)
            name = item.get("name", "tool")
            lines.append(f"[tool:{name}] {content}")
        elif item.get("role") == "assistant":
            content = _clip_text(item.get("content"), 500)
            tool_calls = item.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    lines.append(
                        f"[call] {fn.get('name', '?')}({_clip_text(fn.get('arguments'), 200)})"
                    )
            if content:
                lines.append(f"[assistant] {content}")
        else:
            msg = _clip_text(item.get("content"), 500)
            role = item.get("role", "agent")
            lines.append(f"[{role}] {msg}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n...[middle truncated]...\n" + tail
    return text


def fmt_minibatch_trajectories(
    results: list[RolloutResult],
    *,
    include_target_context: bool = False,
) -> str:
    """Format multiple rollout results for minibatch analyst consumption.

    Parameters
    ----------
    include_target_context : bool
        When True, append target system/user prompts from each result
        (mirrors official ``fmt_minibatch_trajectories`` which reads
        ``target_system_prompt.txt`` from disk).
    """
    parts: list[str] = []
    for idx, result in enumerate(results, 1):
        traj_text = fmt_trajectory(result.trajectory)
        header = (
            f"### Trajectory {idx} (id={result.id})\n"
            f"Task: {result.task_description or result.question}\n"
            f"Task type: {result.task_type}\n"
        )
        if result.fail_reason:
            header += f"Failure reason: {result.fail_reason}\n"
        header += f"Hard: {result.hard}, Soft: {result.soft:.2f}\n"
        header += f"Turns: {result.n_turns}\n"

        if result.reference_text:
            header += f"\n#### Reference\n{result.reference_text[:4000]}\n"
        if result.predicted_answer:
            header += f"\n#### Predicted Answer\n{result.predicted_answer[:2000]}\n"

        # ── Target context (what the agent saw) ──
        if include_target_context:
            target_prompt = getattr(result, "target_system_prompt", "") or ""
            if target_prompt:
                header += f"\n#### Target System Prompt\n{target_prompt[:3000]}\n"
            target_user = getattr(result, "target_user_prompt", "") or ""
            if target_user:
                header += f"\n#### Target User Prompt\n{target_user[:3000]}\n"

        parts.append(header + "\n" + traj_text)

    return "\n\n---\n\n".join(parts)


# ── LLM helpers ──────────────────────────────────────────────────────────

async def _call_llm(
    provider: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
    *,
    retries: int = 3,
    stage: str = "",
    reasoning_effort: str | None = None,
) -> str | None:
    """Call the LLM provider and return the response text.

    Parameters
    ----------
    retries : int
        Number of retry attempts on failure (mirrors official ``chat_optimizer``).
    stage : str
        Pipeline stage label for logging (e.g. ``"reflect"``, ``"merge"``).
    reasoning_effort : str | None
        Optional reasoning effort hint passed to the provider when supported.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            kwargs: dict[str, Any] = dict(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
            )
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            response = await provider.chat_with_retry(**kwargs)
            return response.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "LLM call [stage={}] attempt {}/{} failed: {}",
                    stage, attempt, retries, exc,
                )
    logger.error("LLM call [stage={}] failed after {} retries: {}", stage, retries, last_exc)
    return None


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response (handles markdown fences)."""
    text = text.strip()
    # Try to find JSON in markdown code block
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except (json.JSONDecodeError, ValueError):
                continue
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ── Minibatch analysts ───────────────────────────────────────────────────

async def run_error_analyst_minibatch(
    provider: Any,
    model: str,
    skill_content: str,
    results: list[RolloutResult],
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
) -> RawPatch | None:
    """Analyze a minibatch of failed trajectories in one LLM call.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider.
    model : str
        Model name.
    skill_content : str
        Current skill document.
    results : list[RolloutResult]
        Failed rollout results.
    edit_budget : int
        Maximum edits/suggestions (L).
    system_prompt : str | None
        Custom override; if None, loads from file via resolve_prompt.
    step_buffer_context : str
        Previous steps context within this epoch.
    meta_skill_context : str
        Cross-epoch optimizer memory context.
    update_mode : str
        One of "patch", "rewrite_from_suggestions", "full_rewrite_minibatch".
    """
    mode = normalize_update_mode(update_mode)
    system = resolve_prompt(system_prompt, "error_analyst", mode)
    pkey = payload_key(mode)

    trajectories_text = fmt_minibatch_trajectories(results)
    if not trajectories_text.strip():
        return None

    user = f"## Current Skill\n{skill_content}\n\n"
    user += f"## Edit Budget\nProduce at most L={edit_budget} items.\n\n"
    if meta_skill_context.strip():
        user += f"## Optimizer Meta Skill\n{meta_skill_context}\n\n"
    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"
    user += f"## Failed Trajectories ({len(results)} total)\n{trajectories_text}"

    max_tokens = 64000 if is_full_rewrite_minibatch_mode(mode) else 4096
    response = await _call_llm(
        provider, model, system, user,
        max_tokens=max_tokens, stage="reflect",
    )
    if not response:
        return None

    result = _extract_json(response)
    if result:
        result = _normalize_response(result, pkey)
    if result and pkey in result:
        return RawPatch(
            patch=Patch.from_dict({
                "reasoning": result.get("reasoning", ""),
                "edits": result.get(pkey, []),
            }),
            source_type="failure",
            batch_size=result.get("batch_size", len(results)),
        )
    return None


async def run_success_analyst_minibatch(
    provider: Any,
    model: str,
    skill_content: str,
    results: list[RolloutResult],
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
) -> RawPatch | None:
    """Analyze a minibatch of successful trajectories in one LLM call.

    See :func:`run_error_analyst_minibatch` for parameter descriptions.
    """
    mode = normalize_update_mode(update_mode)
    system = resolve_prompt(system_prompt, "success_analyst", mode)
    pkey = payload_key(mode)

    trajectories_text = fmt_minibatch_trajectories(results)
    if not trajectories_text.strip():
        return None

    user = f"## Current Skill\n{skill_content}\n\n"
    user += f"## Edit Budget\nProduce at most L={edit_budget} items.\n\n"
    if meta_skill_context.strip():
        user += f"## Optimizer Meta Skill\n{meta_skill_context}\n\n"
    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"
    user += f"## Successful Trajectories ({len(results)} total)\n{trajectories_text}"

    max_tokens = 64000 if is_full_rewrite_minibatch_mode(mode) else 4096
    response = await _call_llm(
        provider, model, system, user,
        max_tokens=max_tokens, stage="reflect",
    )
    if not response:
        return None

    result = _extract_json(response)
    if result:
        result = _normalize_response(result, pkey)
    if result and pkey in result:
        return RawPatch(
            patch=Patch.from_dict({
                "reasoning": result.get("reasoning", ""),
                "edits": result.get(pkey, []),
            }),
            source_type="success",
            batch_size=result.get("batch_size", len(results)),
        )
    return None


# ── Minibatch reflect dispatcher ─────────────────────────────────────────

def _split_minibatches(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def run_minibatch_reflect(
    provider: Any,
    model: str,
    results: list[RolloutResult],
    skill_content: str,
    patches_dir: str,
    workers: int = 4,
    minibatch_size: int = 5,
    edit_budget: int = 4,
    *,
    error_system: str | None = None,
    success_system: str | None = None,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
) -> list[RawPatch]:
    """Full minibatch reflect stage: group -> parallel LLM calls -> patches.

    Separates failure/success trajectories, splits each into minibatches,
    runs all minibatches concurrently, and saves patch files.

    Parameters
    ----------
    meta_skill_context : str
        Cross-epoch optimizer memory context.
    update_mode : str
        One of "patch", "rewrite_from_suggestions", "full_rewrite_minibatch".
    """
    os.makedirs(patches_dir, exist_ok=True)

    failures = [r for r in results if not r.hard]
    successes = [r for r in results if r.hard]

    fail_batches = _split_minibatches(failures, minibatch_size)
    succ_batches = _split_minibatches(successes, minibatch_size)

    logger.info(
        "[REFLECT minibatch] failure={}→{} groups  success={}→{} groups  (M={}, L={}, mode={})",
        len(failures), len(fail_batches),
        len(successes), len(succ_batches),
        minibatch_size, edit_budget, normalize_update_mode(update_mode),
    )

    semaphore = asyncio.Semaphore(workers)

    async def _do_fail(idx: int, batch: list[RolloutResult]) -> RawPatch | None:
        async with semaphore:
            patch = await run_error_analyst_minibatch(
                provider, model, skill_content, batch,
                edit_budget=edit_budget,
                system_prompt=error_system,
                step_buffer_context=step_buffer_context,
                meta_skill_context=meta_skill_context,
                update_mode=update_mode,
            )
            if patch:
                path = os.path.join(patches_dir, f"minibatch_fail_{idx:03d}.json")
                with open(path, "w") as f:
                    json.dump(patch.to_dict(), f, ensure_ascii=False, indent=2)
            return patch

    async def _do_succ(idx: int, batch: list[RolloutResult]) -> RawPatch | None:
        async with semaphore:
            patch = await run_success_analyst_minibatch(
                provider, model, skill_content, batch,
                edit_budget=edit_budget,
                system_prompt=success_system,
                step_buffer_context=step_buffer_context,
                meta_skill_context=meta_skill_context,
                update_mode=update_mode,
            )
            if patch:
                path = os.path.join(patches_dir, f"minibatch_succ_{idx:03d}.json")
                with open(path, "w") as f:
                    json.dump(patch.to_dict(), f, ensure_ascii=False, indent=2)
            return patch

    tasks: list = []
    for idx, batch in enumerate(fail_batches):
        tasks.append(_do_fail(idx, batch))
    for idx, batch in enumerate(succ_batches):
        tasks.append(_do_succ(idx, batch))

    raw_results = await asyncio.gather(*tasks)
    return [p for p in raw_results if p is not None]
