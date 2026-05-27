"""SkillOpt slow update — longitudinal comparison + LLM-driven guidance.

Ported from SkillOpt's ``optimizer/slow_update.py`` with SummerClaw LLM
provider adaptation.

At epoch boundaries, compares per-item results between consecutive skill
versions and calls an LLM to produce free-form guidance written into a
**protected** section of the skill document.  This section cannot be
modified by step-level analyst edits — only the slow update process
overwrites it.

Public API
----------
- :func:`inject_empty_slow_update_field` — add empty placeholder (epoch 1)
- :func:`extract_slow_update_field`      — read current content
- :func:`replace_slow_update_field`      — overwrite content
- :func:`has_slow_update_field`          — check if markers are present
- :func:`build_comparison_pairs`         — build structured comparison entries
- :func:`format_comparison_text`         — format pairs for LLM consumption
- :func:`run_slow_update`               — LLM call to produce guidance
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.algorithms.skillopt.reflect import _call_llm, _extract_json
from summerclaw.agent_trainer.types import RolloutResult

from .prompts_loader import load_prompt

# ── Protected field markers ─────────────────────────────────────────────

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

_MAX_TRAJ_CHARS = 3000


# ── Result dataclass ────────────────────────────────────────────────────

@dataclass
class SlowUpdateResult:
    """Output of the epoch-level slow update stage."""

    reasoning: str = ""
    guidance: str = ""
    action: str = ""
    prev_hard: float | None = None
    curr_hard: float | None = None


# ── Field manipulation helpers ──────────────────────────────────────────

def has_slow_update_field(skill: str) -> bool:
    """Check if the skill document contains SLOW_UPDATE markers."""
    return SLOW_UPDATE_START in skill and SLOW_UPDATE_END in skill


def extract_slow_update_field(skill: str) -> str | None:
    """Extract the content within the SLOW_UPDATE region, if present."""
    start = skill.find(SLOW_UPDATE_START)
    end = skill.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return None
    content_start = start + len(SLOW_UPDATE_START)
    return skill[content_start:end].strip()


def _strip_all_slow_update_fields(skill: str) -> str:
    """Remove every SLOW_UPDATE_START/END pair (and content between) from *skill*."""
    while True:
        start = skill.find(SLOW_UPDATE_START)
        if start == -1:
            break
        end = skill.find(SLOW_UPDATE_END, start)
        if end == -1:
            # Orphan start marker — remove it
            skill = skill[:start] + skill[start + len(SLOW_UPDATE_START):]
            break
        skill = skill[:start] + skill[end + len(SLOW_UPDATE_END):]
    # Clean up stray end markers
    skill = skill.replace(SLOW_UPDATE_END, "")
    # Collapse excess blank lines left behind
    while "\n\n\n" in skill:
        skill = skill.replace("\n\n\n", "\n\n")
    return skill.rstrip()


def replace_slow_update_field(skill: str, content: str) -> str:
    """Replace the SLOW_UPDATE region content.

    Removes all existing regions first to guarantee exactly one, then
    appends the new region at the end.
    """
    skill = _strip_all_slow_update_fields(skill)
    block = (
        f"\n\n{SLOW_UPDATE_START}\n"
        f"{content.strip()}\n"
        f"{SLOW_UPDATE_END}\n"
    )
    return skill + block


def inject_empty_slow_update_field(skill: str) -> str:
    """Inject an empty SLOW_UPDATE region if not already present."""
    if has_slow_update_field(skill):
        return skill
    block = f"\n\n{SLOW_UPDATE_START}\n{SLOW_UPDATE_END}\n"
    return skill.rstrip() + block


# ── Trajectory formatting ───────────────────────────────────────────────

def _clip_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value)[:limit]


def _format_trajectory(trajectory: list[dict], max_chars: int = _MAX_TRAJ_CHARS) -> str:
    """Format a rollout trajectory into readable text."""
    lines: list[str] = []
    for entry in trajectory:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "tool_call":
            cmd = _clip_text(entry.get("cmd"), 500)
            obs = _clip_text(entry.get("obs"), 800)
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in entry and "env_feedback" in entry:
            step = entry.get("step", "?")
            reasoning = _clip_text(entry.get("reasoning"), 300)
            action = _clip_text(entry.get("action"), 200)
            feedback = _clip_text(entry.get("env_feedback"), 500)
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif entry.get("role") == "assistant":
            content = _clip_text(entry.get("content"), 500)
            if content:
                lines.append(f"[assistant] {content}")
        elif entry.get("role") not in ("system",):
            msg = _clip_text(entry.get("content"), 500)
            role = entry.get("role", "agent")
            lines.append(f"[{role}] {msg}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


# ── Structured comparison pairs ─────────────────────────────────────────

def build_comparison_pairs(
    prev_results: list[RolloutResult],
    curr_results: list[RolloutResult],
    items: list[dict],
) -> list[dict]:
    """Build a structured list of per-sample comparison entries.

    Each entry bundles the original item, both rollout results, the change
    category, and both trajectories into one dict.

    Returns
    -------
    list[dict]
        One dict per sample with keys:
        ``id, task, category, prev, curr, prev_trajectory, curr_trajectory``
    """
    prev_by_id = {str(r.id): r for r in prev_results}
    curr_by_id = {str(r.id): r for r in curr_results}

    pairs: list[dict] = []
    for item in items:
        tid = str(item.get("id", ""))
        prev = prev_by_id.get(tid)
        curr = curr_by_id.get(tid)
        if prev is None or curr is None:
            continue

        prev_ok = bool(prev.hard)
        curr_ok = bool(curr.hard)

        if not prev_ok and curr_ok:
            category = "improved"
        elif prev_ok and not curr_ok:
            category = "regressed"
        elif not prev_ok and not curr_ok:
            category = "persistent_fail"
        else:
            category = "stable_success"

        pairs.append({
            "id": tid,
            "task": item.get("question", item.get("task_description", tid)),
            "category": category,
            "prev": {
                "hard": int(prev_ok),
                "soft": prev.soft,
                "predicted_answer": prev.predicted_answer or "N/A",
                "fail_reason": prev.fail_reason,
            },
            "curr": {
                "hard": int(curr_ok),
                "soft": curr.soft,
                "predicted_answer": curr.predicted_answer or "N/A",
                "fail_reason": curr.fail_reason,
            },
            "prev_trajectory": _format_trajectory(prev.trajectory),
            "curr_trajectory": _format_trajectory(curr.trajectory),
        })

    return pairs


def save_comparison_pairs(pairs: list[dict], out_path: str) -> None:
    """Persist comparison pairs to JSON (without trajectory text to save space)."""
    slim = []
    for p in pairs:
        slim.append({
            "id": p["id"],
            "task": p["task"][:300] if isinstance(p["task"], str) else str(p["task"])[:300],
            "category": p["category"],
            "prev": p["prev"],
            "curr": p["curr"],
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)


def format_comparison_text(pairs: list[dict]) -> str:
    """Format structured comparison pairs into LLM-readable text."""
    by_cat: dict[str, list[dict]] = {
        "regressed": [],
        "persistent_fail": [],
        "improved": [],
        "stable_success": [],
    }
    for p in pairs:
        by_cat.setdefault(p["category"], []).append(p)

    total = len(pairs)
    parts = [
        f"## Longitudinal Comparison Summary\n"
        f"Total samples: {total}\n"
        f"- Improved (wrong→right): {len(by_cat['improved'])}\n"
        f"- Regressed (right→wrong): {len(by_cat['regressed'])}\n"
        f"- Persistent failures (wrong→wrong): {len(by_cat['persistent_fail'])}\n"
        f"- Stable successes (right→right): {len(by_cat['stable_success'])}\n"
    ]

    categories = [
        ("regressed", "Regressions (right→wrong) — HIGHEST PRIORITY", True),
        ("persistent_fail", "Persistent Failures (wrong→wrong)", True),
        ("improved", "Improvements (wrong→right)", True),
        ("stable_success", "Stable Successes (right→right)", False),
    ]

    for cat_key, label, show_traj in categories:
        entries = by_cat[cat_key]
        if not entries:
            parts.append(f"### {label}\n(none)\n")
            continue

        lines = [f"### {label}"]
        for e in entries[:15]:  # limit entries per category
            prev = e["prev"]
            curr = e["curr"]
            lines.append(
                f"\n#### Task {e['id']}\n"
                f"Task: {str(e.get('task', ''))[:200]}\n"
                f"Prev: hard={prev['hard']}, soft={prev['soft']:.2f}, "
                f"answer={str(prev['predicted_answer'])[:100]}\n"
                f"Curr: hard={curr['hard']}, soft={curr['soft']:.2f}, "
                f"answer={str(curr['predicted_answer'])[:100]}\n"
            )
            if curr.get("fail_reason"):
                lines.append(f"Fail reason: {curr['fail_reason'][:200]}\n")
            if show_traj and e.get("curr_trajectory"):
                traj = e["curr_trajectory"][:1500]
                lines.append(f"Current trajectory:\n{traj}\n")
        parts.append("\n".join(lines))

    return "\n".join(parts)


# ── Main LLM-driven slow update ────────────────────────────────────────

async def run_slow_update(
    provider: Any,
    model: str,
    prev_skill: str,
    curr_skill: str,
    comparison_pairs: list[dict],
    *,
    system_prompt: str | None = None,
) -> SlowUpdateResult:
    """Produce epoch-level slow update guidance via LLM analysis.

    Compares per-item results between consecutive skill versions, then
    calls an LLM to analyze regressions, improvements, and persistent
    failures, producing free-form guidance for the skill's protected region.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider.
    model : str
        Model name for the optimizer call.
    prev_skill : str
        Previous epoch's skill document.
    curr_skill : str
        Current epoch's skill document.
    comparison_pairs : list[dict]
        Output of :func:`build_comparison_pairs`.
    system_prompt : str | None
        Custom override; if None, loads ``slow_update.txt``.

    Returns
    -------
    SlowUpdateResult
        Result with reasoning, guidance text, and action label.
    """
    if not comparison_pairs:
        return SlowUpdateResult(
            reasoning="no comparison pairs available",
            action="skip",
        )

    # Compute basic statistics
    improved = sum(1 for p in comparison_pairs if p["category"] == "improved")
    regressed = sum(1 for p in comparison_pairs if p["category"] == "regressed")
    persistent_fail = sum(1 for p in comparison_pairs if p["category"] == "persistent_fail")
    stable_success = sum(1 for p in comparison_pairs if p["category"] == "stable_success")

    prev_hard_vals = [p["prev"]["hard"] for p in comparison_pairs]
    curr_hard_vals = [p["curr"]["hard"] for p in comparison_pairs]
    prev_hard_mean = sum(prev_hard_vals) / max(len(prev_hard_vals), 1)
    curr_hard_mean = sum(curr_hard_vals) / max(len(curr_hard_vals), 1)

    # Determine action
    delta = curr_hard_mean - prev_hard_mean
    if delta > 0.01:
        action = "improving"
    elif delta < -0.01:
        action = "regressing"
    else:
        action = "stable"

    # Load prompt
    actual_system = system_prompt
    if actual_system is None:
        try:
            actual_system = load_prompt("slow_update")
        except FileNotFoundError:
            logger.warning("slow_update prompt not found; returning statistical summary only")
            return SlowUpdateResult(
                reasoning=f"Epoch comparison: {len(comparison_pairs)} items. "
                          f"Improved: {improved}, Regressed: {regressed}, "
                          f"Persistent fail: {persistent_fail}, Stable success: {stable_success}. "
                          f"Mean hard: {prev_hard_mean:.3f} → {curr_hard_mean:.3f}",
                action=action,
                prev_hard=prev_hard_mean,
                curr_hard=curr_hard_mean,
            )

    # Truncate skill displays
    prev_display = prev_skill[:6000] + "\n...[truncated]..." if len(prev_skill) > 6000 else prev_skill
    curr_display = curr_skill[:6000] + "\n...[truncated]..." if len(curr_skill) > 6000 else curr_skill

    comparison_text = format_comparison_text(comparison_pairs)

    user = (
        f"## Previous Epoch Skill\n{prev_display}\n\n"
        f"## Current Epoch Skill\n{curr_display}\n\n"
        f"{comparison_text}"
    )

    try:
        response = await _call_llm(
            provider, model, actual_system, user,
            max_tokens=4096, stage="slow_update",
        )
        if response:
            result = _extract_json(response)
            if result and (result.get("guidance") or result.get("slow_update_content")):
                guidance_text = result.get("slow_update_content") or result.get("guidance", "")
                return SlowUpdateResult(
                    reasoning=str(result.get("reasoning", "")).strip(),
                    guidance=str(guidance_text).strip(),
                    action=action,
                    prev_hard=prev_hard_mean,
                    curr_hard=curr_hard_mean,
                )
    except Exception as exc:
        logger.error("Slow update LLM call failed: {}", exc)

    # Fallback: statistical summary only
    return SlowUpdateResult(
        reasoning=f"LLM call failed. Stats: {len(comparison_pairs)} items. "
                  f"Improved: {improved}, Regressed: {regressed}, "
                  f"Persistent fail: {persistent_fail}, Stable success: {stable_success}. "
                  f"Mean hard: {prev_hard_mean:.3f} → {curr_hard_mean:.3f}",
        action=action,
        prev_hard=prev_hard_mean,
        curr_hard=curr_hard_mean,
    )
