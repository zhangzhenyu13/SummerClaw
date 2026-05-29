"""Initial skill loader — file-path or LLM-generated bootstrap skill.

SkillOpt training requires an initial skill document.  This module
supports two loading schemes:

1. **File path** — read an existing Markdown/text skill file from disk.
2. **LLM generation** — sample up to 5 training items, ask the LLM to
   synthesize an initial skill document, and persist it to the training
   output directory.

The :func:`resolve_initial_skill` entry point combines both schemes and
is called by the trainer engine and the dashboard/command bootstrap.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from .reflect import _call_llm


# ── Scheme 1: File-path loading ──────────────────────────────────────────

def load_skill_from_file(path: str | Path) -> str | None:
    """Load an initial skill document from a file path.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the skill file.

    Returns
    -------
    str | None
        Skill content string, or *None* if the file does not exist or
        cannot be read.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        logger.warning("[INIT-SKILL] file not found: {}", p)
        return None
    try:
        content = p.read_text(encoding="utf-8")
        if not content.strip():
            logger.warning("[INIT-SKILL] file is empty: {}", p)
            return None
        logger.info("[INIT-SKILL] loaded from file: {} ({} chars)", p, len(content))
        return content
    except OSError as exc:
        logger.error("[INIT-SKILL] failed to read {}: {}", p, exc)
        return None


# ── Scheme 2: LLM-driven generation ─────────────────────────────────────

_MAX_SEED_ITEMS = 5

_GENERATE_SYSTEM_PROMPT = """\
You are a skill-document author for the SummerClaw agent platform.

Given a small sample of training items (question/answer pairs), produce
an **initial SKILL.md** that will serve as the starting point for
iterative skill optimization (SkillOpt).

## Required Format

The document **must** follow this exact structure:

```
---
name: kebab-case-skill-name
description: >
  [2-4 sentences describing when this skill triggers.
   Include concrete trigger phrases, e.g.
   "Use when the user needs to...", "For example: 'help me...'".]
---

# [Skill Title]

## When to Use
[1-3 sentences: specific trigger conditions and usage scenarios]

## When NOT to Use
[Explicitly list scenarios where this skill should NOT activate,
 to prevent false triggers]

## Execution Steps
1. [Concrete step, name the tool if applicable (exec, read_file, web_search, etc.)]
2. [Next step]
...

## Output Format
[Expected output structure or format, if there is a fixed pattern]

## Key Considerations
- [Pitfalls or gotchas observed in the sample data]
- [Error recovery patterns that work]

## Examples
[Brief example showing input and expected output]
```

## Rules

1. **YAML frontmatter is mandatory.**  The `name` field must be
   lowercase-kebab-case (letters, digits, hyphens only, max 64 chars).
   The `description` field is the primary triggering mechanism — make it
   comprehensive with concrete trigger phrases.
2. **Body must be concise** — 300 to 1500 words total.  Every line must
   carry standalone value.  Prefer concise bullets over verbose prose.
3. **Focus on procedural knowledge** — steps, decisions, tool usage.
   Do NOT include information the agent already knows from its base model.
4. **Output only the SKILL.md content** — no preamble, no commentary,
   no markdown code fences around the whole document.
5. **Write in English** unless the training data is clearly in another
   language, in which case match that language.
"""


def _format_seed_items(items: list[dict]) -> str:
    """Format up to *_MAX_SEED_ITEMS* data items for the LLM prompt."""
    sampled = items[:_MAX_SEED_ITEMS]
    parts: list[str] = []
    for idx, item in enumerate(sampled, 1):
        q = item.get("question", item.get("input", item.get("prompt", "")))
        answers = item.get("answers", item.get("answer", item.get("output", "")))
        ctx = item.get("context", "")
        if isinstance(answers, list):
            answers = " | ".join(str(a) for a in answers)
        block = f"### Item {idx}\n**Question:** {q}\n**Expected Answer:** {answers}"
        if ctx:
            block += f"\n**Context:** {ctx}"
        parts.append(block)
    return "\n\n".join(parts)


async def generate_initial_skill_from_data(
    provider: Any,
    model: str,
    items: list[dict],
    out_dir: str | Path,
    *,
    max_items: int = _MAX_SEED_ITEMS,
) -> str:
    """Use the LLM to synthesize an initial skill from sample data.

    The generated Markdown is written to ``<out_dir>/skills/skill_v0000.md``
    and also returned as a string.

    Parameters
    ----------
    provider : LLMProvider
        SummerClaw LLM provider (must support ``chat_with_retry``).
    model : str
        Model name to use for generation.
    items : list[dict]
        Training items — at most *max_items* will be sampled.
    out_dir : str | Path
        Training output directory; the skill is saved under
        ``<out_dir>/skills/skill_v0000.md``.
    max_items : int
        Maximum number of items to send to the LLM (default 5).

    Returns
    -------
    str
        The generated skill document content.

    Raises
    ------
    RuntimeError
        If the LLM call fails after retries.
    """
    seed_items = items[:max_items]
    logger.info(
        "[INIT-SKILL] generating from {} data items (model={})",
        len(seed_items), model,
    )

    user_prompt = (
        "Below are sample training items.  Generate a complete SKILL.md "
        "document with YAML frontmatter (name + description) and the "
        "structured body sections (When to Use, When NOT to Use, "
        "Execution Steps, Output Format, Key Considerations, Examples).\n\n"
        + _format_seed_items(seed_items)
    )

    content = await _call_llm(
        provider=provider,
        model=model,
        system=_GENERATE_SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=4096,
        retries=3,
        stage="init_skill",
    )

    if not content or not content.strip():
        raise RuntimeError(
            "LLM failed to generate an initial skill document "
            "(empty response after 3 retries)."
        )

    # Strip leading/trailing markdown fences if present
    content = content.strip()
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    # Persist
    skills_dir = Path(out_dir) / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skills_dir / "skill_v0000.md"
    skill_path.write_text(content, encoding="utf-8")
    logger.info(
        "[INIT-SKILL] generated and saved to {} ({} chars)",
        skill_path, len(content),
    )
    return content


# ── Unified resolver ─────────────────────────────────────────────────────

async def resolve_initial_skill(
    skill_init_path: str,
    provider: Any,
    model: str,
    data_loader: Any,
    out_dir: str | Path,
) -> str:
    """Resolve the initial skill via file path or LLM generation.

    Resolution order:

    1. If *skill_init_path* is a valid, non-empty file, load it.
    2. Otherwise, if *provider* and *data_loader* are available,
       generate a skill from up to 5 sampled training items.
    3. If neither scheme succeeds, return an empty string and log a
       warning — training will likely fail at a later stage.

    Parameters
    ----------
    skill_init_path : str
        Path to an existing skill file (may be empty/missing).
    provider : LLMProvider
        LLM provider instance (required for scheme 2).
    model : str
        Model name for LLM generation.
    data_loader : DataLoader | None
        Loaded data loader with at least a ``train`` split.
    out_dir : str | Path
        Training output directory for persisting the generated skill.

    Returns
    -------
    str
        Initial skill content (may be empty if both schemes fail).
    """
    # ── Scheme 1: file path ─────────────────────────────────────────
    if skill_init_path:
        content = load_skill_from_file(skill_init_path)
        if content:
            return content
        logger.info(
            "[INIT-SKILL] file path '{}' not usable, trying LLM generation",
            skill_init_path,
        )

    # ── Scheme 2: LLM generation ────────────────────────────────────
    if not provider:
        logger.warning("[INIT-SKILL] no LLM provider; cannot auto-generate skill")
        return ""

    if not data_loader:
        logger.warning("[INIT-SKILL] no data loader; cannot auto-generate skill")
        return ""

    try:
        train_items = data_loader.train.items
    except (KeyError, AttributeError):
        logger.warning("[INIT-SKILL] data loader has no train split")
        return ""

    if not train_items:
        logger.warning("[INIT-SKILL] train split is empty")
        return ""

    try:
        return await generate_initial_skill_from_data(
            provider=provider,
            model=model,
            items=train_items,
            out_dir=out_dir,
        )
    except RuntimeError as exc:
        logger.error("[INIT-SKILL] LLM generation failed: {}", exc)
        return ""
