"""Prompt file loader for SkillOpt algorithm stages.

Loads prompt text from ``templates/trainer/skillopt/<name>.txt`` files.
Supports a two-level priority system:

1. Custom prompt passed directly (non-None) — used as-is.
2. Built-in default prompt loaded from file.

The :func:`resolve_prompt` helper combines a custom override with a
default name and an optional ``update_mode`` suffix to select the
correct prompt variant.
"""
from __future__ import annotations

import functools
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "templates" / "trainer" / "skillopt"


@functools.lru_cache(maxsize=64)
def load_prompt(name: str, *, env: str | None = None) -> str:
    """Load a prompt by *name* from the ``templates/trainer/skillopt/`` directory.

    Parameters
    ----------
    name : str
        Prompt file basename (without ``.txt`` extension).
    env : str | None
        Optional environment label.  When set, tries loading
        ``<name>_<env>.txt`` first, falling back to ``<name>.txt``.
        Mirrors official ``load_prompt(name, env=env)``.

    Returns
    -------
    str
        Prompt text.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    """
    if env:
        variant = _PROMPT_DIR / f"{name}_{env}.txt"
        if variant.is_file():
            return variant.read_text(encoding="utf-8")

    path = _PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_prompt(
    custom: str | None,
    default_name: str,
    update_mode: str = "patch",
) -> str:
    """Resolve a prompt from a custom override or a built-in default.

    Resolution order:

    1. If *custom* is not None, return it directly.
    2. Try loading ``<default_name>_<mode_suffix>`` (if a mode-specific
       variant exists).
    3. Fall back to ``<default_name>`` (the generic default).

    Parameters
    ----------
    custom : str | None
        Caller-supplied override.  If non-None, used as-is.
    default_name : str
        Base prompt name (e.g. ``"error_analyst"``).
    update_mode : str
        The active update mode — used to select mode-specific variants
        (e.g. ``"error_analyst_full_rewrite"``).

    Returns
    -------
    str
        The resolved prompt text.
    """
    if custom is not None:
        return custom

    from summerclaw.agent_trainer.algorithms.skillopt.update_modes import (
        is_full_rewrite_minibatch_mode,
        is_rewrite_mode,
        normalize_update_mode,
    )

    mode = normalize_update_mode(update_mode)

    # Try mode-specific variant first
    if is_full_rewrite_minibatch_mode(mode):
        suffix = "full_rewrite"
    elif is_rewrite_mode(mode):
        suffix = "rewrite"
    else:
        suffix = None

    if suffix:
        variant_name = f"{default_name}_{suffix}"
        try:
            return load_prompt(variant_name)
        except FileNotFoundError:
            pass  # fall through to generic default

    return load_prompt(default_name)
