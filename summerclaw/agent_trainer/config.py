"""SkillOpt Trainer configuration — structured YAML with inheritance.

Supports two config formats:
  1. **Structured** (new): sections like ``model``, ``train``, ``gradient``,
     ``optimizer``, ``evaluation``, ``env`` — with ``_base_`` inheritance.
  2. **Flat** (legacy): all keys at top level — fully backward compatible.

Mirrors the official ``skillopt.config`` module.

Usage::

    from summerclaw.agent_trainer.config import load_config, flatten_config

    cfg = load_config("configs/searchqa_default.yaml")
    flat = flatten_config(cfg)  # always returns flat dict for trainer
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from summerclaw.agent.loop import AgentLoop

logger = logging.getLogger(__name__)

# ── Default training parameters (legacy / flat config) ────────────────────

_DEFAULTS: dict[str, Any] = {
    # Training loop
    "num_epochs": 3,
    "batch_size": 5,
    "edit_budget": 4,
    "seed": 42,
    "workers": 0,  # 0 = auto-derive 80% of provider.maxConcurrency
    "accumulation": 1,

    # SkillOpt-specific
    "minibatch_size": 5,
    "merge_batch_size": 8,
    "optimizer_model": None,
    "update_mode": "patch",
    "lr_mode": "constant",
    "lr_scheduler": "constant",
    "lr_control_mode": "fixed",
    "min_lr": 2,
    "min_edit_budget": 2,
    "skill_update_mode": "patch",
    "max_analyst_rounds": 3,
    "failure_only": False,

    # Slow update / meta skill
    "use_slow_update": True,
    "slow_update_samples": 20,
    "longitudinal_pair_policy": "mixed",
    "use_meta_skill": True,

    # Rejected buffer
    "use_rejected_buffer": True,
    "rejected_buffer_max_size": 10,
    "rejected_buffer_max_summary_chars": 200,

    # Rewrite
    "reasoning_effort": "high",
    "rewrite_reasoning_effort": "",
    "rewrite_max_completion_tokens": 64000,
    "env": None,

    # Evaluation
    "use_gate": True,
    "sel_env_num": 0,
    "test_env_num": 0,
    "eval_test": True,

    # Data
    "data_dir": "",
    "skill_init": "",

    # Dashboard
    "dashboard_port": 443,
    "dashboard_share": True,
}

# ── Section names that indicate a structured config ──────────────────────

_STRUCTURED_SECTIONS = frozenset({
    "model", "train", "gradient", "optimizer", "evaluation", "env",
})

# ── Structured → flat key mapping (mirrors official _FLATTEN_MAP) ─────────

_FLATTEN_MAP: dict[str, str] = {
    # model section
    "model.backend": "model_backend",
    "model.optimizer": "optimizer_model",
    "model.target": "target_model",
    "model.optimizer_backend": "optimizer_backend",
    "model.target_backend": "target_backend",
    "model.reasoning_effort": "reasoning_effort",
    "model.rewrite_reasoning_effort": "rewrite_reasoning_effort",
    "model.rewrite_max_completion_tokens": "rewrite_max_completion_tokens",
    "model.codex_exec_path": "codex_exec_path",
    "model.codex_exec_sandbox": "codex_exec_sandbox",
    "model.codex_exec_profile": "codex_exec_profile",
    "model.codex_exec_full_auto": "codex_exec_full_auto",
    "model.codex_exec_reasoning_effort": "codex_exec_reasoning_effort",
    "model.codex_exec_use_sdk": "codex_exec_use_sdk",
    "model.codex_exec_network_access": "codex_exec_network_access",
    "model.codex_exec_web_search": "codex_exec_web_search",
    "model.codex_exec_approval_policy": "codex_exec_approval_policy",
    "model.claude_code_exec_path": "claude_code_exec_path",
    "model.claude_code_exec_profile": "claude_code_exec_profile",
    "model.claude_code_exec_use_sdk": "claude_code_exec_use_sdk",
    "model.claude_code_exec_effort": "claude_code_exec_effort",
    "model.claude_code_exec_max_thinking_tokens": "claude_code_exec_max_thinking_tokens",
    "model.codex_trace_to_optimizer": "codex_trace_to_optimizer",
    "model.azure_endpoint": "azure_endpoint",
    "model.azure_api_version": "azure_api_version",
    "model.azure_api_key": "azure_api_key",
    "model.azure_openai_endpoint": "azure_openai_endpoint",
    "model.azure_openai_api_version": "azure_openai_api_version",
    "model.azure_openai_api_key": "azure_openai_api_key",
    "model.azure_openai_auth_mode": "azure_openai_auth_mode",
    "model.azure_openai_ad_scope": "azure_openai_ad_scope",
    "model.azure_openai_managed_identity_client_id": "azure_openai_managed_identity_client_id",
    "model.optimizer_azure_openai_endpoint": "optimizer_azure_openai_endpoint",
    "model.optimizer_azure_openai_api_version": "optimizer_azure_openai_api_version",
    "model.optimizer_azure_openai_api_key": "optimizer_azure_openai_api_key",
    "model.optimizer_azure_openai_auth_mode": "optimizer_azure_openai_auth_mode",
    "model.optimizer_azure_openai_ad_scope": "optimizer_azure_openai_ad_scope",
    "model.optimizer_azure_openai_managed_identity_client_id": "optimizer_azure_openai_managed_identity_client_id",
    "model.target_azure_openai_endpoint": "target_azure_openai_endpoint",
    "model.target_azure_openai_api_version": "target_azure_openai_api_version",
    "model.target_azure_openai_api_key": "target_azure_openai_api_key",
    "model.target_azure_openai_auth_mode": "target_azure_openai_auth_mode",
    "model.target_azure_openai_ad_scope": "target_azure_openai_ad_scope",
    "model.target_azure_openai_managed_identity_client_id": "target_azure_openai_managed_identity_client_id",
    "model.qwen_chat_base_url": "qwen_chat_base_url",
    "model.qwen_chat_api_key": "qwen_chat_api_key",
    "model.qwen_chat_temperature": "qwen_chat_temperature",
    "model.qwen_chat_timeout_seconds": "qwen_chat_timeout_seconds",
    "model.qwen_chat_max_tokens": "qwen_chat_max_tokens",
    "model.qwen_chat_enable_thinking": "qwen_chat_enable_thinking",
    # train section
    "train.num_epochs": "num_epochs",
    "train.train_size": "train_size",
    "train.steps_per_epoch": "steps_per_epoch",
    "train.batch_size": "batch_size",
    "train.workers": "workers",
    "train.accumulation": "accumulation",
    "train.seed": "seed",
    # gradient section
    "gradient.minibatch_size": "minibatch_size",
    "gradient.merge_batch_size": "merge_batch_size",
    "gradient.analyst_workers": "analyst_workers",
    "gradient.aggregate_workers": "aggregate_workers",
    "gradient.evaluate_workers": "evaluate_workers",
    "gradient.failure_only": "failure_only",
    "gradient.max_analyst_rounds": "max_analyst_rounds",
    # optimizer section
    "optimizer.learning_rate": "edit_budget",
    "optimizer.min_learning_rate": "min_edit_budget",
    "optimizer.lr_scheduler": "lr_scheduler",
    "optimizer.lr_control_mode": "lr_control_mode",
    "optimizer.skill_update_mode": "skill_update_mode",
    "optimizer.meta_learning_rate": "meta_edit_budget",
    "optimizer.use_slow_update": "use_slow_update",
    "optimizer.slow_update_samples": "slow_update_samples",
    "optimizer.longitudinal_pair_policy": "longitudinal_pair_policy",
    "optimizer.use_meta_skill": "use_meta_skill",
    "optimizer.use_rejected_buffer": "use_rejected_buffer",
    "optimizer.rejected_buffer_max_size": "rejected_buffer_max_size",
    "optimizer.rejected_buffer_max_summary_chars": "rejected_buffer_max_summary_chars",
    # evaluation section
    "evaluation.use_gate": "use_gate",
    "evaluation.sel_env_num": "sel_env_num",
    "evaluation.test_env_num": "test_env_num",
    "evaluation.eval_test": "eval_test",
    # env section
    "env.name": "env",
    "env.skill_init": "skill_init",
    "env.out_root": "out_root",
    "env.memory_algorithm": "memory_algorithm",
    "env.enabled_tools": "enabled_tools",
}


# ── Deep merge ───────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ── YAML loading with _base_ inheritance ─────────────────────────────────

def _load_yaml(path: str, _visited: set[str] | None = None) -> dict:
    """Load a YAML file, resolving ``_base_`` inheritance recursively."""
    abs_path = os.path.abspath(path)
    if _visited is None:
        _visited = set()
    if abs_path in _visited:
        raise ValueError(f"Circular _base_ inheritance: {abs_path}")
    _visited.add(abs_path)

    with open(abs_path) as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = os.path.join(os.path.dirname(abs_path), base_ref)
        base_cfg = _load_yaml(base_path, _visited)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


# ── Format detection ─────────────────────────────────────────────────────

def is_structured(cfg: dict) -> bool:
    """Return True if *cfg* uses the new structured section format."""
    return any(
        key in _STRUCTURED_SECTIONS and isinstance(cfg.get(key), dict)
        for key in cfg
    )


# ── Flatten ──────────────────────────────────────────────────────────────

def flatten_config(cfg: dict) -> dict:
    """Convert a structured config to the flat dict expected by the trainer.

    If *cfg* is already flat, returns a shallow copy unchanged.
    """
    if not is_structured(cfg):
        return dict(cfg)

    flat: dict[str, Any] = {}

    evaluation_section = cfg.get("evaluation", {})
    if isinstance(evaluation_section, dict) and evaluation_section.get("use_gate") is False:
        raise ValueError(
            "Gate validation is mandatory in this branch. Remove "
            "`evaluation.use_gate: false` from the config."
        )

    # Apply the explicit mapping
    for dotted, flat_key in _FLATTEN_MAP.items():
        section, key = dotted.split(".", 1)
        section_dict = cfg.get(section, {})
        if isinstance(section_dict, dict) and key in section_dict:
            flat[flat_key] = section_dict[key]

    # Pass through env-specific keys not in the explicit mapping
    env_section = cfg.get("env", {})
    if isinstance(env_section, dict):
        mapped_env_keys = {
            k.split(".", 1)[1]
            for k in _FLATTEN_MAP
            if k.startswith("env.")
        }
        for key, val in env_section.items():
            if key not in mapped_env_keys:
                flat[key] = val

    return flat


# ── Override application ─────────────────────────────────────────────────

def _cast_value(val_str: str) -> Any:
    """Auto-cast a CLI string value to int / float / bool / str."""
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply ``key=value`` overrides to a structured config (in place).

    Supports both ``section.key=value`` (for structured configs) and
    ``key=value`` (for flat configs or flat keys in env section).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        key, val_str = item.split("=", 1)
        val = _cast_value(val_str)

        if "." in key:
            section, subkey = key.split(".", 1)
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section][subkey] = val
            else:
                cfg.setdefault(section, {})[subkey] = val
        else:
            # Flat key — apply to top level (for legacy compat)
            cfg[key] = val


# ── Public API ───────────────────────────────────────────────────────────

def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> dict:
    """Load a config file with ``_base_`` inheritance and optional overrides.

    Parameters
    ----------
    path : str
        Path to the YAML config file.
    overrides : list[str] | None
        ``key=value`` strings from ``--cfg-options``.

    Returns
    -------
    dict
        The merged config (structured or flat depending on the YAML).
    """
    cfg = _load_yaml(path)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


# ── Legacy integration ───────────────────────────────────────────────────

def build_trainer_config(loop: AgentLoop, algorithm_name: str) -> dict[str, Any]:
    """Build a training configuration dict from the agent loop.

    Merges (in priority order):
      1. Built-in defaults
      2. Default YAML config (``<workspace>/skillopt.yaml`` or ``<algorithm_name>.yaml``)
      3. Explicit YAML config (if ``trainer.config_path`` is set)
      4. Extra ``trainer`` section from loop.channels_config (if present)
      5. Algorithm-specific overrides

    Parameters
    ----------
    loop : AgentLoop
        The running agent loop with workspace/model info.
    algorithm_name : str
        The algorithm name (e.g. "skillopt").

    Returns
    -------
    dict[str, Any]
        Flat configuration dict for the trainer.
    """
    result = dict(_DEFAULTS)
    workspace = Path(loop.workspace).expanduser() if loop.workspace else Path.cwd()

    # Always set expected paths so error messages are informative
    result["data_dir"] = str(workspace / "train-data")

    # Auto-detect initial skill file
    skill_init = workspace / "skills" / "SKILL.md"
    if skill_init.exists():
        result["skill_init"] = str(skill_init)

    # ── Auto-load default YAML config from workspace root ────────────
    # Looks for: <workspace>/<algorithm_name>.yaml  (e.g. skillopt.yaml)
    default_yaml = workspace / f"{algorithm_name}.yaml"
    if default_yaml.is_file():
        try:
            yaml_cfg = load_config(str(default_yaml))
            flat_yaml = flatten_config(yaml_cfg)
            _merge(result, flat_yaml)
            logger.info(
                "Loaded default config: {} ({} keys)",
                default_yaml, len(flat_yaml),
            )
        except Exception as exc:
            logger.warning("Failed to load default config {}: {}", default_yaml, exc)

    # Read extra "trainer" config section from channels_config if present
    channels_cfg = getattr(loop, "channels_config", None)
    extra: dict[str, Any] = {}
    if channels_cfg:
        extra = getattr(channels_cfg, "trainer", None) or {}
        if not isinstance(extra, dict):
            extra = {}

    # ── Explicit config_path overrides default YAML ──────────────────
    config_path = extra.get("config_path") or extra.get("config")
    if config_path:
        config_path = str(Path(config_path).expanduser())
        if not os.path.isabs(config_path):
            config_path = str(workspace / config_path)
        try:
            yaml_cfg = load_config(config_path)
            flat_yaml = flatten_config(yaml_cfg)
            # Explicit YAML values override defaults and default YAML
            _merge(result, flat_yaml)
            logger.info("Loaded explicit config from %s", config_path)
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", config_path, exc)

    # Merge remaining extra keys (non-config-path entries override YAML)
    _merge(result, {k: v for k, v in extra.items() if k not in ("config_path", "config")})

    # Algorithm-specific overrides
    alg_section = result.pop(f"_{algorithm_name}", None)
    if isinstance(alg_section, dict):
        _merge(result, alg_section)

    return result


def _merge(base: dict, overrides: dict) -> None:
    """Merge overrides into base (in-place, shallow)."""
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
