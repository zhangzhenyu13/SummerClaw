"""Unit tests for the SkillOpt trainer config engine.

Covers:
  - YAML loading with ``_base_`` inheritance
  - Structured → flat config flattening
  - CLI override application with auto-casting
  - ``is_structured`` detection
  - ``_deep_merge`` helper
  - Default YAML template loading
"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from summerclaw.agent_trainer.config import (
    _DEFAULTS,
    _FLATTEN_MAP,
    _STRUCTURED_SECTIONS,
    _cast_value,
    _deep_merge,
    _load_yaml,
    apply_overrides,
    flatten_config,
    is_structured,
    load_config,
)


# ── _deep_merge ──────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"model": {"backend": "openai", "optimizer": "gpt-4"}}
        override = {"model": {"optimizer": "gpt-5"}}
        result = _deep_merge(base, override)
        assert result["model"]["backend"] == "openai"
        assert result["model"]["optimizer"] == "gpt-5"

    def test_non_dict_override_replaces(self):
        base = {"model": {"backend": "openai"}}
        override = {"model": "simple"}
        result = _deep_merge(base, override)
        assert result["model"] == "simple"

    def test_does_not_mutate_originals(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}


# ── is_structured ────────────────────────────────────────────────────────

class TestIsStructured:
    def test_flat_config(self):
        assert not is_structured({"num_epochs": 3, "batch_size": 5})

    def test_structured_config(self):
        assert is_structured({"model": {"backend": "openai"}, "train": {"num_epochs": 3}})

    def test_section_not_dict(self):
        assert not is_structured({"model": "not_a_dict"})

    def test_empty(self):
        assert not is_structured({})


# ── _cast_value ──────────────────────────────────────────────────────────

class TestCastValue:
    def test_true(self):
        assert _cast_value("true") is True
        assert _cast_value("True") is True
        assert _cast_value("yes") is True

    def test_false(self):
        assert _cast_value("false") is False
        assert _cast_value("False") is False
        assert _cast_value("no") is False

    def test_int(self):
        assert _cast_value("42") == 42
        assert isinstance(_cast_value("42"), int)

    def test_float(self):
        assert _cast_value("3.14") == 3.14
        assert isinstance(_cast_value("3.14"), float)

    def test_string(self):
        assert _cast_value("hello") == "hello"
        assert isinstance(_cast_value("hello"), str)


# ── _load_yaml with _base_ inheritance ───────────────────────────────────

class TestLoadYaml:
    def test_simple_yaml(self, tmp_path):
        cfg_file = tmp_path / "simple.yaml"
        cfg_file.write_text("num_epochs: 5\nbatch_size: 10\n")
        result = _load_yaml(str(cfg_file))
        assert result == {"num_epochs": 5, "batch_size": 10}

    def test_base_inheritance(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("num_epochs: 3\nbatch_size: 5\nseed: 42\n")
        child = tmp_path / "child.yaml"
        child.write_text('_base_: base.yaml\nnum_epochs: 10\n')
        result = _load_yaml(str(child))
        assert result == {"num_epochs": 10, "batch_size": 5, "seed": 42}

    def test_structured_base_inheritance(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({
            "model": {"backend": "openai", "optimizer": "gpt-4"},
            "train": {"num_epochs": 4},
        }))
        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({
            "_base_": "base.yaml",
            "model": {"optimizer": "gpt-5"},
            "train": {"batch_size": 40},
        }))
        result = _load_yaml(str(child))
        assert result["model"]["backend"] == "openai"
        assert result["model"]["optimizer"] == "gpt-5"
        assert result["train"]["num_epochs"] == 4
        assert result["train"]["batch_size"] == 40

    def test_circular_inheritance(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(f"_base_: b.yaml\nkey: a\n")
        b.write_text(f"_base_: a.yaml\nkey: b\n")
        with pytest.raises(ValueError, match="Circular"):
            _load_yaml(str(a))

    def test_empty_yaml(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert _load_yaml(str(f)) == {}


# ── flatten_config ───────────────────────────────────────────────────────

class TestFlattenConfig:
    def test_flat_passthrough(self):
        flat = {"num_epochs": 3, "batch_size": 5}
        result = flatten_config(flat)
        assert result == flat

    def test_structured_flatten(self):
        structured = {
            "model": {"backend": "openai", "optimizer": "gpt-5", "reasoning_effort": "high"},
            "train": {"num_epochs": 4, "batch_size": 40, "seed": 42},
            "optimizer": {"learning_rate": 4, "min_learning_rate": 2},
        }
        result = flatten_config(structured)
        assert result["model_backend"] == "openai"
        assert result["optimizer_model"] == "gpt-5"
        assert result["reasoning_effort"] == "high"
        assert result["num_epochs"] == 4
        assert result["batch_size"] == 40
        assert result["seed"] == 42
        assert result["edit_budget"] == 4
        assert result["min_edit_budget"] == 2

    def test_env_passthrough(self):
        structured = {
            "env": {
                "name": "alfworld",
                "skill_init": "skills/init.md",
                "max_steps": 50,  # not in _FLATTEN_MAP → passed through
            },
        }
        result = flatten_config(structured)
        assert result["env"] == "alfworld"
        assert result["skill_init"] == "skills/init.md"
        assert result["max_steps"] == 50

    def test_gate_validation_error(self):
        structured = {
            "evaluation": {"use_gate": False},
        }
        with pytest.raises(ValueError, match="Gate validation"):
            flatten_config(structured)


# ── apply_overrides ──────────────────────────────────────────────────────

class TestApplyOverrides:
    def test_flat_override(self):
        cfg = {"num_epochs": 3}
        apply_overrides(cfg, ["num_epochs=10"])
        assert cfg["num_epochs"] == 10

    def test_section_override(self):
        cfg = {"model": {"backend": "openai"}}
        apply_overrides(cfg, ["model.backend=azure_openai"])
        assert cfg["model"]["backend"] == "azure_openai"

    def test_new_section(self):
        cfg = {}
        apply_overrides(cfg, ["model.backend=claude"])
        assert cfg["model"]["backend"] == "claude"

    def test_invalid_override(self):
        with pytest.raises(ValueError, match="key=value"):
            apply_overrides({}, ["bad_format"])

    def test_auto_cast_types(self):
        cfg = {}
        apply_overrides(cfg, [
            "num_epochs=5",
            "use_gate=true",
            "temperature=0.7",
            "model=gpt-5",
        ])
        assert cfg["num_epochs"] == 5
        assert cfg["use_gate"] is True
        assert cfg["temperature"] == 0.7
        assert cfg["model"] == "gpt-5"


# ── load_config (integration) ───────────────────────────────────────────

class TestLoadConfig:
    def test_load_with_overrides(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump({
            "train": {"num_epochs": 4, "batch_size": 40},
            "optimizer": {"learning_rate": 4},
        }))
        result = load_config(str(cfg_file), overrides=["train.num_epochs=10"])
        assert result["train"]["num_epochs"] == 10
        assert result["train"]["batch_size"] == 40

    def test_full_pipeline(self, tmp_path):
        """Load → overrides → flatten end-to-end."""
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({
            "model": {"backend": "openai", "optimizer": "gpt-4"},
            "train": {"num_epochs": 4, "batch_size": 40},
            "optimizer": {"learning_rate": 4},
        }))
        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({
            "_base_": "base.yaml",
            "model": {"optimizer": "gpt-5"},
        }))
        cfg = load_config(str(child), overrides=["train.num_epochs=8"])
        flat = flatten_config(cfg)
        assert flat["optimizer_model"] == "gpt-5"
        assert flat["num_epochs"] == 8
        assert flat["edit_budget"] == 4


# ── Default YAML template ───────────────────────────────────────────────

class TestDefaultTemplate:
    def test_default_yaml_loads(self):
        """Verify the bundled _base_default.yaml can be loaded and flattened."""
        from pathlib import Path
        tpl_dir = Path(__file__).resolve().parents[2] / "summerclaw" / "templates" / "trainer" / "skillopt"
        default_yaml = tpl_dir / "_base_default.yaml"
        if not default_yaml.exists():
            pytest.skip("Template file not found")
        cfg = load_config(str(default_yaml))
        assert is_structured(cfg)
        flat = flatten_config(cfg)
        # Spot-check key mappings
        assert "num_epochs" in flat
        assert "edit_budget" in flat
        assert "minibatch_size" in flat
        assert flat["edit_budget"] == 4
        assert flat["use_slow_update"] is True

    def test_flatten_map_coverage(self):
        """All structured section keys in _FLATTEN_MAP should be valid."""
        for dotted, flat_key in _FLATTEN_MAP.items():
            assert "." in dotted
            section, _ = dotted.split(".", 1)
            assert section in _STRUCTURED_SECTIONS, f"Unknown section: {section}"
