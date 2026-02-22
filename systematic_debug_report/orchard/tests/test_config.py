"""Tests for config loading and validation."""

import os
import tempfile

import pytest

from orchard.config import load_config, _apply_overrides, _parse_override_value
from orchard.enums import EnvType, EncoderType, ModelType, TrainMode, Schedule


def _write_yaml(content: str) -> str:
    """Write content to a temp file and return path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


MINIMAL_YAML = """
env:
  height: 2
  width: 2
  n_agents: 2
  n_apples: 1
  gamma: 0.9
  r_picker: -1.0
  type: deterministic
model:
  input_type: relative
  model_type: mlp
train:
  mode: value_learning
  total_steps: 100
  seed: 42
  lr:
    start: 0.001
    schedule: none
"""


class TestLoadConfig:
    def test_minimal_parse(self):
        path = _write_yaml(MINIMAL_YAML)
        cfg = load_config(path)
        assert cfg.env.env_type == EnvType.DETERMINISTIC
        assert cfg.model.input_type == EncoderType.RELATIVE
        assert cfg.model.model_type == ModelType.MLP
        assert cfg.train.mode == TrainMode.VALUE_LEARNING
        assert cfg.train.total_steps == 100
        os.unlink(path)

    def test_missing_env_section(self):
        path = _write_yaml("model:\n  input_type: relative\n  model_type: mlp\ntrain:\n  mode: value_learning\n  total_steps: 1\n  seed: 1\n  lr:\n    start: 0.001\n    schedule: none\n")
        with pytest.raises(ValueError, match="Missing required config section"):
            load_config(path)
        os.unlink(path)

    def test_invalid_enum(self):
        bad = MINIMAL_YAML.replace("type: deterministic", "type: banana")
        path = _write_yaml(bad)
        with pytest.raises(ValueError, match="Invalid env.type"):
            load_config(path)
        os.unlink(path)

    def test_stochastic_requires_block(self):
        bad = MINIMAL_YAML.replace("type: deterministic", "type: stochastic")
        path = _write_yaml(bad)
        with pytest.raises(ValueError, match="stochastic block required"):
            load_config(path)
        os.unlink(path)


class TestOverrides:
    def test_dot_notation(self):
        raw = {"a": {"b": {"c": 1}}}
        result = _apply_overrides(raw, ["a.b.c=42"])
        assert result["a"]["b"]["c"] == 42

    def test_parse_int(self):
        assert _parse_override_value("42") == 42

    def test_parse_float(self):
        assert _parse_override_value("3.14") == 3.14

    def test_parse_bool(self):
        assert _parse_override_value("true") is True
        assert _parse_override_value("false") is False

    def test_parse_list(self):
        assert _parse_override_value("[64,128]") == [64, 128]

    def test_parse_string(self):
        assert _parse_override_value("hello") == "hello"

    def test_config_with_overrides(self):
        path = _write_yaml(MINIMAL_YAML)
        cfg = load_config(path, overrides=["train.total_steps=999"])
        assert cfg.train.total_steps == 999
        os.unlink(path)
