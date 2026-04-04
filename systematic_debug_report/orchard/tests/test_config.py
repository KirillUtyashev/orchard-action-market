"""Tests for config loading, parsing, enum conversions, and overrides."""

import os
import tempfile
import pytest

from orchard.config import load_config, _apply_overrides, _parse_override_value
from orchard.enums import EncoderType, PickMode, Heuristic, LearningType

def _write_yaml(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path

VALID_YAML = """
env:
  height: 5
  width: 5
  n_agents: 2
  gamma: 0.99
  stochastic:
    spawn_prob: 0.1
    despawn_mode: none
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [64]
train:
  total_steps: 100
  lr:
    start: 0.001
"""

class TestConfigParsing:
    def test_valid_parse(self):
        path = _write_yaml(VALID_YAML)
        cfg = load_config(path)
        
        assert cfg.env.height == 5
        assert cfg.env.gamma == 0.99
        assert cfg.model.encoder == EncoderType.BLIND_TASK_CNN_GRID
        assert cfg.train.total_steps == 100
        
        os.unlink(path)

    def test_missing_section_raises(self):
        bad_yaml = VALID_YAML.replace("env:", "environment:")
        path = _write_yaml(bad_yaml)
        with pytest.raises(ValueError, match="Missing required config section: 'env'"):
            load_config(path)
        os.unlink(path)

    def test_invalid_enum_raises(self):
        bad_yaml = VALID_YAML.replace("blind_task_cnn_grid", "magic_encoder")
        path = _write_yaml(bad_yaml)
        with pytest.raises(ValueError, match="Invalid encoder: 'magic_encoder'"):
            load_config(path)
        os.unlink(path)

    def test_rho_generates_assignments(self):
        yaml_str = VALID_YAML.replace(
            "n_agents: 2", "n_agents: 4\n  n_task_types: 4\n  rho: 0.25"
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.n_task_types == 4
        assert cfg.env.task_assignments is not None
        assert len(cfg.env.task_assignments) == 4
        assert cfg.env.task_assignments[0] == (0,)
        assert cfg.env.task_assignments[3] == (3,)
        os.unlink(path)

class TestBackwardCompatibility:
    def test_n_apples_maps_to_n_tasks(self):
        yaml_str = VALID_YAML.replace("env:\n", "env:\n  n_apples: 7\n")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.n_tasks == 7
        os.unlink(path)
        
    def test_input_type_maps_to_encoder(self):
        yaml_str = VALID_YAML.replace("encoder:", "input_type:")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.model.encoder == EncoderType.BLIND_TASK_CNN_GRID
        os.unlink(path)

class TestOverrides:
    def test_override_dot_notation(self):
        raw = {"env": {"height": 5, "width": 5}}
        result = _apply_overrides(raw, ["env.height=10"])
        assert result["env"]["height"] == 10
        assert result["env"]["width"] == 5

    def test_override_type_coercion(self):
        assert _parse_override_value("42") == 42
        assert _parse_override_value("3.14") == 3.14
        assert _parse_override_value("true") is True
        assert _parse_override_value("false") is False
        assert _parse_override_value("hello") == "hello"

    def test_override_list_parsing(self):
        assert _parse_override_value("[64, 128]") == [64, 128]
        assert _parse_override_value("[]") == []
        assert _parse_override_value("[[4,3], [2,2]]") == [[4, 3], [2, 2]]

    def test_config_load_with_overrides(self):
        path = _write_yaml(VALID_YAML)
        cfg = load_config(path, overrides=["train.total_steps=999", "env.pick_mode=choice"])
        assert cfg.train.total_steps == 999
        assert cfg.env.pick_mode == PickMode.CHOICE
        os.unlink(path)