"""Tests for config loading, parsing, enum conversions, and overrides."""

import os
import tempfile
import pytest

from orchard.config import load_config, _apply_overrides, _parse_override_value
from orchard.enums import AlgorithmName, EncoderType, PickMode, Heuristic, LearningType

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
        assert cfg.train.comm_only_teammates is False
        
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

    def test_actor_critic_nested_blocks_parse(self):
        yaml_str = """
env:
  height: 5
  width: 5
  n_agents: 2
  n_task_types: 2
  gamma: 0.99
  task_assignments: [[0], [1]]
  stochastic:
    spawn_prob: 0.1
    despawn_mode: none
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [64]
actor_model:
  encoder: blind_task_cnn_grid
  mlp_dims: [32]
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  actor_lr:
    start: 0.002
  freeze_critic: true
  following_rates:
    enabled: true
    budget: 1.0
    rho: 0.5
    reallocation_freq: 2
    solver: closed_form
  influencer:
    enabled: true
    budget: 0.25
"""
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.algorithm.name == AlgorithmName.ACTOR_CRITIC
        assert cfg.train.actor_lr is not None
        assert cfg.train.freeze_critic is True
        assert cfg.train.following_rates.enabled is True
        assert cfg.train.influencer.enabled is True
        assert cfg.train.comm_only_teammates is False
        assert cfg.train.following_rates.teammate_budget is None
        assert cfg.train.following_rates.non_teammate_budget is None
        assert cfg.actor_model is not None
        assert cfg.actor_model.mlp_dims == (32,)
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
        cfg = load_config(
            path,
            overrides=[
                "train.total_steps=999",
                "env.pick_mode=choice",
                "train.algorithm.name=actor_critic",
                "train.comm_only_teammates=true",
            ],
        )
        assert cfg.train.total_steps == 999
        assert cfg.env.pick_mode == PickMode.CHOICE
        assert cfg.train.comm_only_teammates is True
        os.unlink(path)


class TestActorCriticValidation:
    def test_actor_critic_rejects_centralized(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  learning_type: centralized
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="requires train.learning_type=decentralized"):
            load_config(path)
        os.unlink(path)

    def test_actor_critic_rejects_comm_weight(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  comm_weight: 0.5
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="train.comm_weight is only supported"):
            load_config(path)
        os.unlink(path)

    def test_influencer_requires_following_rates(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  influencer:
    enabled: true
    budget: 0.5
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="requires train.following_rates.enabled=true"):
            load_config(path)
        os.unlink(path)

    def test_actor_critic_rejects_comm_only_teammates_without_gpu(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  use_gpu: false
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  comm_only_teammates: true
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="only supported for GPU actor-critic"):
            load_config(path)
        os.unlink(path)

    def test_non_actor_critic_rejects_comm_only_teammates(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  comm_only_teammates: true
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="only supported for train.algorithm.name=actor_critic"):
            load_config(path)
        os.unlink(path)

    def test_fixed_following_rates_parse_dual_budgets(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  following_rates:
    enabled: true
    fixed: true
    teammate_budget: 1.5
    non_teammate_budget: 2.5
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
"""
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.following_rates.fixed is True
        assert cfg.train.following_rates.teammate_budget == 1.5
        assert cfg.train.following_rates.non_teammate_budget == 2.5
        os.unlink(path)

    def test_fixed_following_rates_require_both_dual_budgets(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  following_rates:
    enabled: true
    fixed: true
    teammate_budget: 1.5
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="requires both"):
            load_config(path)
        os.unlink(path)

    def test_fixed_following_rates_allow_legacy_budget_field_but_ignore_it(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  following_rates:
    enabled: true
    fixed: true
    budget: 1.0
    teammate_budget: 1.5
    non_teammate_budget: 2.5
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
"""
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.following_rates.fixed is True
        assert cfg.train.following_rates.budget == 1.0
        assert cfg.train.following_rates.teammate_budget == 1.5
        assert cfg.train.following_rates.non_teammate_budget == 2.5
        os.unlink(path)

    def test_non_fixed_following_rates_reject_dual_budgets(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  following_rates:
    enabled: true
    budget: 1.0
    teammate_budget: 1.5
    non_teammate_budget: 2.5
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="only supported when train.following_rates.fixed=true"):
            load_config(path)
        os.unlink(path)

    def test_fixed_following_rates_reject_negative_dual_budgets(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  following_rates:
    enabled: true
    fixed: true
    teammate_budget: -1.0
    non_teammate_budget: 2.5
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="teammate_budget must be >= 0"):
            load_config(path)
        os.unlink(path)

    def test_following_rates_reject_bad_reallocation_frequency(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  following_rates:
    enabled: true
    budget: 1.0
    rho: 0.5
    reallocation_freq: 0
    solver: closed_form
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="reallocation_freq must be >= 1"):
            load_config(path)
        os.unlink(path)

    def test_freeze_critic_rejects_value_algorithm(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  freeze_critic: true
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="train.freeze_critic is only supported"):
            load_config(path)
        os.unlink(path)


class TestWarmupSteps:
    def test_warmup_steps_default_is_zero(self):
        path = _write_yaml(VALID_YAML)
        cfg = load_config(path)
        assert cfg.train.warmup_steps == 0
        os.unlink(path)

    def test_warmup_steps_custom_value_parses(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  warmup_steps: 250
"""
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.warmup_steps == 250
        os.unlink(path)

    def test_warmup_steps_negative_raises(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  warmup_steps: -1
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="train.warmup_steps must be >= 0"):
            load_config(path)
        os.unlink(path)

    def test_warmup_steps_requires_actor_critic(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  warmup_steps: 100
"""
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="train.warmup_steps>0 requires train.algorithm.name=actor_critic"):
            load_config(path)
        os.unlink(path)
