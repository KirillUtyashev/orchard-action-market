"""Tests for config loading, parsing, enum conversions, and overrides."""

import os
import tempfile
import pytest

from orchard.config import load_config, _apply_overrides, _parse_override_value
from orchard.enums import AlgorithmName, EncoderType, Heuristic, LearningType, RewardGeneration, StructureType

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
  encoder: general_dec_cnn_grid
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
        assert cfg.model.encoder == EncoderType.GENERAL_DEC_CNN_GRID
        assert cfg.train.total_steps == 100
        assert cfg.train.comm_only_teammates is False
        assert cfg.train.batch_forced_actor_updates is True
        assert cfg.env.stochastic.reward_generation == RewardGeneration.BASELINE_OFFSET

        os.unlink(path)

    def test_missing_section_raises(self):
        bad_yaml = VALID_YAML.replace("env:", "environment:")
        path = _write_yaml(bad_yaml)
        with pytest.raises(ValueError, match="Missing required config section: 'env'"):
            load_config(path)
        os.unlink(path)

    def test_invalid_enum_raises(self):
        bad_yaml = VALID_YAML.replace("general_dec_cnn_grid", "magic_encoder")
        path = _write_yaml(bad_yaml)
        with pytest.raises(ValueError, match="Invalid encoder: 'magic_encoder'"):
            load_config(path)
        os.unlink(path)

    def test_clustering_specialization_parse(self):
        yaml_str = VALID_YAML.replace(
            "n_agents: 2", "n_agents: 4\n  n_task_types: 4\n  clustering: 1\n  specialization: 2"
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.clustering == 1
        assert cfg.env.specialization == 2
        assert cfg.env.n_task_types == 4
        os.unlink(path)

    def test_structure_parse(self):
        yaml_str = VALID_YAML.replace(
            "n_agents: 2",
            "n_agents: 4\n  n_task_types: 4\n  structure: disjoint_groups\n  structure_group_size: 2",
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.structure == StructureType.DISJOINT_GROUPS
        assert cfg.env.structure_group_size == 2
        assert cfg.env.n_tasks_per_group is None
        os.unlink(path)

    def test_n_tasks_per_group_parse(self):
        yaml_str = VALID_YAML.replace(
            "n_agents: 2",
            (
                "n_agents: 4\n  n_task_types: 2\n  structure: disjoint_groups\n"
                "  structure_group_size: 2\n  n_tasks_per_group: 1"
            ),
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.n_tasks_per_group == 1
        os.unlink(path)

    def test_sigma_a_sigma_b_parse(self):
        yaml_str = VALID_YAML.replace(
            "spawn_prob: 0.1", "spawn_prob: 0.1\n    sigma_a: 0.3\n    sigma_b: 0.5"
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.stochastic.sigma_a == pytest.approx(0.3)
        assert cfg.env.stochastic.sigma_b == pytest.approx(0.5)
        os.unlink(path)

    def test_reward_generation_parse(self):
        yaml_str = VALID_YAML.replace(
            "spawn_prob: 0.1", "spawn_prob: 0.1\n    reward_generation: sampled_mean"
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.stochastic.reward_generation == RewardGeneration.SAMPLED_MEAN
        os.unlink(path)

    def test_actor_critic_nested_blocks_parse(self):
        yaml_str = """
env:
  height: 5
  width: 5
  n_agents: 2
  n_task_types: 2
  gamma: 0.99
  stochastic:
    spawn_prob: 0.1
    despawn_mode: none
model:
  encoder: general_dec_cnn_grid
  mlp_dims: [64]
actor_model:
  encoder: general_dec_cnn_grid
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
        assert cfg.train.batch_forced_actor_updates is True
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
        assert cfg.model.encoder == EncoderType.GENERAL_DEC_CNN_GRID
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
                "train.algorithm.name=actor_critic",
                "train.comm_only_teammates=true",
                "train.batch_forced_actor_updates=false",
            ],
        )
        assert cfg.train.total_steps == 999
        assert cfg.train.comm_only_teammates is True
        assert cfg.train.batch_forced_actor_updates is False
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

    def test_actor_critic_batch_forced_actor_updates_parses(self):
        yaml_str = VALID_YAML + """
train:
  total_steps: 100
  lr:
    start: 0.001
  algorithm:
    name: actor_critic
  batch_forced_actor_updates: false
"""
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.batch_forced_actor_updates is False
        os.unlink(path)

    def test_batch_forced_actor_updates_defaults_to_true(self):
        path = _write_yaml(VALID_YAML)
        cfg = load_config(path)
        assert cfg.train.batch_forced_actor_updates is True
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
