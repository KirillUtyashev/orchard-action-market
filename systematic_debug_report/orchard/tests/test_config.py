"""Tests for config loading and validation (chunk 1)."""

import os
import tempfile

import pytest

from orchard.config import load_config, _apply_overrides, _parse_override_value
from orchard.datatypes import compute_task_assignments
from orchard.enums import (
    EncoderType,
    EnvType,
    Heuristic,
    ModelType,
    PickMode,
    Schedule,
    StoppingCondition,
    TrainMode,
)


def _write_yaml(content: str) -> str:
    """Write content to a temp file and return path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Backward compatibility — old configs still parse
# ---------------------------------------------------------------------------

LEGACY_YAML = """
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


class TestBackwardCompat:
    def test_legacy_parse(self):
        path = _write_yaml(LEGACY_YAML)
        cfg = load_config(path)
        assert cfg.env.env_type == EnvType.DETERMINISTIC
        assert cfg.env.n_tasks == 1  # n_apples maps to n_tasks
        assert cfg.env.n_apples == 1  # backward compat alias
        assert cfg.env.r_picker == -1.0
        assert cfg.env.n_task_types == 1
        assert cfg.env.task_assignments is None  # not set for legacy
        assert cfg.env.pick_mode == PickMode.FORCED
        assert cfg.env.force_pick is True  # backward compat alias
        assert cfg.train.heuristic == Heuristic.NEAREST_TASK  # default for n_task_types=1
        os.unlink(path)

    def test_force_pick_false_legacy(self):
        """force_pick: false should NOT work with n_task_types=1 since choice requires >1."""
        yaml_str = LEGACY_YAML.replace("type: deterministic", "type: deterministic\n  force_pick: false")
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="CHOICE requires n_task_types > 1"):
            load_config(path)
        os.unlink(path)

    def test_n_apples_maps_to_n_tasks(self):
        path = _write_yaml(LEGACY_YAML)
        cfg = load_config(path)
        assert cfg.env.n_tasks == 1
        assert cfg.env.n_apples == 1
        os.unlink(path)

    def test_missing_env_section(self):
        path = _write_yaml("model:\n  input_type: relative\n  model_type: mlp\ntrain:\n  mode: value_learning\n  total_steps: 1\n  seed: 1\n  lr:\n    start: 0.001\n    schedule: none\n")
        with pytest.raises(ValueError, match="Missing required config section"):
            load_config(path)
        os.unlink(path)

    def test_invalid_enum(self):
        bad = LEGACY_YAML.replace("type: deterministic", "type: banana")
        path = _write_yaml(bad)
        with pytest.raises(ValueError, match="Invalid env.type"):
            load_config(path)
        os.unlink(path)

    def test_stochastic_requires_block(self):
        bad = LEGACY_YAML.replace("type: deterministic", "type: stochastic")
        path = _write_yaml(bad)
        with pytest.raises(ValueError, match="stochastic block required"):
            load_config(path)
        os.unlink(path)


# ---------------------------------------------------------------------------
# Task specialization configs
# ---------------------------------------------------------------------------

TASK_SPEC_YAML = """
env:
  height: 9
  width: 9
  n_agents: 4
  n_tasks: 4
  n_task_types: 4
  gamma: 0.99
  r_picker: 1.0
  r_high: 1.0
  r_low: 0.0
  max_tasks_per_type: 3
  pick_mode: forced
  type: stochastic
  stochastic:
    spawn_prob: 0.04
    despawn_mode: probability
    despawn_prob: 0.05
model:
  input_type: task_cnn_grid
  model_type: cnn
  conv_specs: [[16, 3]]
  mlp_dims: [64]
train:
  mode: policy_learning
  td_target: after_state
  total_steps: 100
  seed: 1234
  lr:
    start: 0.001
    schedule: none
  policy_learning:
    epsilon:
      start: 0.05
      end: 0.05
      schedule: none
"""


def _add_env_fields(yaml_str: str, *fields: str) -> str:
    """Insert env-level YAML fields before the 'model:' line."""
    lines = yaml_str.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("model:"):
            for field in reversed(fields):
                lines.insert(i, "  " + field)
            break
    return "\n".join(lines)


class TestTaskSpecConfig:
    def test_rho_generates_assignments(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "rho: 0.25")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.n_task_types == 4
        assert cfg.env.task_assignments is not None
        assert len(cfg.env.task_assignments) == 4
        # rho=0.25, T=4 → |G_i|=1, cyclic: agent 0→{0}, 1→{1}, 2→{2}, 3→{3}
        assert cfg.env.task_assignments[0] == (0,)
        assert cfg.env.task_assignments[1] == (1,)
        assert cfg.env.task_assignments[2] == (2,)
        assert cfg.env.task_assignments[3] == (3,)
        os.unlink(path)

    def test_rho_half(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "rho: 0.5")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        # rho=0.5, T=4 → |G_i|=2, cyclic
        assert cfg.env.task_assignments is not None
        for g in cfg.env.task_assignments:
            assert len(g) == 2
        # agent 0→{0,1}, 1→{1,2}, 2→{2,3}, 3→{3,0}
        assert cfg.env.task_assignments[0] == (0, 1)
        assert cfg.env.task_assignments[3] == (3, 0)
        os.unlink(path)

    def test_rho_one_recovers_all(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "rho: 1.0")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.task_assignments is not None
        for g in cfg.env.task_assignments:
            assert set(g) == {0, 1, 2, 3}
        os.unlink(path)

    def test_explicit_task_assignments(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "task_assignments: [[0], [1], [2], [3]]")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.task_assignments == ((0,), (1,), (2,), (3,))
        os.unlink(path)

    def test_explicit_task_assignments_multi(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "task_assignments: [[0,1,2,3], [0,1,2,3], [0,1,2,3], [0,1,2,3]]")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        for g in cfg.env.task_assignments:
            assert set(g) == {0, 1, 2, 3}
        os.unlink(path)

    def test_both_rho_and_assignments_error(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "rho: 0.25", "task_assignments: [[0],[1],[2],[3]]")
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="not both"):
            load_config(path)
        os.unlink(path)

    def test_neither_rho_nor_assignments_defaults_all(self):
        """When neither specified, default all agents → all types (rho=1)."""
        path = _write_yaml(TASK_SPEC_YAML)
        cfg = load_config(path)
        assert cfg.env.task_assignments is not None
        for g in cfg.env.task_assignments:
            assert set(g) == {0, 1, 2, 3}
        os.unlink(path)

    def test_assignments_missing_type_error(self):
        # Only cover types 0,1,2 — type 3 is uncovered
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "task_assignments: [[0], [1], [2], [0]]")
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="do not cover all types"):
            load_config(path)
        os.unlink(path)

    def test_assignments_type_out_of_range_error(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "task_assignments: [[0], [1], [2], [99]]")
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="contains type 99"):
            load_config(path)
        os.unlink(path)

    def test_rho_out_of_bounds_error(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "rho: 0.1")  # min is 0.25 for T=4
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="out of bounds"):
            load_config(path)
        os.unlink(path)

    def test_rho_non_integer_product_error(self):
        yaml_str = _add_env_fields(TASK_SPEC_YAML, "rho: 0.33")  # 0.33 * 4 = 1.32, not integer
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="not near an integer"):
            load_config(path)
        os.unlink(path)


# ---------------------------------------------------------------------------
# Pick mode
# ---------------------------------------------------------------------------
class TestPickMode:
    def test_forced_explicit(self):
        path = _write_yaml(TASK_SPEC_YAML)
        cfg = load_config(path)
        assert cfg.env.pick_mode == PickMode.FORCED
        os.unlink(path)

    def test_choice_explicit(self):
        yaml_str = TASK_SPEC_YAML.replace("pick_mode: forced", "pick_mode: choice")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.pick_mode == PickMode.CHOICE
        os.unlink(path)

    def test_choice_requires_multi_types(self):
        """pick_mode=choice with n_task_types=1 should fail."""
        yaml_str = TASK_SPEC_YAML.replace("n_task_types: 4", "n_task_types: 1").replace(
            "pick_mode: forced", "pick_mode: choice"
        )
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="CHOICE requires n_task_types > 1"):
            load_config(path)
        os.unlink(path)


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------
class TestHeuristic:
    def test_default_for_multi_types(self):
        path = _write_yaml(TASK_SPEC_YAML)
        cfg = load_config(path)
        assert cfg.train.heuristic == Heuristic.NEAREST_CORRECT_TASK
        os.unlink(path)

    def test_default_for_legacy(self):
        path = _write_yaml(LEGACY_YAML)
        cfg = load_config(path)
        assert cfg.train.heuristic == Heuristic.NEAREST_TASK
        os.unlink(path)

    def test_explicit_heuristic(self):
        yaml_str = TASK_SPEC_YAML + "  heuristic: nearest_task\n"
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.heuristic == Heuristic.NEAREST_TASK
        os.unlink(path)


# ---------------------------------------------------------------------------
# Stopping condition
# ---------------------------------------------------------------------------
class TestStoppingCondition:
    def test_running_max_rps(self):
        yaml_str = TASK_SPEC_YAML + "  stopping_condition: running_max_rps\n"
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.stopping_condition == StoppingCondition.RUNNING_MAX_RPS
        os.unlink(path)


# ---------------------------------------------------------------------------
# New encoder types
# ---------------------------------------------------------------------------
class TestEncoderTypes:
    def test_task_cnn_grid(self):
        path = _write_yaml(TASK_SPEC_YAML)
        cfg = load_config(path)
        assert cfg.model.input_type == EncoderType.TASK_CNN_GRID
        os.unlink(path)

    def test_centralized_task_cnn_grid(self):
        yaml_str = TASK_SPEC_YAML.replace("task_cnn_grid", "centralized_task_cnn_grid")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.model.input_type == EncoderType.CENTRALIZED_TASK_CNN_GRID
        os.unlink(path)

    def test_old_encoder_with_multi_types_warns(self):
        yaml_str = TASK_SPEC_YAML.replace("task_cnn_grid", "cnn_grid")
        path = _write_yaml(yaml_str)
        with pytest.warns(UserWarning, match="old encoder"):
            load_config(path)
        os.unlink(path)


# ---------------------------------------------------------------------------
# use_vmap
# ---------------------------------------------------------------------------
class TestUseVmap:
    def test_default_false(self):
        path = _write_yaml(TASK_SPEC_YAML)
        cfg = load_config(path)
        assert cfg.train.use_vmap is False
        os.unlink(path)

    def test_explicit_true(self):
        yaml_str = TASK_SPEC_YAML + "  use_vmap: true\n"
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.train.use_vmap is True
        os.unlink(path)


# ---------------------------------------------------------------------------
# Overrides (preserved from old tests)
# ---------------------------------------------------------------------------
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
        path = _write_yaml(LEGACY_YAML)
        cfg = load_config(path, overrides=["train.total_steps=999"])
        assert cfg.train.total_steps == 999
        os.unlink(path)


# ---------------------------------------------------------------------------
# compute_task_assignments
# ---------------------------------------------------------------------------
class TestComputeTaskAssignments:
    def test_max_specialization(self):
        result = compute_task_assignments(4, 4, 0.25)
        assert result == ((0,), (1,), (2,), (3,))

    def test_half_specialization(self):
        result = compute_task_assignments(4, 4, 0.5)
        assert result == ((0, 1), (1, 2), (2, 3), (3, 0))

    def test_no_specialization(self):
        result = compute_task_assignments(4, 4, 1.0)
        for g in result:
            assert set(g) == {0, 1, 2, 3}

    def test_coverage_guaranteed(self):
        """All types must be covered."""
        result = compute_task_assignments(7, 7, 1.0 / 7)
        covered = set()
        for g in result:
            covered.update(g)
        assert covered == set(range(7))

    def test_more_types_than_agents(self):
        """T=8, N=4, rho=0.25 → |G_i|=2, spaced by T/N=2."""
        result = compute_task_assignments(4, 8, 0.25)
        assert all(len(g) == 2 for g in result)
        # Starts at 0, 2, 4, 6 — each gets 2 consecutive types
        assert result == ((0, 1), (2, 3), (4, 5), (6, 7))
        covered = set()
        for g in result:
            covered.update(g)
        assert covered == set(range(8))
