"""Integration tests: full train loop runs without errors (chunk 5).

These are smoke tests — they run a tiny number of steps and just check
that the loop completes, produces a metrics.csv, and the csv has the
expected columns. They do NOT check correctness of learning.
"""

import csv
import os
import tempfile

import pytest
import yaml

from orchard.config import load_config


def _write_config(yaml_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(yaml_str)
    return path


# Base configs for integration tests
LEGACY_CONFIG = """
env:
  height: 4
  width: 4
  n_agents: 2
  n_tasks: 2
  gamma: 0.9
  r_picker: 1.0
  type: stochastic
  stochastic:
    spawn_prob: 0.1
    despawn_mode: probability
    despawn_prob: 0.05
model:
  input_type: cnn_grid
  model_type: cnn
  conv_specs: [[4, 3]]
  mlp_dims: []
  activation: leaky_relu
  weight_init: zero_bias
train:
  mode: {mode}
  td_target: after_state
  train_method: backward_view
  td_lambda: 0.3
  total_steps: 50
  seed: 42
  batch_actions: true
  lr:
    start: 0.01
    schedule: none
  policy_learning:
    epsilon:
      start: 0.3
      end: 0.3
      schedule: none
eval:
  rollout_len: 20
  eval_steps: 20
  n_test_states: 5
logging:
  main_csv_freq: 50
  detail_csv_freq: 50
  output_dir: {output_dir}
"""

TASK_SPEC_CONFIG = """
env:
  height: 5
  width: 5
  n_agents: 4
  n_tasks: 2
  n_task_types: 4
  r_high: 1.0
  r_low: 0.0
  gamma: 0.99
  r_picker: 1.0
  pick_mode: {pick_mode}
  max_tasks_per_type: 2
  task_assignments: [[0], [1], [2], [3]]
  type: stochastic
  stochastic:
    spawn_prob: 0.04
    despawn_mode: probability
    despawn_prob: 0.05
model:
  input_type: task_cnn_grid
  model_type: cnn
  conv_specs: [[4, 3]]
  mlp_dims: []
  activation: leaky_relu
  weight_init: zero_bias
train:
  mode: {mode}
  td_target: after_state
  train_method: backward_view
  td_lambda: 0.3
  learning_type: decentralized
  total_steps: 50
  seed: 42
  batch_actions: true
  heuristic: nearest_correct_task
  lr:
    start: 0.01
    schedule: none
  policy_learning:
    epsilon:
      start: 0.3
      end: 0.3
      schedule: none
eval:
  rollout_len: 20
  eval_steps: 20
  n_test_states: 5
logging:
  main_csv_freq: 50
  detail_csv_freq: 50
  output_dir: {output_dir}
"""


def _run_train(yaml_template: str, mode: str, pick_mode: str = "forced", **extra):
    """Run training and return (run_dir, metrics_rows).
    
    Caller is responsible for cleanup of the parent tmpdir.
    """
    from orchard.train import train

    tmpdir = tempfile.mkdtemp()
    yaml_str = yaml_template.format(
        mode=mode, output_dir=tmpdir, pick_mode=pick_mode, **extra
    )
    path = _write_config(yaml_str)
    cfg = load_config(path)
    train(cfg)
    os.unlink(path)

    # Find the run directory (timestamp-named)
    run_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    assert len(run_dirs) == 1, f"Expected 1 run dir, got {run_dirs}"
    run_dir = os.path.join(tmpdir, run_dirs[0])

    # Read metrics.csv
    metrics_path = os.path.join(run_dir, "metrics.csv")
    assert os.path.exists(metrics_path), "metrics.csv not found"
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return run_dir, rows


class TestLegacyIntegration:
    """n_task_types=1 training runs without errors."""

    def test_policy_learning(self):
        _, rows = _run_train(LEGACY_CONFIG, "policy_learning")
        assert len(rows) >= 1
        assert "greedy_rps" in rows[0]
        assert "nearest_task_rps" in rows[0]
        assert "greedy_pps" not in rows[0]
        assert "nearest_pps" not in rows[0]

    def test_reward_learning(self):
        _, rows = _run_train(LEGACY_CONFIG, "reward_learning")
        assert len(rows) >= 1
        assert "mae_avg" in rows[0]


class TestTaskSpecIntegration:
    """n_task_types>1 training runs without errors."""

    def test_policy_learning_forced(self):
        _, rows = _run_train(TASK_SPEC_CONFIG, "policy_learning", "forced")
        assert len(rows) >= 1
        assert "greedy_rps" in rows[0]
        assert "greedy_correct_pps" in rows[0]
        assert "greedy_wrong_pps" in rows[0]

    def test_policy_learning_choice(self):
        _, rows = _run_train(TASK_SPEC_CONFIG, "policy_learning", "choice")
        assert len(rows) >= 1
        assert "greedy_rps" in rows[0]

    def test_reward_learning_forced(self):
        _, rows = _run_train(TASK_SPEC_CONFIG, "reward_learning", "forced")
        assert len(rows) >= 1
        assert "mae_avg" in rows[0]
        # Should have task-specialization categories
        assert "mae_no_pick_avg" in rows[0]

    def test_checkpoint_exists(self):
        run_dir, _ = _run_train(TASK_SPEC_CONFIG, "policy_learning", "forced")
        assert os.path.exists(os.path.join(run_dir, "checkpoints", "final.pt"))
        assert os.path.exists(os.path.join(run_dir, "checkpoints", "step_0.pt"))
