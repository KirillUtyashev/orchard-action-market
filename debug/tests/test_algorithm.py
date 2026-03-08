from __future__ import annotations

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from debug.code.config import load_config
    from debug.code.learning import Learning

import csv
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests"
)

if TORCH_AVAILABLE:
    from debug.code.config import load_config
    from debug.code.enums import L, NUM_AGENTS, W
    from debug.code.environment import MoveAction, Orchard
    from debug.code.helpers import set_all_seeds
    from debug.code.learning import Learning


BASE_CONFIG = Path(__file__).resolve().parents[1] / "code" / "configs" / "base.yaml"


def _write_test_yaml(path: Path, output_dir: Path) -> None:
    yaml_text = f"""
network:
  mlp_dims: [64, 64]
  conv_channels: [16, 32]
  kernel_size: 3
  CNN: false
  input_dim: 0

train:
  alpha: 0.01
  timesteps: 2
  seed: 1234
  epsilon: 0.05
  lmda: -1.0
  schedule_lr: false
  load_weights: false

algorithm:
  forward: false
  eligibility: false
  monte_carlo: false
  random_policy: false
  q_agent: 0.4
  centralized: true
  concat: false

reward:
  picker_r: -1
  supervised: false
  reward_learning: false
  top_k_num_apples: 1

eval:
  num_eval_states: 0
  num_seeds: 10
  variance: 0.0
  debug: true
  action_prob_num_states: 100
  action_prob_burnin: 0
  action_prob_stride: 1

env:
  length: 6
  width: 6
  apple_life: 10.0
  max_apples: 9

logging:
  output_dir: "{output_dir}"
  main_csv_freq: 1
"""
    path.write_text(yaml_text.strip() + "\n")


def _build_interior_state() -> dict:
    apples = np.zeros((W, L), dtype=int)
    agents = np.zeros((W, L), dtype=int)
    base_positions = [(2, 2), (2, 3), (3, 2), (3, 3), (1, 2), (2, 1)]
    if NUM_AGENTS > len(base_positions):
        raise RuntimeError("Increase base_positions to support current NUM_AGENTS")

    agent_positions = np.array(base_positions[:NUM_AGENTS], dtype=int)
    for r, c in agent_positions:
        agents[r, c] += 1

    return {
        "apples": apples,
        "agents": agents,
        "agent_positions": agent_positions,
    }


def _clone_state(state: dict) -> dict:
    return {
        "apples": state["apples"].copy(),
        "agents": state["agents"].copy(),
        "agent_positions": state["agent_positions"].copy(),
    }


class CyclingController:
    """Cycles deterministically through all 5 actions for stable probability checks."""

    def __init__(self):
        self._idx = 0
        self._actions = [
            MoveAction.LEFT,
            MoveAction.RIGHT,
            MoveAction.UP,
            MoveAction.DOWN,
            MoveAction.STAY,
        ]

    def agent_get_action(self, env, agent_id, epsilon=None):
        action = self._actions[self._idx % len(self._actions)]
        self._idx += 1
        return env.agent_positions[agent_id] + action.vector


@pytest.fixture
def learner_with_test_config(tmp_path):
    config_path = tmp_path / "test.yaml"
    output_dir = tmp_path / "runs"
    _write_test_yaml(config_path, output_dir)

    # Follow run_experiments flow: load config -> set seeds -> init Learning.
    cfg = load_config(config_path)
    set_all_seeds(seed=cfg.train.seed)
    learner = Learning(cfg)
    learner.train_start_time = time.time()

    learner.env = Orchard(
        W,
        L,
        NUM_AGENTS,
        learner.reward_module,
        p_apple=0.05,
        d_apple=0.05,
        max_apples=cfg.env.max_apples,
    )
    learner.env.set_positions()
    learner.agent_controller = CyclingController()

    interior_state = _build_interior_state()
    learner._sample_action_probability_states = (
        lambda n: [_clone_state(interior_state) for _ in range(n)]
    )

    yield learner

    learner.main_logger.close()
    for logger in learner.action_prob_loggers.values():
        logger.close()


def _read_action_prob_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_action_probability_csv_schema_and_rollover(learner_with_test_config):
    learner = learner_with_test_config
    learner.evaluate_action_probabilities(step=5)
    learner.evaluate_action_probabilities(step=6)

    for agent_id in range(NUM_AGENTS):
        csv_path = learner.data_dir / f"action_probabilities_agent_{agent_id}.csv"
        assert csv_path.exists()

        rows = _read_action_prob_rows(csv_path)
        assert len(rows) == 2
        assert set(rows[0].keys()) == {"step", "wall_time", "left", "right", "up", "down", "stay"}
        assert rows[0]["step"] == "5"
        assert rows[1]["step"] == "6"
        assert float(rows[1]["wall_time"]) >= float(rows[0]["wall_time"]) >= 0.0

        for row in rows:
            probs = [float(row[a]) for a in ("left", "right", "up", "down", "stay")]
            assert all(0.0 <= p <= 1.0 for p in probs)
            assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_action_probabilities_are_roughly_uniform(learner_with_test_config):
    learner = learner_with_test_config
    learner.evaluate_action_probabilities(step=10)

    target = 1.0 / 5.0
    tolerance = 0.10  # 10% absolute tolerance as requested

    for agent_id in range(NUM_AGENTS):
        csv_path = learner.data_dir / f"action_probabilities_agent_{agent_id}.csv"
        rows = _read_action_prob_rows(csv_path)
        assert rows, f"Expected at least one row in action_probabilities_agent_{agent_id}.csv"
        row = rows[-1]

        for action in ("left", "right", "up", "down", "stay"):
            prob = float(row[action])
            assert abs(prob - target) <= tolerance, f"agent={agent_id} {action} prob {prob:.4f} not within tolerance of {target:.4f}"



def _make_cfg(tmp_path: Path, overrides: list[str]):
    return load_config(
        BASE_CONFIG,
        overrides=[
            f"logging.output_dir={tmp_path.as_posix()}",
            "train.timesteps=1",
            "eval.num_eval_states=0",
            "eval.action_prob_num_states=0",
            "algorithm.random_policy=false",
            "reward.reward_learning=false",
        ]
        + overrides,
    )


def _close_learning_loggers(learning: Learning) -> None:
    learning.main_logger.close()
    for logger in learning.action_prob_loggers.values():
        logger.close()
    for logger in learning.weight_sample_loggers.values():
        logger.close()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_weight_sampling_config_overrides_are_loaded(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "logging.weight_samples_enabled=true",
            "logging.weight_samples_per_tensor=12",
            "logging.weight_samples_freq=7",
        ],
    )

    assert cfg.logging.weight_samples_enabled is True
    assert cfg.logging.weight_samples_per_tensor == 12
    assert cfg.logging.weight_samples_freq == 7


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_weight_sample_indices_initialized_for_cnn_and_mlp(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "network.CNN=true",
            "algorithm.centralized=true",
            "logging.weight_samples_enabled=true",
            "logging.weight_samples_per_tensor=16",
        ],
    )

    learning = Learning(cfg)
    learning.build_experiment()

    assert learning.weight_sample_indices, "weight_sample_indices should be initialized"
    assert 0 in learning.weight_sample_indices

    tensor_map = learning.weight_sample_indices[0]
    assert tensor_map, "Expected sampled tensors for agent/network 0"

    assert any(name.startswith("cnn.") for name in tensor_map), "Missing CNN tensor samples"
    assert any(name.startswith("mlp.") for name in tensor_map), "Missing MLP tensor samples"

    params = dict(learning.critic_networks[0].model.named_parameters())
    for tensor_name, sample_indices in tensor_map.items():
        assert tensor_name in params
        assert sample_indices.ndim == 1
        assert sample_indices.size > 0
        assert sample_indices.size <= learning.weight_samples_per_tensor

        numel = int(params[tensor_name].numel())
        assert sample_indices.max() < numel
        assert sample_indices.min() >= 0
        assert len(set(int(v) for v in sample_indices.tolist())) == sample_indices.size
    _close_learning_loggers(learning)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_weight_sample_logging_respects_frequency(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "network.CNN=true",
            "algorithm.centralized=true",
            "logging.weight_samples_enabled=true",
            "logging.weight_samples_per_tensor=8",
            "logging.weight_samples_freq=2",
        ],
    )

    learning = Learning(cfg)
    learning.build_experiment()

    rows_per_snapshot = sum(
        len(indices)
        for tensor_map in learning.weight_sample_indices.values()
        for indices in tensor_map.values()
    )
    assert rows_per_snapshot > 0

    learning._maybe_log_weight_samples(step=0)
    learning._maybe_log_weight_samples(step=1)
    learning._maybe_log_weight_samples(step=2)

    all_rows = []
    for network_id in range(len(learning.critic_networks)):
        csv_path = learning.data_dir / f"weight_samples_network_{network_id}.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"Expected rows in {csv_path.name}"
        assert {int(r["step"]) for r in rows} == {0, 2}
        all_rows.extend(rows)

    assert len(all_rows) == rows_per_snapshot * 2

    _close_learning_loggers(learning)
