from __future__ import annotations

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from debug.code.core.config import load_config
    from debug.code.training.learning import Learning

import csv
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests"
)

if TORCH_AVAILABLE:
    from debug.code.agents.actor_critic_agent import ACAgent
    from debug.code.agents.controllers import (
        AgentControllerActorCritic,
        AgentControllerCentralized,
        AgentControllerDecentralized,
    )
    import debug.code.training.learning as learning_mod
    import debug.code.training.learning_eval as learning_eval_mod
    import debug.code.training.learning_loop as learning_loop_mod
    import debug.code.training.learning_setup as learning_setup_mod
    from debug.code.core.config import load_config
    from debug.code.env.environment import MoveAction, Orchard
    from debug.code.nn.encoders import CenGridEncoder, DecCenteredGridEncoder, DecGridEncoder
    from debug.code.nn.policy_network import PolicyNetwork
    from debug.code.training.helpers import set_all_seeds
    from debug.code.training.learning import Learning


BASE_CONFIG = Path(__file__).resolve().parents[1] / "code" / "configs" / "base.yaml"
if TORCH_AVAILABLE:
    BASE_CFG = load_config(BASE_CONFIG)
    NUM_AGENTS = int(BASE_CFG.env.num_agents)
    W = int(BASE_CFG.env.width)
    L = int(BASE_CFG.env.length)
else:
    NUM_AGENTS = 2
    W = 6
    L = 6


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
  centralized: false
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
  action_prob_num_states: 20
  action_prob_burnin: 0
  action_prob_stride: 1
  value_track_num_states: 10

env:
  num_agents: 2
  length: 6
  width: 6
  apple_life: 10.0
  max_apples: 9

logging:
  output_dir: "{output_dir}"
  main_csv_freq: 1
  weight_samples_enabled: false
  weight_samples_per_tensor: 16
  weight_samples_freq: 0
"""
    path.write_text(yaml_text.strip() + "\n")


def _write_reward_learning_yaml(path: Path, output_dir: Path) -> None:
    yaml_text = f"""
network:
  mlp_dims: [64, 64]
  conv_channels: [16, 32]
  kernel_size: 3
  CNN: false
  input_dim: 0

train:
  alpha: 0.01
  timesteps: 4
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
  centralized: false
  concat: false

reward:
  picker_r: -1
  supervised: false
  reward_learning: true
  top_k_num_apples: 1

eval:
  num_eval_states: 0
  num_seeds: 10
  variance: 0.0
  debug: true
  reward_eval_num_states: 1000
  reward_eval_zero_frac: 0.25
  action_prob_num_states: 0
  action_prob_burnin: 0
  action_prob_stride: 1
  value_track_num_states: 0

env:
  num_agents: 2
  length: 6
  width: 6
  apple_life: 10.0
  max_apples: 9

logging:
  output_dir: "{output_dir}"
  main_csv_freq: 2
  weight_samples_enabled: false
  weight_samples_per_tensor: 16
  weight_samples_freq: 0
"""
    path.write_text(yaml_text.strip() + "\n")


def _write_supervised_learning_yaml(path: Path, output_dir: Path, weights_path: Path) -> None:
    yaml_text = f"""
network:
  mlp_dims: [64, 64]
  conv_channels: [16, 32]
  kernel_size: 3
  CNN: false
  input_dim: 0

train:
  alpha: 0.01
  timesteps: 4
  seed: 1234
  epsilon: 0.05
  lmda: -1.0
  schedule_lr: false
  load_weights: false

algorithm:
  random_policy: false
  q_agent: 0.4
  centralized: false
  concat: false

reward:
  picker_r: -1
  supervised: true
  reward_learning: false
  top_k_num_apples: 1

supervised:
  weights_path: "{weights_path}"
  CNN: false
  mlp_dims: [64, 64]
  conv_channels: [16, 32]
  kernel_size: 3

eval:
  num_eval_states: 0
  num_seeds: 10
  variance: 0.0
  debug: true
  supervised_eval_num_states: 25
  action_prob_num_states: 0
  action_prob_burnin: 0
  action_prob_stride: 1
  value_track_num_states: 0

env:
  num_agents: 2
  length: 6
  width: 6
  apple_life: 10.0
  max_apples: 9

logging:
  output_dir: "{output_dir}"
  main_csv_freq: 2
  weight_samples_enabled: false
  weight_samples_per_tensor: 16
  weight_samples_freq: 0
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


class DummyEnvState:
    def __init__(self, state: dict, width: int, length: int):
        self._state = {
            "agents": state["agents"].copy(),
            "apples": state["apples"].copy(),
            "agent_positions": state["agent_positions"].copy(),
        }
        self.width = int(width)
        self.length = int(length)
        self.agent_positions = self._state["agent_positions"].copy()

    def get_state(self) -> dict:
        return {
            "agents": self._state["agents"].copy(),
            "apples": self._state["apples"].copy(),
            "agent_positions": self._state["agent_positions"].copy(),
        }


class DummyValueNet:
    def __init__(self, grid_shape: tuple[int, int, int], scalar_dim: int):
        size = int(np.prod(grid_shape))
        self.grid_weights = torch.arange(1, size + 1, dtype=torch.float32).reshape(grid_shape) / float(max(size, 1))
        self.scalar_weights = (
            torch.arange(1, scalar_dim + 1, dtype=torch.float32) / float(max(scalar_dim, 1))
            if scalar_dim > 0
            else None
        )

    def get_value_function(self, enc):
        value = torch.tensor(0.0, dtype=torch.float32)
        if enc.grid is not None:
            value = value + (enc.grid.float() * self.grid_weights).sum()
        if enc.scalar is not None and self.scalar_weights is not None:
            value = value + (enc.scalar.float() * self.scalar_weights).sum()
        return float(value.item())

    def get_value_function_batch(self, enc):
        if enc.grid is not None:
            grid = enc.grid.float()
            if grid.dim() == 3:
                grid = grid.unsqueeze(0)
            values = (grid * self.grid_weights).sum(dim=(1, 2, 3))
        else:
            if enc.scalar is None:
                raise ValueError("Expected grid or scalar inputs.")
            scalar = enc.scalar.float()
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(0)
            values = torch.zeros(scalar.shape[0], dtype=torch.float32)

        if enc.scalar is not None and self.scalar_weights is not None:
            scalar = enc.scalar.float()
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(0)
            values = values + (scalar * self.scalar_weights).sum(dim=1)

        return values.detach().cpu().numpy()


class DummyPolicyAgent:
    def __init__(self, value_net):
        self.policy_value = value_net

    def get_value_function(self, enc):
        return self.policy_value.get_value_function(enc)


def _manual_best_action(controller, env, actor_id: int):
    best_action = env.agent_positions[actor_id].copy()
    best_val = -1_000_000.0
    position = env.agent_positions[actor_id].copy()

    for act in MoveAction:
        curr_state = env.get_state()
        r, c = curr_state["agent_positions"][actor_id]
        nr, nc = r + act.vector[0], c + act.vector[1]
        if not (0 <= nr < env.width and 0 <= nc < env.length):
            continue

        new_pos = np.array([nr, nc], dtype=np.int64)
        curr_state["agents"][tuple(new_pos)] += 1
        curr_state["agents"][tuple(position)] -= 1
        curr_state["agent_positions"][actor_id] = new_pos
        curr_state["actor_id"] = actor_id

        encoded_states = controller.get_all_agent_obs(curr_state)
        val = controller.discount * controller.get_collective_value(encoded_states, actor_id)
        if val > best_val:
            best_action = new_pos
            best_val = val

    return best_action


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


def test_eval_performance_saves_greedy_positions_and_heatmaps(learner_with_test_config):
    pytest.importorskip("matplotlib")

    learner = learner_with_test_config
    learner.agent_controller.epsilon = learner.exp_config.train.epsilon

    class DummyCritic:
        @staticmethod
        def get_lr():
            return 0.01

    learner.critic_networks = [DummyCritic()]

    fake_positions = np.zeros((8, NUM_AGENTS, 2), dtype=np.int16)
    for t in range(fake_positions.shape[0]):
        for agent_id in range(NUM_AGENTS):
            fake_positions[t, agent_id, 0] = (t + agent_id) % W
            fake_positions[t, agent_id, 1] = (2 * t + agent_id) % L

    original_eval_performance = learning_mod.eval_performance

    def fake_eval_performance(**kwargs):
        return {
            "greedy_pps": 10,
            "total_apples": 20,
            "greedy_ratio": 0.5,
            "nearest_pps": 9,
            "nearest_ratio": 0.45,
            "nearest_total_apples": 20,
            "greedy_agent_positions": fake_positions,
        }

    learning_mod.eval_performance = fake_eval_performance
    try:
        learner.eval_performance(step=3)
    finally:
        learning_mod.eval_performance = original_eval_performance

    pos_path = learner.data_dir / "agent_positions" / "greedy_eval_step_000000003.npz"
    assert pos_path.exists()
    with np.load(pos_path) as data:
        saved = data["positions"]
    assert saved.shape == fake_positions.shape
    assert np.array_equal(saved, fake_positions)

    learner._write_last_greedy_position_heatmaps()
    for agent_id in range(NUM_AGENTS):
        heatmap_path = learner.data_dir / "agent_positions" / f"agent_{agent_id}_heatmap.png"
        assert heatmap_path.exists()



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
    for logger in learning.value_track_loggers.values():
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
def test_separate_actor_and_critic_schedule_flags_are_loaded(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "algorithm.actor_critic=true",
            "train.schedule_lr=false",
            "train.critic_schedule_lr=false",
            "train.actor_schedule_lr=true",
        ],
    )

    assert cfg.train.schedule_lr is False
    assert cfg.train.critic_schedule_lr is False
    assert cfg.train.actor_schedule_lr is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_actor_and_critic_can_use_separate_lr_schedulers(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "algorithm.actor_critic=true",
            "train.schedule_lr=false",
            "train.critic_schedule_lr=false",
            "train.actor_schedule_lr=true",
        ],
    )

    learning = Learning(cfg)
    learning.build_experiment()

    assert learning.critic_networks
    assert learning.policy_networks
    assert all(net.scheduler is None for net in learning.critic_networks)
    assert all(net.scheduler is not None for net in learning.policy_networks)

    _close_learning_loggers(learning)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_separate_actor_and_critic_network_configs_are_loaded(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "network.CNN=true",
            "network.MLP=false",
            "network.conv_channels=[8]",
            "critic_network.CNN=true",
            "critic_network.MLP=false",
            "critic_network.conv_channels=[4]",
            "actor_network.CNN=true",
            "actor_network.MLP=true",
            "actor_network.conv_channels=[4,8]",
            "actor_network.mlp_dims=[32]",
        ],
    )

    assert list(cfg.critic_network.conv_channels) == [4]
    assert bool(cfg.critic_network.MLP) is False
    assert list(cfg.actor_network.conv_channels) == [4, 8]
    assert bool(cfg.actor_network.MLP) is True
    assert list(cfg.actor_network.mlp_dims) == [32]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_actor_and_critic_networks_can_use_different_architectures(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "algorithm.actor_critic=true",
            "network.CNN=true",
            "network.self_centered_grid=false",
            "critic_network.CNN=true",
            "critic_network.MLP=false",
            "critic_network.conv_channels=[4]",
            "critic_network.mlp_dims=[0]",
            "actor_network.CNN=true",
            "actor_network.MLP=true",
            "actor_network.conv_channels=[4,8]",
            "actor_network.mlp_dims=[32]",
        ],
    )

    learning = Learning(cfg)
    learning.build_experiment()

    critic_model = learning.critic_networks[0].model
    actor_model = learning.policy_networks[0].model

    critic_conv_layers = [m for m in critic_model.cnn if isinstance(m, torch.nn.Conv2d)]
    actor_conv_layers = [m for m in actor_model.cnn if isinstance(m, torch.nn.Conv2d)]

    assert len(critic_conv_layers) == 1
    assert len(actor_conv_layers) == 2
    assert isinstance(critic_model.head, torch.nn.Linear)
    assert isinstance(actor_model.head, torch.nn.Sequential)

    _close_learning_loggers(learning)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_actor_and_critic_must_share_encoder_style(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "algorithm.actor_critic=true",
            "critic_network.CNN=true",
            "actor_network.CNN=false",
        ],
    )

    with pytest.raises(ValueError, match="actor_network.CNN"):
        Learning(cfg)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_dec_centered_grid_encoder_centers_self_and_marks_outside():
    encoder = DecCenteredGridEncoder(3, 4, 3)
    state = {
        "apples": np.zeros((3, 4), dtype=int),
        "agents": np.zeros((3, 4), dtype=int),
        "agent_positions": np.array([[0, 0], [1, 2], [2, 1]], dtype=int),
        "actor_id": 2,
    }
    state["apples"][2, 3] = 1

    encoded = encoder.encode(state, agent_idx=0)
    grid = encoded.grid.numpy()

    assert grid.shape == (3, 5, 7)
    assert encoded.scalar.tolist() == pytest.approx([0.0])

    assert np.all(grid[:, 0, 0] == -1.0)
    assert np.all(grid[:, 2, 3] == 0.0)
    assert grid[0, 4, 6] == pytest.approx(1.0)
    assert grid[1, 3, 5] == pytest.approx(1.0)
    assert grid[2, 4, 4] == pytest.approx(1.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_dec_centered_grid_encoder_marks_actor_at_center_when_self_is_actor():
    encoder = DecCenteredGridEncoder(4, 4, 2)
    state = {
        "apples": np.zeros((4, 4), dtype=int),
        "agents": np.zeros((4, 4), dtype=int),
        "agent_positions": np.array([[1, 2], [0, 0]], dtype=int),
        "actor_id": 0,
    }

    encoded = encoder.encode(state, agent_idx=0)
    grid = encoded.grid.numpy()

    assert encoded.scalar.tolist() == pytest.approx([1.0])
    assert grid[2, encoder.center_r, encoder.center_c] == pytest.approx(1.0)
    assert grid[1, encoder.center_r, encoder.center_c] == pytest.approx(0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_decentralized_grid_controller_batched_action_selection_matches_manual():
    state = {
        "apples": np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "agents": np.zeros((4, 4), dtype=int),
        "agent_positions": np.array([[1, 1], [2, 2]], dtype=int),
    }
    for r, c in state["agent_positions"]:
        state["agents"][r, c] += 1

    encoder = DecGridEncoder(4, 4, 2)
    agents = [
        DummyPolicyAgent(DummyValueNet((4, 4, 4), 1)),
        DummyPolicyAgent(DummyValueNet((4, 4, 4), 1)),
    ]
    controller = AgentControllerDecentralized(agents, encoder, discount=0.99, epsilon=0.0)
    env = DummyEnvState(state, width=4, length=4)

    expected = _manual_best_action(controller, env, actor_id=0)
    actual = controller.get_best_action(env, actor_id=0)
    assert np.array_equal(actual, expected)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_centralized_grid_controller_batched_action_selection_matches_manual():
    state = {
        "apples": np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ],
            dtype=int,
        ),
        "agents": np.zeros((4, 4), dtype=int),
        "agent_positions": np.array([[1, 1], [0, 3]], dtype=int),
    }
    for r, c in state["agent_positions"]:
        state["agents"][r, c] += 1

    encoder = CenGridEncoder(4, 4, 2)
    agents = [DummyPolicyAgent(DummyValueNet((3, 4, 4), 0))]
    controller = AgentControllerCentralized(agents, encoder, discount=0.99, epsilon=0.0)
    env = DummyEnvState(state, width=4, length=4)

    expected = _manual_best_action(controller, env, actor_id=0)
    actual = controller.get_best_action(env, actor_id=0)
    assert np.array_equal(actual, expected)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_self_centered_controller_batched_forward_fallback_matches_manual():
    state = {
        "apples": np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=int,
        ),
        "agents": np.zeros((4, 4), dtype=int),
        "agent_positions": np.array([[1, 1], [2, 3]], dtype=int),
    }
    for r, c in state["agent_positions"]:
        state["agents"][r, c] += 1

    encoder = DecCenteredGridEncoder(4, 4, 2)
    agents = [
        DummyPolicyAgent(DummyValueNet((3, encoder.H, encoder.W), 1)),
        DummyPolicyAgent(DummyValueNet((3, encoder.H, encoder.W), 1)),
    ]
    controller = AgentControllerDecentralized(agents, encoder, discount=0.99, epsilon=0.0)
    env = DummyEnvState(state, width=4, length=4)

    expected = _manual_best_action(controller, env, actor_id=0)
    actual = controller.get_best_action(env, actor_id=0)
    assert np.array_equal(actual, expected)


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
def test_self_centered_grid_encoder_is_selected_for_decentralized_cnn(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "network.CNN=true",
            "network.self_centered_grid=true",
            "algorithm.centralized=false",
        ],
    )

    learning = Learning(cfg)
    learning.build_experiment()

    assert isinstance(learning.encoder, DecCenteredGridEncoder)
    assert learning.encoder.grid_channels() == 3
    assert learning.encoder.H == (2 * learning.width - 1)
    assert learning.encoder.W == (2 * learning.length - 1)

    _close_learning_loggers(learning)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for algorithm/network tests")
def test_self_centered_grid_flag_requires_decentralized_cnn(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "network.CNN=false",
            "network.self_centered_grid=true",
        ],
    )

    with pytest.raises(ValueError, match="self_centered_grid"):
        Learning(cfg)


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


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


@pytest.fixture
def learner_with_config(tmp_path):
    config_path = tmp_path / "test.yaml"
    output_dir = tmp_path / "runs"
    _write_test_yaml(config_path, output_dir)

    cfg = load_config(config_path)
    set_all_seeds(seed=cfg.train.seed)
    learner = Learning(cfg)
    learner.train_start_time = time.time()

    yield learner

    learner.main_logger.close()
    for logger in learner.action_prob_loggers.values():
        logger.close()
    for logger in learner.value_track_loggers.values():
        logger.close()


@pytest.fixture
def reward_learning_learner(tmp_path):
    config_path = tmp_path / "reward_learning_test.yaml"
    output_dir = tmp_path / "runs"
    _write_reward_learning_yaml(config_path, output_dir)

    cfg = load_config(config_path)
    set_all_seeds(seed=cfg.train.seed)
    learner = Learning(cfg)
    learner.train_start_time = time.time()

    yield learner

    learner.main_logger.close()
    for logger in learner.action_prob_loggers.values():
        logger.close()
    for logger in learner.value_track_loggers.values():
        logger.close()
    for logger in learner.weight_sample_loggers.values():
        logger.close()


@pytest.fixture
def actor_critic_learner(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "algorithm.actor_critic=true",
            "algorithm.centralized=false",
            "train.actor_alpha=0.02",
            "reward.reward_learning=false",
        ],
    )

    set_all_seeds(seed=cfg.train.seed)
    learner = Learning(cfg)
    learner.train_start_time = time.time()

    yield learner

    _close_learning_loggers(learner)


def test_policy_network_masks_illegal_actions_and_normalizes():
    encoder = DecGridEncoder(4, 4, 2)
    network = PolicyNetwork(
        encoder,
        output_dim=len(MoveAction),
        alpha=0.01,
        discount=0.99,
        mlp_dims=(16,),
        num_training_steps=10,
    )
    state = {
        "apples": np.zeros((4, 4), dtype=int),
        "agents": np.zeros((4, 4), dtype=int),
        "agent_positions": np.array([[0, 0], [2, 2]], dtype=int),
        "actor_id": 0,
    }
    for r, c in state["agent_positions"]:
        state["agents"][r, c] += 1

    enc = encoder.encode(state, 0)
    mask = AgentControllerActorCritic.legal_action_mask_from_position(state["agent_positions"][0], 4, 4)
    probs = network.get_action_probabilities(enc, mask)

    assert probs.shape == (len(MoveAction),)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert probs[MoveAction.LEFT.idx] == pytest.approx(0.0, abs=1e-12)
    assert probs[MoveAction.UP.idx] == pytest.approx(0.0, abs=1e-12)
    assert probs[MoveAction.RIGHT.idx] > 0.0
    assert probs[MoveAction.DOWN.idx] > 0.0
    assert probs[MoveAction.STAY.idx] > 0.0


def test_actor_critic_controller_masks_corner_actions():
    class DummyPolicy:
        def __init__(self):
            self.last_mask = None

        def sample_action(self, enc, legal_mask):
            self.last_mask = np.asarray(legal_mask, dtype=bool)
            return MoveAction.RIGHT.idx, np.asarray(legal_mask, dtype=float)

        def get_action_probabilities(self, enc, legal_mask):
            probs = np.zeros(len(MoveAction), dtype=float)
            probs[MoveAction.RIGHT.idx] = 1.0
            return probs

    class DummyValue:
        @staticmethod
        def get_value_function(enc):
            return 0.0

    encoder = DecGridEncoder(4, 4, 2)
    actor = DummyPolicy()
    agents = [
        ACAgent(lambda pos: pos, 0, DummyValue(), actor, np.zeros(2, dtype=float)),
        ACAgent(lambda pos: pos, 1, DummyValue(), DummyPolicy(), np.zeros(2, dtype=float)),
    ]
    controller = AgentControllerActorCritic(agents, encoder, discount=0.99, epsilon=0.0)

    env = DummyEnvState(
        {
            "apples": np.zeros((4, 4), dtype=int),
            "agents": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
            "agent_positions": np.array([[0, 0], [1, 1]], dtype=int),
        },
        width=4,
        length=4,
    )

    new_pos, action_idx, legal_mask = controller.sample_action(env, 0)

    assert action_idx == MoveAction.RIGHT.idx
    assert np.array_equal(new_pos, np.array([0, 1], dtype=np.int64))
    assert not bool(legal_mask[MoveAction.LEFT.idx])
    assert not bool(legal_mask[MoveAction.UP.idx])
    assert bool(legal_mask[MoveAction.RIGHT.idx])
    assert bool(legal_mask[MoveAction.DOWN.idx])
    assert bool(legal_mask[MoveAction.STAY.idx])
    assert np.array_equal(actor.last_mask, legal_mask)


def test_actor_critic_build_experiment_initializes_actors(actor_critic_learner):
    learner = actor_critic_learner
    learner.build_experiment()

    assert len(learner.policy_networks) == learner.num_agents
    assert len(learner.agents) == learner.num_agents
    assert all(isinstance(agent, ACAgent) for agent in learner.agents)
    assert isinstance(learner.agent_controller, AgentControllerActorCritic)
    assert learner._networks_for_eval == learner.policy_networks


def test_actor_critic_transition_stores_social_advantage(actor_critic_learner):
    learner = actor_critic_learner

    class DummyEncoder:
        @staticmethod
        def encode(state, agent_id):
            return (state["tag"], int(agent_id))

    class DummyCritic:
        def __init__(self, values):
            self.values = values
            self.experiences = []

        def get_value_function(self, enc):
            tag, agent_id = enc
            return float(self.values[tag][agent_id])

        def add_experience(self, *args, **kwargs):
            self.experiences.append((args, kwargs))

        def train(self):
            return 0.0

    class DummyActor:
        def __init__(self):
            self.experiences = []

        def add_experience(self, state, legal_mask, action, advantage):
            self.experiences.append(
                {
                    "state": state,
                    "legal_mask": np.asarray(legal_mask, dtype=bool),
                    "action": int(action),
                    "advantage": float(advantage),
                }
            )

        def train(self):
            return {"loss": 0.25, "advantage_mean": self.experiences[-1]["advantage"], "entropy_mean": 0.5}

    learner.encoder = DummyEncoder()
    learner.critic_networks = [
        DummyCritic({"curr": {0: 1.0, 1: 2.0}, "next": {0: 3.0, 1: 4.0}}),
        DummyCritic({"curr": {0: 1.0, 1: 2.0}, "next": {0: 3.0, 1: 4.0}}),
    ]
    learner.policy_networks = [DummyActor(), DummyActor()]
    learner.discount_factor = 0.99

    curr_state = {"tag": "curr", "actor_id": 0}
    s_moved = {"tag": "moved", "actor_id": 0}
    s_next = {"tag": "next", "actor_id": 1}
    legal_mask = np.array([False, True, True, False, True], dtype=bool)

    learner._train_actor_critic_transition(
        curr_state,
        s_moved,
        s_next,
        pick_rewards=[0.0, 0.0],
        on_apple=False,
        actor_idx=0,
        action_idx=MoveAction.RIGHT.idx,
        legal_mask=legal_mask,
    )

    expected_advantage = learner.discount_factor * (3.0 + 4.0) - (1.0 + 2.0)
    assert len(learner.policy_networks[0].experiences) == 1
    assert len(learner.policy_networks[1].experiences) == 0
    assert learner.policy_networks[0].experiences[0]["action"] == MoveAction.RIGHT.idx
    assert learner.policy_networks[0].experiences[0]["advantage"] == pytest.approx(expected_advantage)
    assert np.array_equal(learner.policy_networks[0].experiences[0]["legal_mask"], legal_mask)
    assert len(learner.critic_networks[0].experiences) == 1
    assert len(learner.critic_networks[1].experiences) == 1
    assert learner._last_actor_training_stats["actor_loss_mean"] == pytest.approx(0.25)
    assert learner._last_actor_training_stats["advantage_mean"] == pytest.approx(expected_advantage)
    assert learner._last_actor_training_stats["policy_entropy_mean"] == pytest.approx(0.5)


def test_actor_critic_metrics_header_includes_actor_columns(actor_critic_learner):
    learner = actor_critic_learner
    header = _read_header(learner.data_dir / "metrics.csv")
    assert "current_lr" in header
    assert "actor_lr" in header
    assert "actor_loss_mean" in header
    assert "advantage_mean" in header
    assert "policy_entropy_mean" in header


def test_actor_critic_action_probability_logging_uses_policy_probs(actor_critic_learner):
    learner = actor_critic_learner
    learner.action_prob_eval_states = [_build_interior_state(), _build_interior_state()]
    learner.action_prob_num_states = 2
    learner.env = object()

    class DummyActorController:
        def get_action_probabilities_from_state(self, state, agent_id):
            probs = np.zeros(len(MoveAction), dtype=float)
            probs[MoveAction.RIGHT.idx] = 0.6
            probs[MoveAction.STAY.idx] = 0.4
            return probs

    learner.agent_controller = DummyActorController()
    learner.evaluate_action_probabilities(step=9)

    for agent_id in range(NUM_AGENTS):
        rows = _read_rows(learner.data_dir / f"action_probabilities_agent_{agent_id}.csv")
        assert rows
        row = rows[-1]
        assert float(row["left"]) == pytest.approx(0.0)
        assert float(row["right"]) == pytest.approx(0.6)
        assert float(row["up"]) == pytest.approx(0.0)
        assert float(row["down"]) == pytest.approx(0.0)
        assert float(row["stay"]) == pytest.approx(0.4)


def test_actor_critic_save_networks_includes_actor_payload(actor_critic_learner, tmp_path):
    learner = actor_critic_learner

    class ExportOnlyNet:
        def __init__(self, tag):
            self.tag = tag

        def export_net_state(self):
            return {"tag": self.tag}

    learner.critic_networks = [ExportOnlyNet("critic-0"), ExportOnlyNet("critic-1")]
    learner.policy_networks = [ExportOnlyNet("actor-0"), ExportOnlyNet("actor-1")]

    out_dir = tmp_path / "weights"
    learner.save_networks(out_dir)

    payload = torch.load(out_dir / "weights.pt", map_location="cpu")
    assert [item["blob"]["tag"] for item in payload["critics"]] == ["critic-0", "critic-1"]
    assert [item["blob"]["tag"] for item in payload["actors"]] == ["actor-0", "actor-1"]


def test_freeze_critics_requires_a_checkpoint_source(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "train.freeze_critics=true",
            "train.load_weights=false",
            "train.critic_weights_path=",
        ],
    )

    with pytest.raises(ValueError, match="freeze_critics"):
        Learning(cfg)


def test_frozen_critics_load_from_explicit_weights_folder_and_disable_gradients(tmp_path):
    source_cfg = _make_cfg(
        tmp_path / "source",
        [
            "algorithm.actor_critic=true",
            "train.actor_schedule_lr=false",
            "train.critic_schedule_lr=false",
        ],
    )
    source = Learning(source_cfg)
    source.build_experiment()
    source_weights_dir = tmp_path / "pretrained" / "weights"
    source.save_networks(source_weights_dir)
    _close_learning_loggers(source)

    cfg = _make_cfg(
        tmp_path / "target",
        [
            "algorithm.actor_critic=true",
            f"train.critic_weights_path={source_weights_dir.as_posix()}",
            "train.freeze_critics=true",
            "train.load_weights=false",
        ],
    )

    learning = Learning(cfg)
    learning.build_experiment()

    assert learning.loaded_critic_checkpoint_path == source_weights_dir / "weights.pt"
    assert len(learning.critic_networks) == NUM_AGENTS
    assert len(learning.policy_networks) == NUM_AGENTS
    for critic in learning.critic_networks:
        assert critic.model.training is False
        assert all(param.requires_grad is False for param in critic.model.parameters())
    for actor in learning.policy_networks:
        assert any(param.requires_grad is True for param in actor.model.parameters())

    _close_learning_loggers(learning)


def test_loaded_frozen_critics_run_greedy_smoke_test_with_critic_controller(tmp_path):
    source_cfg = _make_cfg(
        tmp_path / "source_smoke",
        [
            "algorithm.actor_critic=true",
        ],
    )
    source = Learning(source_cfg)
    source.build_experiment()
    source_weights_dir = tmp_path / "pretrained_smoke" / "weights"
    source.save_networks(source_weights_dir)
    _close_learning_loggers(source)

    cfg = _make_cfg(
        tmp_path / "target_smoke",
        [
            "algorithm.actor_critic=true",
            f"train.critic_weights_path={source_weights_dir.as_posix()}",
            "train.freeze_critics=true",
            "train.load_weights=false",
        ],
    )
    learning = Learning(cfg)
    learning.build_experiment()

    called = {"controller": None}
    original_eval_performance = learning_eval_mod.eval_performance

    def fake_eval_performance(**kwargs):
        called["controller"] = kwargs["agent_controller"]
        return {
            "greedy_pps": 12.0,
            "greedy_ratio": 0.6,
            "nearest_pps": 10.0,
            "nearest_ratio": 0.5,
            "total_apples": 20,
            "nearest_total_apples": 20,
        }

    learning_eval_mod.eval_performance = fake_eval_performance
    try:
        result = learning._maybe_run_loaded_critic_smoke_test()
    finally:
        learning_eval_mod.eval_performance = original_eval_performance

    assert isinstance(called["controller"], AgentControllerDecentralized)
    assert not isinstance(called["controller"], AgentControllerActorCritic)
    assert result is not None
    assert result["greedy_pps"] == pytest.approx(12.0)
    assert learning.loaded_critic_smoke_test_result["greedy_ratio"] == pytest.approx(0.6)

    _close_learning_loggers(learning)


def test_frozen_critic_metrics_header_includes_smoke_columns(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "train.freeze_critics=true",
            "train.load_weights=true",
        ],
    )
    learner = Learning(cfg)

    header = _read_header(learner.data_dir / "metrics.csv")
    assert "loaded_critic_smoke_greedy_pps" in header
    assert "loaded_critic_smoke_greedy_ratio" in header
    assert "loaded_critic_smoke_nearest_pps" in header
    assert "loaded_critic_smoke_nearest_ratio" in header

    _close_learning_loggers(learner)


def test_eval_performance_logs_loaded_critic_smoke_metrics_at_step_zero(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        [
            "train.freeze_critics=true",
            "train.load_weights=true",
        ],
    )
    learner = Learning(cfg)

    class DummyController:
        epsilon = 0.05

    class DummyCritic:
        @staticmethod
        def get_lr():
            return 0.01

    class DummyEnv:
        d_apple = 0.1
        p_apple = 0.1
        max_apples = 9

    learner.agent_controller = DummyController()
    learner.critic_networks = [DummyCritic(), DummyCritic()]
    learner.env = DummyEnv()
    learner.loaded_critic_smoke_test_result = {
        "greedy_pps": 12.0,
        "greedy_ratio": 0.6,
        "nearest_pps": 10.0,
        "nearest_ratio": 0.5,
        "total_apples": 20,
        "nearest_total_apples": 20,
    }

    original_eval_performance = learning_mod.eval_performance

    def fake_eval_performance(**kwargs):
        return {
            "greedy_pps": 9.0,
            "total_apples": 18,
            "greedy_ratio": 0.45,
            "nearest_pps": 8.0,
            "nearest_ratio": 0.4,
            "nearest_total_apples": 18,
            "greedy_agent_positions": np.zeros((2, NUM_AGENTS, 2), dtype=np.int16),
        }

    learning_mod.eval_performance = fake_eval_performance
    try:
        learner.eval_performance(step=0)
    finally:
        learning_mod.eval_performance = original_eval_performance

    rows = _read_rows(learner.data_dir / "metrics.csv")
    assert rows
    row = rows[-1]
    assert row["step"] == "0"
    assert float(row["loaded_critic_smoke_greedy_pps"]) == pytest.approx(12.0)
    assert float(row["loaded_critic_smoke_greedy_ratio"]) == pytest.approx(0.6)
    assert float(row["loaded_critic_smoke_nearest_pps"]) == pytest.approx(10.0)
    assert float(row["loaded_critic_smoke_nearest_ratio"]) == pytest.approx(0.5)

    _close_learning_loggers(learner)


def test_value_tracking_state_generation_per_agent(learner_with_config):
    learner = learner_with_config
    learner._ensure_value_track_states()

    assert learner.value_track_states_by_agent is not None
    assert set(learner.value_track_states_by_agent.keys()) == set(range(NUM_AGENTS))

    for agent_id, states in learner.value_track_states_by_agent.items():
        assert len(states) == learner.value_track_num_states == 10
        for st in states:
            assert st["actor_id"] == agent_id
            assert "apples" in st and "agents" in st and "agent_positions" in st
            assert st["apples"].shape == st["agents"].shape
            assert st["agent_positions"].shape[0] == NUM_AGENTS


def test_value_tracking_logs_per_agent_csv(learner_with_config):
    learner = learner_with_config
    learner._ensure_value_track_states()

    # Avoid full NN dependency in this unit test: stub predictor + minimal guards.
    learner.encoder = object()
    learner.critic_networks = [object() for _ in range(NUM_AGENTS)]
    learner._predict_state_value = lambda state, agent_id: float(
        agent_id + state["apples"].sum() * 0.01 + state["agent_positions"][agent_id][0] * 0.001
    )

    learner.evaluate_tracked_state_values(step=0)
    learner.evaluate_tracked_state_values(step=50)

    expected_cols = {"step", "wall_time"} | {f"state_{i}" for i in range(10)}

    for agent_id in range(NUM_AGENTS):
        csv_path = learner.data_dir / f"tracked_state_values_agent_{agent_id}.csv"
        assert csv_path.exists()
        rows = _read_rows(csv_path)
        assert len(rows) == 2
        assert set(rows[0].keys()) == expected_cols
        assert rows[0]["step"] == "0"
        assert rows[1]["step"] == "50"
        assert float(rows[1]["wall_time"]) >= float(rows[0]["wall_time"]) >= 0.0

        for row in rows:
            for idx in range(10):
                float(row[f"state_{idx}"])


def test_value_learning_metrics_csv_uses_value_columns_only(learner_with_config):
    learner = learner_with_config
    header = _read_header(learner.data_dir / "metrics.csv")
    assert "current_lr" in header
    assert "greedy_pps" in header
    assert "nearest_pps" in header
    assert "reward_acc_mean" not in header
    assert "reward_mae_mean" not in header


def test_reward_learning_metrics_csv_uses_reward_columns_only(reward_learning_learner):
    learner = reward_learning_learner
    header = _read_header(learner.data_dir / "metrics.csv")
    assert "current_lr" in header
    assert "reward_acc_mean" in header
    assert "reward_mae_mean" in header
    assert "greedy_pps" not in header
    assert "nearest_pps" not in header


def test_reward_learning_does_not_create_action_prob_or_tracked_value_csvs(reward_learning_learner):
    learner = reward_learning_learner
    assert learner.action_prob_loggers == {}
    assert learner.value_track_loggers == {}

    action_prob_paths = list(learner.data_dir.glob("action_probabilities_agent_*.csv"))
    tracked_value_paths = list(learner.data_dir.glob("tracked_state_values_agent_*.csv"))
    assert action_prob_paths == []
    assert tracked_value_paths == []


def test_reward_learning_flag_is_passed_to_critics(reward_learning_learner):
    learner = reward_learning_learner
    learner.build_experiment()
    assert learner.critic_networks, "Expected critic networks after build_experiment"
    assert all(getattr(net, "reward_learning", False) is True for net in learner.critic_networks)


def test_vnetwork_train_dispatches_to_reward_supervised(reward_learning_learner):
    learner = reward_learning_learner
    learner.build_experiment()
    net = learner.critic_networks[0]

    called = {"reward": 0, "td0": 0, "lambda": 0}

    def fake_reward_supervised():
        called["reward"] += 1
        return 123.0

    def fake_td0():
        called["td0"] += 1
        return -1.0

    def fake_lambda():
        called["lambda"] += 1
        return -2.0

    net.reward_supervised = fake_reward_supervised
    net.train_td0 = fake_td0
    net.train_lambda = fake_lambda

    out = net.train()
    assert out == pytest.approx(123.0)
    assert called["reward"] == 1
    assert called["td0"] == 0
    assert called["lambda"] == 0


def test_reward_learning_generates_1000_labeled_states(reward_learning_learner):
    learner = reward_learning_learner
    learner._generate_evaluation_states_reward_learning()

    assert isinstance(learner.evaluation_states, list)
    assert len(learner.evaluation_states) == 1000
    expected_zero = int(round(learner.reward_eval_num_states * learner.reward_eval_zero_frac))
    zero_count = 0

    for st in learner.evaluation_states:
        assert "actor_id" in st
        assert "true_rewards" in st
        assert st["reward_eval_source"] in {"curr_state", "s_moved"}
        true_rewards = np.asarray(st["true_rewards"], dtype=float)
        assert true_rewards.shape == (NUM_AGENTS,)
        actor_id = int(st["actor_id"])
        actor_pos = st["agent_positions"][actor_id]
        expected = learner.reward_module.get_reward(st, actor_id, actor_pos, mode=1)
        assert np.allclose(true_rewards, expected)
        if np.allclose(true_rewards, 0.0):
            zero_count += 1
            assert st["reward_eval_source"] == "curr_state"
        else:
            assert st["reward_eval_source"] == "s_moved"

    assert zero_count == expected_zero


def test_reward_learning_eval_state_uses_curr_state_for_zero_reward(reward_learning_learner):
    learner = reward_learning_learner
    curr_state = _build_interior_state()
    s_moved = _clone_state(curr_state)
    s_moved["agent_positions"][0] = np.array([1, 1], dtype=int)
    s_moved["agents"] = np.zeros_like(curr_state["agents"])
    for r, c in s_moved["agent_positions"]:
        s_moved["agents"][r, c] += 1

    eval_state = learner._make_reward_eval_state(
        curr_state,
        s_moved,
        actor_id=0,
        pick_rewards=np.zeros(NUM_AGENTS, dtype=np.float32),
    )

    assert eval_state["reward_eval_source"] == "curr_state"
    assert np.array_equal(eval_state["agents"], curr_state["agents"])
    assert np.array_equal(eval_state["agent_positions"], curr_state["agent_positions"])


def test_reward_learning_eval_state_uses_s_moved_for_nonzero_reward(reward_learning_learner):
    learner = reward_learning_learner
    curr_state = _build_interior_state()
    s_moved = _clone_state(curr_state)
    s_moved["agent_positions"][0] = np.array([1, 1], dtype=int)
    s_moved["agents"] = np.zeros_like(curr_state["agents"])
    for r, c in s_moved["agent_positions"]:
        s_moved["agents"][r, c] += 1

    rewards = np.zeros(NUM_AGENTS, dtype=np.float32)
    rewards[0] = 1.0

    eval_state = learner._make_reward_eval_state(
        curr_state,
        s_moved,
        actor_id=0,
        pick_rewards=rewards,
    )

    assert eval_state["reward_eval_source"] == "s_moved"
    assert np.array_equal(eval_state["agents"], s_moved["agents"])
    assert np.array_equal(eval_state["agent_positions"], s_moved["agent_positions"])


def test_reward_learning_eval_logs_mean_accuracy_to_csv(reward_learning_learner):
    learner = reward_learning_learner
    learner._generate_evaluation_states_reward_learning()

    class DummyEncoder:
        @staticmethod
        def encode(state, agent_id):
            return state, agent_id

    class PerfectRewardNet:
        @staticmethod
        def get_value_function(encoded):
            state, agent_id = encoded
            return float(state["true_rewards"][agent_id])

        @staticmethod
        def get_lr():
            return 0.01

    learner.encoder = DummyEncoder()
    learner.critic_networks = [PerfectRewardNet() for _ in range(NUM_AGENTS)]

    learner.evaluate_networks_reward(step=7)

    rows = _read_rows(learner.data_dir / "metrics.csv")
    assert rows, "Expected at least one row in metrics.csv"
    row = rows[-1]
    assert row["step"] == "7"
    assert float(row["reward_acc_mean"]) == pytest.approx(1.0, abs=1e-9)
    assert float(row["reward_mae_mean"]) == pytest.approx(0.0, abs=1e-9)


def test_reward_learning_training_path_uses_nearest_policy(reward_learning_learner):
    learner = reward_learning_learner
    learner.trajectory_length = 3
    learner.evaluation_states = []
    learner.evaluate_networks_reward = lambda **_: {}

    class DummyEncoder:
        @staticmethod
        def encode(state, agent_id):
            return np.zeros(4, dtype=np.float32)

    class DummyNet:
        def add_experience(self, *args, **kwargs):
            return None

        def train(self):
            return 0.0

        @staticmethod
        def get_lr():
            return 0.01

    class ForbiddenController:
        @staticmethod
        def agent_get_action(*args, **kwargs):
            raise AssertionError("agent_get_action should not be called in reward_learning mode")

    learner.encoder = DummyEncoder()
    learner.critic_networks = [DummyNet() for _ in range(NUM_AGENTS)]
    learner.agent_controller = ForbiddenController()
    learner.env = Orchard(
        W,
        L,
        NUM_AGENTS,
        learner.reward_module,
        p_apple=0.05,
        d_apple=0.05,
        max_apples=learner.exp_config.env.max_apples,
    )
    learner.env.set_positions()

    call_counter = {"n": 0}

    def fake_nearest_policy(agent_pos, apples_matrix):
        call_counter["n"] += 1
        return np.array(agent_pos, copy=True)

    original_nearest_policy = learning_loop_mod.nearest_apple_policy
    learning_loop_mod.nearest_apple_policy = fake_nearest_policy
    try:
        learner.step_and_collect_observation()
    finally:
        learning_loop_mod.nearest_apple_policy = original_nearest_policy

    assert call_counter["n"] > 0


def test_supervised_build_experiment_enforces_teacher_student_count_match(tmp_path, monkeypatch):
    weights_path = tmp_path / "teacher.pt"
    weights_path.write_bytes(b"placeholder")
    config_path = tmp_path / "supervised_test.yaml"
    output_dir = tmp_path / "runs"
    _write_supervised_learning_yaml(config_path, output_dir, weights_path)

    cfg = load_config(config_path)
    learner = Learning(cfg)

    def fake_load(_path, map_location=None):
        return {"critics": [{"blob": {}}]}  # 1 teacher only; students are per-agent (2)

    monkeypatch.setattr(learning_setup_mod.torch, "load", fake_load)
    with pytest.raises(ValueError, match="mismatch"):
        learner.build_experiment()
    _close_learning_loggers(learner)


def test_supervised_state_generation_uses_configured_count(tmp_path):
    weights_path = tmp_path / "teacher.pt"
    weights_path.write_bytes(b"placeholder")
    config_path = tmp_path / "supervised_test.yaml"
    output_dir = tmp_path / "runs"
    _write_supervised_learning_yaml(config_path, output_dir, weights_path)

    cfg = load_config(config_path)
    learner = Learning(cfg)
    learner._generate_evaluation_states_supervised()

    assert isinstance(learner.supervised_evaluation_states, list)
    assert len(learner.supervised_evaluation_states) == 25
    for st in learner.supervised_evaluation_states[:5]:
        assert "apples" in st and "agents" in st and "agent_positions" in st and "actor_id" in st
    _close_learning_loggers(learner)


def test_supervised_eval_performance_logs_both_policy_and_fit_metrics(tmp_path):
    weights_path = tmp_path / "teacher.pt"
    weights_path.write_bytes(b"placeholder")
    config_path = tmp_path / "supervised_test.yaml"
    output_dir = tmp_path / "runs"
    _write_supervised_learning_yaml(config_path, output_dir, weights_path)

    cfg = load_config(config_path)
    learner = Learning(cfg)

    class DummyController:
        epsilon = 0.05

    class DummyCritic:
        @staticmethod
        def get_lr():
            return 0.01

    class DummyEnv:
        d_apple = 0.1
        p_apple = 0.1
        max_apples = 9

    learner.agent_controller = DummyController()
    learner.critic_networks = [DummyCritic(), DummyCritic()]
    learner.env = DummyEnv()

    called = {"n": 0}

    def fake_supervised_eval():
        called["n"] += 1
        return {"supervised_mae_mean": 0.25, "supervised_rmse_mean": 0.5}

    learner.evaluate_networks_supervised = fake_supervised_eval
    original_eval_performance = learning_mod.eval_performance

    def fake_eval_performance(**kwargs):
        return {
            "greedy_pps": 10,
            "total_apples": 20,
            "greedy_ratio": 0.5,
            "nearest_pps": 9,
            "nearest_ratio": 0.45,
            "nearest_total_apples": 20,
            "greedy_agent_positions": np.zeros((2, NUM_AGENTS, 2), dtype=np.int16),
        }

    learning_mod.eval_performance = fake_eval_performance
    try:
        learner.eval_performance(step=11)
    finally:
        learning_mod.eval_performance = original_eval_performance

    rows = _read_rows(learner.data_dir / "metrics.csv")
    assert rows, "Expected metrics row for supervised eval."
    row = rows[-1]
    assert row["step"] == "11"
    assert float(row["greedy_pps"]) == pytest.approx(10.0)
    assert float(row["supervised_mae_mean"]) == pytest.approx(0.25)
    assert float(row["supervised_rmse_mean"]) == pytest.approx(0.5)
    assert called["n"] == 1
    _close_learning_loggers(learner)
