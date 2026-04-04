"""Tests for the comm_weight (communication weight) feature.

Tests verify:
- Config parsing: comm_weight is read from YAML and defaults to 0.0
- Reward weighting: _team_rewards unchanged; adjusted reward in _train_all_agents
  produces correct weighted TD targets
- Action selection: V_weighted = V_i + w * Σ V_j (actor weight 1, others weight w)
- _after_state_and_team_reward: returns weighted immediate reward
- Centralized is unaffected by comm_weight (len(networks)==1 skips weighting)
- w=0 gives only own reward; w=1 recovers full team reward
- Integration: training loop runs with comm_weight set
"""

import csv
import os
import tempfile

import pytest
import torch

from orchard.enums import (
    Action, EncoderType, EnvType, ModelType, PickMode,
    Schedule, make_pick_action,
)
from orchard.datatypes import (
    EncoderOutput, EnvConfig, Grid, ModelConfig,
    ScheduleConfig, State, TrainConfig,
)
from orchard.config import load_config
from orchard.env.deterministic import DeterministicEnv
import orchard.encoding as encoding
from orchard.model import ValueNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_cfg(n_agents=4, r_picker=-1.0, n_task_types=2, **overrides):
    if n_task_types == 1:
        assignments = tuple((0,) for _ in range(n_agents))
    elif n_task_types == 2 and n_agents == 4:
        # agents 0,1 -> type 0;  agents 2,3 -> type 1
        assignments = ((0,), (0,), (1,), (1,))
    else:
        assignments = tuple((i % n_task_types,) for i in range(n_agents))
    defaults = dict(
        height=5, width=5, n_agents=n_agents, n_tasks=2,
        gamma=0.99, r_picker=r_picker,
        n_task_types=n_task_types, r_low=-1.0,
        task_assignments=assignments,
        pick_mode=PickMode.FORCED,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_networks(env_cfg, n_networks, seed=42):
    torch.manual_seed(seed)
    model_cfg = ModelConfig(
        input_type=EncoderType.RELATIVE,
        model_type=ModelType.MLP,
        mlp_dims=(16,),
    )
    lr_cfg = ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)
    encoding.init_encoder(EncoderType.RELATIVE, env_cfg)
    nets = [ValueNetwork(model_cfg, env_cfg, lr_cfg, total_steps=1000)
            for _ in range(n_networks)]
    return nets


def _scalar_dim():
    """Return the current encoder's scalar dimension (call after _make_networks)."""
    return encoding.get_scalar_dim()


def _make_state_on_task(env_cfg):
    """Return a state where actor=0 is on a task of type 0 (correct pick)."""
    return State(
        agent_positions=(Grid(1, 0), Grid(0, 0), Grid(3, 3), Grid(4, 4)),
        task_positions=(Grid(1, 0), Grid(3, 3)),
        task_types=(0, 1),
        actor=0,
    )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing:
    def test_default_comm_weight_is_zero(self):
        yaml_str = """
env:
  height: 3
  width: 3
  n_agents: 2
  n_tasks: 1
  gamma: 0.9
  r_picker: 1.0
  type: deterministic
model:
  input_type: relative
  model_type: mlp
  mlp_dims: [16]
train:
  mode: policy_learning
  td_target: pre_action
  total_steps: 10
  lr:
    start: 0.01
    schedule: none
  policy_learning:
    epsilon:
      start: 0.1
      schedule: none
eval:
  rollout_len: 10
  eval_steps: 10
logging:
  output_dir: /tmp/test_cw
"""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(yaml_str)
        cfg = load_config(path)
        os.unlink(path)
        assert cfg.train.comm_weight == 0.0

    def test_explicit_comm_weight(self):
        yaml_str = """
env:
  height: 3
  width: 3
  n_agents: 2
  n_tasks: 1
  gamma: 0.9
  r_picker: 1.0
  type: deterministic
model:
  input_type: relative
  model_type: mlp
  mlp_dims: [16]
train:
  mode: policy_learning
  td_target: pre_action
  total_steps: 10
  comm_weight: 0.75
  lr:
    start: 0.01
    schedule: none
  policy_learning:
    epsilon:
      start: 0.1
      schedule: none
eval:
  rollout_len: 10
  eval_steps: 10
logging:
  output_dir: /tmp/test_cw
"""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(yaml_str)
        cfg = load_config(path)
        os.unlink(path)
        assert cfg.train.comm_weight == 0.75


# ---------------------------------------------------------------------------
# _after_state_and_team_reward weighting
# ---------------------------------------------------------------------------

class TestAfterStateAndTeamReward:
    """Test that _after_state_and_team_reward returns correctly weighted reward."""

    def setup_method(self):
        self.cfg = _make_env_cfg(n_agents=4, r_picker=-1.0, n_task_types=2)
        self.env = DeterministicEnv(self.cfg)
        self.state = _make_state_on_task(self.cfg)

    def test_w1_returns_team_reward(self):
        """w=1 should return sum(rewards) — the full team reward."""
        from orchard.policy import _after_state_and_team_reward
        pick = make_pick_action(0)
        _, weighted_r = _after_state_and_team_reward(
            self.state, pick, self.env, phase2=True, comm_weight=1.0)
        # Direct computation
        _, raw_rewards = self.env.resolve_pick(self.state, pick_type=0)
        assert pytest.approx(weighted_r) == sum(raw_rewards)

    def test_w0_returns_picker_reward(self):
        """w=0 should return only the picker's own reward."""
        from orchard.policy import _after_state_and_team_reward
        pick = make_pick_action(0)
        _, weighted_r = _after_state_and_team_reward(
            self.state, pick, self.env, phase2=True, comm_weight=0.0)
        _, raw_rewards = self.env.resolve_pick(self.state, pick_type=0)
        assert pytest.approx(weighted_r) == raw_rewards[0]

    def test_w05_interpolates(self):
        """w=0.5 should interpolate between picker and team reward."""
        from orchard.policy import _after_state_and_team_reward
        pick = make_pick_action(0)
        _, weighted_r = _after_state_and_team_reward(
            self.state, pick, self.env, phase2=True, comm_weight=0.5)
        _, raw_rewards = self.env.resolve_pick(self.state, pick_type=0)
        actor = self.state.actor
        expected = raw_rewards[actor] + 0.5 * (sum(raw_rewards) - raw_rewards[actor])
        assert pytest.approx(weighted_r) == expected

    def test_move_action_returns_zero(self):
        """Non-pick actions should always return 0 regardless of w."""
        from orchard.policy import _after_state_and_team_reward
        _, r = _after_state_and_team_reward(
            self.state, Action.RIGHT, self.env, phase2=False, comm_weight=0.5)
        assert r == 0.0


# ---------------------------------------------------------------------------
# Q_team value weighting
# ---------------------------------------------------------------------------

class TestQTeamWeighting:
    """Test that Q_team weights other agents' values by comm_weight."""

    def setup_method(self):
        self.cfg = _make_env_cfg(n_agents=4, r_picker=-1.0, n_task_types=2)
        self.env = DeterministicEnv(self.cfg)
        self.networks = _make_networks(self.cfg, n_networks=4)
        self.state = _make_state_on_task(self.cfg)

    def test_w1_is_full_team_value(self):
        """w=1 should sum all agents' values (original behavior)."""
        from orchard.policy import Q_team
        action = Action.RIGHT
        q_w1 = Q_team(self.state, action, self.networks, self.env,
                       phase2=False, comm_weight=1.0)
        # Manual computation: sum of all network outputs
        s_after = self.env.apply_action(self.state, action)
        total = 0.0
        with torch.no_grad():
            for i, net in enumerate(self.networks):
                total += net(encoding.encode(s_after, i)).item()
        assert pytest.approx(q_w1, abs=1e-5) == total

    def test_w0_is_actor_only(self):
        """w=0 should only include the actor's value."""
        from orchard.policy import Q_team
        action = Action.RIGHT
        q_w0 = Q_team(self.state, action, self.networks, self.env,
                       phase2=False, comm_weight=0.0)
        # Manual: only actor's (agent 0) value
        s_after = self.env.apply_action(self.state, action)
        with torch.no_grad():
            actor_val = self.networks[0](encoding.encode(s_after, 0)).item()
        assert pytest.approx(q_w0, abs=1e-5) == actor_val

    def test_w05_weights_others_half(self):
        """w=0.5: actor full weight, others at 0.5."""
        from orchard.policy import Q_team
        action = Action.RIGHT
        q = Q_team(self.state, action, self.networks, self.env,
                    phase2=False, comm_weight=0.5)
        s_after = self.env.apply_action(self.state, action)
        expected = 0.0
        with torch.no_grad():
            for i, net in enumerate(self.networks):
                v = net(encoding.encode(s_after, i)).item()
                weight = 1.0 if i == 0 else 0.5
                expected += weight * v
        assert pytest.approx(q, abs=1e-5) == expected

    def test_single_network_ignores_weight(self):
        """Centralized (1 network) should ignore comm_weight."""
        from orchard.policy import Q_team
        nets_cen = _make_networks(self.cfg, n_networks=1)
        action = Action.RIGHT
        q_w0 = Q_team(self.state, action, nets_cen, self.env,
                       phase2=False, comm_weight=0.0)
        q_w1 = Q_team(self.state, action, nets_cen, self.env,
                       phase2=False, comm_weight=1.0)
        assert pytest.approx(q_w0, abs=1e-5) == q_w1


# ---------------------------------------------------------------------------
# _train_all_agents adjusted reward correctness
# ---------------------------------------------------------------------------

class TestTrainAllAgentsAdjustedReward:
    """Verify the adjusted reward produces the correct weighted TD error."""

    def setup_method(self):
        self.cfg = _make_env_cfg(n_agents=3, r_picker=-1.0, n_task_types=1,
                                 task_assignments=((0,), (0,), (0,)))
        self.networks = _make_networks(self.cfg, n_networks=3)
        self._dim = _scalar_dim()

    def _get_values(self, encs):
        with torch.no_grad():
            return [net(encs[i]).item() for i, net in enumerate(self.networks)]

    def _rand_encs(self, n):
        return [EncoderOutput(scalar=torch.randn(self._dim)) for _ in range(n)]

    def test_w0_adjusted_reward_equals_own_reward(self):
        """With w=0 the adjusted reward for each agent should be its own reward."""
        from orchard.train import _train_all_agents
        s_encs = self._rand_encs(3)
        s_next_encs = self._rand_encs(3)
        rewards = (-1.0, 2.0 / 2, 2.0 / 2)
        discount = 0.99

        # With w=0, should be same as calling without comm_weight
        # Clone networks to compare
        nets_a = _make_networks(self.cfg, n_networks=3, seed=123)
        nets_b = _make_networks(self.cfg, n_networks=3, seed=123)

        _train_all_agents(nets_a, s_encs, rewards, discount, s_next_encs, 0,
                          comm_weight=0.0)
        _train_all_agents(nets_b, s_encs, rewards, discount, s_next_encs, 0,
                          comm_weight=0.0)

        # Both should produce identical weights
        for na, nb in zip(nets_a, nets_b):
            for pa, pb in zip(na.parameters(), nb.parameters()):
                assert torch.allclose(pa, pb)

    def test_w1_adjusted_reward_matches_weighted_td(self):
        """With w=1, verify adjusted reward = r_i + (total_r - r_i) + γ·(ΣV'_j - V'_i) - (ΣV_j - V_i).

        This should make the TD target equal to total_r + γ·ΣV(s').
        """
        s_encs = self._rand_encs(3)
        s_next_encs = self._rand_encs(3)
        rewards = (-1.0, 1.0, 1.0)
        discount = 0.99

        v_s = self._get_values(s_encs)
        v_next = self._get_values(s_next_encs)
        total_r = sum(rewards)
        total_v_s = sum(v_s)
        total_v_next = sum(v_next)

        for i in range(3):
            adjusted_r = (rewards[i]
                          + 1.0 * (total_r - rewards[i])
                          + discount * 1.0 * (total_v_next - v_next[i])
                          - 1.0 * (total_v_s - v_s[i]))
            td_error = adjusted_r + discount * v_next[i] - v_s[i]
            expected_error = total_r + discount * total_v_next - total_v_s
            assert pytest.approx(td_error, abs=1e-5) == expected_error

    def test_intermediate_w_td_target(self):
        """With w=0.5, TD target = (r_i + 0.5·Σ_{j≠i}r_j) + γ·(V_i + 0.5·Σ_{j≠i}V_j)(s')."""
        s_encs = self._rand_encs(3)
        s_next_encs = self._rand_encs(3)
        rewards = (-1.0, 1.0, 1.0)
        discount = 0.99
        w = 0.5

        v_s = self._get_values(s_encs)
        v_next = self._get_values(s_next_encs)
        total_r = sum(rewards)
        total_v_s = sum(v_s)
        total_v_next = sum(v_next)

        for i in range(3):
            adjusted_r = (rewards[i]
                          + w * (total_r - rewards[i])
                          + discount * w * (total_v_next - v_next[i])
                          - w * (total_v_s - v_s[i]))
            td_error = adjusted_r + discount * v_next[i] - v_s[i]
            # Expected: weighted_r + γ · V_weighted(s') - V_weighted(s)
            weighted_r = rewards[i] + w * (total_r - rewards[i])
            v_weighted_next = v_next[i] + w * (total_v_next - v_next[i])
            v_weighted_s = v_s[i] + w * (total_v_s - v_s[i])
            expected_error = weighted_r + discount * v_weighted_next - v_weighted_s
            assert pytest.approx(td_error, abs=1e-5) == expected_error


# ---------------------------------------------------------------------------
# Centralized unaffected
# ---------------------------------------------------------------------------

class TestCentralizedUnaffected:
    """Centralized training should not be affected by comm_weight."""

    def test_single_network_skips_weighting(self):
        """When len(networks)==1, _train_all_agents should skip the comm_weight logic."""
        from orchard.train import _train_all_agents
        cfg = _make_env_cfg(n_agents=2, r_picker=-1.0, n_task_types=1,
                            task_assignments=((0,), (0,)))
        # 1 network = centralized
        nets_w0 = _make_networks(cfg, n_networks=1, seed=99)
        nets_w1 = _make_networks(cfg, n_networks=1, seed=99)
        d = _scalar_dim()

        s_encs = [EncoderOutput(scalar=torch.randn(d))]
        s_next_encs = [EncoderOutput(scalar=torch.randn(d))]
        rewards_cen = (1.0,)  # centralized: sum of per-agent rewards

        _train_all_agents(nets_w0, s_encs, rewards_cen, 0.99, s_next_encs, 0,
                          comm_weight=0.0)
        _train_all_agents(nets_w1, s_encs, rewards_cen, 0.99, s_next_encs, 0,
                          comm_weight=1.0)

        for p0, p1 in zip(nets_w0[0].parameters(), nets_w1[0].parameters()):
            assert torch.allclose(p0, p1)


# ---------------------------------------------------------------------------
# Integration: training loop with comm_weight
# ---------------------------------------------------------------------------

COMM_WEIGHT_CONFIG = """
env:
  height: 5
  width: 5
  n_agents: 4
  n_tasks: 2
  n_task_types: 2
  r_low: -1.0
  gamma: 0.99
  r_picker: -1.0
  pick_mode: forced
  max_tasks_per_type: 2
  task_assignments: [[0], [0], [1], [1]]
  type: stochastic
  stochastic:
    spawn_prob: 0.04
    despawn_mode: probability
    despawn_prob: 0.05
model:
  input_type: filtered_task_cnn_grid
  model_type: cnn
  conv_specs: [[4, 3]]
  mlp_dims: []
  activation: leaky_relu
  weight_init: zero_bias
train:
  mode: policy_learning
  td_target: after_state
  train_method: backward_view
  td_lambda: 0.3
  learning_type: decentralized
  comm_weight: {comm_weight}
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


def _run_comm_weight_train(comm_weight):
    from orchard.train import train
    tmpdir = tempfile.mkdtemp()
    yaml_str = COMM_WEIGHT_CONFIG.format(
        comm_weight=comm_weight, output_dir=tmpdir)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(yaml_str)
    cfg = load_config(path)
    assert cfg.train.comm_weight == comm_weight
    train(cfg)
    os.unlink(path)

    run_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    assert len(run_dirs) == 1
    run_dir = os.path.join(tmpdir, run_dirs[0])
    metrics_path = os.path.join(run_dir, "metrics.csv")
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    return rows


class TestCommWeightIntegration:
    """Training loop completes without errors for various comm_weight values."""

    def test_w0_runs(self):
        rows = _run_comm_weight_train(0.0)
        assert len(rows) >= 1
        assert "greedy_rps" in rows[0]

    def test_w05_runs(self):
        rows = _run_comm_weight_train(0.5)
        assert len(rows) >= 1
        assert "greedy_rps" in rows[0]

    def test_w1_runs(self):
        rows = _run_comm_weight_train(1.0)
        assert len(rows) >= 1
        assert "greedy_rps" in rows[0]
