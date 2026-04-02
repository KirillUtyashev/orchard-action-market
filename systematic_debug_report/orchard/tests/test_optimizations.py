"""Tests for optimization correctness (vmap and vectorized encoding).

These tests verify that optimized code paths produce identical results
to the sequential/loop-based baselines.
"""

import pytest
import torch

from orchard.enums import (
    Action, Activation, EncoderType, EnvType, LearningType, ModelType,
    PickMode, Schedule, TrainMethod, WeightInit, make_pick_action,
)
from orchard.datatypes import (
    EnvConfig, Grid, ModelConfig, ScheduleConfig, State, StochasticConfig,
)
from orchard.encoding.grid import TaskGridEncoder, CentralizedTaskGridEncoder, BasicGridEncoder
from orchard.env.deterministic import DeterministicEnv
import orchard.encoding as encoding
from orchard.model import ValueNetwork, create_networks


def _make_cfg(n_task_types=4, n_agents=4, pick_mode=PickMode.FORCED):
    return EnvConfig(
        height=5, width=5, n_agents=n_agents, n_tasks=3,
        gamma=0.99, r_picker=1.0,
        n_task_types=n_task_types, r_low=0.0,
        task_assignments=tuple((i,) for i in range(n_task_types)),
        pick_mode=pick_mode,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )


def _make_state():
    return State(
        agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
        task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
        actor=0,
        task_types=(0, 1, 2),
    )


def _make_legacy_cfg():
    return EnvConfig(
        height=4, width=4, n_agents=2, n_tasks=2,
        gamma=0.9, r_picker=1.0,
        pick_mode=PickMode.FORCED,
        max_tasks=4, env_type=EnvType.DETERMINISTIC,
        task_assignments=((0,), (0,)),
    )


# ---------------------------------------------------------------------------
# Vectorized encoding correctness
# ---------------------------------------------------------------------------
class TestVectorizedEncodingCorrectness:
    """Batch encoding must match individual encode() calls."""

    def test_task_encoder_movement_batch(self):
        """TaskGridEncoder batch encoding matches loop for movement actions."""
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)

        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid, atol=1e-6), \
                f"Grid mismatch at action {k}"
            assert torch.allclose(batch.scalar[k], single.scalar, atol=1e-6), \
                f"Scalar mismatch at action {k}"

    def test_task_encoder_pick_batch(self):
        """TaskGridEncoder batch encoding matches loop for pick after-states."""
        cfg = _make_cfg(pick_mode=PickMode.CHOICE)
        enc = TaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0), Grid(1, 1)),
            actor=0,
            task_types=(0, 1),
        )

        # Mix of movement and pick after-states
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        s_picked, _ = env.resolve_pick(s, pick_type=0)
        after_states.append(s_picked)

        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)
        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid, atol=1e-6), \
                f"Grid mismatch at action {k}"

    def test_centralized_encoder_batch(self):
        """CentralizedTaskGridEncoder batch matches loop."""
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)

        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid, atol=1e-6)
            assert torch.allclose(batch.scalar[k], single.scalar, atol=1e-6)

    def test_legacy_encoder_batch(self):
        """BasicGridEncoder batch matches loop (backward compat)."""
        cfg = _make_legacy_cfg()
        enc = BasicGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0), Grid(1, 1)),
            actor=0,
        )
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)

        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid, atol=1e-6)

    def test_all_agents_batch_match(self):
        """Batch encoding matches for every agent, not just agent 0."""
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]

        for agent_idx in range(4):
            batch = enc.encode_batch_for_actions(s, agent_idx=agent_idx, after_states=after_states)
            for k, s_after in enumerate(after_states):
                single = enc.encode(s_after, agent_idx=agent_idx)
                assert torch.allclose(batch.grid[k], single.grid, atol=1e-6), \
                    f"Agent {agent_idx}, action {k}: grid mismatch"


# ---------------------------------------------------------------------------
# Vmap correctness
# ---------------------------------------------------------------------------
class TestVmapCorrectness:
    """Vmap-batched forward passes must match sequential."""

    def _setup_networks(self, cfg):
        """Create networks and init encoder."""
        model_cfg = ModelConfig(
            input_type=EncoderType.TASK_CNN_GRID,
            model_type=ModelType.CNN,
            mlp_dims=(16,),
            conv_specs=((4, 3),),
            activation=Activation.LEAKY_RELU,
            weight_init=WeightInit.ZERO_BIAS,
        )
        lr_cfg = ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)
        encoding.init_encoder(EncoderType.TASK_CNN_GRID, cfg)
        networks = create_networks(
            model_cfg, cfg, lr_cfg, total_steps=100,
            td_lambda=0.3, train_method=TrainMethod.BACKWARD_VIEW,
            n_networks=cfg.n_agents,
        )
        return networks

    def test_vmap_matches_sequential(self):
        """VmapHelper forward produces same values as sequential loop."""
        from orchard.policy import VmapHelper

        cfg = _make_cfg()
        networks = self._setup_networks(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]

        # Sequential: for each network, encode and forward all actions
        seq_values = torch.zeros(cfg.n_agents, 5)
        with torch.no_grad():
            for i, net in enumerate(networks):
                batch_enc = encoding.encode_batch_for_actions(s, i, after_states)
                seq_values[i] = net(batch_enc)

        # Vmap: batch all networks
        helper = VmapHelper(networks)
        all_grids = []
        all_scalars = []
        for i in range(cfg.n_agents):
            batch_enc = encoding.encode_batch_for_actions(s, i, after_states)
            all_grids.append(batch_enc.grid)
            all_scalars.append(batch_enc.scalar)
        grids_stacked = torch.stack(all_grids)
        scalars_stacked = torch.stack(all_scalars)

        with torch.no_grad():
            vmap_values = helper.forward_batched(grids_stacked, scalars_stacked)

        assert torch.allclose(seq_values, vmap_values, atol=1e-5), \
            f"Max diff: {(seq_values - vmap_values).abs().max().item()}"

    def test_vmap_team_value_matches(self):
        """Team Q-value from vmap matches sequential computation."""
        from orchard.policy import VmapHelper

        cfg = _make_cfg()
        networks = self._setup_networks(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]

        # Sequential team values
        seq_team = torch.zeros(5)
        with torch.no_grad():
            for i, net in enumerate(networks):
                batch_enc = encoding.encode_batch_for_actions(s, i, after_states)
                seq_team += net(batch_enc)

        # Vmap team values
        helper = VmapHelper(networks)
        all_grids = []
        all_scalars = []
        for i in range(cfg.n_agents):
            batch_enc = encoding.encode_batch_for_actions(s, i, after_states)
            all_grids.append(batch_enc.grid)
            all_scalars.append(batch_enc.scalar)

        with torch.no_grad():
            vmap_values = helper.forward_batched(
                torch.stack(all_grids), torch.stack(all_scalars)
            )
            vmap_team = vmap_values.sum(dim=0)

        assert torch.allclose(seq_team, vmap_team, atol=1e-5)

    def test_vmap_greedy_same_action(self):
        """Vmap greedy selects same action as sequential greedy,
        or the Q-values are within float tolerance (tie)."""
        from orchard.policy import (
            argmax_a_Q_team_batched, argmax_a_Q_team_vmap,
            init_vmap, get_all_actions, Q_team,
        )

        cfg = _make_cfg()
        networks = self._setup_networks(cfg)
        env = DeterministicEnv(cfg)
        init_vmap(networks)

        s = _make_state()
        for _ in range(5):
            a_seq = argmax_a_Q_team_batched(s, networks, env)
            a_vmap = argmax_a_Q_team_vmap(s, networks, env)

            if a_seq != a_vmap:
                # Disagreement is OK only if Q-values are nearly tied
                q_seq = Q_team(s, a_seq, networks, env)
                q_vmap = Q_team(s, a_vmap, networks, env)
                assert abs(q_seq - q_vmap) < 1e-4, \
                    f"Sequential chose {a_seq} (Q={q_seq:.6f}), " \
                    f"vmap chose {a_vmap} (Q={q_vmap:.6f}), gap={abs(q_seq-q_vmap):.8f}"

            tr = env.step(s, a_seq)
            s = tr.s_t_next

    def test_vmap_refresh_after_weight_change(self):
        """After modifying weights, refresh() updates vmap cached params."""
        from orchard.policy import VmapHelper

        cfg = _make_cfg()
        networks = self._setup_networks(cfg)
        helper = VmapHelper(networks)

        # Get initial output
        s = _make_state()
        env = DeterministicEnv(cfg)
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]

        all_grids = []
        all_scalars = []
        for i in range(cfg.n_agents):
            batch_enc = encoding.encode_batch_for_actions(s, i, after_states)
            all_grids.append(batch_enc.grid)
            all_scalars.append(batch_enc.scalar)
        grids = torch.stack(all_grids)
        scalars = torch.stack(all_scalars)

        with torch.no_grad():
            out_before = helper.forward_batched(grids, scalars).clone()

        # Modify network 0 weights
        with torch.no_grad():
            for p in networks[0].parameters():
                p.add_(1.0)

        # Without refresh — should use OLD cached params
        with torch.no_grad():
            out_stale = helper.forward_batched(grids, scalars)
        assert torch.allclose(out_stale, out_before, atol=1e-6), \
            "Stale params should give same output as before"

        # After refresh — should reflect new weights
        helper.refresh()
        with torch.no_grad():
            out_after = helper.forward_batched(grids, scalars)
        assert not torch.allclose(out_after, out_before, atol=1e-3), \
            "After refresh, output should differ from before weight change"
