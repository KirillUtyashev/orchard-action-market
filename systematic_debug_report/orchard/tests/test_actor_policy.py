"""Tests for the orchard actor policy scaffolding."""

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.actor_critic.action_space import (
    action_to_policy_index,
    build_phase1_legal_mask,
    build_phase2_legal_mask,
    full_action_head_dim,
    policy_index_to_action,
)
from orchard.actor_critic.policy_network import PolicyNetwork
from orchard.datatypes import EnvConfig, Grid, ModelConfig, State, StochasticConfig
from orchard.enums import Action, Activation, EncoderType, PickMode, WeightInit, make_pick_action


def _make_env_cfg(n_task_types: int = 3) -> EnvConfig:
    return EnvConfig(
        height=5,
        width=5,
        n_agents=2,
        n_tasks=2,
        gamma=0.99,
        r_picker=1.0,
        n_task_types=n_task_types,
        pick_mode=PickMode.CHOICE,
        max_tasks_per_type=2,
        task_assignments=((0, 1), (1, 2)),
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
    )


def _make_forced_env_cfg(n_task_types: int = 3) -> EnvConfig:
    cfg = _make_env_cfg(n_task_types=n_task_types)
    return EnvConfig(
        height=cfg.height,
        width=cfg.width,
        n_agents=cfg.n_agents,
        n_tasks=cfg.n_tasks,
        gamma=cfg.gamma,
        r_picker=cfg.r_picker,
        n_task_types=cfg.n_task_types,
        r_low=cfg.r_low,
        task_assignments=cfg.task_assignments,
        pick_mode=PickMode.FORCED,
        max_tasks_per_type=cfg.max_tasks_per_type,
        stochastic=cfg.stochastic,
    )


def _make_model_cfg() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(16,),
        conv_specs=((4, 3),),
        activation=Activation.RELU,
        weight_init=WeightInit.ZERO_BIAS,
    )


def _make_phase1_state() -> State:
    return State(
        agent_positions=(Grid(0, 0), Grid(4, 4)),
        task_positions=(Grid(2, 2),),
        actor=0,
        task_types=(0,),
    )


def _make_phase2_state() -> State:
    return State(
        agent_positions=(Grid(2, 2), Grid(4, 4)),
        task_positions=(Grid(2, 2), Grid(2, 2)),
        actor=0,
        task_types=(0, 2),
        pick_phase=True,
    )


class TestActionHeadMapping:
    def test_head_dimension_is_derived_from_action_encoding(self):
        cfg = _make_env_cfg(n_task_types=3)

        assert full_action_head_dim(cfg) == make_pick_action(cfg.n_task_types - 1).value + 1

    def test_forced_head_dimension_contains_only_movement_actions(self):
        cfg = _make_forced_env_cfg(n_task_types=3)

        assert full_action_head_dim(cfg) == Action.STAY.value + 1

    def test_action_index_round_trip_matches_action_values(self):
        actions = [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.STAY,
            make_pick_action(0),
            make_pick_action(2),
        ]

        for action in actions:
            idx = action_to_policy_index(action)
            assert idx == action.value
            assert policy_index_to_action(idx) == action


class TestMasks:
    def test_phase1_mask_allows_all_moves_and_masks_all_picks(self):
        cfg = _make_env_cfg(n_task_types=3)
        mask = build_phase1_legal_mask(_make_phase1_state(), cfg)

        assert mask.shape == (full_action_head_dim(cfg),)
        assert mask[: Action.STAY.value + 1].tolist() == [True, True, True, True, True]
        assert mask[Action.PICK.value :].tolist() == [False, False, False]

    def test_forced_phase1_mask_has_no_pick_slots(self):
        cfg = _make_forced_env_cfg(n_task_types=3)
        mask = build_phase1_legal_mask(_make_phase1_state(), cfg)

        assert mask.shape == (Action.STAY.value + 1,)
        assert mask.tolist() == [True, True, True, True, True]

    def test_phase2_mask_allows_stay_and_present_pick_types_only(self):
        cfg = _make_env_cfg(n_task_types=3)
        mask = build_phase2_legal_mask(_make_phase2_state(), cfg)

        assert mask.shape == (full_action_head_dim(cfg),)
        assert mask[: Action.STAY.value].tolist() == [False, False, False, False]
        assert mask[Action.STAY.value]
        assert mask[make_pick_action(0).value]
        assert not mask[make_pick_action(1).value]
        assert mask[make_pick_action(2).value]


class TestPolicyNetwork:
    def test_masked_probabilities_sum_to_one_and_zero_invalid_entries(self):
        env_cfg = _make_env_cfg(n_task_types=3)
        model_cfg = _make_model_cfg()
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        policy = PolicyNetwork(model_cfg, env_cfg, lr=0.01)

        state = _make_phase2_state()
        enc = encoding.encode(state, state.actor)
        mask = build_phase2_legal_mask(state, env_cfg)
        probs = policy.get_action_probabilities(enc, mask)

        assert probs.shape == (full_action_head_dim(env_cfg),)
        assert np.isclose(probs.sum(), 1.0)
        assert np.allclose(probs[~mask], 0.0)

    def test_sampling_never_returns_masked_action(self):
        env_cfg = _make_env_cfg(n_task_types=3)
        model_cfg = _make_model_cfg()
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        policy = PolicyNetwork(model_cfg, env_cfg, lr=0.01)

        state = _make_phase2_state()
        enc = encoding.encode(state, state.actor)
        mask = build_phase2_legal_mask(state, env_cfg)

        for _ in range(50):
            action, _ = policy.sample_action(enc, mask)
            assert mask[action_to_policy_index(action)]

    def test_train_batch_returns_metrics_and_updates_parameters(self):
        env_cfg = _make_env_cfg(n_task_types=3)
        model_cfg = _make_model_cfg()
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        policy = PolicyNetwork(model_cfg, env_cfg, lr=0.05)

        phase1_state = _make_phase1_state()
        phase2_state = _make_phase2_state()
        enc1 = encoding.encode(phase1_state, phase1_state.actor)
        enc2 = encoding.encode(phase2_state, phase2_state.actor)
        mask1 = build_phase1_legal_mask(phase1_state, env_cfg)
        mask2 = build_phase2_legal_mask(phase2_state, env_cfg)

        policy.add_experience(enc1, mask1, Action.RIGHT, advantage=1.0)
        policy.add_experience(enc2, mask2, make_pick_action(2), advantage=0.5)

        params_before = [param.detach().clone() for param in policy.parameters()]
        metrics = policy.train_batch()
        params_after = list(policy.parameters())

        assert metrics is not None
        assert metrics["loss"] > 0.0
        assert metrics["entropy_mean"] >= 0.0
        assert any(not torch.allclose(before, after) for before, after in zip(params_before, params_after))
