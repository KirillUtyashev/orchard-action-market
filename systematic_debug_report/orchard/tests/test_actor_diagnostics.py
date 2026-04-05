"""Tests for actor policy diagnostics and sampled state suites."""

import json

from orchard.actor_critic.action_space import full_action_head_dim
from orchard.actor_critic.policy_eval_states import (
    generate_phase2_policy_eval_states,
    sample_phase1_policy_eval_states,
    serialize_state,
)
from orchard.actor_critic.policy_logging import (
    build_phase1_policy_prob_row,
    build_phase2_policy_prob_row,
)
from orchard.datatypes import EnvConfig, Grid, State, StochasticConfig
from orchard.enums import PickMode


def _make_env_cfg(n_task_types: int = 3, n_agents: int = 2) -> EnvConfig:
    return EnvConfig(
        height=5,
        width=5,
        n_agents=n_agents,
        n_tasks=2,
        gamma=0.95,
        r_picker=1.0,
        n_task_types=n_task_types,
        pick_mode=PickMode.CHOICE,
        max_tasks_per_type=2,
        task_assignments=((0, 1), (1, 2)) if n_agents == 2 else tuple((0, 1) for _ in range(n_agents)),
        stochastic=StochasticConfig(spawn_prob=0.05, despawn_mode=None, despawn_prob=0.0),
    )


class TestSerializeState:
    def test_serialize_state_is_stable_and_canonical(self):
        state = State(
            agent_positions=(Grid(1, 2), Grid(3, 4)),
            task_positions=(Grid(2, 2), Grid(0, 1)),
            actor=1,
            task_types=(2, 0),
            pick_phase=True,
        )

        serialized = serialize_state(state)

        assert serialized == (
            '{"actor":1,"agent_positions":[[1,2],[3,4]],"pick_phase":true,'
            '"tasks":[[0,1,0],[2,2,2]]}'
        )


class TestSamplePhase1States:
    def test_phase1_sampling_is_deterministic_and_returns_pre_move_states(self):
        cfg = _make_env_cfg()

        sampled_a = sample_phase1_policy_eval_states(cfg, num_states=5, burnin=3, stride=2, seed=123)
        sampled_b = sample_phase1_policy_eval_states(cfg, num_states=5, burnin=3, stride=2, seed=123)

        assert sampled_a == sampled_b
        assert len(sampled_a) == 5
        assert all(not state.pick_phase for state in sampled_a)

    def test_phase1_row_contains_state_json_and_probability_columns(self):
        cfg = _make_env_cfg()
        state = sample_phase1_policy_eval_states(cfg, num_states=1, burnin=0, stride=1, seed=321)[0]
        probs = [0.0] * full_action_head_dim(cfg)
        probs[0] = 0.6
        probs[4] = 0.4

        row = build_phase1_policy_prob_row(10, 1.25, 0, state, probs, cfg)

        assert row["step"] == 10
        assert row["actor_id"] == state.actor
        assert row["state_json"] == serialize_state(state)
        assert row["prob_up"] == 0.6
        assert row["prob_stay"] == 0.4
        assert row["prob_pick_0"] == 0.0


class TestGeneratePhase2States:
    def test_phase2_state_suite_is_deterministic_and_complete(self):
        cfg = _make_env_cfg(n_task_types=3, n_agents=2)

        states_a = generate_phase2_policy_eval_states(cfg)
        states_b = generate_phase2_policy_eval_states(cfg)

        assert states_a == states_b
        assert len(states_a) == 14  # 2 actors * (3 singletons + 3 pairs + 1 all-types)

        for label, state in states_a:
            assert label
            assert state.pick_phase
            assert state.is_agent_on_task(state.actor)

    def test_phase2_row_contains_present_and_assigned_type_flags(self):
        cfg = _make_env_cfg(n_task_types=3, n_agents=2)
        label, state = generate_phase2_policy_eval_states(cfg)[0]
        probs = [0.0] * full_action_head_dim(cfg)
        probs[4] = 0.25
        probs[5] = 0.75

        row = build_phase2_policy_prob_row(20, 2.5, "case_0", label, state, probs, cfg)

        assert row["state_id"] == "case_0"
        assert row["state_label"] == label
        assert row["state_json"] == serialize_state(state)
        assert row["assigned_type_0"] == 1
        assert row["assigned_type_2"] == 0
        assert row["present_type_0"] == 1
        assert row["present_type_1"] == 0
        assert row["prob_stay"] == 0.25
        assert row["prob_pick_0"] == 0.75
