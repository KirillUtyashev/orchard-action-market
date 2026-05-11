"""Tests for evaluation rollouts, 2-phase transitions, and metrics calculation."""

import pytest
from orchard.enums import Action, DespawnMode, make_pick_action
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.eval import rollout_trajectory, evaluate_policy_metrics
from orchard.seed import set_all_seeds


def _make_eval_cfg(n_agents=2, n_task_types=2) -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=n_agents, n_tasks=2, gamma=0.9,
        n_task_types=n_task_types, max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )


class TestRolloutTrajectory:
    def test_no_pick_yields_one_transition(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)

        # Agent 0 at (0,0), task at (4,4) - far away
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(4, 4),),
            actor=0, task_types=(0,)
        )

        def dummy_policy(state):
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, dummy_policy, env, n_steps=1))

        # Should be exactly 1 transition (just the move)
        assert len(transitions) == 1
        t = transitions[0]
        assert t.action == Action.RIGHT
        assert t.discount == pytest.approx(0.9)  # Regular gamma for movement
        assert t.rewards == (0.0, 0.0)

    def test_pick_yields_two_transitions(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg(n_agents=2, n_task_types=2)
        env = StochasticEnv(cfg)

        # Agent 0 at (0,0), task at (0,1) with type 0
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )

        # Policy: move RIGHT then pick type 0
        def policy(state):
            if state.pick_phase:
                return make_pick_action(0)
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, policy, env, n_steps=1))

        # Should be 2 transitions: move + pick
        assert len(transitions) == 2
        t_move, t_pick = transitions

        assert t_move.action == Action.RIGHT
        assert t_move.discount == pytest.approx(0.9)
        assert sum(t_move.rewards) == 0.0

        assert t_pick.action == make_pick_action(0)
        assert t_pick.discount == pytest.approx(1.0)  # Pick discount is always 1.0

    def test_stay_during_pick_phase_yields_no_reward(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )

        # Policy: move RIGHT, then STAY in pick phase (decline pick)
        def policy(state):
            if state.pick_phase:
                return Action.STAY
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, policy, env, n_steps=1))

        assert len(transitions) == 2
        t_move, t_stay = transitions
        assert t_stay.action == Action.STAY
        assert t_stay.discount == pytest.approx(1.0)
        assert sum(t_stay.rewards) == 0.0


class TestEvaluatePolicyMetrics:
    def test_rps_and_team_rps_returned(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(4, 4),),
            actor=0, task_types=(0,)
        )

        def dummy_policy(state):
            if state.pick_phase:
                return Action.STAY
            return Action.STAY

        metrics = evaluate_policy_metrics(s, dummy_policy, env, n_steps=5)

        assert "rps" in metrics
        assert "team_rps" in metrics
        assert "tasks_picked_per_step" in metrics
        assert "correct_pps" not in metrics
        assert "wrong_pps" not in metrics

    def test_zero_reward_policy_has_zero_rps(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(4, 4),),
            actor=0, task_types=(0,)
        )

        def zero_policy(state):
            return Action.STAY

        metrics = evaluate_policy_metrics(s, zero_policy, env, n_steps=10)
        assert metrics["rps"] == pytest.approx(0.0)
        assert metrics["team_rps"] == pytest.approx(0.0)
        assert metrics["tasks_picked_per_step"] == pytest.approx(0.0)

    def test_successful_pick_counts_task_per_step(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )

        def pick_policy(state):
            if state.pick_phase:
                return make_pick_action(0)
            return Action.RIGHT

        metrics = evaluate_policy_metrics(s, pick_policy, env, n_steps=1)
        assert metrics["tasks_picked_per_step"] == pytest.approx(1.0)

    def test_stay_on_task_does_not_count_task_pick(self):
        set_all_seeds(0)
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )

        def stay_policy(state):
            if state.pick_phase:
                return Action.STAY
            return Action.RIGHT

        metrics = evaluate_policy_metrics(s, stay_policy, env, n_steps=1)
        assert metrics["tasks_picked_per_step"] == pytest.approx(0.0)
