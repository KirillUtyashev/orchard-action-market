"""Tests for heuristic policies and action space helpers."""

import numpy as np
import pytest
from orchard.enums import Action, Heuristic, make_pick_action, DespawnMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.policy import (
    get_all_actions,
    get_phase2_actions,
    nearest_action,
    heuristic_action,
    clairvoyant_rollout_action,
    raw_stochastic_mpc_action,
    stochastic_mpc_action,
    survival_adjusted_assignment,
    survival_adjusted_hungarian_action,
    survival_adjusted_utilities,
)
from orchard.seed import rng, set_all_seeds


def _make_cfg(n_agents=4, n_task_types=2, relatedness_width=0, proficiency_width=0) -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=n_agents, n_tasks=2, gamma=0.99,
        n_task_types=n_task_types, relatedness_width=relatedness_width, proficiency_width=proficiency_width,
        max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )


def _make_env(cfg: EnvConfig) -> StochasticEnv:
    set_all_seeds(0)
    return StochasticEnv(cfg)


class TestActionMasking:
    def test_get_all_actions_always_returns_moves(self):
        cfg = _make_cfg()
        actions = get_all_actions(cfg)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_get_phase2_actions_not_on_task(self):
        cfg = _make_cfg()
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(0,)
        )
        # Not on a task cell, so phase 2 should return an empty list
        actions = get_phase2_actions(s, env)
        assert actions == []

    def test_get_phase2_actions_on_task_with_phi_match(self):
        # proficiency_width=4 → all agents have phi > 0 for all types
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 1)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(0,)
        )
        actions = get_phase2_actions(s, env)
        assert Action.STAY in actions
        assert make_pick_action(0) in actions

    def test_get_phase2_actions_no_phi_match_returns_stay_only(self):
        # Agent 0 has proficiency_width=0 → phi[0, κ]=1 only for κ=0
        # Task type=1 → phi[0,1]=0 → not eligible → only STAY
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=0)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 1)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(1,)  # agent 0 not eligible for type 1
        )
        actions = get_phase2_actions(s, env)
        # phi[0,1] = 0 → only STAY
        assert actions == [Action.STAY]

    def test_get_phase2_actions_stacked_tasks(self):
        # proficiency_width=4 → all agents see all types
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 1)),
            task_positions=(Grid(2, 2), Grid(2, 2)),
            actor=0, task_types=(0, 1)
        )
        actions = get_phase2_actions(s, env)
        # Both types present, both eligible → STAY + pick(0) + pick(1)
        assert len(actions) == 3
        assert Action.STAY in actions
        assert make_pick_action(0) in actions
        assert make_pick_action(1) in actions


class TestNearestAction:
    def test_moves_toward_eligible_task(self):
        # proficiency_width=4 → agent 0 eligible for all types
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)

        # Actor 0 at (0,0), task at (0,1): nearest is RIGHT
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )
        action = nearest_action(s, env)
        assert action == Action.RIGHT

    def test_stays_when_no_tasks(self):
        cfg = _make_cfg(n_agents=2, n_task_types=1, proficiency_width=0)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(2, 2)),
            task_positions=(),
            actor=0, task_types=()
        )
        action = nearest_action(s, env)
        assert action == Action.STAY

    def test_phase2_picks_eligible_type(self):
        # proficiency_width=4 → phi[0,0]=phi[0,1]=1
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(0,),
            pick_phase=True,
        )
        action = nearest_action(s, env)
        assert action == make_pick_action(0)

    def test_phase2_stays_when_no_eligible_type(self):
        # proficiency_width=0 → phi[0, kappa]=1 only for kappa=0
        # Task at actor's cell is type 1 → not eligible → STAY
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=0)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(1,),  # type 1 not eligible for agent 0
            pick_phase=True,
        )
        action = nearest_action(s, env)
        assert action == Action.STAY

    def test_heuristic_dispatch(self):
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )
        a = heuristic_action(s, env, Heuristic.NEAREST)
        assert a == Action.RIGHT


class TestSurvivalAdjustedHungarianAction:
    @staticmethod
    def _env(despawn_prob: float = 0.5) -> StochasticEnv:
        cfg = EnvConfig(
            height=5,
            width=5,
            n_agents=2,
            n_tasks=0,
            gamma=0.99,
            n_task_types=2,
            relatedness_width=1,
            proficiency_width=1,
            max_tasks_per_type=2,
            stochastic=StochasticConfig(
                spawn_prob=0.0,
                despawn_mode=DespawnMode.PROBABILITY,
                despawn_prob=despawn_prob,
            ),
        )
        return _make_env(cfg)

    def test_utility_is_reward_times_survival_to_manhattan_distance(self):
        env = self._env(despawn_prob=0.5)
        env.category_rewards[:] = np.array([[4.0, 4.0], [2.0, 2.0]])
        state = State(
            agent_positions=(Grid(0, 0), Grid(0, 4)),
            task_positions=(Grid(0, 2), Grid(0, 3)),
            actor=0,
            task_types=(0, 1),
        )

        utilities = survival_adjusted_utilities(state, env)

        np.testing.assert_allclose(
            utilities,
            np.array([[2.0, 0.5], [2.0, 2.0]]),
        )

    def test_assignment_is_global_and_each_task_is_unique(self):
        env = self._env(despawn_prob=0.5)
        env.category_rewards[:] = np.array([[4.0, 4.0], [2.0, 2.0]])
        state = State(
            agent_positions=(Grid(0, 0), Grid(0, 4)),
            task_positions=(Grid(0, 2), Grid(0, 3)),
            actor=0,
            task_types=(0, 1),
        )

        assert survival_adjusted_assignment(state, env) == (0, 1)
        assert survival_adjusted_hungarian_action(state, env) == Action.RIGHT
        assert heuristic_action(state, env, Heuristic.HUNGARIAN) == Action.RIGHT

    def test_agents_can_stay_instead_of_accepting_nonpositive_tasks(self):
        env = self._env(despawn_prob=0.2)
        env.category_rewards[:] = -1.0
        state = State(
            agent_positions=(Grid(1, 1), Grid(4, 4)),
            task_positions=(Grid(1, 2),),
            actor=0,
            task_types=(0,),
        )

        assert survival_adjusted_assignment(state, env) == (None, None)
        assert survival_adjusted_hungarian_action(state, env) == Action.STAY

    def test_more_agents_than_tasks_leaves_unassigned_agent_staying(self):
        env = self._env(despawn_prob=0.5)
        env.category_rewards[:] = 1.0
        state = State(
            agent_positions=(Grid(0, 0), Grid(0, 4)),
            task_positions=(Grid(0, 1),),
            actor=1,
            task_types=(0,),
        )

        assert survival_adjusted_assignment(state, env) == (0, None)
        assert survival_adjusted_hungarian_action(state, env) == Action.STAY

    def test_pick_phase_selects_best_positive_task_at_actor_cell(self):
        env = self._env(despawn_prob=0.5)
        env.category_rewards[:] = np.array([[1.0, 1.0], [3.0, 3.0]])
        state = State(
            agent_positions=(Grid(2, 2), Grid(4, 4)),
            task_positions=(Grid(2, 2), Grid(2, 2)),
            actor=0,
            task_types=(0, 1),
            pick_phase=True,
        )

        assert survival_adjusted_hungarian_action(state, env) == make_pick_action(1)


class TestClairvoyantRolloutAction:
    @staticmethod
    def _spawn_env() -> StochasticEnv:
        cfg = EnvConfig(
            height=1,
            width=3,
            n_agents=1,
            n_tasks=0,
            gamma=0.99,
            n_task_types=1,
            relatedness_width=0,
            proficiency_width=0,
            max_tasks_per_type=1,
            stochastic=StochasticConfig(
                spawn_prob=1.0,
                despawn_mode=DespawnMode.NONE,
                despawn_prob=0.0,
                spawn_on_agent_cells=True,
                spawn_at_round_end=True,
            ),
        )
        return _make_env(cfg)

    @staticmethod
    def _empty_state() -> State:
        return State(
            agent_positions=(Grid(0, 0),),
            task_positions=(),
            actor=0,
            task_types=(),
        )

    def test_uses_spawn_inside_horizon_to_intercept_future_task(self):
        env = self._spawn_env()
        state = self._empty_state()

        rng.seed(4)
        assert clairvoyant_rollout_action(state, env, horizon=1) == Action.STAY

        rng.seed(4)
        assert clairvoyant_rollout_action(state, env, horizon=2) == Action.RIGHT

    def test_planning_does_not_consume_rng_or_mutate_round_counter(self):
        env = self._spawn_env()
        state = self._empty_state()
        env._rounds_elapsed = 7
        rng.seed(4)
        expected_rng_state = rng.getstate()

        clairvoyant_rollout_action(state, env, horizon=3)

        assert rng.getstate() == expected_rng_state
        assert env._rounds_elapsed == 7

    def test_dispatch_uses_default_clairvoyant_horizon(self):
        env = self._spawn_env()
        state = self._empty_state()
        rng.seed(4)

        action = heuristic_action(state, env, Heuristic.CLAIRVOYANT_ROLLOUT)

        assert action == Action.RIGHT

    def test_rejects_nonpositive_horizon(self):
        env = self._spawn_env()
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            clairvoyant_rollout_action(self._empty_state(), env, horizon=0)


class TestStochasticMPCAction:
    @staticmethod
    def _planning_env() -> StochasticEnv:
        cfg = EnvConfig(
            height=1,
            width=5,
            n_agents=1,
            n_tasks=0,
            gamma=0.99,
            n_task_types=2,
            relatedness_width=0,
            proficiency_width=2,
            max_tasks_per_type=1,
            stochastic=StochasticConfig(
                spawn_prob=0.0,
                despawn_mode=DespawnMode.NONE,
                despawn_prob=0.0,
            ),
        )
        env = _make_env(cfg)
        env.category_rewards[:] = np.array([[2.0], [10.0]])
        env._precompute_pick_rewards()
        return env

    def test_finite_horizon_return_can_override_hungarian_target(self):
        env = self._planning_env()
        state = State(
            agent_positions=(Grid(0, 2),),
            task_positions=(Grid(0, 1), Grid(0, 4)),
            actor=0,
            task_types=(0, 1),
        )

        assert survival_adjusted_hungarian_action(state, env) == Action.RIGHT
        assert stochastic_mpc_action(
            state,
            env,
            horizon=1,
            n_rollouts=4,
            n_candidates=4,
        ) == Action.LEFT

    def test_sampled_planning_preserves_live_rng_and_environment_counter(self):
        env = TestClairvoyantRolloutAction._spawn_env()
        state = TestClairvoyantRolloutAction._empty_state()
        env._rounds_elapsed = 9
        rng.seed(11)
        expected_rng_state = rng.getstate()

        action = stochastic_mpc_action(
            state,
            env,
            horizon=2,
            n_rollouts=8,
        )

        assert action.is_move()
        assert rng.getstate() == expected_rng_state
        assert env._rounds_elapsed == 9

    def test_dispatch_uses_stochastic_mpc(self):
        env = self._planning_env()
        state = State(
            agent_positions=(Grid(0, 2),),
            task_positions=(Grid(0, 1), Grid(0, 4)),
            actor=0,
            task_types=(0, 1),
        )

        action = heuristic_action(state, env, Heuristic.STOCHASTIC_MPC)

        assert action.is_move()

    @pytest.mark.parametrize(
        ("horizon", "n_rollouts", "message"),
        [
            (0, 1, "horizon must be >= 1"),
            (1, 0, "n_rollouts must be >= 1"),
        ],
    )
    def test_rejects_invalid_search_parameters(
        self,
        horizon,
        n_rollouts,
        message,
    ):
        env = self._planning_env()
        with pytest.raises(ValueError, match=message):
            stochastic_mpc_action(
                State((Grid(0, 2),), (), 0, ()),
                env,
                horizon=horizon,
                n_rollouts=n_rollouts,
            )


class TestRawStochasticMPCAction:
    def test_raw_action_tree_can_override_hungarian_target(self):
        env = TestStochasticMPCAction._planning_env()
        state = State(
            agent_positions=(Grid(0, 2),),
            task_positions=(Grid(0, 1), Grid(0, 4)),
            actor=0,
            task_types=(0, 1),
        )

        assert survival_adjusted_hungarian_action(state, env) == Action.RIGHT
        assert raw_stochastic_mpc_action(
            state,
            env,
            horizon=1,
            n_rollouts=4,
        ) == Action.LEFT

    def test_raw_planning_preserves_live_rng_and_environment_counter(self):
        env = TestClairvoyantRolloutAction._spawn_env()
        state = TestClairvoyantRolloutAction._empty_state()
        env._rounds_elapsed = 9
        rng.seed(11)
        expected_rng_state = rng.getstate()

        action = raw_stochastic_mpc_action(
            state,
            env,
            horizon=2,
            n_rollouts=8,
        )

        assert action.is_move()
        assert rng.getstate() == expected_rng_state
        assert env._rounds_elapsed == 9

    def test_dispatch_uses_raw_stochastic_mpc(self):
        env = TestStochasticMPCAction._planning_env()
        state = State(
            agent_positions=(Grid(0, 2),),
            task_positions=(Grid(0, 1), Grid(0, 4)),
            actor=0,
            task_types=(0, 1),
        )

        action = heuristic_action(state, env, Heuristic.RAW_STOCHASTIC_MPC)

        assert action.is_move()

    @pytest.mark.parametrize(
        ("horizon", "n_rollouts", "message"),
        [
            (0, 1, "horizon must be >= 1"),
            (1, 0, "n_rollouts must be >= 1"),
        ],
    )
    def test_rejects_invalid_search_parameters(
        self,
        horizon,
        n_rollouts,
        message,
    ):
        env = TestStochasticMPCAction._planning_env()
        with pytest.raises(ValueError, match=message):
            raw_stochastic_mpc_action(
                State((Grid(0, 2),), (), 0, ()),
                env,
                horizon=horizon,
                n_rollouts=n_rollouts,
            )
