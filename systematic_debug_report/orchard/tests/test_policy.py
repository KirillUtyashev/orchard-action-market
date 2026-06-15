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
    eps_nearest_action,
    nearest_rewarding_task_action,
    nearest_task_action,
    heuristic_action,
)
from orchard.seed import set_all_seeds


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

    def test_eps_nearest_with_zero_random_prob_matches_nearest(self):
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,),
        )
        assert eps_nearest_action(s, env, random_action_prob=0.0) == nearest_action(s, env)

    def test_eps_nearest_with_full_random_prob_returns_valid_action(self):
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,),
        )
        assert eps_nearest_action(s, env, random_action_prob=1.0).is_move()

    def test_eps_nearest_dispatch_uses_default_random_mixture(self):
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=4)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,),
        )
        assert heuristic_action(s, env, Heuristic.EPS_NEAREST).is_move()

    def test_nearest_rewarding_task_picks_best_positive_team_reward_here(self):
        cfg = _make_cfg(n_agents=3, n_task_types=3, proficiency_width=3)
        env = _make_env(cfg)
        env.category_rewards = np.array([
            [-2.0, -2.0, -2.0],
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ], dtype=np.float32)
        env._precompute_pick_rewards()
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(2, 2), Grid(2, 2), Grid(2, 2)),
            actor=0,
            task_types=(0, 1, 2),
            pick_phase=True,
        )
        action = nearest_rewarding_task_action(s, env)
        assert action == make_pick_action(2)

    def test_nearest_rewarding_task_moves_by_reward_over_distance(self):
        cfg = _make_cfg(n_agents=3, n_task_types=3, proficiency_width=3)
        env = _make_env(cfg)
        env.category_rewards = np.array([
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [-100.0, 0.0, 0.0],
        ], dtype=np.float32)
        env._precompute_pick_rewards()
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 2), Grid(2, 3), Grid(2, 1)),
            actor=0,
            task_types=(1, 0, 2),
        )
        action = nearest_rewarding_task_action(s, env)
        assert action == Action.UP

    def test_nearest_task_ignores_reward_and_moves_to_nearest_proficient_task(self):
        cfg = _make_cfg(n_agents=3, n_task_types=3, proficiency_width=3)
        env = _make_env(cfg)
        env.category_rewards = np.array([
            [-100.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        env._precompute_pick_rewards()
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 2), Grid(2, 3)),
            actor=0,
            task_types=(1, 0),
        )
        action = nearest_task_action(s, env)
        assert action == Action.RIGHT

    def test_nearest_rewarding_task_dispatch_returns_move_action(self):
        cfg = _make_cfg(n_agents=2, n_task_types=2, proficiency_width=2)
        env = _make_env(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )
        action = heuristic_action(s, env, Heuristic.NEAREST_REWARDING_TASK)
        assert action.is_move()
