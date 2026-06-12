"""Tests for optimistic oracle evaluation."""

import pytest

from orchard.datatypes import EnvConfig, Grid, State, StochasticConfig
from orchard.env.stochastic import StochasticEnv
from orchard.enums import DespawnMode
from orchard.oracle_eval import evaluate_oracle_metrics, oracle_step
from orchard.seed import set_all_seeds


def _make_oracle_env() -> StochasticEnv:
    cfg = EnvConfig(
        height=4,
        width=4,
        n_agents=2,
        n_tasks=0,
        gamma=0.9,
        n_task_types=3,
        clustering=1,
        specialization=3,
        max_tasks_per_type=3,
        stochastic=StochasticConfig(
            spawn_prob=0.0,
            despawn_mode=DespawnMode.NONE,
            despawn_prob=0.0,
        ),
    )
    set_all_seeds(0)
    env = StochasticEnv(cfg)
    env.category_rewards[:] = [
        [-1.0, -1.0],
        [0.25, 0.75],
        [2.0, -0.5],
    ]
    env._precompute_pick_rewards()
    return env


def test_oracle_picks_best_positive_available_task_ignoring_position():
    env = _make_oracle_env()
    state = State(
        agent_positions=(Grid(0, 0), Grid(3, 3)),
        task_positions=(Grid(0, 1), Grid(2, 2), Grid(3, 0)),
        actor=0,
        task_types=(0, 1, 2),
    )

    next_state, rewards, tasks_picked = oracle_step(state, env)

    assert rewards == pytest.approx((2.0, -0.5))
    assert tasks_picked == 1
    assert next_state.actor == 1
    assert next_state.task_positions == (Grid(0, 1), Grid(2, 2))
    assert next_state.task_types == (0, 1)


def test_oracle_picks_nothing_when_all_available_tasks_are_non_positive():
    env = _make_oracle_env()
    state = State(
        agent_positions=(Grid(0, 0), Grid(3, 3)),
        task_positions=(Grid(0, 1),),
        actor=0,
        task_types=(0,),
    )

    next_state, rewards, tasks_picked = oracle_step(state, env)

    assert rewards == pytest.approx((0.0, 0.0))
    assert tasks_picked == 0
    assert next_state.actor == 1
    assert next_state.task_positions == state.task_positions
    assert next_state.task_types == state.task_types


def test_evaluate_oracle_metrics_uses_greedy_team_rps_units():
    env = _make_oracle_env()
    state = State(
        agent_positions=(Grid(0, 0), Grid(3, 3)),
        task_positions=(Grid(2, 2),),
        actor=0,
        task_types=(2,),
    )

    metrics = evaluate_oracle_metrics(state, env, n_steps=2)

    assert metrics["team_rps"] == pytest.approx(0.75)
    assert metrics["rps"] == pytest.approx(1.0)
    assert metrics["tasks_picked_per_step"] == pytest.approx(0.5)
