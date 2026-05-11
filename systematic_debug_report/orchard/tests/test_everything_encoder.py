"""Tests for the raw-binary EverythingEncoder."""

from __future__ import annotations

import torch

import orchard.encoding as encoding
from orchard.datatypes import EnvConfig, Grid, State, StochasticConfig
from orchard.enums import DespawnMode, EncoderType


def _make_env_cfg() -> EnvConfig:
    return EnvConfig(
        height=4,
        width=5,
        n_agents=3,
        n_tasks=2,
        gamma=0.99,
        n_task_types=2,
        stochastic=StochasticConfig(
            spawn_prob=0.0,
            despawn_mode=DespawnMode.NONE,
            despawn_prob=0.0,
        ),
    )


def _make_state(*, pick_phase: bool = False) -> State:
    return State(
        agent_positions=(Grid(0, 0), Grid(2, 3), Grid(3, 4)),
        task_positions=(Grid(1, 1), Grid(2, 3), Grid(3, 0)),
        actor=1,
        task_types=(0, 1, 1),
        pick_phase=pick_phase,
    )


class _EnvStub:
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg


def _init_encoder(n_networks: int = 3) -> None:
    encoding.init_encoder(
        EncoderType.EVERYTHING_CNN_GRID,
        _EnvStub(_make_env_cfg()),
        n_networks=n_networks,
    )


def test_encode_has_expected_binary_channels():
    _init_encoder()
    state = _make_state()
    out = encoding.encode(state, agent_idx=0)

    T = 2
    N = 3
    assert out.grid.shape == (T + N + 1, 4, 5)
    assert out.scalar.shape == (N + 1,)

    expected = torch.zeros(T + N + 1, 4, 5)
    expected[0, 1, 1] = 1.0
    expected[1, 2, 3] = 1.0
    expected[1, 3, 0] = 1.0
    expected[T + 0, 0, 0] = 1.0
    expected[T + 1, 2, 3] = 1.0
    expected[T + 2, 3, 4] = 1.0
    expected[T + N, 2, 3] = 1.0

    assert torch.equal(out.grid, expected)
    assert torch.equal(out.scalar, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert set(out.grid.unique().tolist()) <= {0.0, 1.0}


def test_encode_sets_pick_phase_scalar():
    _init_encoder()
    out = encoding.encode(_make_state(pick_phase=True), agent_idx=2)
    assert torch.equal(out.scalar, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_encode_all_agents_broadcasts_identical_binary_view():
    _init_encoder(n_networks=3)
    state = _make_state()
    single = encoding.encode(state, agent_idx=0)
    grids, scalars = encoding.encode_all_agents(state)

    assert grids.shape == (3, 6, 4, 5)
    assert scalars.shape == (3, 4)
    for i in range(3):
        assert torch.equal(grids[i], single.grid)
        assert torch.equal(scalars[i], single.scalar)


def test_encode_batch_for_actions_updates_actor_position_and_pick_phase():
    _init_encoder()
    state = _make_state()
    moved = State(
        agent_positions=(Grid(0, 0), Grid(2, 4), Grid(3, 4)),
        task_positions=state.task_positions,
        actor=1,
        task_types=state.task_types,
    )
    pick_after = State(
        agent_positions=state.agent_positions,
        task_positions=(Grid(1, 1), Grid(3, 0)),
        actor=1,
        task_types=(0, 1),
        pick_phase=True,
    )

    out = encoding.encode_batch_for_actions(state, agent_idx=0, after_states=[moved, pick_after])

    T = 2
    N = 3
    assert out.grid.shape == (2, T + N + 1, 4, 5)
    assert out.scalar.shape == (2, N + 1)

    assert out.grid[0, T + 1, 2, 4].item() == 1.0
    assert out.grid[0, T + N, 2, 4].item() == 1.0
    assert out.grid[0, T + 1].sum().item() == 1.0
    assert out.grid[0, T + N].sum().item() == 1.0
    assert out.scalar[0, -1].item() == 0.0

    assert out.grid[1, 1, 2, 3].item() == 0.0
    assert out.grid[1, 0, 1, 1].item() == 1.0
    assert out.grid[1, 1, 3, 0].item() == 1.0
    assert out.scalar[1, -1].item() == 1.0


def test_encode_all_agents_for_actions_broadcasts_action_batch_to_networks():
    _init_encoder(n_networks=2)
    state = _make_state()
    after_states = [
        State(
            agent_positions=(Grid(0, 0), Grid(2, 4), Grid(3, 4)),
            task_positions=state.task_positions,
            actor=1,
            task_types=state.task_types,
        ),
        State(
            agent_positions=state.agent_positions,
            task_positions=(Grid(1, 1), Grid(3, 0)),
            actor=1,
            task_types=(0, 1),
            pick_phase=True,
        ),
    ]

    grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
    batch = encoding.encode_batch_for_actions(state, agent_idx=0, after_states=after_states)

    assert grids.shape == (2, 2, 6, 4, 5)
    assert scalars.shape == (2, 2, 4)
    for i in range(2):
        assert torch.equal(grids[i], batch.grid)
        assert torch.equal(scalars[i], batch.scalar)
