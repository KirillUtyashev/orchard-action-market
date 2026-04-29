"""Prerequisite for skip_sort_transient: encoder must be sort-invariant.

Verifies that encode_all_agents_for_actions produces bit-identical output
regardless of the order of tasks in State.task_positions / task_types.
If this test fails, skip_sort_transient must NOT be enabled.
"""

import itertools
import torch
import orchard.encoding as encoding
from orchard.datatypes import EnvConfig, Grid, State, StochasticConfig, sort_tasks
from orchard.enums import DespawnMode, EncoderType, PickMode


def _make_env_cfg() -> EnvConfig:
    return EnvConfig(
        height=9, width=9, n_agents=4, n_tasks=4,
        gamma=0.99, r_picker=-1.0,
        n_task_types=2, r_low=-1.0,
        task_assignments=((0,), (0,), (1,), (1,)),
        pick_mode=PickMode.CHOICE,
        max_tasks_per_type=10,
        stochastic=StochasticConfig(
            spawn_prob=0.01, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=0.0
        ),
    )


def _base_state(env_cfg: EnvConfig) -> State:
    return State(
        agent_positions=(Grid(0,0), Grid(1,1), Grid(2,2), Grid(3,3)),
        task_positions=(Grid(4,4), Grid(5,5), Grid(6,6), Grid(7,7)),
        task_types=(0, 1, 0, 1),
        actor=0,
        pick_phase=False,
    )


def _permute_state(state: State, perm: list[int]) -> State:
    tp = tuple(state.task_positions[i] for i in perm)
    tt = tuple(state.task_types[i] for i in perm)     # type: ignore[index]
    return State(
        agent_positions=state.agent_positions,
        task_positions=tp,
        task_types=tt,
        actor=state.actor,
        pick_phase=state.pick_phase,
    )


def _after_states(state: State, env_cfg: EnvConfig) -> list[State]:
    from orchard.env.base import BaseEnv
    from orchard.env import create_env
    from orchard.policy import get_all_actions
    env = create_env(env_cfg)
    actions = get_all_actions(env_cfg)
    after = []
    for a in actions:
        s = env.apply_action(state, a)
        after.append(s.with_pick_phase() if s.is_agent_on_task(s.actor) else s)
    return after


def test_encode_all_agents_sort_invariant():
    env_cfg = _make_env_cfg()
    encoding.init_encoder(EncoderType.FILTERED_TASK_CNN_GRID, env_cfg)

    state_sorted = _base_state(env_cfg)
    after_sorted = _after_states(state_sorted, env_cfg)

    # Try every permutation of the 4 tasks
    for perm in itertools.permutations(range(4)):
        state_perm = _permute_state(state_sorted, list(perm))
        after_perm = [_permute_state(s, list(perm))
                      if s.task_positions != state_sorted.task_positions
                      else s for s in after_sorted]

        grids_ref, scalars_ref = encoding.encode_all_agents_for_actions(state_sorted, after_sorted)
        grids_p,   scalars_p   = encoding.encode_all_agents_for_actions(state_perm,   after_sorted)

        assert torch.equal(grids_ref, grids_p), (
            f"encode_all_agents_for_actions NOT sort-invariant for perm={perm}: "
            f"max diff = {(grids_ref - grids_p).abs().max().item()}"
        )
        assert torch.equal(scalars_ref, scalars_p), (
            f"scalars NOT sort-invariant for perm={perm}"
        )


def test_encode_all_agents_train_sort_invariant():
    """encode_all_agents (used in TD path) must also be sort-invariant."""
    env_cfg = _make_env_cfg()
    encoding.init_encoder(EncoderType.FILTERED_TASK_CNN_GRID, env_cfg)
    state_sorted = _base_state(env_cfg)

    for perm in itertools.permutations(range(4)):
        state_perm = _permute_state(state_sorted, list(perm))
        grids_ref, scalars_ref = encoding.encode_all_agents(state_sorted)
        grids_p,   scalars_p   = encoding.encode_all_agents(state_perm)
        assert torch.equal(grids_ref, grids_p), (
            f"encode_all_agents NOT sort-invariant for perm={perm}: "
            f"max diff = {(grids_ref - grids_p).abs().max().item()}"
        )
        assert torch.equal(scalars_ref, scalars_p), f"scalars NOT sort-invariant for perm={perm}"
