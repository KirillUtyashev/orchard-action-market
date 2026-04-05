"""State sampling helpers for actor policy diagnostics."""

from __future__ import annotations

import json
import random

import numpy as np
import torch

from orchard.actor_critic.action_space import (
    build_phase1_legal_mask,
    build_phase2_legal_mask,
    policy_index_to_action,
)
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks
from orchard.env import create_env
from orchard.enums import PickMode


def serialize_state(state: State) -> str:
    """Stable compact JSON representation for logging policy-eval states."""
    tasks = []
    if state.task_types is None:
        task_types = (0,) * len(state.task_positions)
    else:
        task_types = state.task_types
    for pos, task_type in sorted(
        zip(state.task_positions, task_types),
        key=lambda item: (item[0].row, item[0].col, item[1]),
    ):
        tasks.append([int(pos.row), int(pos.col), int(task_type)])

    payload = {
        "actor": int(state.actor),
        "pick_phase": bool(state.pick_phase),
        "agent_positions": [[int(pos.row), int(pos.col)] for pos in state.agent_positions],
        "tasks": tasks,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _preferred_cells(env_cfg: EnvConfig) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for row in range(1, max(1, env_cfg.height - 1)):
        for col in range(1, max(1, env_cfg.width - 1)):
            if row < env_cfg.height - 1 and col < env_cfg.width - 1:
                cells.append((row, col))
    for row in range(env_cfg.height):
        for col in range(env_cfg.width):
            if row in {0, env_cfg.height - 1} or col in {0, env_cfg.width - 1}:
                cells.append((row, col))
    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if cell not in seen:
            deduped.append(cell)
            seen.add(cell)
    return deduped


def _interior_anchor(env_cfg: EnvConfig) -> tuple[int, int]:
    row = max(0, min(env_cfg.height - 1, env_cfg.height // 2))
    col = max(0, min(env_cfg.width - 1, env_cfg.width // 2))
    if env_cfg.height > 2:
        row = max(1, min(env_cfg.height - 2, row))
    if env_cfg.width > 2:
        col = max(1, min(env_cfg.width - 2, col))
    return row, col


def _build_canonical_state(
    env_cfg: EnvConfig,
    actor_id: int,
    actor_pos: tuple[int, int],
    task_specs: list[tuple[int, int, int]],
    other_anchor_positions: list[tuple[int, int]] | None = None,
    *,
    pick_phase: bool = False,
) -> State:
    if other_anchor_positions is None:
        other_anchor_positions = []

    actor_pos = (int(actor_pos[0]), int(actor_pos[1]))
    task_cells = {(int(row), int(col)) for row, col, _ in task_specs}

    agent_positions: list[Grid | None] = [None] * env_cfg.n_agents
    agent_positions[actor_id] = Grid(*actor_pos)
    used_positions = {actor_pos}
    reserved_positions = set(task_cells)

    other_ids = [idx for idx in range(env_cfg.n_agents) if idx != actor_id]
    for other_id, pos in zip(other_ids, other_anchor_positions):
        pos_t = (int(pos[0]), int(pos[1]))
        if pos_t in used_positions or pos_t in reserved_positions:
            continue
        agent_positions[other_id] = Grid(*pos_t)
        used_positions.add(pos_t)

    for other_id in other_ids:
        if agent_positions[other_id] is not None:
            continue
        for pos in _preferred_cells(env_cfg):
            if pos not in used_positions and pos not in reserved_positions:
                agent_positions[other_id] = Grid(*pos)
                used_positions.add(pos)
                break

    if any(pos is None for pos in agent_positions):
        raise RuntimeError("Could not assign deterministic positions for all agents.")

    task_positions, task_types = sort_tasks(
        [Grid(int(row), int(col)) for row, col, _ in task_specs],
        [int(task_type) for _, _, task_type in task_specs],
    )
    return State(
        agent_positions=tuple(agent_positions),  # type: ignore[arg-type]
        task_positions=task_positions,
        actor=actor_id,
        task_types=task_types,
        pick_phase=pick_phase,
    )


def _save_rng_state() -> dict[str, object]:
    import orchard.env.stochastic as stochastic_module
    import orchard.seed as seed_module

    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "seed_rng": seed_module.rng.getstate(),
        "stochastic_rng": stochastic_module.rng.getstate(),
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    import orchard.env.stochastic as stochastic_module
    import orchard.seed as seed_module

    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    seed_module.rng.setstate(state["seed_rng"])  # type: ignore[arg-type]
    stochastic_module.rng.setstate(state["stochastic_rng"])  # type: ignore[arg-type]


def _seed_rngs(seed: int) -> None:
    import orchard.env.stochastic as stochastic_module
    import orchard.seed as seed_module

    random.seed(seed)
    torch.manual_seed(seed)
    seed_module.rng.seed(seed)
    stochastic_module.rng.seed(seed)


def sample_phase1_policy_eval_states(
    env_cfg: EnvConfig,
    *,
    num_states: int = 100,
    burnin: int = 500,
    stride: int = 5,
    seed: int = 42069,
) -> list[State]:
    """Sample deterministic rollout states using the same random-rollout principle as debug."""
    if num_states <= 0:
        return []

    saved_rng_state = _save_rng_state()
    local_rng = random.Random(seed)
    try:
        _seed_rngs(seed)
        env = create_env(env_cfg)
        state = env.init_state()
        sampled: list[State] = []
        total_turns = burnin + stride * num_states

        for turn_idx in range(total_turns):
            phase1_mask = build_phase1_legal_mask(state, env_cfg)
            phase1_indices = np.flatnonzero(phase1_mask)
            chosen_idx = int(phase1_indices[local_rng.randrange(len(phase1_indices))])
            move_action = policy_index_to_action(chosen_idx)

            s_moved = env.apply_action(state, move_action)
            if s_moved.is_agent_on_task(s_moved.actor):
                if env_cfg.pick_mode == PickMode.FORCED:
                    s_after, _ = env.resolve_pick(s_moved)
                else:
                    phase2_state = s_moved.with_pick_phase()
                    phase2_mask = build_phase2_legal_mask(phase2_state, env_cfg)
                    phase2_indices = np.flatnonzero(phase2_mask)
                    pick_idx = int(phase2_indices[local_rng.randrange(len(phase2_indices))])
                    pick_action = policy_index_to_action(pick_idx)
                    s_after, _ = env.resolve_pick(
                        s_moved,
                        pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
                    )
            else:
                s_after = s_moved

            state = env.advance_actor(env.spawn_and_despawn(s_after))
            if turn_idx >= burnin and ((turn_idx - burnin) % stride == 0):
                sampled.append(state)
                if len(sampled) >= num_states:
                    break

        return sampled
    finally:
        _restore_rng_state(saved_rng_state)


def generate_phase2_policy_eval_states(env_cfg: EnvConfig) -> list[tuple[str, State]]:
    """Create deterministic curated states for inspecting pick decisions."""
    actor_pos = _interior_anchor(env_cfg)
    other_anchor = (max(0, env_cfg.height - 2), max(0, env_cfg.width - 2))
    cases: list[tuple[str, State]] = []

    for actor_id in range(env_cfg.n_agents):
        for tau in range(env_cfg.n_task_types):
            label = f"actor_{actor_id}_singleton_{tau}"
            cases.append(
                (
                    label,
                    _build_canonical_state(
                        env_cfg,
                        actor_id,
                        actor_pos,
                        [(actor_pos[0], actor_pos[1], tau)],
                        [other_anchor],
                        pick_phase=True,
                    ),
                )
            )

        for tau_a in range(env_cfg.n_task_types):
            for tau_b in range(tau_a + 1, env_cfg.n_task_types):
                label = f"actor_{actor_id}_pair_{tau_a}_{tau_b}"
                cases.append(
                    (
                        label,
                        _build_canonical_state(
                            env_cfg,
                            actor_id,
                            actor_pos,
                            [
                                (actor_pos[0], actor_pos[1], tau_a),
                                (actor_pos[0], actor_pos[1], tau_b),
                            ],
                            [other_anchor],
                            pick_phase=True,
                        ),
                    )
                )

        if env_cfg.n_task_types > 2:
            label = f"actor_{actor_id}_all_types_present"
            cases.append(
                (
                    label,
                    _build_canonical_state(
                        env_cfg,
                        actor_id,
                        actor_pos,
                        [
                            (actor_pos[0], actor_pos[1], tau)
                            for tau in range(env_cfg.n_task_types)
                        ],
                        [other_anchor],
                        pick_phase=True,
                    ),
                )
            )

    return cases


__all__ = [
    "serialize_state",
    "sample_phase1_policy_eval_states",
    "generate_phase2_policy_eval_states",
]
