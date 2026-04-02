"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

from orchard.enums import DespawnMode, PickMode
from orchard.env.base import BaseEnv
from orchard.seed import rng
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks


class StochasticEnv(BaseEnv):
    """Stochastic spawn/despawn with configurable modes."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        assert cfg.stochastic is not None, "StochasticEnv requires stochastic config"
        self.stoch = cfg.stochastic

    def init_state(self) -> State:
        """Random placement, no overlaps between agents and tasks."""
        cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
        ]

        # Place agents first
        agent_positions = tuple(rng.sample(cells, self.cfg.n_agents))
        occupied = set(agent_positions)

        # Spawn initial tasks per type
        all_task_positions: list[Grid] = []
        all_task_types: list[int] = []

        for tau in range(self.cfg.n_task_types):
            count = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
            available = [c for c in cells if c not in occupied]
            to_spawn = min(count, len(available))
            chosen_cells = rng.sample(available, to_spawn)
            for cell in chosen_cells:
                all_task_positions.append(cell)
                all_task_types.append(tau)
                occupied.add(cell)

        tp, tt = sort_tasks(all_task_positions, all_task_types)
        return State(
            agent_positions=agent_positions,
            task_positions=tp,
            actor=0,
            task_types=tt,
        )

    def spawn_and_despawn(self, state: State) -> State:
        """Despawn phase then spawn phase."""
        return self._spawn_and_despawn_multi(state)

    def _spawn_and_despawn_multi(self, state: State) -> State:
        """Spawn/despawn for n_task_types > 1."""
        positions = list(state.task_positions)
        types = list(state.task_types)

        # --- Despawn phase ---
        if self.stoch.despawn_mode == DespawnMode.PROBABILITY:
            keep = [
                i for i in range(len(positions))
                if rng.random() >= self.stoch.despawn_prob
            ]
            positions = [positions[i] for i in keep]
            types = [types[i] for i in keep]

        # --- Spawn phase (per type) ---
        # Resolve effective spawn mode:
        #   explicit task_spawn_mode overrides; None → auto based on pick_mode
        from orchard.enums import TaskSpawnMode
        tsm = self.stoch.task_spawn_mode
        if tsm is None:
            tsm = (
                TaskSpawnMode.GLOBAL_UNIQUE
                if self.cfg.pick_mode == PickMode.FORCED
                else TaskSpawnMode.PER_TYPE_UNIQUE
            )

        for tau in range(self.cfg.n_task_types):
            n_tau = sum(1 for t in types if t == tau)
            if n_tau >= self.cfg.max_tasks_per_type:
                continue

            if tsm == TaskSpawnMode.GLOBAL_UNIQUE:
                # No task of ANY type, no agent
                task_set = set(positions)
                agent_set = set(state.agent_positions)
                empty_cells = [
                    Grid(r, c)
                    for r in range(self.cfg.height)
                    for c in range(self.cfg.width)
                    if Grid(r, c) not in task_set
                    and Grid(r, c) not in agent_set
                ]
            else:
                # PER_TYPE_UNIQUE: no task of type τ at this cell, no agent
                cells_with_tau = set(
                    positions[i] for i in range(len(positions))
                    if types[i] == tau
                )
                agent_set = set(state.agent_positions)
                empty_cells = [
                    Grid(r, c)
                    for r in range(self.cfg.height)
                    for c in range(self.cfg.width)
                    if Grid(r, c) not in cells_with_tau
                    and Grid(r, c) not in agent_set
                ]

            for cell in empty_cells:
                if n_tau >= self.cfg.max_tasks_per_type:
                    break
                if rng.random() < self.stoch.spawn_prob:
                    positions.append(cell)
                    types.append(tau)
                    n_tau += 1

        tp, tt = sort_tasks(positions, types)
        return State(
            agent_positions=state.agent_positions,
            task_positions=tp,
            actor=state.actor,
            task_types=tt,
        )
