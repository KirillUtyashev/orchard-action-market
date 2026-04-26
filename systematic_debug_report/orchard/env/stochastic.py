"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

from orchard.enums import DespawnMode, PickMode, TaskSpawnMode
from orchard.env.base import BaseEnv
from orchard.seed import rng
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks


class StochasticEnv(BaseEnv):
    """Stochastic spawn/despawn with configurable modes."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        assert cfg.stochastic is not None, "StochasticEnv requires stochastic config"
        self.stoch = cfg.stochastic
        self._all_cells: list[Grid] = [
            Grid(r, c)
            for r in range(cfg.height)
            for c in range(cfg.width)
        ]

    def init_state(self) -> State:
        """Random placement, no overlaps between agents and tasks."""
        cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
        ]

        if self.stoch.old_init_rng:
            # Single combined sample matching old code's RNG pattern:
            # rng.sample(cells, n_agents + n_tasks) in one call, then split.
            # Only valid for n_task_types=1; for multi-type falls through to
            # per-type placement below so each type gets its initial tasks.
            n_tasks = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
            chosen = rng.sample(cells, self.cfg.n_agents + n_tasks)
            agent_positions = tuple(chosen[:self.cfg.n_agents])
            if self.cfg.n_task_types == 1:
                task_cells = chosen[self.cfg.n_agents:]
                all_task_positions = list(task_cells)
                all_task_types = [0] * len(task_cells)
            else:
                # Multi-type: agents placed via old RNG, tasks placed per type
                # using PER_TYPE_UNIQUE (types may share cells, matching runtime).
                agent_set = set(agent_positions)
                all_task_positions = []
                all_task_types = []
                for tau in range(self.cfg.n_task_types):
                    count = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
                    cells_with_tau: set[Grid] = set()
                    available = [c for c in cells if c not in agent_set and c not in cells_with_tau]
                    to_spawn = min(count, len(available))
                    for cell in rng.sample(available, to_spawn):
                        all_task_positions.append(cell)
                        all_task_types.append(tau)
        else:
            # Multi-type: place agents first, then tasks per type using
            # PER_TYPE_UNIQUE (types may share cells, matching runtime behavior).
            agent_positions = tuple(rng.sample(cells, self.cfg.n_agents))
            agent_set = set(agent_positions)
            all_task_positions = []
            all_task_types = []
            for tau in range(self.cfg.n_task_types):
                count = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
                cells_with_tau = {p for p, t in zip(all_task_positions, all_task_types) if t == tau}
                available = [c for c in cells if c not in agent_set and c not in cells_with_tau]
                to_spawn = min(count, len(available))
                for cell in rng.sample(available, to_spawn):
                    all_task_positions.append(cell)
                    all_task_types.append(tau)

        tp, tt = sort_tasks(all_task_positions, all_task_types)
        return State(
            agent_positions=agent_positions,
            task_positions=tp,
            actor=0,
            task_types=tt,
        )

    def spawn_and_despawn(self, state: State) -> State:
        """Despawn phase then spawn phase.

        If spawn_at_round_end is True, this is a no-op for every actor except
        the last one (actor == n_agents - 1), so tasks only change once per round.
        """
        if self.stoch.spawn_at_round_end and state.actor != self.cfg.n_agents - 1:
            return state
        return self._spawn_and_despawn_multi(state)

    def _spawn_and_despawn_multi(self, state: State) -> State:
        """Spawn/despawn for n_task_types > 1."""
        positions = list(state.task_positions)
        assert state.task_types is not None, "task_types must be set"
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
        tsm = self.stoch.task_spawn_mode
        if tsm is None:
            tsm = TaskSpawnMode.PER_TYPE_UNIQUE

        # Build shared structures once (avoids O(n_types) redundant passes).
        agent_set = set(state.agent_positions)
        block_agents = not self.stoch.spawn_on_agent_cells
        n_tau_counts = [0] * self.cfg.n_task_types
        cells_by_type: list[set[Grid]] = [set() for _ in range(self.cfg.n_task_types)]
        for pos, tau in zip(positions, types):
            n_tau_counts[tau] += 1
            cells_by_type[tau].add(pos)

        for tau in range(self.cfg.n_task_types):
            n_tau = n_tau_counts[tau]
            if n_tau >= self.cfg.max_tasks_per_type:
                continue

            if tsm == TaskSpawnMode.GLOBAL_UNIQUE:
                task_set = set(positions)
                empty_cells = [
                    c for c in self._all_cells
                    if c not in task_set and (not block_agents or c not in agent_set)
                ]
            else:
                empty_cells = [
                    c for c in self._all_cells
                    if c not in cells_by_type[tau] and (not block_agents or c not in agent_set)
                ]

            rng.shuffle(empty_cells)
            for cell in empty_cells:
                if n_tau >= self.cfg.max_tasks_per_type:
                    break
                if rng.random() < self.stoch.spawn_prob:
                    positions.append(cell)
                    types.append(tau)
                    cells_by_type[tau].add(cell)
                    n_tau += 1

        tp, tt = sort_tasks(positions, types)
        return State(
            agent_positions=state.agent_positions,
            task_positions=tp,
            actor=state.actor,
            task_types=tt,
        )
