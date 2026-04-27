"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

import random as _random

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
        # Per-type RNG objects for exact T=1 vs T=M equivalence testing.
        # Each type k uses _type_rngs[k] for init, despawn, spawn, AND epsilon-greedy
        # (the trainer reads this same list). None → fall back to global rng.
        if cfg.stochastic.per_type_seeds is not None:
            assert len(cfg.stochastic.per_type_seeds) == cfg.n_task_types, (
                f"per_type_seeds must have one entry per task type "
                f"(got {len(cfg.stochastic.per_type_seeds)}, expected {cfg.n_task_types})"
            )
            self._type_rngs: list[_random.Random] | None = [
                _random.Random(s) for s in cfg.stochastic.per_type_seeds
            ]
            # Per-type spawn timing: type k spawns when its team's LAST agent acts.
            # This ensures type-k dynamics match a standalone T=1 run exactly.
            assert cfg.task_assignments is not None
            self._per_type_last_agent: list[int] | None = [
                max(i for i in range(cfg.n_agents) if tau in set(cfg.task_assignments[i]))
                for tau in range(cfg.n_task_types)
            ]
        else:
            self._type_rngs = None
            self._per_type_last_agent = None

    def init_state(self) -> State:
        """Random placement, no overlaps between agents and tasks."""
        cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
        ]

        if self._type_rngs is not None:
            # Per-team init: each team k uses _type_rngs[k] to place its agents
            # and type-k tasks independently. Matches T=1 run k exactly.
            assert self.cfg.task_assignments is not None
            n_tasks = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
            agent_positions_by_idx: dict[int, Grid] = {}
            all_task_positions: list[Grid] = []
            all_task_types: list[int] = []
            for tau in range(self.cfg.n_task_types):
                trng = self._type_rngs[tau]
                team_agent_indices = [
                    i for i in range(self.cfg.n_agents)
                    if tau in set(self.cfg.task_assignments[i])
                ]
                n_team = len(team_agent_indices)
                chosen = trng.sample(cells, n_team + n_tasks)
                print(f"[init_state] tau={tau} team_agents={team_agent_indices} "
                      f"agent_cells={[(c.row,c.col) for c in chosen[:n_team]]} "
                      f"task_cells={[(c.row,c.col) for c in chosen[n_team:]]}")
                for idx, agent_i in enumerate(team_agent_indices):
                    agent_positions_by_idx[agent_i] = chosen[idx]
                for cell in chosen[n_team:]:
                    all_task_positions.append(cell)
                    all_task_types.append(tau)
            agent_positions = tuple(
                agent_positions_by_idx[i] for i in range(self.cfg.n_agents)
            )
        elif self.stoch.old_init_rng:
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

        If spawn_at_round_end is True with per_type_seeds, each type k spawns
        only when its team's last agent acts (per _per_type_last_agent). Without
        per_type_seeds the original behaviour applies: all types spawn once when
        the global last agent (n_agents-1) acts.
        """
        if not self.stoch.spawn_at_round_end:
            return self._spawn_and_despawn_multi(state, active_types=None)

        if self._per_type_last_agent is not None:
            active_types = [k for k in range(self.cfg.n_task_types)
                            if self._per_type_last_agent[k] == state.actor]
            if not active_types:
                return state
            return self._spawn_and_despawn_multi(state, active_types=active_types)

        if state.actor != self.cfg.n_agents - 1:
            return state
        return self._spawn_and_despawn_multi(state, active_types=None)

    def _spawn_and_despawn_multi(self, state: State, active_types: list[int] | None = None) -> State:
        """Spawn/despawn for n_task_types > 1.

        active_types: if set, only process these type indices (for per-type spawn timing).
        """
        positions = list(state.task_positions)
        assert state.task_types is not None, "task_types must be set"
        types = list(state.task_types)
        active_set = set(active_types) if active_types is not None else None

        # --- Despawn phase (per type when using isolated RNGs) ---
        if self.stoch.despawn_mode == DespawnMode.PROBABILITY:
            if self._type_rngs is not None:
                # Draw despawn decision from each task's type RNG so team k's
                # despawn sequence matches T=1 run k exactly.
                keep = [
                    i for i in range(len(positions))
                    if active_set is not None and types[i] not in active_set
                    or self._type_rngs[types[i]].random() >= self.stoch.despawn_prob
                ]
            else:
                keep = [
                    i for i in range(len(positions))
                    if active_set is not None and types[i] not in active_set
                    or rng.random() >= self.stoch.despawn_prob
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
            if active_set is not None and tau not in active_set:
                continue
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

            _trng = self._type_rngs[tau] if self._type_rngs is not None else rng
            _trng.shuffle(empty_cells)
            for cell in empty_cells:
                if n_tau >= self.cfg.max_tasks_per_type:
                    break
                if _trng.random() < self.stoch.spawn_prob:
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
