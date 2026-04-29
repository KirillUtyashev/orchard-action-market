"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

import random as _random

from orchard.enums import DespawnMode, PickMode, SpawnZoneMode, TaskSpawnMode
from orchard.env.base import BaseEnv
from orchard.seed import rng
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks


def edge_zone_positions(
    height: int, width: int, size: int, n_zones: int
) -> tuple[tuple[int, int], ...]:
    """Compute n_zones spawn-zone top-left corners spread evenly around the grid perimeter.

    Valid corner (r, c): a size×size zone fits, so r ∈ [0, height-size], c ∈ [0, width-size].
    Corners are traced clockwise and n_zones evenly-spaced points are selected.

    Examples (9×9 grid, size=3 → max_r=max_c=6):
      n=1 → (0, 0)
      n=2 → (0, 0), (6, 6)
      n=4 → (0,0), (0,6), (6,6), (6,0)
      n=8 → corners + edge midpoints
    """
    max_r = height - size
    max_c = width - size
    perimeter: list[tuple[int, int]] = []
    for c in range(0, max_c + 1):
        perimeter.append((0, c))
    for r in range(1, max_r + 1):
        perimeter.append((r, max_c))
    for c in range(max_c - 1, -1, -1):
        perimeter.append((max_r, c))
    for r in range(max_r - 1, 0, -1):
        perimeter.append((r, 0))
    n = len(perimeter)
    return tuple(perimeter[round(i * n / n_zones) % n] for i in range(n_zones))


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
        if cfg.stochastic.per_type_seeds is not None:
            assert len(cfg.stochastic.per_type_seeds) == cfg.n_task_types, (
                f"per_type_seeds must have one entry per task type "
                f"(got {len(cfg.stochastic.per_type_seeds)}, expected {cfg.n_task_types})"
            )
            self._type_rngs: list[_random.Random] | None = [
                _random.Random(s) for s in cfg.stochastic.per_type_seeds
            ]
        else:
            self._type_rngs = None

        if cfg.stochastic.spawn_area_size is not None:
            size = cfg.stochastic.spawn_area_size
            assert size <= cfg.height and size <= cfg.width, (
                f"spawn_area_size={size} exceeds grid ({cfg.height}x{cfg.width})"
            )
            self._valid_corners: list[tuple[int, int]] | None = [
                (r, c)
                for r in range(cfg.height - size + 1)
                for c in range(cfg.width - size + 1)
            ]
            self._spawn_area_corners: list[tuple[int, int]] | None = [(0, 0)] * cfg.n_task_types
            self._spawn_area_cells: list[list[Grid]] | None = [[] for _ in range(cfg.n_task_types)]

            if cfg.stochastic.spawn_zone_mode == SpawnZoneMode.EDGE_SWITCH:
                self._place_zones_on_edge()
            else:
                for tau in range(cfg.n_task_types):
                    _rng = self._type_rngs[tau] if self._type_rngs is not None else rng
                    r0, c0 = _rng.choice(self._valid_corners)
                    self._spawn_area_corners[tau] = (r0, c0)
                    self._spawn_area_cells[tau] = self._cells_from_corner(r0, c0)
        else:
            self._valid_corners = None
            self._spawn_area_corners = None
            self._spawn_area_cells = None

        self._rounds_elapsed: int = 0
        self._eval_mode: bool = False
        self._eval_rounds_elapsed: int = 0
        self._saved_spawn_area_corners: list[tuple[int, int]] | None = None
        self._saved_spawn_area_cells: list[list[Grid]] | None = None
        self._saved_rng_state = None
        self._saved_type_rng_states: list | None = None

    def _cells_from_corner(self, r0: int, c0: int) -> list[Grid]:
        size = self.stoch.spawn_area_size
        return [Grid(r0 + dr, c0 + dc) for dr in range(size) for dc in range(size)]

    def _spawn_cells_for_type(self, tau: int) -> list[Grid]:
        if self._spawn_area_cells is None:
            return self._all_cells
        return self._spawn_area_cells[tau]

    def _place_zones_on_edge(self) -> None:
        """Place all spawn zones on the grid border.

        Zones are evenly spaced around the perimeter but rotated to a random starting
        offset, then randomly assigned to task types. The random rotation means a single
        zone lands at a genuinely random border position, not always (0,0).
        """
        size = self.stoch.spawn_area_size
        n_types = self.cfg.n_task_types
        H, W = self.cfg.height, self.cfg.width
        max_r, max_c = H - size, W - size

        perimeter: list[tuple[int, int]] = []
        for c in range(0, max_c + 1):
            perimeter.append((0, c))
        for r in range(1, max_r + 1):
            perimeter.append((r, max_c))
        for c in range(max_c - 1, -1, -1):
            perimeter.append((max_r, c))
        for r in range(max_r - 1, 0, -1):
            perimeter.append((r, 0))

        n = len(perimeter)
        offset = rng.randrange(n)
        positions = [perimeter[(round(i * n / n_types) + offset) % n] for i in range(n_types)]
        rng.shuffle(positions)

        for tau, (r0, c0) in enumerate(positions):
            self._spawn_area_corners[tau] = (r0, c0)
            self._spawn_area_cells[tau] = self._cells_from_corner(r0, c0)

    def _flip_spawn_areas(self) -> None:
        """Flip each spawn zone to its antipodal position: (r,c) → (H-S-r, W-S-c)."""
        size = self.stoch.spawn_area_size
        H, W = self.cfg.height, self.cfg.width
        for tau in range(self.cfg.n_task_types):
            r0, c0 = self._spawn_area_corners[tau]
            r1, c1 = H - size - r0, W - size - c0
            self._spawn_area_corners[tau] = (r1, c1)
            self._spawn_area_cells[tau] = self._cells_from_corner(r1, c1)

    def set_eval_mode(
        self,
        eval_mode: bool,
        seed: int | None = None,
        fixed_spawn_zones: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        if eval_mode:
            self._saved_spawn_area_corners = (
                list(self._spawn_area_corners)
                if self._spawn_area_corners is not None else None
            )
            self._saved_spawn_area_cells = (
                [list(cells) for cells in self._spawn_area_cells]
                if self._spawn_area_cells is not None else None
            )
            self._saved_rng_state = rng.getstate()
            self._saved_type_rng_states = (
                [r.getstate() for r in self._type_rngs]
                if self._type_rngs is not None else None
            )
            self._eval_rounds_elapsed = 0
            if seed is not None:
                rng.seed(seed)
                if self._type_rngs is not None:
                    for i, r in enumerate(self._type_rngs):
                        r.seed(seed + i + 1)
            if self.stoch.eval_spawn_zone_mode == SpawnZoneMode.EDGE_SWITCH and self._spawn_area_cells is not None:
                self._place_zones_on_edge()
            if fixed_spawn_zones is not None and self._spawn_area_cells is not None:
                for tau, (r0, c0) in enumerate(fixed_spawn_zones):
                    self._spawn_area_corners[tau] = (r0, c0)
                    self._spawn_area_cells[tau] = self._cells_from_corner(r0, c0)
        else:
            self._spawn_area_corners = self._saved_spawn_area_corners
            self._saved_spawn_area_corners = None
            self._spawn_area_cells = self._saved_spawn_area_cells
            self._saved_spawn_area_cells = None
            rng.setstate(self._saved_rng_state)
            if self._type_rngs is not None and self._saved_type_rng_states is not None:
                for r, state in zip(self._type_rngs, self._saved_type_rng_states):
                    r.setstate(state)
            self._saved_rng_state = None
            self._saved_type_rng_states = None
        self._eval_mode = eval_mode

    def _relocate_spawn_areas(self) -> None:
        for tau in range(self.cfg.n_task_types):
            _rng = self._type_rngs[tau] if self._type_rngs is not None else rng
            r0, c0 = _rng.choice(self._valid_corners)
            self._spawn_area_corners[tau] = (r0, c0)
            self._spawn_area_cells[tau] = self._cells_from_corner(r0, c0)

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
                spawn_cells = self._spawn_cells_for_type(tau)
                if spawn_cells is self._all_cells:
                    chosen = trng.sample(cells, n_team + n_tasks)
                    for idx, agent_i in enumerate(team_agent_indices):
                        agent_positions_by_idx[agent_i] = chosen[idx]
                    for cell in chosen[n_team:]:
                        all_task_positions.append(cell)
                        all_task_types.append(tau)
                else:
                    chosen_agents = trng.sample(cells, n_team)
                    for idx, agent_i in enumerate(team_agent_indices):
                        agent_positions_by_idx[agent_i] = chosen_agents[idx]
                    if self.stoch.spawn_on_agent_cells:
                        available = list(spawn_cells)
                    else:
                        agent_set_tau = set(chosen_agents)
                        available = [c for c in spawn_cells if c not in agent_set_tau]
                    for cell in trng.sample(available, min(n_tasks, len(available))):
                        all_task_positions.append(cell)
                        all_task_types.append(tau)
            agent_positions = tuple(
                agent_positions_by_idx[i] for i in range(self.cfg.n_agents)
            )
        else:
            # Place agents first, then tasks per type using
            # PER_TYPE_UNIQUE (types may share cells, matching runtime behavior).
            agent_positions = tuple(rng.sample(cells, self.cfg.n_agents))
            agent_set = set(agent_positions)
            block_agents_init = not self.stoch.spawn_on_agent_cells
            all_task_positions = []
            all_task_types = []
            for tau in range(self.cfg.n_task_types):
                count = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
                cells_with_tau = {p for p, t in zip(all_task_positions, all_task_types) if t == tau}
                available = [
                    c for c in self._spawn_cells_for_type(tau)
                    if (not block_agents_init or c not in agent_set) and c not in cells_with_tau
                ]
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

        If spawn_at_round_end is True, only fires after the last agent in each
        round (actor == n_agents - 1). Per-type RNGs (if set) guarantee each
        type's sequence matches a standalone T=1 run regardless of when in the
        round this fires.
        """
        if state.actor == self.cfg.n_agents - 1:
            if self._eval_mode:
                self._eval_rounds_elapsed += 1
                rounds = self._eval_rounds_elapsed
                zone_mode = self.stoch.eval_spawn_zone_mode
                zone_interval = self.stoch.eval_spawn_zone_interval
                flip_interval = self.stoch.eval_spawn_flip_interval
            else:
                self._rounds_elapsed += 1
                rounds = self._rounds_elapsed
                zone_mode = self.stoch.spawn_zone_mode
                zone_interval = self.stoch.spawn_zone_interval
                flip_interval = self.stoch.spawn_flip_interval

            if self._spawn_area_cells is not None:
                if zone_mode == SpawnZoneMode.EDGE_SWITCH and zone_interval > 0 and rounds % zone_interval == 0:
                    self._place_zones_on_edge()
                elif zone_mode == SpawnZoneMode.RANDOM and zone_interval > 0 and rounds % zone_interval == 0:
                    self._relocate_spawn_areas()
                elif flip_interval > 0 and rounds % flip_interval == 0:
                    self._flip_spawn_areas()

        if self.stoch.spawn_at_round_end and state.actor != self.cfg.n_agents - 1:
            return state
        return self._spawn_and_despawn_multi(state)

    def _spawn_and_despawn_multi(self, state: State) -> State:
        """Spawn/despawn all task types."""
        positions = list(state.task_positions)
        assert state.task_types is not None, "task_types must be set"
        types = list(state.task_types)

        # --- Despawn phase ---
        if self.stoch.despawn_mode == DespawnMode.PROBABILITY:
            if self._type_rngs is not None:
                keep = [
                    i for i in range(len(positions))
                    if self._type_rngs[types[i]].random() >= self.stoch.despawn_prob
                ]
            else:
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

            candidate_cells = self._spawn_cells_for_type(tau)
            if tsm == TaskSpawnMode.GLOBAL_UNIQUE:
                task_set = set(positions)
                empty_cells = [
                    c for c in candidate_cells
                    if c not in task_set and (not block_agents or c not in agent_set)
                ]
            else:
                empty_cells = [
                    c for c in candidate_cells
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
