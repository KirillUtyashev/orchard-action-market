"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

from orchard.enums import DespawnMode
from orchard.env.base import BaseEnv
from orchard.seed import rng
from orchard.datatypes import EnvConfig, Grid, State


class StochasticEnv(BaseEnv):
    """Stochastic spawn/despawn with configurable modes."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        assert cfg.stochastic is not None, "StochasticEnv requires stochastic config"
        self.stoch = cfg.stochastic

    def init_state(self) -> State:
        """Random placement, no overlaps between agents and apples."""
        cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
        ]
        total_needed = self.cfg.n_agents + self.cfg.n_apples
        assert total_needed <= len(cells), (
            f"Need {total_needed} cells but grid is {self.cfg.height}x{self.cfg.width}"
        )
        chosen = rng.sample(cells, total_needed)
        agent_positions = tuple(chosen[: self.cfg.n_agents])
        apple_positions = tuple(sorted(chosen[self.cfg.n_agents :]))

        apple_ages: tuple[int, ...] | None = None
        if self.stoch.despawn_mode == DespawnMode.LIFETIME:
            apple_ages = tuple(0 for _ in apple_positions)
        apple_ids = tuple(range(len(apple_positions)))

        return State(
            agent_positions=agent_positions,
            apple_positions=apple_positions,
            actor=0,
            apple_ages=apple_ages,
            apple_ids=apple_ids,
        )

    def spawn_and_despawn(self, state: State) -> State:
        """Despawn phase then spawn phase."""
        # --- Despawn phase ---
        apples = list(state.apple_positions)
        ages: list[int] | None = list(state.apple_ages) if state.apple_ages is not None else None
        ids: list[int] | None = list(state.apple_ids) if state.apple_ids is not None else None

        if self.stoch.despawn_mode == DespawnMode.PROBABILITY:
            keep_indices = [
                i for i in range(len(apples))
                if rng.random() >= self.stoch.despawn_prob
            ]
            apples = [apples[i] for i in keep_indices]
            if ages is not None:
                ages = [ages[i] for i in keep_indices]
            if ids is not None:
                ids = [ids[i] for i in keep_indices]

        elif self.stoch.despawn_mode == DespawnMode.LIFETIME:
            assert ages is not None
            keep_indices = [
                i for i in range(len(apples))
                if ages[i] < self.stoch.apple_lifetime
            ]
            apples = [apples[i] for i in keep_indices]
            ages = [ages[i] for i in keep_indices]
            if ids is not None:
                ids = [ids[i] for i in keep_indices]

        # Age surviving apples
        if ages is not None:
            ages = [a + 1 for a in ages]

        # --- Spawn phase ---
        occupied = set(state.agent_positions) | set(apples)
        empty_cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
            if Grid(r, c) not in occupied
        ]

        # Pre-compute available IDs for spawning
        available_ids: list[int] = []
        if ids is not None:
            available_ids = sorted(set(range(self.cfg.max_apples)) - set(ids))

        for cell in empty_cells:
            if len(apples) >= self.cfg.max_apples:
                break
            if rng.random() < self.stoch.spawn_prob:
                apples.append(cell)
                if ages is not None:
                    ages.append(0)
                if ids is not None and available_ids:
                    ids.append(available_ids.pop(0))

        new_apple_positions = tuple(sorted(apples))
        # Re-sort ages to match sorted apple positions
        new_ages: tuple[int, ...] | None = None
        if ages is not None:
            pos_age = dict(zip(apples, ages))
            new_ages = tuple(pos_age[p] for p in new_apple_positions)
        # Re-sort ids to match sorted apple positions
        new_ids: tuple[int, ...] | None = None
        if ids is not None:
            pos_id = dict(zip(apples, ids))
            new_ids = tuple(pos_id[p] for p in new_apple_positions)

        return State(
            agent_positions=state.agent_positions,
            apple_positions=new_apple_positions,
            actor=state.actor,
            apple_ages=new_ages,
            apple_ids=new_ids,
        )
