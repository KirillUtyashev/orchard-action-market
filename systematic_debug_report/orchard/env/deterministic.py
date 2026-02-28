"""Deterministic environment: row-major init, first-empty-cell spawn, no despawn."""

from __future__ import annotations

from orchard.env.base import BaseEnv
from orchard.datatypes import EnvConfig, Grid, State


class DeterministicEnv(BaseEnv):
    """Deterministic spawn: first empty cell in row-major order."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)

    def init_state(self) -> State:
        """Agents fill first cells row-major, apples fill next."""
        cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
        ]
        total_needed = self.cfg.n_agents + self.cfg.n_apples
        assert total_needed <= len(cells), (
            f"Need {total_needed} cells but grid is {self.cfg.height}x{self.cfg.width} "
            f"= {len(cells)} cells"
        )
        agent_positions = tuple(cells[: self.cfg.n_agents])
        apple_positions = tuple(sorted(cells[self.cfg.n_agents : total_needed]))
        apple_ids = tuple(range(len(apple_positions)))
        return State(
            agent_positions=agent_positions,
            apple_positions=apple_positions,
            actor=0,
            apple_ids=apple_ids,
        )

    def spawn_and_despawn(self, state: State) -> State:
        """If apple count < n_apples, spawn at first empty cell row-major. No despawn."""
        if len(state.apple_positions) >= self.cfg.n_apples:
            return state

        occupied = set(state.agent_positions) | set(state.apple_positions)
        for r in range(self.cfg.height):
            for c in range(self.cfg.width):
                pos = Grid(r, c)
                if pos not in occupied:
                    new_apples = tuple(sorted(state.apple_positions + (pos,)))
                    new_ids: tuple[int, ...] | None = None
                    if state.apple_ids is not None:
                        used = set(state.apple_ids)
                        new_id = min(set(range(self.cfg.max_apples)) - used)
                        pos_id = dict(zip(state.apple_positions, state.apple_ids))
                        pos_id[pos] = new_id
                        new_ids = tuple(pos_id[p] for p in new_apples)
                    return State(
                        agent_positions=state.agent_positions,
                        apple_positions=new_apples,
                        actor=state.actor,
                        apple_ages=state.apple_ages,
                        apple_ids=new_ids,
                    )

        # No empty cell found (shouldn't happen in normal configs)
        return state
