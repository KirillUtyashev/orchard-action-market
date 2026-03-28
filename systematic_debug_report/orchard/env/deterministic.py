"""Deterministic environment: row-major init, first-empty-cell spawn, no despawn."""

from __future__ import annotations

from orchard.env.base import BaseEnv
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks


class DeterministicEnv(BaseEnv):
    """Deterministic spawn: first empty cell in row-major order."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)

    def init_state(self) -> State:
        """Agents fill first cells row-major, tasks fill next."""
        cells = [
            Grid(r, c)
            for r in range(self.cfg.height)
            for c in range(self.cfg.width)
        ]

        if self.cfg.n_task_types == 1:
            # --- Legacy path ---
            total_needed = self.cfg.n_agents + self.cfg.n_tasks
            assert total_needed <= len(cells), (
                f"Need {total_needed} cells but grid is "
                f"{self.cfg.height}x{self.cfg.width} = {len(cells)} cells"
            )
            agent_positions = tuple(cells[:self.cfg.n_agents])
            task_positions = tuple(sorted(
                cells[self.cfg.n_agents:total_needed]
            ))
            return State(
                agent_positions=agent_positions,
                task_positions=task_positions,
                actor=0,
                task_types=None,
            )

        # --- Task specialization path ---
        agent_positions = tuple(cells[:self.cfg.n_agents])
        occupied = set(agent_positions)

        all_task_positions: list[Grid] = []
        all_task_types: list[int] = []

        for tau in range(self.cfg.n_task_types):
            count = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
            spawned = 0
            for cell in cells:
                if spawned >= count:
                    break
                if cell not in occupied:
                    all_task_positions.append(cell)
                    all_task_types.append(tau)
                    occupied.add(cell)
                    spawned += 1

        tp, tt = sort_tasks(all_task_positions, all_task_types)
        return State(
            agent_positions=agent_positions,
            task_positions=tp,
            actor=0,
            task_types=tt,
        )

    def spawn_and_despawn(self, state: State) -> State:
        """If task count < n_tasks, spawn at first empty cell row-major. No despawn."""
        if self.cfg.n_task_types == 1:
            return self._spawn_legacy(state)
        return self._spawn_multi(state)

    def _spawn_legacy(self, state: State) -> State:
        """Legacy spawn for n_task_types == 1."""
        if len(state.task_positions) >= self.cfg.n_tasks:
            return state

        occupied = set(state.agent_positions) | set(state.task_positions)
        for r in range(self.cfg.height):
            for c in range(self.cfg.width):
                pos = Grid(r, c)
                if pos not in occupied:
                    new_tasks = tuple(sorted(state.task_positions + (pos,)))
                    return State(
                        agent_positions=state.agent_positions,
                        task_positions=new_tasks,
                        actor=state.actor,
                        task_types=None,
                    )
        return state

    def _spawn_multi(self, state: State) -> State:
        """Spawn for n_task_types > 1: one task per under-capacity type."""
        positions = list(state.task_positions)
        types = list(state.task_types)

        for tau in range(self.cfg.n_task_types):
            n_tau = sum(1 for t in types if t == tau)
            if n_tau >= self.cfg.max_tasks_per_type:
                continue

            occupied = set(state.agent_positions) | set(positions)
            for r in range(self.cfg.height):
                for c in range(self.cfg.width):
                    pos = Grid(r, c)
                    if pos not in occupied:
                        positions.append(pos)
                        types.append(tau)
                        break
                else:
                    continue
                break

        tp, tt = sort_tasks(positions, types)
        return State(
            agent_positions=state.agent_positions,
            task_positions=tp,
            actor=state.actor,
            task_types=tt,
        )
