"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

import numpy as np

from orchard.enums import DespawnMode
from orchard.env.base import BaseEnv, circular_task_distance
from orchard.seed import rng
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks


class StochasticEnv(BaseEnv):
    """Stochastic spawn/despawn with random uniform task placement."""

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg)
        assert cfg.stochastic is not None, "StochasticEnv requires stochastic config"
        self.stoch = cfg.stochastic
        self._all_cells: list[Grid] = [
            Grid(r, c)
            for r in range(cfg.height)
            for c in range(cfg.width)
        ]
        self._rounds_elapsed: int = 0
        self._eval_mode: bool = False
        self._saved_rng_state = None

        # Generate fixed per-category reward vectors from sigma_a, sigma_b and seed.
        # category_rewards[kappa] = r'^(kappa), shape (T, N).
        seed = rng.randint(0, 2**31)
        self.category_rewards: np.ndarray = self._generate_category_rewards(
            seed,
            cfg.n_task_types,
            cfg.n_agents,
            cfg.stochastic.sigma_a,
            cfg.stochastic.sigma_b,
            cfg.relatedness_width,
            tuple(int(x) for x in self.task_centers),
            cfg.stochastic.constant_reward_value,
        )
        self._precompute_pick_rewards()

    @staticmethod
    def _generate_category_rewards(
        seed: int,
        n_task_types: int,
        N: int,
        sigma_a: float,
        sigma_b: float,
        relatedness_width: int,
        task_centers: tuple[int, ...],
        constant_reward_value: float | None = None,
    ) -> np.ndarray:
        """Generate task reward vectors r'^(k), shape (T, N).

        r'^(k)_j is nonzero when task k is inside agent j's interest window in
        task-type space. Agent j's window is centered at task_centers[j]; for
        T=N, task_centers[j] == j and this reduces to the previous behavior.
        """
        rng_np = np.random.default_rng(seed)

        if len(task_centers) != N:
            raise ValueError(f"Expected {N} task centers, got {len(task_centers)}")

        # Standardized per-task baseline noise. It is scaled by 1/g_k for each
        # task, where g_k is the number of agents interested in task k. Tasks
        # with no interested agents remain exactly zero.
        if sigma_b > 0 and n_task_types >= 2:
            while True:
                b_raw = rng_np.standard_normal(n_task_types)
                b_std = b_raw.std()
                if b_std > 1e-10:
                    b_z = (b_raw - b_raw.mean()) / b_std
                    break
        else:
            b_z = np.zeros(n_task_types, dtype=np.float64)

        rewards = np.zeros((n_task_types, N), dtype=np.float32)
        for kappa in range(n_task_types):
            caring = [
                j for j, center in enumerate(task_centers)
                if circular_task_distance(center, kappa, n_task_types) <= relatedness_width
            ]
            g = len(caring)
            if g == 0:
                continue
            if constant_reward_value is not None:
                for j in caring:
                    rewards[kappa, j] = np.float32(constant_reward_value)
                continue

            b_kappa = float(1.0 / g + b_z[kappa] * (sigma_b / g))

            # a^(kappa): zero-mean, std sigma_a over the caring agents only.
            if sigma_a > 0 and g >= 2:
                while True:
                    a_raw = rng_np.standard_normal(g)
                    a_std = a_raw.std()
                    if a_std > 1e-10:
                        a = (a_raw - a_raw.mean()) / a_std * sigma_a
                        break
            else:
                a = np.zeros(g)

            for idx, j in enumerate(caring):
                rewards[kappa, j] = np.float32(b_kappa + a[idx])

        return rewards

    def set_eval_mode(
        self,
        eval_mode: bool,
        seed: int | None = None,
        fixed_spawn_zones: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        if eval_mode:
            self._saved_rng_state = rng.getstate()
            if seed is not None:
                rng.seed(seed)
        else:
            if self._saved_rng_state is not None:
                rng.setstate(self._saved_rng_state)
                self._saved_rng_state = None
        self._eval_mode = eval_mode

    def init_state(self) -> State:
        """Random placement of agents and tasks."""
        cells = self._all_cells

        agent_positions = tuple(rng.sample(cells, self.cfg.n_agents))
        agent_set = set(agent_positions)
        block_agents = not self.stoch.spawn_on_agent_cells

        all_task_positions: list[Grid] = []
        all_task_types: list[int] = []
        for tau in range(self.cfg.n_task_types):
            count = min(self.cfg.n_tasks, self.cfg.max_tasks_per_type)
            cells_with_tau = {p for p, t in zip(all_task_positions, all_task_types) if t == tau}
            available = [
                c for c in cells
                if (not block_agents or c not in agent_set) and c not in cells_with_tau
            ]
            for cell in rng.sample(available, min(count, len(available))):
                all_task_positions.append(cell)
                all_task_types.append(tau)

        tp, tt = sort_tasks(all_task_positions, all_task_types)
        return State(agent_positions=agent_positions, task_positions=tp, actor=0, task_types=tt)

    def spawn_and_despawn(self, state: State) -> State:
        """Despawn then spawn. If spawn_at_round_end, only fires after the last agent acts."""
        if self.stoch.spawn_at_round_end and state.actor != self.cfg.n_agents - 1:
            return state
        if state.actor == self.cfg.n_agents - 1:
            if self._eval_mode:
                pass
            else:
                self._rounds_elapsed += 1
        return self._do_spawn_and_despawn(state)

    def _do_spawn_and_despawn(self, state: State) -> State:
        positions = list(state.task_positions)
        assert state.task_types is not None, "task_types must be set"
        types = list(state.task_types)

        # Despawn
        if self.stoch.despawn_mode == DespawnMode.PROBABILITY:
            keep = [i for i in range(len(positions)) if rng.random() >= self.stoch.despawn_prob]
            positions = [positions[i] for i in keep]
            types = [types[i] for i in keep]

        # Spawn — PER_TYPE_UNIQUE: at most 1 task per type per cell
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
