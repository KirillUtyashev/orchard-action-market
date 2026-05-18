"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

import numpy as np

from orchard.enums import DespawnMode, RewardGeneration
from orchard.env.base import BaseEnv
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

        self.category_reward_seed_attempts = 0
        self._initialize_category_rewards()
        self._precompute_pick_rewards()

    def _initialize_category_rewards(self) -> None:
        """Generate fixed per-category reward vectors, retrying seeds if configured."""
        assert self.cfg.stochastic is not None
        stoch = self.cfg.stochastic
        max_attempts = max(1, stoch.reward_seed_max_attempts)

        for attempt in range(1, max_attempts + 1):
            seed = rng.randint(0, 2**31)
            (
                rewards,
                baseline_raw,
                baseline_standardized,
                baselines,
                agent_offsets,
            ) = self._generate_category_reward_components(
                seed,
                self.cfg.n_task_types,
                self.cfg.n_agents,
                stoch.sigma_a,
                stoch.sigma_b,
                stoch.reward_generation,
            )
            if (
                not stoch.require_positive_diagonal_rewards
                or self._diagonal_rewards_are_positive(rewards)
            ):
                self.category_reward_seed = seed
                self.category_reward_seed_attempts = attempt
                self.category_rewards = rewards
                self.category_reward_baseline_raw = baseline_raw
                self.category_reward_baseline_standardized = baseline_standardized
                self.category_reward_baselines = baselines
                self.category_reward_agent_offsets = agent_offsets
                return

        raise ValueError(
            "Failed to generate strictly positive diagonal reward entries "
            f"after {max_attempts} attempts"
        )

    @staticmethod
    def _diagonal_rewards_are_positive(rewards: np.ndarray) -> bool:
        diagonal_len = min(rewards.shape)
        if diagonal_len == 0:
            return True
        diag = rewards[np.arange(diagonal_len), np.arange(diagonal_len)]
        return bool(np.all(diag > 0.0))

    @staticmethod
    def _generate_category_rewards(
        seed: int,
        T: int,
        N: int,
        sigma_a: float,
        sigma_b: float,
        reward_generation: RewardGeneration = RewardGeneration.BASELINE_OFFSET,
    ) -> np.ndarray:
        return StochasticEnv._generate_category_reward_components(
            seed,
            T,
            N,
            sigma_a,
            sigma_b,
            reward_generation,
        )[0]

    @staticmethod
    def _generate_category_reward_components(
        seed: int,
        T: int,
        N: int,
        sigma_a: float,
        sigma_b: float,
        reward_generation: RewardGeneration = RewardGeneration.BASELINE_OFFSET,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate reward components for each category kappa.

        BASELINE_OFFSET: r'^(kappa) = a^(kappa) + b^(kappa) * 1_N where:
          b^(kappa) — scalar baseline: T values standardized to (mean=0, std=sigma_b/N), shifted by 1/N
                      so that team reward std = N * std(b) = sigma_b
          a^(kappa) — agent variance: N values standardized to (mean=0, std=sigma_a), zero-sum

        SAMPLED_MEAN: r'^(kappa) preserves the empirical mean of samples from
        Normal(1, sigma_a^2), while forcing empirical row std to sigma_a.
        sigma_b is intentionally unused in this mode.
        """
        rng_np = np.random.default_rng(seed)

        if reward_generation == RewardGeneration.SAMPLED_MEAN:
            return StochasticEnv._generate_sampled_mean_reward_components(
                rng_np, T, N, sigma_a
            )

        # Baseline b: draw T samples, standardize to std=sigma_b/N so team reward std=sigma_b
        if sigma_b > 0:
            while True:
                b_raw = rng_np.standard_normal(T)
                b_std = b_raw.std()
                if b_std > 1e-10:
                    b_standardized = (b_raw - b_raw.mean()) / b_std
                    b = b_standardized * (sigma_b / N) + 1.0 / N
                    break
        else:
            b_raw = np.zeros(T, dtype=np.float64)
            b_standardized = np.zeros(T, dtype=np.float64)
            b = np.full(T, 1.0 / N, dtype=np.float64)

        # Agent variance a: draw N samples per category, standardize
        rewards = np.zeros((T, N), dtype=np.float32)
        agent_offsets = np.zeros((T, N), dtype=np.float32)
        for kappa in range(T):
            b_kappa = float(b[kappa])
            if sigma_a > 0:
                while True:
                    a_raw = rng_np.standard_normal(N)
                    a_std = a_raw.std()
                    if a_std > 1e-10:
                        a = (a_raw - a_raw.mean()) / a_std * sigma_a
                        break
            else:
                a = np.zeros(N)
            agent_offsets[kappa] = a.astype(np.float32)
            rewards[kappa] = (a + b_kappa).astype(np.float32)

        return (
            rewards,
            b_raw.astype(np.float32),
            b_standardized.astype(np.float32),
            b.astype(np.float32),
            agent_offsets,
        )

    @staticmethod
    def _generate_sampled_mean_reward_components(
        rng_np: np.random.Generator,
        T: int,
        N: int,
        sigma_a: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate reward rows with sampled mean near 1 and exact std sigma_a."""
        if sigma_a > 0 and N < 2:
            raise ValueError("sampled_mean reward generation requires n_agents >= 2 when sigma_a > 0")

        rewards = np.zeros((T, N), dtype=np.float32)
        baselines = np.zeros(T, dtype=np.float32)
        baseline_raw = np.zeros(T, dtype=np.float32)
        baseline_standardized = np.zeros(T, dtype=np.float32)
        agent_offsets = np.zeros((T, N), dtype=np.float32)

        for kappa in range(T):
            if sigma_a > 0:
                while True:
                    z = rng_np.normal(loc=1.0, scale=sigma_a, size=N)
                    z_std = z.std()
                    if z_std > 1e-10:
                        z_mean = float(z.mean())
                        row = z_mean + sigma_a * (z - z_mean) / z_std
                        break
            else:
                z_mean = 1.0
                row = np.ones(N, dtype=np.float64)

            row = row.astype(np.float32)
            rewards[kappa] = row
            baselines[kappa] = float(row.mean())
            baseline_raw[kappa] = z_mean
            agent_offsets[kappa] = row - baselines[kappa]

        return (
            rewards,
            baseline_raw,
            baseline_standardized,
            baselines,
            agent_offsets,
        )

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
