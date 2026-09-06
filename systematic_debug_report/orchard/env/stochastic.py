"""Stochastic environment: random init, per-cell spawn, configurable despawn."""

from __future__ import annotations

import numpy as np

from orchard.enums import DespawnMode
from orchard.env.base import BaseEnv
from orchard.seed import rng
from orchard.datatypes import EnvConfig, Grid, State, sort_tasks


def generate_rewards_all_to_all(
    num_agents: int,
    sigma_a: float,
    sigma_b: float,
    mean_team_reward: float = 1.0,
) -> np.ndarray:
    """Return a deterministic (agents, task types) reward matrix.

    The two orthogonal Fourier frequencies control within-task agent variation
    and across-task team-total variation independently. Cyclic shifts make
    every agent row and task column balanced. Individual rewards are not
    constrained to be non-negative; increasing mean_team_reward shifts all
    entries upward without changing any of the controlled variances.
    """
    N = num_agents
    if N < 5:
        raise ValueError("all-to-all Fourier reward generation requires num_agents >= 5")
    if sigma_a < 0 or sigma_b < 0:
        raise ValueError("sigma_a and sigma_b must be non-negative")

    indices = np.arange(N, dtype=np.float64)
    deviation_pattern = (
        np.sqrt(2.0) * sigma_a * np.cos(2.0 * np.pi * indices / N)
    )
    task_totals = (
        mean_team_reward
        + np.sqrt(2.0) * sigma_b * np.cos(4.0 * np.pi * indices / N)
    )
    task_baselines = task_totals / N

    agent_ids = np.arange(N)[:, None]
    task_ids = np.arange(N)[None, :]
    deviation_indices = (agent_ids - task_ids) % N
    rewards = deviation_pattern[deviation_indices] + task_baselines[None, :]

    assert np.allclose(rewards.std(axis=0, ddof=0), sigma_a)
    assert np.isclose(rewards.sum(axis=0).std(ddof=0), sigma_b)
    assert np.allclose(rewards.mean(axis=1), mean_team_reward / N)
    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(sigma_a**2 + sigma_b**2 / N**2),
    )
    return rewards


def generate_rewards_circulant(
    num_agents: int,
    relatedness_width: int,
    sigma_a: float,
    sigma_b: float,
    mean_team_reward: float = 1.0,
) -> np.ndarray:
    """Return a deterministic Fourier reward matrix for any interest width.

    The returned matrix has shape (agents, task types). Every task uses the
    same Fourier task-total vector as generate_rewards_all_to_all, so changing
    relatedness_width does not change the task-level reward landscape. For
    partial interest, rewards are zero outside the task's circular caring
    group. Within that group, the first Fourier mode is centered and rescaled
    to have standard deviation sigma_a.

    The full-interest case delegates to generate_rewards_all_to_all to preserve
    its matrix exactly.
    """
    N = num_agents
    if N < 5:
        raise ValueError(
            "circulant Fourier reward generation requires num_agents >= 5"
        )
    if relatedness_width < 0:
        raise ValueError("relatedness_width must be non-negative")
    if sigma_a < 0 or sigma_b < 0:
        raise ValueError("sigma_a and sigma_b must be non-negative")

    g = min(2 * relatedness_width + 1, N)
    if g == N:
        return generate_rewards_all_to_all(
            N,
            sigma_a,
            sigma_b,
            mean_team_reward=mean_team_reward,
        )

    indices = np.arange(N, dtype=np.float64)
    task_totals = (
        mean_team_reward
        + np.sqrt(2.0) * sigma_b * np.cos(4.0 * np.pi * indices / N)
    )
    task_baselines = task_totals / g

    relative_offsets = np.arange(N)
    circular_distances = np.minimum(relative_offsets, N - relative_offsets)
    caring_pattern = circular_distances <= relatedness_width
    assert caring_pattern.sum() == g

    deviation_pattern = np.zeros(N, dtype=np.float64)
    if sigma_a > 0 and g >= 2:
        caring_deviations = np.cos(
            2.0 * np.pi * relative_offsets[caring_pattern] / N
        )
        caring_deviations -= caring_deviations.mean()
        caring_deviations *= sigma_a / caring_deviations.std(ddof=0)
        deviation_pattern[caring_pattern] = caring_deviations

    agent_ids = np.arange(N)[:, None]
    task_ids = np.arange(N)[None, :]
    deviation_indices = (agent_ids - task_ids) % N
    caring = caring_pattern[deviation_indices]
    rewards = np.where(
        caring,
        deviation_pattern[deviation_indices] + task_baselines[None, :],
        0.0,
    )

    assert np.allclose(rewards.sum(axis=0), task_totals)
    assert np.isclose(rewards.sum(axis=0).std(ddof=0), sigma_b)
    if g >= 2:
        caring_stds = np.array(
            [
                rewards[:, task][caring[:, task]].std(ddof=0)
                for task in range(N)
            ]
        )
        assert np.allclose(caring_stds, sigma_a)
    return rewards


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
            cfg.stochastic.reward_generation,
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
        reward_generation: str = "independent",
    ) -> np.ndarray:
        """Generate the task reward vectors r'^(k). Returns (n_task_types, N) array.

        Implements the spec directly: r'^(k)_j = I[j ∈ C^(k)] · (b^(k) + a^(k)_j),
        where C^(k) = {j : circular_dist(k, j) ≤ relatedness_width} is the set of
        agents that care about task k (size g = min(2*relatedness_width+1, N)),
        keyed on the TASK k (not the actor). With T=N, task k lives on the agent
        ring, so this circular distance is the same one used for relatedness.

          b^(k) — per-task baseline: mean 1/g, std sigma_b/g over tasks.
          a^(k) — per-agent deviation: mean 0, std sigma_a over j ∈ C^(k) (the
                  caring agents only); entries for j ∉ C^(k) are exactly 0.

        Baking the C^(k) mask into r' means the pick reward needs no separate
        relatedness factor: r_j(actor, k) = proficiency[actor, k] · r'^(k)_j.
        """
        g = min(2 * relatedness_width + 1, N)
        if reward_generation == "circulant_all_to_all":
            if n_task_types != N:
                raise ValueError(
                    "circulant_all_to_all reward generation requires "
                    "n_task_types == num_agents"
                )
            # The environment stores (task types, agents), while the public
            # generator intentionally exposes (agents, task types).
            return generate_rewards_circulant(
                N,
                relatedness_width,
                sigma_a,
                sigma_b,
            ).T

        rng_np = np.random.default_rng(seed)

        # Baseline b: draw n_task_types samples, standardize to std=sigma_b/g.
        # With a single task type there is no across-task spread to impose, so b
        # is just the baseline 1/g (a one-element std is identically 0, and the
        # reject loop below could never satisfy std>0).
        if sigma_b > 0 and n_task_types >= 2:
            while True:
                b_raw = rng_np.standard_normal(n_task_types)
                b_std = b_raw.std()
                if b_std > 1e-10:
                    b = (b_raw - b_raw.mean()) / b_std * (sigma_b / g) + 1.0 / g
                    break
        else:
            b = np.full(n_task_types, 1.0 / g, dtype=np.float64)

        rewards = np.zeros((n_task_types, N), dtype=np.float32)
        for kappa in range(n_task_types):
            b_kappa = float(b[kappa])
            # C^(kappa): agents that care about task kappa (circular distance on the ring).
            caring = [j for j in range(N) if min(abs(kappa - j), N - abs(kappa - j)) <= relatedness_width]

            # a^(kappa): zero-mean, std sigma_a over the caring agents only.
            # A single carer (g=1, e.g. relatedness_width=0) has no within-set
            # spread to impose — a=0 is the only consistent value (and a
            # one-element std is identically 0, so the reject loop can't proceed).
            if sigma_a > 0 and len(caring) >= 2:
                while True:
                    a_raw = rng_np.standard_normal(len(caring))
                    a_std = a_raw.std()
                    if a_std > 1e-10:
                        a = (a_raw - a_raw.mean()) / a_std * sigma_a
                        break
            else:
                a = np.zeros(len(caring))

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
