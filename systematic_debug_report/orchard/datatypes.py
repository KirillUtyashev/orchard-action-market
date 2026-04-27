"""Core types: Grid, State, Transition, EncoderOutput, and config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from orchard.enums import (
    Action,
    Activation,
    AlgorithmName,
    DespawnMode,
    EncoderType,
    Heuristic,
    LearningType,
    PickMode,
    Schedule,
    StoppingCondition,
    TaskSpawnMode,
    WeightInit,
)


# ---------------------------------------------------------------------------
# Grid coordinate
# ---------------------------------------------------------------------------
class Grid(NamedTuple):
    row: int
    col: int


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class State:
    agent_positions: tuple[Grid, ...]       # length = n_agents
    task_positions: tuple[Grid, ...]        # sorted by (row, col)
    actor: int                              # index of agent whose turn it is
    task_types: tuple[int, ...] | None = None  # parallel to task_positions
    # task_types[k] = type τ ∈ {0, ..., T-1} of the task at task_positions[k]
    # None when n_task_types == 1 (legacy mode)
    pick_phase: bool = False  # True = agent is on task, pick not yet resolved

    def is_agent_on_task(self, agent_idx: int, my_types: frozenset[int] | None = None) -> bool:
        """Check if agent is on a task cell.

        my_types: if provided, only counts tasks whose type is in this set.
        None means any task type (legacy behaviour).
        """
        pos = self.agent_positions[agent_idx]
        if pos not in self.task_positions:
            return False
        if my_types is None:
            return True
        for tp, tt in zip(self.task_positions, self.task_types or ()):
            if tp == pos and tt in my_types:
                return True
        return False

    def task_type_at(self, pos: Grid) -> int | None:
        """Type of task at pos. None if no task there."""
        for i, tp in enumerate(self.task_positions):
            if tp == pos:
                return self.task_types[i] if self.task_types is not None else 0
        return None

    def tasks_at(self, pos: Grid) -> list[tuple[int, int]]:
        """List of (task_index, type) at pos. For choice pick with multiple types."""
        result = []
        for i, tp in enumerate(self.task_positions):
            if tp == pos:
                t = self.task_types[i] if self.task_types is not None else 0
                result.append((i, t)) # t is type, i is index
        return result

    @property
    def n_agents(self) -> int:
        return len(self.agent_positions)

    def with_pick_phase(self) -> State:
        """Copy of this state with pick_phase=True (agent on task, pick pending)."""
        return State(
            agent_positions=self.agent_positions,
            task_positions=self.task_positions,
            actor=self.actor,
            task_types=self.task_types,
            pick_phase=True,
        )


# ---------------------------------------------------------------------------
# Parallel array helper
# ---------------------------------------------------------------------------
def sort_tasks(
    positions: tuple[Grid, ...] | list[Grid],
    types: tuple[int, ...] | list[int] | None = None,
) -> tuple[tuple[Grid, ...], tuple[int, ...] | None]:
    """Sort tasks by (row, col), reordering types array consistently."""
    if not positions:
        return (), types if types is None else ()
    order = sorted(range(len(positions)), key=lambda i: (positions[i].row, positions[i].col))
    sorted_pos = tuple(positions[i] for i in order)
    sorted_types = tuple(types[i] for i in order) if types is not None else None
    return sorted_pos, sorted_types


# ---------------------------------------------------------------------------
# Encoder output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderOutput:
    """Output of an encoder. Supports batching: scalar/grid may have batch dim."""
    scalar: torch.Tensor | None = None     # shape: (D,) or (B, D)
    grid: torch.Tensor | None = None       # shape: (C, H, W) or (B, C, H, W)


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transition:
    s_t: State
    action: Action
    s_t_after: State
    s_t_next: State
    rewards: tuple[float, ...]  # r_{t+1} for each agent; length = n_agents
    discount: float             # γ_{t+1} for this transition


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StochasticConfig:
    spawn_prob: float           # per empty cell per type per turn
    despawn_mode: DespawnMode
    despawn_prob: float         # only meaningful if despawn_mode == PROBABILITY
    task_spawn_mode: TaskSpawnMode | None = None  # None = auto-select based on pick_mode
    spawn_on_agent_cells: bool = False  # if True, agent positions don't block task spawning (removes inter-team spawn coupling)
    spawn_at_round_end: bool = False    # if True, spawn/despawn only fires after the last agent in a round acts (keeps dec sub-problem identical across T)
    per_type_seeds: tuple[int, ...] | None = None  # one seed per task type; enables per-team RNG isolation for exact T=1 vs T=M equivalence testing


@dataclass(frozen=True)
class EnvConfig:
    height: int
    width: int
    n_agents: int
    n_tasks: int                    # initial tasks per type
    gamma: float
    r_picker: float                 # reward to picker on correct pick
    n_task_types: int = 1
    r_low: float = 0.0             # reward for picking wrong task type
    task_assignments: tuple[tuple[int, ...], ...] | None = None
    pick_mode: PickMode = PickMode.FORCED
    max_tasks_per_type: int = 3
    stochastic: StochasticConfig | None = None
    allow_cross_type_picks: bool = True


@dataclass(frozen=True)
class ScheduleConfig:
    start: float
    end: float
    schedule: Schedule = Schedule.NONE
    step_size: int = 0
    step_factor: float = 1.0
    step_start: int = 0


@dataclass(frozen=True)
class StoppingConfig:
    condition: StoppingCondition = StoppingCondition.NONE
    patience_steps: int = 10000
    improvement_threshold: float = 0.01
    min_steps_before_stop: int = 0


@dataclass(frozen=True)
class AlgorithmConfig:
    name: AlgorithmName = AlgorithmName.VALUE


@dataclass(frozen=True)
class FollowingRatesConfig:
    enabled: bool = False
    budget: float = 0.0
    teammate_budget: float | None = None
    non_teammate_budget: float | None = None
    rho: float = 0.0
    reallocation_freq: int = 1
    solver: str = "closed_form"
    fixed: bool = False


@dataclass(frozen=True)
class InfluencerConfig:
    enabled: bool = False
    budget: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int
    seed: int
    lr: ScheduleConfig
    epsilon: ScheduleConfig
    actor_lr: ScheduleConfig | None = None
    freeze_critic: bool = False
    algorithm: AlgorithmConfig = AlgorithmConfig()
    following_rates: FollowingRatesConfig = FollowingRatesConfig()
    influencer: InfluencerConfig = InfluencerConfig()
    learning_type: LearningType = LearningType.DECENTRALIZED
    use_gpu: bool = True
    td_lambda: float = 0.0
    comm_only_teammates: bool = False
    heuristic: Heuristic = Heuristic.NEAREST_TASK
    stopping: StoppingConfig = StoppingConfig()
    warmup_steps: int = 0
    train_only_teammates: bool = False
    simulate_stranger_gap: int = 0
    # simulate_stranger_gap: for T=1 ≡ T=M verification with new dec gamma accumulation.
    # Set to n_total_agents - n_own_team_agents so T=1 artificially accumulates
    # the same gamma that would build up from stranger move-steps in T=M.
    # Only meaningful when train_only_teammates=True. See ValueTrainerBase.


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderType
    mlp_dims: tuple[int, ...]
    conv_specs: tuple[tuple[int, int], ...] | None = None
    activation: Activation = Activation.LEAKY_RELU
    weight_init: WeightInit = WeightInit.ZERO_BIAS


@dataclass(frozen=True)
class EvalConfig:
    eval_steps: int = 1000
    n_test_states: int = 50
    checkpoint_freq: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "runs/"
    main_csv_freq: int = 10000
    detail_csv_freq: int = 50000
    timing_csv_freq: int = 0
    alpha_state_log_freq: int = 0
    env_trace: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig
    model: ModelConfig
    actor_model: ModelConfig | None
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig


# ---------------------------------------------------------------------------
# Task assignment generation
# ---------------------------------------------------------------------------
def compute_task_assignments(
    n_agents: int, n_task_types: int, rho: float
) -> tuple[tuple[int, ...], ...]:
    """Generate G_i for each agent from rho.

    |G_i| = max(1, round(rho * T)). Must be integer.
    Assignments are cyclic: agent i gets types {i, i+1, ..., i+|G_i|-1} mod T.
    Every type must be covered by at least one agent.
    """
    g_size = max(1, round(rho * n_task_types))
    assignments = []
    for i in range(n_agents):
        start = (i * n_task_types) // n_agents
        agent_types = tuple((start + k) % n_task_types for k in range(g_size))
        assignments.append(agent_types)

    covered = set()
    for g in assignments:
        covered.update(g)
    if covered != set(range(n_task_types)):
        missing = set(range(n_task_types)) - covered
        raise ValueError(
            f"Task assignments from rho={rho} do not cover all types. "
            f"Missing: {missing}"
        )

    return tuple(assignments)
