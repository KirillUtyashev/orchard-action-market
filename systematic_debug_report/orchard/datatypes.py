"""Core types: Grid, State, Transition, EncoderOutput, and config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from orchard.enums import (
    Action,
    Activation,
    DespawnMode,
    EncoderType,
    EnvType,
    Heuristic,
    LearningType,
    ModelType,
    PickMode,
    Schedule,
    StoppingCondition,
    TDTarget,
    TrainMode,
    TrainMethod,
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
    task_positions: tuple[Grid, ...]        # sorted by (row, col); was apple_positions
    actor: int                              # index of agent whose turn it is
    task_types: tuple[int, ...] | None = None  # parallel to task_positions
    # task_types[k] = type τ ∈ {0, ..., T-1} of the task at task_positions[k]
    # None when n_task_types == 1 (legacy mode)
    phase2_pending: bool = False  # True = agent is on task, phase 2 not yet resolved

    def is_agent_on_task(self, agent_idx: int) -> bool:
        """Check if agent is on any task cell."""
        return self.agent_positions[agent_idx] in self.task_positions

    def task_type_at(self, pos: Grid) -> int | None:
        """Type of task at pos. None if no task there.
        For forced pick (at most 1 task per cell), this is unambiguous.
        For choice pick with multiple types at same cell, returns the first found.
        """
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
                result.append((i, t))
        return result

    @property
    def n_agents(self) -> int:
        return len(self.agent_positions)

    # Backward compat aliases for code not yet updated
    @property
    def apple_positions(self) -> tuple[Grid, ...]:
        return self.task_positions

    def is_agent_on_apple(self, agent_idx: int) -> bool:
        return self.is_agent_on_task(agent_idx)


# ---------------------------------------------------------------------------
# Parallel array helper
# ---------------------------------------------------------------------------
def sort_tasks(
    positions: tuple[Grid, ...] | list[Grid],
    types: tuple[int, ...] | list[int] | None = None,
) -> tuple[tuple[Grid, ...], tuple[int, ...] | None]:
    """Sort tasks by (row, col), reordering types array consistently.
    Returns (sorted_positions, sorted_types).
    """
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
    """Output of an encoder. Note this supports batching across all after states
    for actions, so scalar may really be a list of scalars, and same for grid."""
    scalar: torch.Tensor | None = None     # shape: (D,)
    grid: torch.Tensor | None = None       # shape: (C, H, W)


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


@dataclass(frozen=True)
class NStepTransition:
    s_enc: EncoderOutput
    reward: float
    discount: float


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StochasticConfig:
    spawn_prob: float           # per empty cell per type per turn
    despawn_mode: DespawnMode
    despawn_prob: float         # only meaningful if despawn_mode == PROBABILITY
    task_spawn_mode: object = None  # TaskSpawnMode | None — None = auto-select
    # apple_lifetime removed — LIFETIME despawn mode no longer supported


@dataclass(frozen=True)
class EnvConfig:
    height: int
    width: int
    n_agents: int
    n_tasks: int                    # was n_apples; starting count per type (or total if n_task_types=1)
    gamma: float
    # --- Unified reward ---
    r_picker: float                 # reward to picker on correct pick; groupmates
                                    # get (1 - r_picker) / (n_τ - 1) each
    # --- Task specialization ---
    n_task_types: int = 1           # T
    r_low: float = 0.0              # reward for picking wrong task type (τ ∉ G_actor)
    task_assignments: tuple[tuple[int, ...], ...] | None = None
    # task_assignments[i] = G_i.  Always populated (for n_task_types==1: every
    # agent is assigned to type 0, forming one big group).
    # rho is DERIVED from assignments: rho = |G_i| / T. Not a config field.
    # --- Pick mode ---
    pick_mode: PickMode = PickMode.FORCED
    # --- Spawning ---
    max_tasks_per_type: int = 3     # n_τ^max (used when n_task_types > 1)
    max_tasks: int = 12             # was max_apples; total hard cap (used when n_task_types == 1)
    env_type: EnvType = EnvType.STOCHASTIC
    stochastic: StochasticConfig | None = None

    # Backward compat aliases
    @property
    def n_apples(self) -> int:
        return self.n_tasks

    @property
    def max_apples(self) -> int:
        return self.max_tasks

    @property
    def force_pick(self) -> bool:
        return self.pick_mode == PickMode.FORCED


@dataclass(frozen=True)
class ScheduleConfig:
    start: float
    end: float
    schedule: Schedule
    step_size: int = 0              # only for Schedule.STEP
    step_factor: float = 1.0        # only for Schedule.STEP
    step_start: int = 0             # only for Schedule.STEP. When we start the stepping.


@dataclass(frozen=True)
class ValueLearningConfig:
    reset_freq: int


@dataclass(frozen=True)
class PolicyLearningConfig:
    epsilon: ScheduleConfig


@dataclass(frozen=True)
class TrainConfig:
    mode: TrainMode
    td_target: TDTarget
    total_steps: int
    seed: int
    lr: ScheduleConfig
    nstep: int = 1
    td_lambda: float = 0.0
    comm_weight: float = 0.0  # w: communication weight for decentralized value learning
    train_method: TrainMethod = TrainMethod.NSTEP
    learning_type: LearningType = LearningType.DECENTRALIZED
    value_learning: ValueLearningConfig | None = None
    policy_learning: PolicyLearningConfig | None = None
    stopping_condition: StoppingCondition = StoppingCondition.NONE
    patience_steps: int = 10000
    improvement_threshold: float = 0.01
    min_steps_before_stop: int = 0
    batch_actions: bool = True
    heuristic: Heuristic = Heuristic.NEAREST_TASK
    use_vmap: bool = False
    use_vec_encode: bool = True  # vectorized encoding (disable for timing baseline)
    use_gpu_batched: bool = False  # GPU-batched TD(λ) training via vmap
    time_debug: bool = False  # per-step timing breakdown to timing.csv
    time_csv_freq: int = 100  # how often to write timing averages


@dataclass(frozen=True)
class ModelConfig:
    input_type: EncoderType
    model_type: ModelType
    mlp_dims: tuple[int, ...]
    conv_specs: tuple[tuple[int, int], ...] | None = None
    k_nearest: int | None = None
    activation: Activation = Activation.RELU
    weight_init: WeightInit = WeightInit.DEFAULT


@dataclass(frozen=True)
class EvalConfig:
    rollout_len: int
    eval_steps: int
    n_test_states: int
    checkpoint_freq: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str
    main_csv_freq: int
    detail_csv_freq: int


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig
    model: ModelConfig
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
        # Start offset spaced by T/N to ensure full coverage when T > N
        start = (i * n_task_types) // n_agents
        agent_types = tuple((start + k) % n_task_types for k in range(g_size))
        assignments.append(agent_types)

    # Validate: every type covered
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
