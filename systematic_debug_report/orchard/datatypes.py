"""Core types: Grid, State, Transition, EncoderOutput, and config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from orchard.enums import (
    Action,
    DespawnMode,
    EncoderType,
    EnvType,
    LearningType,
    ModelType,
    Schedule,
    StoppingCondition,
    TDTarget,
    TrainMode,
    TrainMethod,
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
    apple_positions: tuple[Grid, ...]       # sorted, length = current apple count
    actor: int                              # index of agent whose turn it is
    apple_ages: tuple[int, ...] | None = None  # parallel to apple_positions; None if not tracking
    apple_ids: tuple[int, ...] | None = None   # parallel to apple_positions; persistent slot ID per apple. needed for input.

    def is_agent_on_apple(self, agent_idx: int) -> bool:
        """Check if agent is on an apple cell."""
        return self.agent_positions[agent_idx] in self.apple_positions

    @property
    def n_agents(self) -> int:
        return len(self.agent_positions)


# ---------------------------------------------------------------------------
# Encoder output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderOutput:
    """Output of an encoder. Exactly one of scalar/grid is non-None."""
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
    rewards: tuple[float, ...] # r_{t+1} for each agent; length = n_agents
    discount: float                  # γ_{t+1} for this transition
    
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
    spawn_prob: float           # per empty cell per turn
    despawn_mode: DespawnMode
    despawn_prob: float         # only meaningful if despawn_mode == PROBABILITY
    apple_lifetime: int         # only meaningful if despawn_mode == LIFETIME


@dataclass(frozen=True)
class EnvConfig:
    height: int
    width: int
    n_agents: int
    n_apples: int
    gamma: float
    r_picker: float
    force_pick: bool
    max_apples: int
    env_type: EnvType
    stochastic: StochasticConfig | None = None  # None if deterministic


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
    td_lambda: float = 0.0  # only for TD(λ) backward view; ignored otherwise
    train_method: TrainMethod = TrainMethod.NSTEP 
    learning_type: LearningType = LearningType.DECENTRALIZED
    value_learning: ValueLearningConfig | None = None   # only if mode == VALUE_LEARNING
    policy_learning: PolicyLearningConfig | None = None  # only if mode == POLICY_LEARNING
    stopping_condition: StoppingCondition = StoppingCondition.NONE
    patience_steps: int = 10000
    improvement_threshold: float = 0.01
    min_steps_before_stop: int = 0


@dataclass(frozen=True)
class ModelConfig:
    input_type: EncoderType
    model_type: ModelType
    mlp_dims: tuple[int, ...]
    conv_specs: tuple[tuple[int, int], ...] | None = None
    k_nearest: int | None = None  # only for relative_k encoder; ignored otherwise. If None, defaults to n_agents - 1.


@dataclass(frozen=True)
class EvalConfig:
    rollout_len: int
    eval_steps: int
    n_test_states: int
    checkpoint_freq: int = 0  # 0 means no periodic checkpoints (final always saved)


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str             # the one allowed string — it's a path
    main_csv_freq: int
    detail_csv_freq: int


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig
