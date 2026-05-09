"""All enums used throughout the orchard RL project."""

from __future__ import annotations

from enum import Enum, auto


class DespawnMode(Enum):
    NONE = auto()
    PROBABILITY = auto()


class Heuristic(Enum):
    NEAREST = auto()  # value-aware: argmax φ(i,κ)·Σ_j R(i,j)·r'_j; moves toward best task


class EncoderType(Enum):
    GENERAL_DEC_CNN_GRID = auto()   # dec: T+3 channels, 3 scalars
    GENERAL_CEN_CNN_GRID = auto()   # cen: T+N+1 channels, N+1 scalars


class Activation(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    NONE = auto()


class WeightInit(Enum):
    DEFAULT = auto()
    ZERO_BIAS = auto()


class LearningType(Enum):
    DECENTRALIZED = auto()
    CENTRALIZED = auto()


class AlgorithmName(Enum):
    VALUE = auto()
    ACTOR_CRITIC = auto()


class StoppingCondition(Enum):
    NONE = auto()
    RUNNING_MAX_RPS = auto()


class Schedule(Enum):
    NONE = auto()
    LINEAR = auto()
    STEP = auto()


# ---------------------------------------------------------------------------
# Action — regular class (not Enum) to support dynamic pick actions
# ---------------------------------------------------------------------------

_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),   # UP
    1: (1, 0),    # DOWN
    2: (0, -1),   # LEFT
    3: (0, 1),    # RIGHT
    4: (0, 0),    # STAY
}

_ACTION_NAMES: dict[int, str] = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'STAY',
}


class Action:
    """Action with integer value. Movement actions 0-4, pick actions 5+."""
    __slots__ = ('_value',)

    # Class-level singletons — declared for Pylance
    UP: Action
    DOWN: Action
    LEFT: Action
    RIGHT: Action
    STAY: Action

    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @property
    def delta(self) -> tuple[int, int]:
        """Movement offset. (0,0) for STAY and all pick actions."""
        return _ACTION_DELTAS.get(self._value, (0, 0))

    @property
    def name(self) -> str:
        return _ACTION_NAMES.get(self._value, f'PICK_{self._value - 5}')

    def is_move(self) -> bool:
        return self._value <= 4

    def is_pick(self) -> bool:
        return self._value >= 5

    def pick_type(self) -> int | None:
        """For pick actions, returns the task type index. None for movement."""
        return self._value - 5 if self._value >= 5 else None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Action):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f'Action.{self.name}'


# Class-level singletons
Action.UP = Action(0)
Action.DOWN = Action(1)
Action.LEFT = Action(2)
Action.RIGHT = Action(3)
Action.STAY = Action(4)


def make_pick_action(task_type: int) -> Action:
    """Create pick(τ) action for choice pick mode."""
    return Action(5 + task_type)


NUM_MOVE_ACTIONS: int = 5


def num_actions(n_task_types: int) -> int:
    """Total action count: 5 move + T pick options."""
    return 5 + n_task_types


# Tie-break priority for greedy action selection (movement only)
ACTION_PRIORITY: list[Action] = [
    Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP, Action.STAY,
]
