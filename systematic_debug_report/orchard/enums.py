"""All enums used throughout the orchard RL project."""

from enum import Enum, auto

class TDTarget(Enum):
    PRE_ACTION = auto()
    AFTER_STATE = auto()

class EnvType(Enum):
    DETERMINISTIC = auto()
    STOCHASTIC = auto()


class DespawnMode(Enum):
    NONE = auto()
    PROBABILITY = auto()
    LIFETIME = auto()


class EncoderType(Enum):
    RELATIVE = auto() # only for deterministic spawn and pre_action td target.
    RELATIVE_K = auto()
    CNN_GRID = auto()


class ModelType(Enum):
    MLP = auto()
    CNN = auto()


class TrainMode(Enum):
    VALUE_LEARNING = auto()
    POLICY_LEARNING = auto()


class Schedule(Enum):
    NONE = auto()
    LINEAR = auto()
    STEP = auto()


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

    @property
    def delta(self) -> tuple[int, int]:
        return _ACTION_DELTAS[self]


_ACTION_DELTAS: dict["Action", tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}

# Tie-break priority for greedy action selection
ACTION_PRIORITY: list[Action] = [
    Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP, Action.STAY,
]

NUM_ACTIONS: int = len(Action)
