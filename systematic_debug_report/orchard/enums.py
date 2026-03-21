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
    RELATIVE_K = auto() # nearest K
    POSITIONAL_K = auto()  # fixed-slot: apples by grid position, agents by index
    GRID_MLP = auto()      # flattened CNN grid as MLP input (baseline)
    STABLE_ID = auto()     # SMAC-style: persistent apple IDs as slot indices
    CNN_GRID = auto()
    CENTRALIZED_CNN_GRID = auto()
    EGOCENTRIC_CNN_GRID = auto()
    NO_REDUNDANT_AGENT_GRID = auto()


class ModelType(Enum):
    MLP = auto()
    CNN = auto()


class TrainMode(Enum):
    VALUE_LEARNING = auto()
    REWARD_LEARNING = auto()
    POLICY_LEARNING = auto()


class Schedule(Enum):
    NONE = auto()
    LINEAR = auto()
    STEP = auto()

class TrainMethod(Enum):
    NSTEP = auto()
    BACKWARD_VIEW = auto()
    
class LearningType(Enum):
    DECENTRALIZED = auto()
    CENTRALIZED = auto()

class StoppingCondition(Enum):
    NONE = auto()
    RUNNING_MAX_PPS = auto()
    RUNNING_MIN_MAE = auto()

class Activation(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    NONE = auto()

class WeightInit(Enum):
    DEFAULT = auto()
    ZERO_BIAS = auto()

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PICK = 5

    @property
    def delta(self) -> tuple[int, int]:
        return _ACTION_DELTAS[self]


_ACTION_DELTAS: dict["Action", tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
    Action.PICK: (0, 0),
}

# Tie-break priority for greedy action selection
ACTION_PRIORITY: list[Action] = [
    Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP, Action.STAY,
]

NUM_ACTIONS: int = 5 # Exclude PICK action from the count of actions for the agent to choose from.
