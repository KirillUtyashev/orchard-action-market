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
    # LIFETIME removed — was unused and complicated State with apple_ages


class TaskSpawnMode(Enum):
    GLOBAL_UNIQUE = auto()   # at most 1 task of ANY type per cell (forced pick)
    PER_TYPE_UNIQUE = auto() # at most 1 task per TYPE per cell; types may coexist (choice pick)


class PickMode(Enum):
    FORCED = auto()   # auto-pick when stepping on task cell
    CHOICE = auto()   # explicit pick(τ) action required


class Heuristic(Enum):
    NEAREST_TASK = auto()                    # move toward any nearest task (legacy)
    NEAREST_CORRECT_TASK = auto()            # move toward nearest task with τ ∈ G_actor; phase 2: always pick
    NEAREST_CORRECT_TASK_STAY_WRONG = auto() # same phase 1; phase 2: pick if correct type, STAY if wrong


class EncoderType(Enum):
    RELATIVE = auto()
    RELATIVE_K = auto()
    GRID_MLP = auto()
    CNN_GRID = auto()
    CENTRALIZED_CNN_GRID = auto()
    EGOCENTRIC_CNN_GRID = auto()
    NO_REDUNDANT_AGENT_GRID = auto()
    # New task-specialization encoders
    TASK_CNN_GRID = auto()              # dec: T+3 channels, 1 scalar
    CENTRALIZED_TASK_CNN_GRID = auto()  # cen: T+N+1 channels, N scalars


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
    RUNNING_MAX_RPS = auto()  # for task specialization experiments


class Activation(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    NONE = auto()


class WeightInit(Enum):
    DEFAULT = auto()
    ZERO_BIAS = auto()


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
    5: 'PICK',
}


class Action:
    """Action with integer value. Movement actions 0-4, pick actions 5+."""
    __slots__ = ('_value',)

    def __init__(self, value: int):
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

    def __eq__(self, other):
        if isinstance(other, Action):
            return self._value == other._value
        return NotImplemented

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        return f'Action.{self.name}'


# Class-level singletons for movement actions
Action.UP = Action(0)
Action.DOWN = Action(1)
Action.LEFT = Action(2)
Action.RIGHT = Action(3)
Action.STAY = Action(4)
Action.PICK = Action(5)  # generic pick, used in forced mode transitions


def make_pick_action(task_type: int) -> Action:
    """Create pick(τ) action for choice pick mode."""
    return Action(5 + task_type)


NUM_MOVE_ACTIONS: int = 5


def num_actions(pick_mode: PickMode, n_task_types: int) -> int:
    """Total action count: 5 for forced, 5+T for choice."""
    if pick_mode == PickMode.FORCED:
        return 5
    return 5 + n_task_types


# Tie-break priority for greedy action selection (movement only)
ACTION_PRIORITY: list[Action] = [
    Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP, Action.STAY,
]

# Backward compat alias
NUM_ACTIONS: int = NUM_MOVE_ACTIONS
