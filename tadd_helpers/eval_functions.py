import sys

sys.path.append("../../orchard/")
from tadd_helpers.env_functions import OldState, State
from orchard.environment import MoveAction
import numpy as np


def random_policy(s: State, agent_idx: int):
    return MoveAction.get_random_action()


def nearest_policy(s: State, agent_idx: int):
    agent_pos = s.agent_position(agent_idx)
    apple_positions = np.argwhere(s.apples > 0)
    if len(apple_positions) == 0:
        return MoveAction.get_random_action()
    distances = np.linalg.norm(apple_positions - agent_pos, axis=1)
    nearest_apple_idx = np.argmin(distances)
    nearest_apple_pos = apple_positions[nearest_apple_idx]
    direction = nearest_apple_pos - agent_pos
    if abs(direction[0]) > abs(direction[1]):
        return MoveAction.DOWN if direction[0] > 0 else MoveAction.UP
    else:
        return MoveAction.RIGHT if direction[1] > 0 else MoveAction.LEFT


def old_random_policy(s: OldState, agent_idx: int, agent_positions: np.ndarray):
    return MoveAction.get_random_action()


def old_nearest_policy(s: OldState, agent_idx: int, agent_positions: np.ndarray):
    agent_pos = agent_positions[agent_idx]
    apple_positions = np.argwhere(s.apples > 0)
    if len(apple_positions) == 0:
        return MoveAction.get_random_action()
    distances = np.linalg.norm(apple_positions - agent_pos, axis=1)
    nearest_apple_idx = np.argmin(distances)
    nearest_apple_pos = apple_positions[nearest_apple_idx]
    direction = nearest_apple_pos - agent_pos
    if abs(direction[0]) > abs(direction[1]):
        return MoveAction.DOWN if direction[0] > 0 else MoveAction.UP
    else:
        return MoveAction.RIGHT if direction[1] > 0 else MoveAction.LEFT
