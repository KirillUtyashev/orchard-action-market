from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Position:
    row: int
    col: int

    def np_array(self):
        return np.array([self.row, self.col])


@dataclass
class State:
    apples: np.ndarray[Any, np.dtype[np.int_]]
    agents: np.ndarray[Any, np.dtype[np.int_]]
    name: str = "State"

    def __hash__(self):
        # Convert the numpy arrays to their byte representation, which is hashable
        return hash((self.agents.tobytes(), self.apples.tobytes()))

    def __eq__(self, other):
        # Define how to check for equality between two State objects
        return np.array_equal(self.agents, other.agents) and np.array_equal(
            self.apples, other.apples
        )

    def copy(self):
        return State(
            apples=self.apples.copy(), agents=self.agents.copy(), name=self.name
        )


def spawn_apples(s: State, p_cell: float):
    H, L = s.apples.shape  # ★ rows = width, cols = length

    rand_mat = np.random.rand(H, L)  # ★ new RNG draw, same shape
    spawn_mask = rand_mat < p_cell
    spawn_mask = spawn_mask & (s.agents == 0)
    spawn_mask = spawn_mask & (s.apples == 0)
    s.apples[spawn_mask] += 1
    return s


def despawn_apples(s: State, q_despawn: float):
    """
    One 'second-boundary' update:
      • despawn apples with geom-hazard q = 1/T
      • spawn new apples in empty cells with p_cell = r · s_target
    Call exactly once after every n micro-ticks.
    """
    H, L = s.apples.shape  # ★ rows = width, cols = length

    # ---- despawn ----
    rand_mat = np.random.rand(H, L)  # ★ same shape as s.apples
    mask_apples = s.apples != 0
    removal_mask = mask_apples & (rand_mat < q_despawn)
    s.apples[removal_mask] -= 1
    return removal_mask.sum()


def init_empty_state(height: int, width: int):
    return State(
        apples=np.zeros((height, width), dtype=int),
        agents=np.zeros((height, width), dtype=int),
        name="Empty State",
    )


def place_agents_randomly(s: State, num_agents: int):
    H, L = s.agents.shape
    empty_positions = np.argwhere(s.agents == 0)
    chosen_indices = np.random.choice(
        len(empty_positions), size=num_agents, replace=False
    )
    for index in chosen_indices:
        pos = empty_positions[index]
        s.agents[tuple(pos)] = 1
    return s
