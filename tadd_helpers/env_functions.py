from dataclasses import dataclass
import random
from typing import Any
import collections

import numpy as np


@dataclass
class Position:
    row: int
    col: int

    def np_array(self):
        return np.array([self.row, self.col])


@dataclass
class State:
    _apples: np.ndarray[Any, np.dtype[np.int_]]
    _agents: dict[int, tuple[int, int]]
    name: str = "State"

    @property
    def apples(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        """Returns the apples matrix."""
        return self._apples

    @apples.setter
    def apples(self, new_apples: np.ndarray[Any, np.dtype[np.int_]]):
        """Sets the apples matrix."""
        self._apples = new_apples

    def remove_apple_at(self, position: np.ndarray):
        """Removes an apple at the given position (row, col) if present."""
        if self._apples[position[0], position[1]] > 0:
            self._apples[position[0], position[1]] -= 1

    def set_agent_position(self, agent_idx: int, position: np.ndarray):
        """Sets the position of agent with index agent_idx to the given position."""
        self._agents[agent_idx] = (position[0], position[1])

    def agent_position(self, agent_idx: int) -> np.ndarray:
        """Returns the position of agent with index agent_idx as a numpy array [row, col]."""
        return np.array(self._agents[agent_idx])

    def agent_position_tuple(self, agent_idx: int) -> tuple[int, int]:
        """Returns the position of agent with index agent_idx as a tuple (row, col)."""
        return self._agents[agent_idx]

    @property
    def agents(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        """Returns the agents matrix."""
        H, L = self.H, self.L
        agents_matrix = np.zeros((H, L), dtype=int)
        for pos in self._agents.values():
            agents_matrix[pos] = 1
        return agents_matrix

    @property
    def H(self) -> int:
        return self._apples.shape[0]

    @property
    def L(self) -> int:
        return self._apples.shape[1]

    def get_random_agent_id(self) -> int:
        """
        Returns a random agent ID from the _agents dictionary.
        """
        # 1. Get the keys (agent IDs) as a list
        agent_ids = list(self._agents.keys())

        # 2. Check if the list is empty
        if not agent_ids:
            raise ValueError("There are no agents in this state")

        # 3. Use random.choice() to pick a random element
        return random.choice(agent_ids)

    def __str__(self):
        """
        Provides a comprehensive visual representation of the state.

        1. Agent Locations: A clear map of each agent's ID to its position.
        2. Agents Grid: A matrix showing the COUNT of agents in each cell.
        3. Apples Grid: A matrix showing the COUNT of apples in each cell.
        """
        output_lines = []
        header = f"--- {self.name} (Grid: {self.H}x{self.L}) ---"
        output_lines.append(header)

        # 1. Display the Agent ID to Position Mapping
        output_lines.append("\n--- Agent Locations ---")
        if not self._agents:
            output_lines.append("  (No agents in this state)")
        else:
            # Sort by agent ID for consistent output
            for agent_id, pos in sorted(self._agents.items()):
                pos = (int(pos[0]), int(pos[1]))
                output_lines.append(f"  Agent {agent_id}: {pos}")

        # 2. Build and Display the Agents Count Grid
        output_lines.append("\n--- Agents (Count) ---")
        # Use collections.Counter for a fast and elegant way to count agents at each position
        agent_pos_counts = collections.Counter(self._agents.values())
        for r in range(self.H):
            row_str = []
            for c in range(self.L):
                count = agent_pos_counts.get((r, c), 0)
                if count > 0:
                    row_str.append(str(count))
                else:
                    row_str.append(".")
            output_lines.append(" ".join(row_str))

        # 3. Build and Display the Apples Count Grid
        output_lines.append("\n--- Apples (Count) ---")
        for r in range(self.H):
            row_str = []
            for c in range(self.L):
                apple_count = self._apples[r, c]
                if apple_count > 0:
                    row_str.append(str(apple_count))
                else:
                    row_str.append(".")
            output_lines.append(" ".join(row_str))

        return "\n".join(output_lines)

    def __hash__(self):
        """
        Creates a unique, hashable representation of the state.

        This allows State objects to be used as keys in a dictionary or elements in a set,
        which is essential for algorithms that need to track visited states (e.g., Q-learning).
        """
        apples_hashable = self._apples.tobytes()

        agents_hashable = frozenset(self._agents.items())

        # 3. Combine the hashable components into a tuple and hash the tuple.
        return hash((agents_hashable, apples_hashable))

    def __eq__(self, other):
            if not isinstance(other, State):
                return NotImplemented

            # 1. Check for dictionary equality (Agent IDs -> Position)
            agents_equal = self._agents == other._agents
            
            # 2. Check for numpy array equality (Apples matrix)
            # Note: You can use self._apples directly or the getter self.apples, 
            # but using the internal attribute is more direct for comparison.
            apples_equal = np.array_equal(self._apples, other._apples)

            return agents_equal and apples_equal

    def copy(self):
        return State(
            _apples=self._apples.copy(), _agents=self._agents.copy(), name=self.name
        )


@dataclass
class OldState:
    # deprecated, use State instead
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
        return OldState(
            apples=self.apples.copy(), agents=self.agents.copy(), name=self.name
        )


def spawn_apples(s: State, p_cell: float):
    """spawns apples in place"""
    H, L = s.H, s.L
    rand_mat = np.random.rand(H, L)
    spawn_mask = rand_mat < p_cell
    spawn_mask = spawn_mask & (s.agents == 0)
    spawn_mask = spawn_mask & (s.apples == 0)
    s.apples[spawn_mask] += 1


def despawn_apples(s: State, q_despawn: float):
    """despawns apples in place"""
    H, L = s.H, s.L
    rand_mat = np.random.rand(H, L)
    mask_apples = s.apples != 0
    removal_mask = mask_apples & (rand_mat < q_despawn)
    s.apples[removal_mask] -= 1


def init_empty_state(height: int, width: int, num_agents: int):
    s = State(
        _apples=np.zeros((height, width), dtype=int),
        _agents={},
        name="Empty State",
    )
    for c in range(num_agents):
        # get random position for each agent
        row = np.random.randint(0, height)
        col = np.random.randint(0, width)
        # place agent in the state
        s._agents[c] = (int(row), int(col))
    return s


def old_place_agents_randomly(s: OldState, num_agents: int):
    H, L = s.agents.shape
    empty_positions = np.argwhere(s.agents == 0)
    chosen_indices = np.random.choice(
        len(empty_positions), size=num_agents, replace=False
    )
    for index in chosen_indices:
        pos = empty_positions[index]
        s.agents[tuple(pos)] = 1
    return s


# deprecated
def old_spawn_apples(s: OldState, p_cell: float):
    """spawns apples in place"""
    H, L = s.apples.shape  # ★ rows = width, cols = length

    rand_mat = np.random.rand(H, L)  # ★ new RNG draw, same shape
    spawn_mask = rand_mat < p_cell
    spawn_mask = spawn_mask & (s.agents == 0)
    spawn_mask = spawn_mask & (s.apples == 0)
    s.apples[spawn_mask] += 1
    return s


# deprecated
def old_despawn_apples(s: OldState, q_despawn: float):
    """
    despawns apples in place
    """
    H, L = s.apples.shape  # ★ rows = width, cols = length

    # ---- despawn ----
    rand_mat = np.random.rand(H, L)  # ★ same shape as s.apples
    mask_apples = s.apples != 0
    removal_mask = mask_apples & (rand_mat < q_despawn)
    s.apples[removal_mask] -= 1
    return removal_mask.sum()


# deprecated
def old_init_empty_state(height: int, width: int):
    return OldState(
        apples=np.zeros((height, width), dtype=int),
        agents=np.zeros((height, width), dtype=int),
        name="Empty State",
    )
