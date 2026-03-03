from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import random
from enum import Enum, auto

from debug.code.reward import Reward

"""
The Orchard environment. Includes provisions for transition actions, spawning, and despawning.
action_algo: an algorithm that is used to process actions (agent movements). Defaults to just updating the environment from the singular agent action.
"""

class ActionMixin:
    """Mixin providing common methods for action enums."""

    @property
    def vector(self):
        """Return the action vector as a NumPy array."""
        return np.array(self._vector, dtype=np.int8)

    @property
    def idx(self):
        """Return the numeric index of the action."""
        return self._idx

    @property
    def one_hot(self):
        """Return a one-hot encoding of the action."""
        n = len(self.__class__)
        oh = np.zeros(n, dtype=np.float32)
        oh[self.idx] = 1.0
        return oh

    @classmethod
    def from_idx(cls, idx):
        """Lookup action by its numeric index."""
        try:
            return next(a for a in cls if a.idx == idx)
        except StopIteration:
            raise ValueError(f"Invalid action index: {idx}")


class MoveAction(ActionMixin, Enum):
    LEFT = (0, [0, -1])
    RIGHT = (1, [0, 1])
    STAY = (2, [0, 0])
    UP = (3, [-1, 0])
    DOWN = (4, [1, 0])

    def __init__(self, idx, vector):
        self._idx = idx
        self._vector = vector

    @classmethod
    def get_random_action(cls):
        return random.choice(list(cls))


@dataclass
class ProcessAction:
    reward_vector: np.ndarray
    picked: bool


@dataclass
class ConsumeResult:
    consumed: bool
    owner_id: Optional[int] = None
    apple_pos: np.ndarray = None


class Orchard:
    def __init__(
            self,
            length: int,
            width: int,
            num_agents: int,
            reward: Reward | None,
            p_apple: float = 0.1,
            d_apple: float = 0.0,
            start_agents_map: Optional[np.ndarray] = None,
            start_apples_map: Optional[np.ndarray] = None,
            start_agent_positions: Optional[np.ndarray] = None,
    ):
        self.length = length
        self.width = width
        self.n = num_agents
        self.reward_module = reward
        self.p_apple = p_apple
        self.d_apple = d_apple

        # Statistics
        self.total_apples = 0
        self.total_despawned = 0
        self.apples_spawned = 0
        self.total_picked = 0

        # Initialize agent positions (n x 2 array)
        if start_agent_positions is not None:
            self.agent_positions = start_agent_positions.copy()
        else:
            self.agent_positions = np.zeros((num_agents, 2), dtype=int)

        # Initialize grid maps
        if start_agents_map is not None:
            self.agents = start_agents_map.copy()
        else:
            self.agents = np.zeros((width, length), dtype=int)

        if start_apples_map is not None:
            self.apples = start_apples_map.copy()
        else:
            self.apples = np.zeros((width, length), dtype=int)
            self.spawn_apples()


    def set_positions(self, agent_pos: Optional[np.ndarray] = None) -> None:
        """Place agents on the grid at specified or random positions."""
        for i in range(self.n):
            if agent_pos is not None:
                position = agent_pos[i]
            else:
                position = np.random.randint(0, [self.width, self.length])

            self.agent_positions[i] = position
            self.agents[tuple(position)] += 1

    def clear_positions(self):
        self.agents = np.zeros((self.width, self.length), dtype=int)
        self.agent_positions = np.zeros((self.n, 2), dtype=int)

    def get_state(self) -> dict[str, np.ndarray]:
        """Get the current state of the environment."""
        return {"agents": self.agents.copy(), "apples": self.apples.copy(), "agent_positions": self.agent_positions.copy()}

    def _apply_move(self, position: np.ndarray, new_pos: np.ndarray | None) -> np.ndarray:
        """Apply movement action and update grid."""
        if new_pos is not None:
            # Update grid
            self.agents[tuple(new_pos)] += 1
            self.agents[tuple(position)] -= 1
            return new_pos
        else:
            return position

    def process_action(
            self, actor_id: int, new_pos: np.ndarray | None, mode) -> ProcessAction:
        """Process an agent's action and return reward and pick status."""

        position = self.agent_positions[actor_id]
        new_pos = self._apply_move(position, new_pos)

        self.agent_positions[actor_id] = new_pos

        reward_vector = self.reward_module.get_reward(
            self.get_state(), actor_id, new_pos, mode
        )

        picked = reward_vector.sum() != 0
        if picked:
            self.remove_apple(new_pos)
            self.total_picked += 1

        return ProcessAction(reward_vector=reward_vector, picked=picked)

    def remove_apple(self, pos: np.ndarray) -> None:
        """Remove an apple at the given position if present."""
        if self.apples[tuple(pos)] >= 1:
            self.apples[tuple(pos)] -= 1

    def get_sum_apples(self) -> int:
        """Return the total number of apples in the orchard."""
        return int(np.sum(self.apples))

    def spawn_apples(self) -> int:
        spawn_mask = np.random.rand(*self.apples.shape) < self.p_apple
        spawned = int(spawn_mask.sum())
        self.apples[spawn_mask] += 1
        self.apples_spawned += spawned
        return spawned

    def despawn_apples(self) -> int:
        removed = np.random.binomial(self.apples, self.d_apple)
        total_removed = int(removed.sum())
        self.apples -= removed
        return total_removed

    def apply_action(self, actor_id: int, new_pos: np.ndarray) -> dict:
        """Move agent, return s_moved state (pre-pick, pre-spawn)."""
        position = self.agent_positions[actor_id]
        self._apply_move(position, new_pos)
        self.agent_positions[actor_id] = new_pos
        s_moved = self.get_state()
        s_moved["actor_id"] = actor_id
        return s_moved

    def is_on_apple(self, state: dict, actor_id: int) -> bool:
        """Check if actor is currently on an apple."""
        pos = tuple(state["agent_positions"][actor_id])
        return state["apples"][pos] >= 1

    def resolve_pick(self, actor_id: int) -> tuple[dict, np.ndarray]:
        """Process pick action, return (s_picked state, reward_vector)."""
        pos = self.agent_positions[actor_id]
        reward_vector = self.reward_module.get_reward(
            self.get_state(), actor_id, pos, mode=1
        )
        picked = reward_vector.sum() != 0
        if picked:
            self.remove_apple(pos)
            self.total_picked += 1
        s_picked = self.get_state()
        s_picked["actor_id"] = actor_id
        return s_picked, reward_vector

    def advance_actor(self, actor_id: int, num_agents: int) -> tuple[dict, int]:
        """Spawn/despawn apples at end of round, advance to next actor."""
        next_actor_idx = (actor_id + 1) % num_agents
        self.despawn_apples()
        self.spawn_apples()
        s_next = self.get_state()
        s_next["actor_id"] = next_actor_idx
        return s_next, next_actor_idx

