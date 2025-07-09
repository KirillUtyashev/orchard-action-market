import math

import numpy as np
import random
from enum import Enum, auto
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


class Action1D(ActionMixin, Enum):
    LEFT = (0, [0,  -1])
    RIGHT = (1, [0,  1])
    STAY = (2, [0,  0])

    def __init__(self, idx, vector):
        self._idx = idx
        self._vector = vector


class Action2D(ActionMixin, Enum):
    LEFT = (0, [0,  -1])
    RIGHT = (1, [0,  1])
    STAY = (2, [0,  0])
    UP = (3, [-1,  0])
    DOWN = (4, [1, 0])

    def __init__(self, idx, vector):
        self._idx = idx
        self._vector = vector


class Orchard:
    def __init__(self,
                 length,
                 width,
                 num_agents,
                 agents_list=None,
                 action_algo=None,
                 spawn_algo=None,
                 despawn_algo=None,
                 s_target=0.1,
                 apple_mean_lifetime=None):
        self.length = length
        self.width = width
        if width == 1:
            self.available_actions = Action1D
        else:
            self.available_actions = Action2D

        self.n = num_agents

        self.agents = np.zeros((self.width, self.length), dtype=int)
        self.apples = np.zeros((self.width, self.length), dtype=int)

        assert self.n == len(agents_list)
        self.agents_list = agents_list

        self.spawn_algorithm = spawn_algo
        self.despawn_algorithm = despawn_algo

        if action_algo is None:
            self.action_algorithm = self.process_action
        else:
            self.action_algorithm = action_algo

        self.total_apples = 0
        self.apples_despawned = 0

        self.spawn_rate = (self.n / (self.length * self.width)) * s_target
        if apple_mean_lifetime is None:
            self.despawn_rate = min(1.0, 1.0 / ((math.ceil(1 / (2 * math.sqrt((self.n / (self.length * self.width))))) + 1) if self.width > 1 else (1 / (2 * (self.n / (self.length * self.width))) + 1)))
        else:
            self.despawn_rate = 1 / apple_mean_lifetime
        # Variables needed when visualizing Orchard environment
        self._rendering_initialized = False
        self.render_mode = None

    def initialize(self, agents_list, agent_pos=None, apples=None):
        """
        Populate the Orchard environment with agents in agent_list and randomly spawn the first apple
        """
        self.agents = np.zeros((self.width, self.length), dtype=int)
        if apples is None:
            self.apples = np.zeros((self.width, self.length), dtype=int)
        else:
            self.apples = apples  # This has to be an array
        self.agents_list = agents_list
        self.set_positions(agent_pos)
        # self.spawn_algorithm(self)

    def set_positions(self, agent_pos=None):
        for i in range(self.n):
            if agent_pos is not None:
                position = agent_pos[i]
            else:
                position = np.random.randint(0, [self.width, self.length])
            self.agents_list[i].position = position
            self.agents[position[0], position[1]] += 1

    def get_state(self):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }

    def process_action(self, position, action):
        """

        :param position:
        :param action:
        :return:
        """
        # Find the new position of the agent based on their old position and their action
        if action is not None:
            new_pos = np.clip(position + self.available_actions.from_idx(action).vector, [0, 0], [self.width-1, self.length-1])
            self.agents[new_pos[0], new_pos[1]] += 1
            self.agents[position[0], position[1]] -= 1
        else:
            new_pos = position
        if self.apples[new_pos[0], new_pos[1]] >= 1:
            self.apples[new_pos[0], new_pos[1]] -= 1
            return 1, new_pos
        return 0, new_pos

    def main_step(self, position, action):
        reward, new_position = self.action_algorithm(position, action)
        # self.total_apples += self.spawn_algorithm(self)
        # self.apples_despawned += self.despawn_algorithm(self)
        return reward, new_position

    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer((self.width, self.length))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")
