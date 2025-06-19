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
    UP = (2, [1,  0])
    DOWN = (3, [-1, 0])
    STAY = (4, [0,  0])

    def __init__(self, idx, vector):
        self._idx = idx
        self._vector = vector


class Orchard:
    def __init__(self,
                 length,
                 width,
                 num_agents,
                 s=None,
                 phi=None,
                 agents_list=None,
                 action_algo=None,
                 spawn_algo=None,
                 despawn_algo=None):
        self.length = length
        self.width = width
        if width == 1:
            self.available_actions = Action1D
        else:
            self.available_actions = Action2D

        self.n = num_agents

        self.agents = np.zeros((self.width, self.length), dtype=int)
        self.apples = np.zeros((self.width, self.length), dtype=int)
        self._apple_ages = None

        if agents_list is None:
            self.agents_list = None
        else:
            assert self.n == len(agents_list)
            self.agents_list = agents_list

        """
        If spawn algorithm / despawn algo / etc. are none, then default to the Phi / S configuration.
        """
        if spawn_algo is None:
            assert np.array_equal(s.shape, np.array([self.length, self.width]))
            self.S = np.array(s)
            self.spawn_algorithm = self.spawn_apples
        else:
            self.spawn_algorithm = spawn_algo

        if despawn_algo is None:
            self.phi = phi
            self.despawn_algorithm = self.despawn_apples
        else:
            self.despawn_algorithm = despawn_algo

        if action_algo is None:
            self.action_algorithm = self.process_action
        else:
            self.action_algorithm = action_algo

        self.total_apples = 0
        self.apples_despawned = 0

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
        self._apple_ages = [[[] for _ in range(self.apples.shape[1])] for _ in range(self.apples.shape[0])]
        self.set_positions(agent_pos)
        self.spawn_algorithm(self, self._apple_ages)

    def set_positions(self, agent_pos=None):
        for i in range(self.n):
            if agent_pos is not None:
                position = np.array(agent_pos[i])
            else:
                position = np.random.randint(0, [self.width, self.length])
            self.agents_list[i].position = position
            self.agents[position[0], position[1]] += 1

    def get_state(self):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }

    def spawn_apples(self):
        """
        Spawn apples. Not used if there is a specific algorithm.
        :return:
        """
        apples = 0
        for i in range(self.width):
            for j in range(self.length):
                chance = random.random()
                if chance < self.S[i, j]:
                    self.apples[i, j] += 1
                    apples += 1
        return apples

    def despawn_apples(self):
        """
        Despawn apples. Not used if there is a specific algorithm.
        :return:
        """
        for i in range(self.width):
            for j in range(self.length):
                count = self.apples[i, j]
                for k in range(int(count)):
                    chance = random.random()
                    if chance < self.phi:
                        self.apples[i, j] -= 1

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
            # Randomly pick one apple from the position
            # index = random.randint(0, len(self._apple_ages[new_pos[0]][new_pos[1]]) - 1)
            # self._apple_ages[new_pos[0]][new_pos[1]].pop(index)
            return 1, new_pos
        return 0, new_pos

    def main_step(self, position, action):
        reward, new_position = self.action_algorithm(position, action)
        self.total_apples += self.spawn_algorithm(self, self._apple_ages)
        self.apples_despawned += self.despawn_algorithm(self, self._apple_ages)

        return reward, new_position

    def validate_agents(self):
        return sum(self.agents) == self.n

    def validate_apples(self):
        for i in range(self.width):
            for j in range(self.length):
                assert self.apples[i, j] >= 0

    def validate_agent_pos(self, agents_list):
        for i in range(self.n):
            assert 0 <= agents_list[i].position[0] < self.length and 0 <= agents_list[i].position[1] < self.width

    def validate_agent_consistency(self, agents_list):
        verifier = self.agents.copy()
        print(verifier)
        for i in range(self.n):
            verifier[agents_list[i].position[0], agents_list[i].position[1]] -= 1
            assert verifier[agents_list[i].position[0], agents_list[i].position[1]] >= 0
        assert sum(verifier.flatten()) == 0

    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer((self.width, self.length))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")
