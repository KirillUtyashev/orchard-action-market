import numpy as np
import random
random.seed(10)

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]
class Orchard:
    def __init__(self, side_length, num_agents, s=None, phi=None, agents_list=None, one=False, action_algo=None, spawn_algo=None, despawn_algo=None):
        self.length = side_length

        if one:
            self.width = 1
        else:
            self.width = side_length

        self.n = num_agents

        self.agents = np.zeros((self.length, self.width), dtype=int)
        self.apples = np.zeros((self.length, self.width), dtype=int)

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

    def initialize(self, agents_list, agent_pos=None, apples=None):
        self.agents = np.zeros((self.length, self.width), dtype=int)
        if apples is None:
            self.apples = np.zeros((self.length, self.width), dtype=int)
        else:
            self.apples = apples
        self.agents_list = agents_list
        self.set_positions(agent_pos)
        self.spawn_algorithm(self)

    def set_positions(self, agent_pos=None):
        self.agents = np.zeros((self.length, self.width), dtype=int)
        for i in range(self.n):
            if agent_pos is not None:
                position = np.array(agent_pos[i])
            else:
                position = np.random.randint(0, [self.length, self.width])
            self.agents_list[i].position = position
            self.agents[position[0], position[1]] += 1

    def get_state_only(self):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }

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
        for i in range(self.length):
            for j in range(self.width):
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
        for i in range(self.length):
            for j in range(self.width):
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
        new_pos = np.clip(position + action_vectors[action], [0, 0], [self.length-1, self.width-1])

        self.agents[new_pos[0], new_pos[1]] += 1
        self.agents[position[0], position[1]] -= 1

        if self.apples[new_pos[0], new_pos[1]] >= 1:
            self.apples[new_pos[0], new_pos[1]] -= 1
            return 1, new_pos
        return 0, new_pos

    def main_step(self, position, action):
        reward, new_position = self.action_algorithm(position, action)
        self.total_apples += self.spawn_algorithm(self)
        self.despawn_algorithm(self)

        return reward, new_position

    def validate_agents(self):
        return sum(self.agents) == self.n

    def validate_apples(self):
        for i in range(self.length):
            for j in range(self.width):
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
