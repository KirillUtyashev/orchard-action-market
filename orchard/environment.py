import numpy as np
import random

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]
class Orchard:
    def __init__(self, side_length, num_agents, S, phi, one=False, spawn_algo=None, despawn_algo=None):
        self.length = side_length

        if one:
            self.width = 1
        else:
            self.width = side_length

        self.n = num_agents

        self.agents = np.zeros((self.length, self.width), dtype=int)
        self.apples = np.zeros((self.length, self.width), dtype=int)

        assert np.array_equal(S.shape, np.array([self.length, self.width]))
        self.S = np.array(S)
        self.phi = phi

        self.spawn_algorithm = spawn_algo
        self.despawn_algorithm = despawn_algo

        self.total_apples = 0




    def initialize(self, agents_list, agent_pos=None):
        self.agents = np.zeros((self.length, self.width), dtype=int)
        self.apples = np.zeros((self.length, self.width), dtype=int)

        for i in range(self.n):
            if agent_pos is not None:
                position = np.array(agent_pos[i])
            else:
                position = np.random.randint(0, [self.length, self.width])
            agents_list[i].position = position
            self.agents[position[0], position[1]] += 1

        self.spawn_apples()

    def get_state(self):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }

    def get_state_only(self):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }

    def spawn_apples_wrapper(self):
        if self.spawn_algorithm is None:
            self.spawn_apples()
        else:
            spawned = self.spawn_algorithm(self)
            self.total_apples += spawned

    def despawn_apples_wrapper(self):
        if self.despawn_algorithm is None:
            self.despawn_apples()
        else:
            self.despawn_algorithm(self)

    def spawn_apples(self):
        for i in range(self.length):
            for j in range(self.width):
                chance = random.random()
                if chance < self.S[i, j]:
                    self.apples[i, j] += 1
                    self.total_apples += 1

    def despawn_apples(self):
        for i in range(self.length):
            for j in range(self.width):
                count = self.apples[i, j]
                for k in range(int(count)):
                    chance = random.random()
                    if chance < self.phi:
                        self.apples[i, j] -= 1

    def process_action(self, position, action):
        new_pos = np.clip(position + action_vectors[action], [0, 0], [self.length-1, self.width-1])

        self.agents[new_pos[0], new_pos[1]] += 1
        self.agents[position[0], position[1]] -= 1

        if self.apples[new_pos[0], new_pos[1]] >= 1:
            self.apples[new_pos[0], new_pos[1]] -= 1
            return 1, new_pos
        return 0, new_pos


    def main_step(self, position, action):
        reward, new_position = self.process_action(position, action)
        self.spawn_apples_wrapper()
        self.despawn_apples_wrapper()
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
        for i in range(self.n):
            verifier[agents_list[i].position[0], agents_list[i].position[1]] -= 1
            assert verifier[agents_list[i].position[0], agents_list[i].position[1]] >= 0
        assert sum(verifier.flatten()) == 0




#S = np.zeros((5, 5))
#for i in range(5):
#    for j in range(5):
#        S[i, j] = 0.25

#env = Orchard(5, 5, S, 0.25)


#env.initialize()