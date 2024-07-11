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
    def __init__(self, side_length, num_agents, S, phi):
        self.length = side_length
        self.n = num_agents

        self.agents = np.zeros((self.length, self.length))
        self.agent_positions = np.array([[0, 0]] * self.n)

        self.apples = np.zeros((self.length, self.length))

        self.S = np.array(S)
        self.phi = phi

        self.total_apples = 0


    def initialize(self):
        self.agents = np.zeros((self.length, self.length))
        self.apples = np.zeros((self.length, self.length))

        for i in range(self.n):
            position = np.random.randint(0, self.length, 2)
            self.agent_positions[i] = position
            self.agents[position[0], position[1]] += 1

        self.spawn_apples()

    def get_state(self, agent):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }, self.agent_positions[agent].copy()

    def spawn_apples(self):
        for i in range(self.n):
            for j in range(self.n):
                chance = random.random()
                if chance < self.S[i, j]:
                    self.apples[i, j] += 1
                    self.total_apples += 1

    def despawn_apples(self):
        for i in range(self.n):
            for j in range(self.n):
                count = self.apples[i, j]
                for k in range(int(count)):
                    chance = random.random()
                    if chance < self.phi:
                        self.apples[i, j] -= 1

    def process_action(self, agent, action):
        agent_pos = self.agent_positions[agent]
        new_pos = np.clip(agent_pos + action_vectors[action], [0, 0], [self.length-1, self.length-1])
        self.agent_positions[agent] = new_pos
        self.agents[new_pos[0], new_pos[1]] += 1
        self.agents[agent_pos[0], agent_pos[1]] -= 1
        if self.apples[new_pos[0], new_pos[1]] >= 1:
            self.apples[new_pos[0], new_pos[1]] -= 1
            return 1
        return 0



    def main_step(self, agent, action):
        reward = self.process_action(agent, action)
        self.spawn_apples()
        self.despawn_apples()
        return reward

    def validate_agents(self):
        return sum(self.agents) == self.n

    def validate_apples(self):
        for i in range(self.length):
            for j in range(self.length):
                assert self.apples[i, j] >= 0

    def validate_agent_pos(self):
        for i in range(self.n):
            assert 0 <= self.agent_positions[i][0] <= self.length and 0 <= self.agent_positions[i][1] <= self.length

    def validate_agent_consistency(self):
        verifier = self.agents.copy()
        for i in range(self.n):
            verifier[self.agent_positions[i][0], self.agent_positions[i][1]] -= 1
            assert verifier[self.agent_positions[i][0], self.agent_positions[i][1]] >= 0
        assert sum(verifier.flatten()) == 0




S = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        S[i, j] = 0.25

env = Orchard(5, 5, S, 0.25)


env.initialize()