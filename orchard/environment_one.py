import numpy as np
import random

action_vectors = [
            -1,
            1,
            0
        ]
class Orchard:
    def __init__(self, side_length, num_agents, S, phi):
        self.length = side_length
        self.n = num_agents

        self.agents = np.zeros(self.length)
        self.apples = np.zeros(self.length)

        self.agent_positions = np.array([0] * self.n)

        self.S = np.array(S)
        self.phi = phi


    def initialize(self):
        """
        Initializes the A and B matrices (agents and apples). Places agents in random locations.
        :return:
        """
        self.agents = np.zeros(self.length)
        self.apples = np.zeros(self.length)

        for i in range(self.n):
            position = np.random.randint(0, self.length, 1)
            self.agent_positions[i] = position
            self.agents[position] += 1
        self.spawn_apples()

    def get_state(self, agent):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
        }, self.agent_positions[agent].copy()

    def spawn_apples(self):
        """
        Spawns apples. Each "tree" (coordinate) has an S[i] chance of spawning an apple.
        :return:
        """
        for i in range(self.n):
            chance = random.random()
            if chance < self.S[i]:
                self.apples[i] += 1

    def despawn_apples(self):
        """
        Despawns apples. Each apple has a phi chance of rotting.
        :return:
        """
        for i in range(self.n):
            count = int(self.apples[i])
            for k in range(count):
                chance = random.random()
                if chance < self.phi:
                    self.apples[i] -= 1

    def process_action(self, agent, action):
        """
        Processes the action of an agent.
        :param agent:
        :param action: Chosen action
        :return: Reward sustained in the action (i.e. whether an apple is picked or not).
        """
        agent_pos = self.agent_positions[agent]
        new_pos = np.clip(agent_pos + action_vectors[action], 0, self.length-1)
        self.agent_positions[agent] = new_pos
        self.agents[new_pos] += 1
        self.agents[agent_pos] -= 1
        if self.apples[new_pos] >= 1:
            self.apples[new_pos] -= 1
            return 1
        return 0


    def main_step(self, agent, action):
        """
        The main time-step in this environment.
        Agent action is processed first; then, apples are spawned and then despawned.
        :param agent:
        :param action:
        :return: reward sustained by agent.
        """
        reward = self.process_action(agent, action)
        self.spawn_apples()
        self.despawn_apples()
        return reward

    def validate_agents(self):
        """
        Validates that the agent count is consistent.
        :return:
        """
        return sum(self.agents) == self.n

    def validate_apples(self):
        """
        Makes sure there are no negative apples.
        :return:
        """
        for i in range(self.length):
            assert self.apples[i] >= 0

    def validate_agent_pos(self):
        """
        Makes sure no Agents' coordinates are outside the Orchard.
        :return: None
        """
        for i in range(self.n):
            assert 0 <= self.agent_positions[i] <= self.length

    def validate_agent_consistency(self):
        """
        Verifies that the agent positions and agent matrices are consistent with each other.
        :return: None
        """
        verifier = self.agents.copy()
        for i in range(self.n):
            verifier[self.agent_positions[i]] -= 1
            assert verifier[self.agent_positions[i]] >= 0
        assert sum(verifier.flatten()) == 0




S = np.zeros(5)
for i in range(5):
    S[i] = 0.25

env = Orchard(5, 2, S, 0.25)


env.initialize()