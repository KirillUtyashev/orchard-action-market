from environment import *
import numpy as np

S = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        S[i, j] = 0.25

env = Orchard(2, 2, S, 0.25)


env.initialize()
env.validate_agents()
env.validate_agent_pos()
env.validate_apples()
env.validate_agent_consistency()

"""
Sample Action Test:
We have a 2x2 grid, and an agent moves into a space that has an apple. We make sure the
agent has moved and the apple has been picked.
"""
env.apples = np.array([[0, 1], [1, 0]])
env.agents = np.array([[1, 0], [0, 1]])
env.agent_positions = [[0, 0], [1, 1]]

env.validate_agent_consistency()
env.process_action(0, 2)
assert np.array_equal(env.agents, np.array([[0, 1], [0, 1]]))
assert np.array_equal(np.array(env.agent_positions), np.array([[0, 1], [1, 1]]))
assert np.array_equal(env.apples, np.array([[0, 0], [1, 0]]))



