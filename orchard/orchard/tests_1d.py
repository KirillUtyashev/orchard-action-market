from environment_one import *
import numpy as np

S = np.zeros(5)
for i in range(5):
    S[i] = 0.25

env = Orchard(5, 2, S, 0.25)


env.initialize()
env.validate_agents()
env.validate_agent_pos()
env.validate_apples()
env.validate_agent_consistency()

"""
Sample Action Test:
We have a 5x1 grid, and an agent moves into a space that has an apple. We make sure the
agent has moved and the apple has been picked.
"""
env.apples = np.array([0, 1, 1, 1, 0])
env.agents = np.array([1, 0, 0, 0, 1])
env.agent_positions = [0, 4]

env.validate_agent_consistency()
env.process_action(0, 1)
assert np.array_equal(env.agents, np.array([0, 1, 0, 0, 1]))
assert np.array_equal(np.array(env.agent_positions), np.array([1, 4]))
assert np.array_equal(env.apples, np.array([0, 0, 1, 1, 0]))

"""
No Movement Test:
An agent stays in the same place to keep picking from the same spot.
"""
env.apples = np.array([0, 5, 1, 1, 0])
env.agents = np.array([0, 1, 0, 0, 1])
env.agent_positions = [1, 4]

env.validate_agent_consistency()
env.process_action(0, 2)
assert np.array_equal(env.agents, np.array([0, 1, 0, 0, 1]))
assert np.array_equal(np.array(env.agent_positions), np.array([1, 4]))
assert np.array_equal(env.apples, np.array([0, 4, 1, 1, 0]))
"""
Border Test:
An agent tries to leave the Orchard. It is not allowed to, and picks up an apple in its own space as a result.
"""
env.apples = np.array([1, 5, 1, 1, 0])
env.agents = np.array([1, 0, 0, 0, 1])
env.agent_positions = [0, 4]

env.validate_agent_consistency()
env.process_action(0, 0)
assert np.array_equal(env.agents, np.array([1, 0, 0, 0, 1]))
assert np.array_equal(np.array(env.agent_positions), np.array([0, 4]))
assert np.array_equal(env.apples, np.array([0, 5, 1, 1, 0]))


