from orchard.environment import *
import numpy as np
import matplotlib.pyplot as plt
import random
from policies.random_policy import random_policy_1d, random_policy
from policies.nearest import nearest_1d, nearest

def step(environment: Orchard, policy):
    agent = random.randint(1, environment.n-1)
    state, agent_pos = environment.get_state(agent)
    action = policy(state, agent_pos)
    reward = environment.main_step(agent, action)
    return agent, reward

def run_environment(policy):
    env = Orchard(side_length, num_agents, S, phi)
    env.initialize()
    reward = 0
    for i in range(1000):
        agent, i_reward = step(env, policy)
        reward += i_reward
    print("Reward: ", reward)


side_length = 5
num_agents = 5

S = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        S[i, j] = 0.25

phi = 0.25

run_environment(random_policy)

run_environment(nearest)

