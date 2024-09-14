from orchard.environment import *
import numpy as np
import matplotlib.pyplot as plt
import orchard.environment_one
import random
from policies.nearest_uniform import replace_agents_1d
from policies.random_policy import random_policy_1d, random_policy
from policies.nearest import nearest_1d, nearest
from metrics.metrics import append_metrics, plot_metrics, append_positional_metrics, plot_agent_specific_metrics
from agents.simple_agent import SimpleAgent as Agent


same_actions = 0

def step(agents_list, environment: Orchard):
    agent = random.randint(0, environment.n-1)

    state = environment.get_state()
    action = agents_list[agent].get_action(state, agents_list=agents_list)
    global same_actions
    same_actions += (action == nearest(state, agents_list[agent].position))
    reward, new_position = environment.main_step(agents_list[agent].position.copy(), action)
    agents_list[agent].position = new_position
    # print(new_position)
    return agent, reward

def run_environment(policy):
    """
    Not usable - broken as of right now.
    """
    env = Orchard(side_length, num_agents, S, phi)
    env.initialize()
    reward = 0
    for i in range(20000):
        agent, i_reward = step(env, policy)
        reward += i_reward
    print("Reward: ", reward)
    print("Total Apples: ", env.total_apples)

def run_environment_1d(num_agents, policy, side_length, S, phi, name="Default", experiment="Default", timesteps=5000, agents_list=None, action_algo=None, spawn_algo=None, despawn_algo=None):
    metrics = []
    agent_metrics = []
    for j in range(5):
        metrics.append([])
    for j in range(num_agents):
        agent_metrics.append([])

    if agents_list is None:
        agents_list = []
        for _ in range(num_agents):
            agents_list.append(Agent(policy=policy))

    env = Orchard(side_length, num_agents, S, phi, agents_list=agents_list, one=True, action_algo=action_algo, spawn_algo=spawn_algo, despawn_algo=despawn_algo)
    env.initialize(agents_list) #, agent_pos=[np.array([1, 0]), np.array([3, 0])]) #, agent_pos=[np.array([2, 0]), np.array([5, 0]), np.array([8, 0])])
    reward = 0
    for i in range(timesteps):
        old_state = env.get_state()
        apples = old_state["apples"]
        agent, i_reward = step(agents_list, env)
        reward += i_reward
        if name != "test" and experiment != "test":
            metrics = append_metrics(metrics, env.get_state_only(), reward, i)
            agent_metrics = append_positional_metrics(agent_metrics, agents_list)
        new_state = env.get_state()
        # if np.sum(apples) == 0:
        #     assert np.array_equal(old_state["agents"], new_state["agents"])
        # else:
        #     assert not np.array_equal(old_state["agents"], new_state["agents"]) or not np.array_equal(old_state["apples"], new_state["apples"])

    if name != "test" and experiment != "test":
        print("Same Actions:", same_actions)
    print("Results for", name)
    print("Reward: ", reward)
    print("Total Apples: ", env.total_apples)
    print("Average Reward: ", reward / timesteps)
    if name != "test" and experiment != "test":
        plot_agent_specific_metrics(agent_metrics, experiment, name)
        plot_metrics(metrics, name, experiment)
    return reward


def all_three_1d(num_agents, length, S, phi, experiment, time=5000):
    run_environment_1d(num_agents, nearest, length, S, phi, "Nearest", experiment, time)
    run_environment_1d(num_agents, nearest, length, S, phi, "Nearest-Uniform", experiment, time, action_algo=replace_agents_1d)
    run_environment_1d(num_agents, random_policy_1d, length, S, phi, "Random", experiment, time)


if __name__ == "__main__":
    side_length = 20
    num_agents = 5

    """
    Different Possible S values
    """
    S = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            S[i, j] = 0.05

    S2 = np.zeros((side_length, 1))
    for i in range(side_length):
        S2[i] = 0.05

    S3 = np.zeros((side_length, 1))
    S3[0] = 0
    S3[side_length - 1] = 0

    phi = 0.05




    run_environment_1d(num_agents, nearest, side_length, S2, phi, "Nearest", "Uniform")
    run_environment_1d(num_agents, nearest, side_length, S2, phi, "Nearest-Uniform", "Uniform", action_algo=replace_agents_1d)
    run_environment_1d(num_agents, random_policy_1d, side_length, S2, phi, "Random", "Uniform")

    all_three_1d(num_agents, side_length, S3, phi, "No_Apples")

    S3[0] = 0.5
    phi = 0.1

    all_three_1d(num_agents, side_length, S3, phi, "Corner_Single")

    S3[0] = 0.25
    S3[side_length - 1] = 0.25

    all_three_1d(num_agents, side_length, S3, phi, "Corner_Double")

    S3 = np.zeros((side_length, 1))
    from orchard.algorithms import single_apple_spawn, single_apple_despawn, single_apple_spawn_malicious
    #
    run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest", "Single_Apple", spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest-Uniform", "Single_Apple", action_algo=replace_agents_1d, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    run_environment_1d(num_agents, random_policy_1d, side_length, S3, phi, "Random", "Single_Apple", spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)

    run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest", "SAM", spawn_algo=single_apple_spawn_malicious, despawn_algo=single_apple_despawn)
    run_environment_1d(num_agents, nearest,  side_length, S3, phi, "Nearest-Uniform", "SAM", action_algo=replace_agents_1d, spawn_algo=single_apple_spawn_malicious, despawn_algo=single_apple_despawn)
    run_environment_1d(num_agents, random_policy_1d,  side_length, S3, phi, "Random", "SAM", spawn_algo=single_apple_spawn_malicious, despawn_algo=single_apple_despawn)



