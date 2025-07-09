from matplotlib import pyplot as plt

from agents.communicating_agent import CommAgent
from agents.simple_agent import SimpleAgent
from orchard.environment import *
import numpy as np
import random
from policies.nearest_uniform import replace_agents_1d
from orchard.algorithms import mean_distances
from policies.random_policy import random_policy
from metrics.metrics import append_metrics, append_positional_metrics, plot_agent_specific_metrics
from value_function_learning.controllers import AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController

same_actions = 0


def step(agents_list, environment: Orchard, agent_controller):
    agent = random.randint(0, environment.n - 1)
    state = environment.get_state()
    if agents_list[agent].policy == "value_function":
        action = agent_controller.get_best_action(state, agent, environment.available_actions)
    elif agents_list[agent].policy is random_policy:
        action = agents_list[agent].policy(environment.available_actions)
    else:
        action = agents_list[agent].policy(state, agents_list[agent].position)
    reward, new_position = environment.main_step(agents_list[agent].position.copy(), action)
    agents_list[agent].position = new_position
    # print(new_position)
    return agent, reward


def run_environment_1d_acting_rate(num_agents, policy, side_length, S, phi, name="Default", experiment="Default", timesteps=5000, agents_list=None, action_algo=None, spawn_algo=None, despawn_algo=None):
    metrics = []
    agent_metrics = []
    for j in range(5):
        metrics.append([])
    for j in range(num_agents):
        agent_metrics.append([])

    # if agents_list is None:
    #     agents_list = []
    #     for _ in range(num_agents):
    #         agents_list.append(Agent(policy=policy))

    env = Orchard(side_length, num_agents, S, phi, agents_list=agents_list, action_algo=action_algo, spawn_algo=spawn_algo, despawn_algo=despawn_algo)
    env.initialize(agents_list) #, agent_pos=[np.array([1, 0]), np.array([3, 0])]) #, agent_pos=[np.array([2, 0]), np.array([5, 0]), np.array([8, 0])])
    reward = 0
    for i in range(timesteps):
        agent = i % num_agents  # random.randint(0, env.n - 1)

        state = env.get_state()
        # state["agents"][agents_list[0].position[0]] -= 1
        action = agents_list[agent].get_action(state, agents_list=agents_list)
        val = random.random()
        if val < agents_list[agent].acting_rate:
            i_reward, new_position = env.main_step(agents_list[agent].position.copy(), action)
            acted = True
        else:
            i_reward, new_position = env.main_step_without_action(agents_list[agent].position.copy())
        agents_list[agent].position = new_position
        # print(new_position)
        reward += i_reward
        if name != "test" and experiment != "test":
            metrics = append_metrics(metrics, env.get_state(), reward, i)
            agent_metrics = append_positional_metrics(agent_metrics, agents_list)
        new_state = env.get_state()
        # if np.sum(apples) == 0:
        #     assert np.array_equal(old_state["agents"], new_state["agents"])
        # else:
        #     assert not np.array_equal(old_state["agents"], new_state["agents"]) or not np.array_equal(old_state["apples"], new_state["apples"])
    print("Results for", name)
    print("Reward: ", reward)
    print("Total Apples: ", env.total_apples)
    print("Average Reward: ", reward / timesteps)
    print("Apple Ratio: ", reward / env.total_apples)
    if name != "test" and experiment != "test":
        plot_agent_specific_metrics(agent_metrics, experiment, name)
        # plot_metrics(metrics, name, experiment)
    return reward


def run_environment_1d(num_agents, side_length, width, S, phi, name="Default", experiment="Default", timesteps=5000, agents_list=None, action_algo=None, spawn_algo=None, despawn_algo=None, vision=None, s_target=0.1, apple_mean_lifetime=0.35):
    metrics = []
    agent_metrics = []
    apple_metrics = []
    for j in range(5):
        metrics.append([])
    for j in range(num_agents):
        agent_metrics.append([])
    env = Orchard(side_length, width, num_agents, agents_list=agents_list, action_algo=action_algo, spawn_algo=spawn_algo, despawn_algo=despawn_algo, s_target=s_target, apple_mean_lifetime=apple_mean_lifetime)
    env.initialize(agents_list) #, agent_pos=[np.array([1, 0]), np.array([3, 0])]) #, agent_pos=[np.array([2, 0]), np.array([5, 0]), np.array([8, 0])])
    reward = 0
    apples_picked = []
    apples_dropped = []

    # Calculate mean distance between agents and their nearest neighbors at each timestep
    nearest_neighbour_mean_distance = []

    num_of_apples_per_second = []

    # Function to compute distance between two points
    def distance(pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    # Function to find nearest neighbor distance for one agent
    def get_nearest_neighbor_distance(agent_idx, agents):
        current_pos = agents[agent_idx].position
        distances = []
        for i, other_agent in enumerate(agents):
            if i != agent_idx:
                distances.append(distance(current_pos, other_agent.position))
        return min(distances) if distances else float('inf')
    if type(agents_list[0]) is CommAgent:
        agent_controller = AgentControllerDecentralized(agents_list, ViewController(vision))
    else:
        agent_controller = AgentControllerCentralized(agents_list, ViewController(vision))
    for i in range(timesteps):
        num_of_apples_per_second.append(env.apples.sum())
        before = env.total_apples
        after = env.total_apples
        apples_dropped.append(after - before)
        apples_per_second = 0
        for tick in range(num_agents):
            agent, i_reward = step(agents_list, env, agent_controller)
            reward += i_reward
            apples_per_second += i_reward
            if tick == num_agents - 1:
                env.apples_despawned += env.despawn_algorithm(env, env.despawn_rate)
                env.total_apples += env.spawn_algorithm(env, env.spawn_rate)
                # env.apples_despawned += env.despawn_algorithm(env, i)
                # env.total_apples += env.spawn_algorithm(env, i)
            if name != "test" and experiment != "test":
                agent_metrics = append_positional_metrics(agent_metrics, agents_list)
                to_add = []
                for j in range(env.apples.shape[1]):
                    if env.apples[0, j] != 0:
                        to_add.append(j)
                apple_metrics.append(to_add)
            apples_picked.append(apples_per_second)
        # Calculate mean nearest neighbor distance for this timestep
        timestep_distances = []
        for agent_idx in range(len(agents_list)):
            nearest_dist = get_nearest_neighbor_distance(agent_idx, agents_list)
            timestep_distances.append(nearest_dist)
        nearest_neighbour_mean_distance.append(np.mean(timestep_distances))
        if i % 1000 == 0:
            print(i)
    # raw per-tick log (0,1,2… apples picked that tick)
    rate_per_tick = np.asarray(apples_picked) / num_agents
    cum_mean = np.cumsum(rate_per_tick) / np.arange(1, len(rate_per_tick)+1)
    print(cum_mean[-1])

    sec_axis = np.arange(len(cum_mean))
    plt.plot(sec_axis, cum_mean, linewidth=1.5)

    apples_per_tick = np.asarray(apples_dropped) / num_agents
    cum_mean = np.cumsum(apples_per_tick) / np.arange(1, len(apples_per_tick)+1)
    sec_axis = np.arange(len(cum_mean))   # seconds
    plt.plot(sec_axis, cum_mean, linewidth=1.5)
    plt.ylim(0, 0.5)

    plt.xlabel('simulated seconds')
    plt.ylabel('cumulative mean  apples/agent/sec')
    plt.tight_layout(); plt.show(); plt.close()
    print("Average number of apples per second: ", np.mean(num_of_apples_per_second))
    print("Average distance:", np.mean(nearest_neighbour_mean_distance))
    print("Results for", name)
    print("Reward: ", reward)
    print("Average distance from spawned apple to nearest agent:", np.mean(mean_distances))
    print("Total Apples: ", env.total_apples)
    print("Apples per agent:", reward / num_agents)
    print("Average Reward: ", reward / env.total_apples)
    print("Picked vs Spawned per agent", (reward / num_agents) / (env.total_apples / num_agents))
    if name != "test" and experiment != "test":
        plot_agent_specific_metrics(agent_metrics, apple_metrics, experiment, name)
        # plot_metrics(metrics, name, experiment)

    return env.total_apples, reward, reward / num_agents, (reward / num_agents) / (env.total_apples / num_agents), np.mean(nearest_neighbour_mean_distance), np.mean(num_of_apples_per_second)


def all_three_1d(num_agents, length, S, phi, experiment, time=5000):
    run_environment_1d(num_agents, nearest, length, S, phi, "Nearest", experiment, time)
    run_environment_1d(num_agents, nearest, length, S, phi, "Nearest-Uniform", experiment, time, action_algo=replace_agents_1d)
    run_environment_1d(num_agents, random_policy, length, S, phi, "Random", experiment, time)


if __name__ == "__main__":
    side_length = 40
    num_agents = int(side_length * 0.2)

    from orchard.algorithms import single_apple_spawn, single_apple_despawn
    #
    agents_list = []
    for i in range(num_agents):
        main_int = 1
        agents_list.append(
            SimpleAgent(policy=random_policy_1d, debug=False, num=num_agents))
    # run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest", "Single_Apple", spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    # run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest-Uniform", "Single_Apple", action_algo=replace_agents_1d, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    run_environment_1d(num_agents, random_policy_1d, side_length, None, None, "Random", "Single_Apple", spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=40000, agents_list=agents_list)
