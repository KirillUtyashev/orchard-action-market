from matplotlib import pyplot as plt

from agents.communicating_agent import CommAgent
from agents.simple_agent import SimpleAgent
from orchard.environment import *
import numpy as np
import random

from policies.nearest import nearest_policy
from policies.nearest_uniform import replace_agents_1d
from orchard.algorithms import mean_distances
from policies.random_policy import random_policy
from metrics.metrics import append_metrics, append_positional_metrics, \
    append_y_coordinates, plot_agent_specific_metrics
from value_function_learning.controllers import AgentControllerActorCritic, \
    AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController


def step(agents_list, environment: Orchard, agent_controller, epsilon):
    agent = random.randint(0, environment.n - 1)
    state = environment.get_state()
    if agents_list[agent].policy == "value_function":
        if random.random() < epsilon:
            action = random_policy(environment.available_actions)
        else:
            action = agent_controller.get_best_action(state, agent, environment.available_actions)
    elif agents_list[agent].policy == "learned_policy":
        action = agent_controller.get_best_action(state, agent, environment.available_actions)
    elif agents_list[agent].policy is random_policy:
        action = agents_list[agent].policy(environment.available_actions)
    else:
        action = agents_list[agent].policy(state, agents_list[agent].position)
    reward, new_position = environment.main_step(agents_list[agent].position.copy(), action)
    agents_list[agent].position = new_position
    # print(new_position)
    return agent, reward, 1 if action == nearest_policy(state, agents_list[agent].position) else 0, 1 if action == 2 else 0


def run_environment_1d(num_agents, side_length, width, S, phi, name="Default", experiment="Default", timesteps=5000, agents_list=None, action_algo=None, spawn_algo=None, despawn_algo=None, vision=None, s_target=0.1, apple_mean_lifetime=0.35, epsilon=0.1):
    metrics = []
    agent_x_coordinates = [[] for _ in range(num_agents)]
    agent_y_coordinates = [[] for _ in range(num_agents)]
    apple_x_coordinates = []
    apple_y_coordinates = []

    nearest_apple_actions = 0
    idle_actions = 0

    for j in range(5):
        metrics.append([])
    env = Orchard(side_length, width, num_agents, agents_list=agents_list, action_algo=action_algo, spawn_algo=spawn_algo, despawn_algo=despawn_algo, s_target=s_target, apple_mean_lifetime=apple_mean_lifetime)
    env.initialize(agents_list)
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
    elif type(agents_list[0]) is SimpleAgent:
        agent_controller = AgentControllerCentralized(agents_list, ViewController(vision))
    else:
        agent_controller = AgentControllerActorCritic(agents_list, ViewController(vision))
    for i in range(timesteps):
        num_of_apples_per_second.append(env.apples.sum())
        before = env.total_apples
        after = env.total_apples
        apples_dropped.append(after - before)
        apples_per_second = 0
        for tick in range(num_agents):
            agent, i_reward, same_action, idle = step(agents_list, env, agent_controller, epsilon)
            nearest_apple_actions += same_action
            idle_actions += idle
            reward += i_reward
            apples_per_second += i_reward
            if tick == num_agents - 1:
                env.apples_despawned += env.despawn_algorithm(env, env.despawn_rate)
                env.total_apples += env.spawn_algorithm(env, env.spawn_rate)
            if name != "test" and experiment != "test":
                agent_x_coordinates = append_positional_metrics(agent_x_coordinates, agents_list)
                agent_y_coordinates = append_y_coordinates(agent_y_coordinates, agents_list)
                to_add_x = []
                to_add_y = []
                for k in range(env.apples.shape[0]):
                    for j in range(env.apples.shape[1]):
                        if env.apples[k, j] != 0:
                            to_add_x.append(j)
                            to_add_y.append(k)
                apple_x_coordinates.append(to_add_x)
                apple_y_coordinates.append(to_add_y)
            apples_picked.append(apples_per_second)
        # Calculate mean nearest neighbor distance for this timestep
        timestep_distances = []
        for agent_idx in range(len(agents_list)):
            nearest_dist = get_nearest_neighbor_distance(agent_idx, agents_list)
            timestep_distances.append(nearest_dist)
        nearest_neighbour_mean_distance.append(np.mean(timestep_distances))
        if i % 1000 == 0:
            print(i)
    print("Average number of apples per second: ", np.mean(num_of_apples_per_second))
    print("Average distance:", np.mean(nearest_neighbour_mean_distance))
    print("Number of nearest actions: ", nearest_apple_actions)
    print("Number of idle actions: ", idle_actions)
    print("Results for", name)
    print("Reward: ", reward)
    print("Average distance from spawned apple to nearest agent:", np.mean(mean_distances))
    print("Total Apples: ", env.total_apples)
    print("Apples per agent:", reward / num_agents)
    print("Average Reward: ", reward / env.total_apples)
    print("Picked vs Spawned per agent", (reward / num_agents) / (env.total_apples / num_agents))
    plot_agent_specific_metrics(agent_x_coordinates, apple_x_coordinates, experiment, name, "x")
    plot_agent_specific_metrics(agent_y_coordinates, apple_y_coordinates, experiment, name, "y")

    return env.total_apples, reward, reward / num_agents, (reward / num_agents) / (env.total_apples / num_agents), np.mean(nearest_neighbour_mean_distance), np.mean(num_of_apples_per_second), nearest_apple_actions, idle_actions
