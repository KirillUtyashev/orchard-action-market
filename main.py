import os

from matplotlib import pyplot as plt
from orchard.environment import *
import numpy as np
import random
from orchard.algorithms import mean_distances
from policies.random_policy import random_policy
from metrics.metrics import PositionRecorder, append_positional_metrics, \
    append_y_coordinates, plot_agent_heatmap_alpha, plot_agent_specific_metrics, \
    plot_agents_trajectories

adv_plot = []


def step(agents_list, environment: Orchard, agent_controller, epsilon):
    agent = random.randint(0, environment.n - 1)
    state = environment.get_state()
    if agents_list[agent].policy is not random_policy:
        action = agent_controller.agent_get_action(environment, agent, epsilon)
    else:
        action = random_policy(environment.available_actions)
    action_result = environment.process_action(agent, agents_list[agent].position.copy(), action)
    agents_list[agent].position = action_result.new_position
    # global adv_plot
    return action_result


def add_distances(agent_i, agents_list):
    stack = []
    for id_, agent in enumerate(agents_list):
        distance = np.linalg.norm(agent.position - agents_list[agent_i].position)
        stack.append(distance)
    return stack


def plot_smoothed(series_list, labels=None, title="", xlabel="Step", ylabel="Value", num_points=40):
    """
    Plot one or more time series averaged into ~num_points bins and return the Figure.
    """
    if labels is None:
        labels = [f"Series {i}" for i in range(len(series_list))]

    # Guard: empty or length-0 series
    if not series_list or min(len(s) for s in series_list) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

    T = min(len(s) for s in series_list)
    win = max(1, T // max(1, num_points))
    nwin = max(1, T // win)  # ensure at least one bin
    x = (np.arange(nwin) * win + win / 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    for s, lab in zip(series_list, labels):
        s = np.asarray(s)[:nwin * win]
        s_avg = s.reshape(nwin, win).mean(axis=1)
        ax.plot(x, s_avg, label=lab)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_raw(series_list, labels=None, title="", xlabel="Step", ylabel="Value"):
    """
    Plot one or more raw time series without smoothing.

    series_list : list of 1D np.arrays (all of length T, or will be truncated to min length)
    labels      : list of labels for each series
    """
    if labels is None:
        labels = [f"Series {i}" for i in range(len(series_list))]

    # Truncate to same length
    T = min(len(s) for s in series_list)
    x = np.arange(T)

    plt.figure(figsize=(10, 4))
    for s, lab in zip(series_list, labels):
        s = np.asarray(s)[:T]
        plt.plot(x, s, label=lab)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.show()


def eval_performance(num_agents, agent_controller, env, name, timesteps=5000, agents_list=None, epsilon=0.1):
    agent_x_coordinates = [[] for _ in range(num_agents)]
    agent_y_coordinates = [[] for _ in range(num_agents)]
    agent_distance_hist = {i: np.zeros((0, num_agents), dtype=float) for i in range(num_agents)}
    alpha_ema = {i: np.zeros((0, num_agents), dtype=float) for i in range(num_agents)}
    personal_q_values = {i: [] for i in range(num_agents)}
    apple_x_coordinates = []
    apple_y_coordinates = []

    nearest_apple_actions = 0
    idle_actions = 0

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

    os.makedirs("positions", exist_ok=True)
    with PositionRecorder(num_agents, timesteps * num_agents + 1, f"positions/{name}_pos.npy") as rec:
        rec.log(agents_list)
        for i in range(timesteps):
            num_of_apples_per_second.append(env.apples.sum())
            before = env.total_apples
            after = env.total_apples
            apples_dropped.append(after - before)
            apples_per_second = 0
            for tick in range(num_agents):
                apples_before = env.get_sum_apples()
                action_result = step(agents_list, env, agent_controller, epsilon)
                reward += (apples_before - env.get_sum_apples())
                if tick == num_agents - 1:
                    env.apples_despawned += env.despawn_algorithm(env, env.despawn_rate)
                    env.total_apples += env.spawn_algorithm(env, env.spawn_rate)
                rec.log(agents_list)
                if name != "test":
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
                for num, agent in enumerate(agents_list):
                    pass
                    # stack = add_distances(num, agents_list)
                    # agent_distance_hist[num] = np.vstack([agent_distance_hist[num], stack])
                    # stack = np.asarray(agents_list[num].agent_alphas, dtype=float).reshape(1, -1)
                    # alpha_ema[num] = np.concatenate([alpha_ema[num], stack], axis=0)
                    # personal_q_values[num].append(agent.personal_q_value)

            # Calculate mean nearest neighbor distance for this timestep
            # for num, agent in enumerate(agents_list):
            #     # Update betas
            #     algo._record_rates(num, agent.agent_alphas)
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
    plot_agent_specific_metrics(agent_x_coordinates, apple_x_coordinates, name, "x")
    plot_agent_specific_metrics(agent_y_coordinates, apple_y_coordinates, name, "y")

    # global adv_plot
    # plt.plot(adv_plot)
    # plt.show()

    # for agent_id in range(num_agents):
    #     arr = agent_distance_hist[agent_id]  # shape [T, num_agents]
    #     series_list = [arr[:, j] for j in range(num_agents)]
    #     labels = [f"Distance to Agent {j}" for j in range(num_agents)]
    #     plot_smoothed(series_list, labels, title=f"Distances of Agent {agent_id}", xlabel="Training Step", ylabel="Distance")
    # for agent_id in range(num_agents):
    #     arr = alpha_ema[agent_id]  # shape [T, num_agents]
    #     series_list = [arr[:, j] for j in range(num_agents)]
    #     labels = [f"Q-value from Agent {j}" for j in range(num_agents)]
    #     fig = plot_smoothed(series_list, labels, title=f"Observed Q-values Over Time, Agent {agent_id}", xlabel="Training Step", ylabel="Q-value")
    #     fig.show()
    #
    def average_every_n(arr, n=500):
        """Average consecutive n points in 1D array."""
        length = len(arr) // n * n  # trim remainder
        arr = arr[:length]
        return arr.reshape(-1, n).mean(axis=1)

    # plt.figure()
    # for agent_id in range(num_agents):
    #     arr = personal_q_values[agent_id]
    #     smoothed = average_every_n(np.array(arr), n=150)
    #     x_axis = np.arange(len(smoothed)) * 50  # step numbers
    #     plt.plot(x_axis, smoothed, label=f"Agent {agent_id}")
    #
    # plt.xlabel("Step")
    # plt.ylabel("Q-value")
    # plt.legend()
    # plt.show()

    # positions = np.load(f"positions/{name}_pos.npy")      # shape (T, N, 2)
    # fig, ax = plt.subplots(figsize=(6, 5))
    # plot_agents_heatmap_alpha(
    #     positions,
    #     agent_ids=[0, 1, 2, 3],                      # pick any subset
    #     colors=["royalblue", "crimson", "gold", "purple"],  # one hue per agent
    #     ax=ax
    # )
    # plt.show()
    #
    # for agent_id in range(num_agents):
    #     plot_agent_heatmap_alpha(positions, agent_ids=[agent_id], colors=["red"])
    #     plt.show()
    #
    # for agent_id in range(num_agents):
    #     plot_agents_trajectories(positions, agent_ids=[agent_id], colors=["royalblue"])
    #     plt.show()

    return env.total_apples, reward, reward / num_agents, (reward / num_agents) / (env.total_apples / num_agents), np.mean(nearest_neighbour_mean_distance), np.mean(num_of_apples_per_second), nearest_apple_actions, idle_actions
