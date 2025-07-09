from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def avg_apples_picked(state, total_reward, timestep):
    return total_reward / timestep


def avg_apples_on_field(state, total_reward, timestep):
    agents = state["agents"]
    apples = state["apples"]
    total_apples = np.sum(apples.flatten())
    return total_apples / agents.size


def smallest_distance_helper(mat, agent_pos):
    """
    A line-by-line copy from the first part of the move-to-nearest algorithm, but with a


    :param mat: a matrix, of either apple or agent spots
    :param agent_pos:
    :return:
    """
    agent_pos = np.array(agent_pos)
    target = (-1, -1)
    smallest_distance = np.linalg.norm(mat.size)
    for point, count in np.ndenumerate(mat):
        if np.linalg.norm(point - agent_pos) < smallest_distance and count > 0:
            target = point
            smallest_distance = np.linalg.norm(point - agent_pos)
    return smallest_distance


def distance_to_nearest_apple(state, total_reward, timestep):
    agents = state["agents"]
    dist_list = []
    if np.sum(state["apples"].flatten()) == 0:
        return -1
    for i in range(agents.shape[0]):
        for j in range(agents.shape[1]):
            for k in range(agents[i, j]):
                dist_list.append(smallest_distance_helper(state["apples"], (i, j)))
    dist_list = np.array(dist_list)
    return np.sum(dist_list) / dist_list.size


def average_agent_distance(state, total_reward, timestep):
    agents = state["agents"]
    dist_list = []
    for i in range(agents.shape[0]):
        for j in range(agents.shape[1]):
            for k in range(agents[i, j]):
                agents[i, j] -= 1
                dist_list.append(smallest_distance_helper(agents, (i, j)))
                agents[i, j] += 1
    dist_list = np.array(dist_list)
    return np.sum(dist_list) / dist_list.size


def average_agent_x(state, total_reward, timestep):
    agents = state["agents"]
    dist_list = []
    for i in range(agents.shape[0]):
        for j in range(agents.shape[1]):
            for k in range(agents[i, j]):
                dist_list.append(i)
    dist_list = np.array(dist_list)
    return np.sum(dist_list) / dist_list.size


def append_metrics(metrics, state, total_reward, timestep, avg_picked=True, avg_field=True, dist_to_apple=True,
                 dist_to_agent=True, avg_x=True):
    if timestep == 0:
        return metrics
    if avg_picked:
        metrics[0].append(avg_apples_picked(state, total_reward, timestep))
    if avg_field:
        metrics[1].append(avg_apples_on_field(state, total_reward, timestep))
    if dist_to_apple:
        metrics[2].append(distance_to_nearest_apple(state, total_reward, timestep))
    if dist_to_agent:
        metrics[3].append(average_agent_distance(state, total_reward, timestep))
    if avg_x:
        metrics[4].append(average_agent_x(state, total_reward, timestep))

    return metrics


def append_positional_metrics(agent_metrics, agents_list):
    for i, metric in enumerate(agent_metrics):
        agent_metrics[i].append(agents_list[i].position[1])

    return agent_metrics


metric_titles = [
    "Average Apples Picked per Timestep",
    "Average Apples on Field",
    "Average Distance to Nearest Apple",
    "Average Distance to Nearest Agent",
    "Average X of Agents"
]


def plot_metrics(metrics, name, experiment):

    graph_folder = Path("graphs")
    graph_folder.mkdir(parents=True, exist_ok=True)

    for i, series in enumerate(metrics):
        plt.figure(str(i) + str(experiment), figsize=(10, 5))
        if metric_titles[i] == "Average Distance to Nearest Apple":
            series = series[2000:3000].copy()
        else:
            series = series[2:].copy()
        plt.plot(series, label=name)
        plt.legend()
        plt.title(metric_titles[i])
        filename = graph_folder / f"graph_{experiment}_{str(i)}.png"
        plt.savefig(str(filename))


def plot_agent_specific_metrics(agent_metrics, apples, experiment, name):
    """Plot per-agent x-coords and overlay apples as a scatter cloud.

    Parameters
    ----------
    agent_metrics : list[list[float]]
        Outer list = agents.  Inner list = episode values of the metric.
    apples : list[float] or np.ndarray
        Same episode index as agent_metrics.  One value per episode.
    experiment : str
        Experiment name for the figure handle / file name.
    name : str
        Extra tag for the plot title and file name.
    """
    graph_folder = Path("graphs")
    graph_folder.mkdir(parents=True, exist_ok=True)
    fig_id = f"{experiment}_{name}"
    plt.figure(fig_id, figsize=(6, 5))

    # --- agents: keep as connected lines ------------------------------
    for i, series in enumerate(agent_metrics):
        series1 = series[1:1000]        # or whatever window you like
        plt.plot(series1, label=f"agent {i}")

    check = apples[1:1000]

    xs, ys = [], []

    for t, vals in enumerate(check):
        xs.extend([t] * len(vals))   # repeat the timestep for *each* value
        ys.extend(vals)              # append every number in the list


    # --- apples: unconnected scatter ---------------------------------
    plt.scatter(xs, ys, c=ys, cmap='tab10', s=30, alpha=0.8, edgecolors='none')

    # -----------------------------------------------------------------
    plt.title(f"Agent X-Axis Coordinates Under Policy {name}")
    plt.legend()
    filename = graph_folder / f"graph_{name}_{experiment}_distances.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig_id)
