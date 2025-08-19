from pathlib import Path
from contextlib import AbstractContextManager

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

graph = 0


class PositionRecorder(AbstractContextManager):
    """
    Logs (x, y) grid positions for N agents at every tick and
    saves the result as a .npy file when you're done.

    Usage
    -----
    with PositionRecorder(num_agents=N, total_steps=T,
                          outfile="run1_pos.npy") as rec:
        for _ in range(T):
            env.step() # advance your sim
            rec.log(agents) # where `agents[i].position -> (x, y)`
    # file is saved automatically on exit
    """

    def __init__(self, num_agents: int, total_steps: int, outfile: str):
        self.num_agents = num_agents
        self.total_steps = total_steps
        self.outfile = Path(outfile)
        self._pos = np.empty((total_steps, num_agents, 2), dtype=np.int16)
        self._t = 0

    # -------- public API ----------------------------------------------------
    def log(self, agents):
        """
        Record positions of *all* agents for the current timestep.
        `agents` must be an iterable whose i-th element has `.position`
        returning (x, y) grid indices.
        """
        if self._t >= self.total_steps:
            raise IndexError("All timesteps already logged!")
        for i, agent in enumerate(agents):
            self._pos[self._t, i] = agent.position
        self._t += 1

    def save(self):
        """Write the tensor to disk immediately (optional)."""
        np.save(self.outfile, self._pos[:self._t])
        print(f"[PositionRecorder]   saved → {self.outfile} "
              f"shape={self._pos[:self._t].shape}")

    # -------- context-manager plumbing -------------------------------------
    def __exit__(self, exc_type, exc, tb):
        # Always save on exit, even if an exception occurred
        try:
            self.save()
        finally:
            return False            # propagate any exception


def plot_agents_heatmap_alpha(positions: np.ndarray,
                              agent_ids: list[int] | None = None,
                              colors: list[str] | None = None,
                              ax: plt.Axes | None = None,
                              grid_color: str = "white") -> plt.Axes:
    """
    Overlay single-colour α-heat-maps for several agents on one axes.

    Parameters
    ----------
    positions : (T, N, 2) ndarray
        Logged grid coordinates for all agents.
    agent_ids : list[int] | None
        Which agents to plot; default = *all*.
    colors     : list[str] | None
        One Matplotlib colour per agent (cycled if shorter).
        Default palette = tab10.
    ax        : matplotlib Axes | None
        Plot on this axes; create new fig/ax if None.
    grid_color : str
        Colour of grid lines.

    Returns
    -------
    ax : matplotlib Axes
        The axes the heat-maps were drawn on.
    """
    if agent_ids is None:
        agent_ids = list(range(positions.shape[1]))

    if colors is None:
        # repeat a default palette if fewer colours than agents
        base = plt.cm.get_cmap("tab10").colors
        colors = [base[i % len(base)] for i in range(len(agent_ids))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))

    # assume square grid; infer side length
    L = int(positions[..., :2].max()) + 1

    # draw one RGBA layer per agent
    for agent_idx, colour in zip(agent_ids, colors):
        xy = positions[:, agent_idx]                 # (T, 2)
        counts = np.zeros((L, L), dtype=np.int32)
        xs, ys = xy[:, 0], xy[:, 1]
        np.add.at(counts, (ys, xs), 1)

        alpha = counts / counts.max() if counts.max() else counts
        r, g, b, _ = to_rgba(colour)
        rgba = np.zeros((L, L, 4), dtype=float)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
        rgba[..., 3] = alpha

        ax.imshow(rgba, origin="upper", extent=[0, L, 0, L])

    # cosmetics (done once)
    ax.set_xticks(range(L + 1))
    ax.set_yticks(range(L + 1))
    ax.grid(which="both", color=grid_color, lw=0.5)
    ax.set_aspect("equal")
    ax.set_title("Occupancy (α ∝ visits)")
    plt.tight_layout()
    return ax


def plot_agents_trajectories(positions: np.ndarray,
                             agent_ids: list[int] | None = None,
                             colors: list[str] | None = None,
                             ax: plt.Axes | None = None,
                             linewidth: float = 2,
                             alpha: float = 0.9,
                             show_markers: bool = True) -> plt.Axes:
    """
    Overlay plain trajectories (no time-colour) for one or more agents.

    Parameters
    ----------
    positions : ndarray, shape (T, N, 2)
        Logged (x, y) integer grid positions for all agents.
    agent_ids : list[int] | None
        Agents to plot; default = all.
    colors : list[str] | None
        Base colours cycled if shorter than agent_ids; default = tab10.
    ax : matplotlib Axes | None
        Pass an existing Axes to draw on; if None, a new fig/ax is created.
    linewidth : float
        Width of the trajectory lines.
    alpha : float
        Global opacity for all trajectories.
    show_markers : bool
        If True, draw start (○) and end (✕) markers.

    Returns
    -------
    ax : matplotlib Axes
        The axes on which the trajectories were drawn.
    """
    if agent_ids is None:
        agent_ids = list(range(positions.shape[1]))        # all agents

    if colors is None:
        palette = plt.cm.get_cmap("tab10").colors
        colors  = [palette[i % len(palette)] for i in range(len(agent_ids))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # infer square grid size from max coordinate
    L = int(positions[..., :2].max()) + 1

    for aid, colour in zip(agent_ids, colors):
        xy = positions[:, aid] + 0.5           # cell centres
        xs, ys = xy[:, 0], xy[:, 1]

        ax.plot(xs, ys, color=colour,
                lw=linewidth, alpha=alpha,
                label=f"agent {aid}")

        if show_markers:
            ax.scatter(xs[0],  ys[0],  s=60, c=colour,
                       marker="o", edgecolors="k", linewidths=.5)
            ax.scatter(xs[-1], ys[-1], s=80, c=colour,
                       marker="X", edgecolors="k", linewidths=.5)

    # cosmetics
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_xticks(range(L + 1)); ax.set_yticks(range(L + 1))
    ax.grid(which="both", color="gray", lw=.4, alpha=.3)
    ax.set_aspect("equal")
    ax.set_title("Agent trajectories (start ○  end ✕)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    return ax


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


def append_y_coordinates(agent_y_coordinates, agents_list):
    for i, metric in enumerate(agent_y_coordinates):
        agent_y_coordinates[i].append(agents_list[i].position[0])

    return agent_y_coordinates

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


def plot_agent_specific_metrics(agent_metrics, apples, experiment, name, vector):
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
    global graph
    graph += 1
    graph_folder = Path(f"graphs/{name}")
    graph_folder.mkdir(parents=True, exist_ok=True)
    fig_id = f"{experiment}_{name}_{str(graph)}"
    plt.figure(fig_id, figsize=(6, 5))

    # --- agents: keep as connected lines ------------------------------
    for i, series in enumerate(agent_metrics):
        series1 = series[8000:10000]        # or whatever window you like
        plt.plot(series1, label=f"agent {i}")

    check = apples[8000:10000]

    xs, ys = [], []

    for t, vals in enumerate(check):
        xs.extend([t] * len(vals))   # repeat the timestep for *each* value
        ys.extend(vals)              # append every number in the list


    # --- apples: unconnected scatter ---------------------------------
    plt.scatter(xs, ys, c=ys, cmap='tab10', s=30, alpha=0.1, edgecolors='none')

    # -----------------------------------------------------------------
    if vector == "x":
        plt.title(f"Agent X-Axis Coordinates")
    else:
        plt.title(f"Agent Y-Axis Coordinates")
    plt.legend()
    filename = graph_folder / f"graph_{name}_distances_{str(graph)}_{vector}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close("all")
