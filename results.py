from evaluate_policies import evaluate_factory
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path


def plot_connected(x, y, label, label_, **plot_kwargs):
    valid = ~np.isnan(y)
    idx = np.where(valid)[0]

    # Draw only the line segments (no label here)
    for i, j in zip(idx, idx[1:]):
        plt.plot(
            x[[i, j]],
            y[[i, j]],
            linestyle=plot_kwargs.get("linestyle", "-"),
            color=plot_kwargs.get("color", None),
            marker="",      # no marker on the segments
            # note: no label
        )

    # Draw the markers once, with the label
    plt.scatter(
        x[valid],
        y[valid],
        label=label,
        marker=plot_kwargs.get("marker", "o"),
        color=plot_kwargs.get("color", None),
    )
    plt.xlabel(label_)


def interpolate_nans(x, y):
    """
    Linearly interpolates over NaNs in the y array.
    """
    x = np.array(x)
    y = np.array(y, dtype=np.float64)
    mask = np.isnan(y)
    if mask.all():
        return y  # All values are NaN, nothing to interpolate
    y[mask] = np.interp(x[mask], x[~mask], y[~mask])
    return y


def plot_apple_picking(widths, series_dict, title, label_x, label_y):
    """
    Plots apple-picking ratios for different approaches.
    NaNs in data are interpolated to connect lines.
    """
    plt.figure()
    for label, ratios in series_dict.items():
        y_interp = interpolate_nans(widths, ratios)
        if 'Centralized' == label:
            color = 'blue'
        elif 'Decentralized' == label:
            color = 'orange'
        elif 'Random' == label:
            color = 'red'
        else:
            color = None
        plt.plot(widths, y_interp, marker='o', label=label, color=color)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.legend()
    plt.xticks(widths)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def parse_log_metrics(architecture: str, num_agents: int, length: int, width: int, hidden_dimensions: int, dimensions: int) -> tuple | None:
    """
    Parse the log file for a specific experimental setup and return:
    (last_ratio_picked, mean_distance, total_apples)

    Parameters:
        architecture (str): "Centralized" or "Decentralized"
        num_agents (int): number of agents
        length (int)
        width (int)
        hidden_dimensions (int)
        dimensions (int)

    Returns:
        tuple: (last_ratio_picked, mean_distance, total_apples) or None if not found
    """

    log_dir = Path("train_scripts/logs")

    if architecture == "Centralized":
        pattern = (
            f"Centralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<.*?>"
            f"-discount-<.*?>-hidden_dimensions-<{hidden_dimensions}>-dimensions-<{dimensions}>.log"
        )
    elif architecture == "Decentralized":
        pattern = (
            f"Decentralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{hidden_dimensions}>-<{dimensions}>.log"
        )
    else:
        raise ValueError("Architecture must be either 'Centralized' or 'Decentralized'")

    for log_file in log_dir.glob("*.log"):
        if re.fullmatch(pattern, log_file.name):
            # Found the matching file
            last_ratio = None
            mean_distance = None
            total_apples = None
            picked_per_agents = None
            total_picked = None

            with open(log_file, "r") as f:
                for line in f:
                    if "Ratio picked" in line:
                        match = re.search(r"Ratio picked: ([0-9.eE+-]+)", line)
                        if match:
                            last_ratio = float(match.group(1))
                    elif "Mean distance" in line:
                        match = re.search(r"Mean distance: ([0-9.eE+-]+)", line)
                        if match:
                            mean_distance = float(match.group(1))
                    elif "Total apples" in line:
                        match = re.search(r"Total apples: ([0-9.eE+-]+)", line)
                        if match:
                            total_apples = float(match.group(1))
                    elif "Picked per agents" in line:
                        match = re.search(r"Picked per agents: ([0-9.eE+-]+)", line)
                        if match:
                            picked_per_agents = float(match.group(1))
                    elif "Total picked" in line:
                        match = re.search(r"Total picked: ([0-9.eE+-]+)", line)
                        if match:
                            total_picked = float(match.group(1))

            return last_ratio, mean_distance, total_apples, picked_per_agents, total_picked

    return None  # if no file matched


def sweep_logs(base_config: dict, sweep_params: dict):
    """
    Sweeps over experiment configs and returns separate metric lists per architecture.

    Parameters:
        base_config (dict): Fixed config values.
        sweep_params (dict): Dict of parameter(s) to sweep, e.g. {"hidden_dimensions": [16, 32, 64]}

    Returns:
        dict: {
            "centralized": {
                "sweep_values": [...],
                "mean_distances": [...],
                "total_apples": [...],
                "last_ratios": [...]
            },
            "decentralized": {
                ...
            }
        }
    """
    results = {
        "centralized": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": []
        },
        "decentralized": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": []
        }
    }

    # Identify which parameter is being swept (we assume 1 for simplicity)
    assert len(sweep_params) == 1, "Only one sweep parameter is supported for now"
    sweep_key = list(sweep_params.keys())[0]
    sweep_values = sweep_params[sweep_key]

    for val in sweep_values:
        config = {**base_config, sweep_key: val}

        for arch in ["Centralized", "Decentralized"]:
            result = parse_log_metrics(
                architecture=arch,
                num_agents=config["num_agents"],
                length=config["length"],
                width=config["width"],
                hidden_dimensions=config["hidden_dimensions"],
                dimensions=config["dimensions"]
            )

            arch_key = "centralized" if arch == "Centralized" else "decentralized"
            results[arch_key]["sweep_values"].append(val)
            results[arch_key]["last_ratios"].append(result[0] if result else None)
            results[arch_key]["mean_distances"].append(result[1] if result else None)
            results[arch_key]["total_apples"].append(result[2] if result else None)
            results[arch_key]["picked_per_agent"].append(result[3] if result else None)
            results[arch_key]["total_picked"].append(result[4] if result else None)

    return results


def init_dicts():
    return {
        "Decentralized": [],
        "Centralized": [],
        "Random": []
    }


def run(base_config: dict, sweep_params: dict):
    # init the dictionaries
    ratios = init_dicts()
    total_apples = init_dicts()
    mean_distances = init_dicts()
    total_picked = init_dicts()
    picked_per_agents = init_dicts()

    # first, get the results for random
    if "hidden_dimensions" in sweep_params or "dimensions" in sweep_params:
        random_res = evaluate_factory(base_config["length"], base_config["width"], base_config["num_agents"])
        for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
            ratios["Random"].append(random_res["ratio_per_agent"])
            total_apples["Random"].append(random_res["total_apples"])
            mean_distances["Random"].append(random_res["mean_distance"])
            picked_per_agents["Random"].append(random_res["picked_per_agent"])
            total_picked["Random"].append(random_res["total_picked"])
    else:
        for width in sweep_params["width"]:
            random_res = evaluate_factory(base_config["length"], width, base_config["num_agents"])
            ratios["Random"].append(random_res["ratio_per_agent"])
            total_apples["Random"].append(random_res["total_apples"])
            mean_distances["Random"].append(random_res["mean_distance"])
            picked_per_agents["Random"].append(random_res["picked_per_agent"])
            total_picked["Random"].append(random_res["total_picked"])
    result = sweep_logs(base_config, sweep_params)
    ratios["Centralized"].extend(result["centralized"]["last_ratios"])
    ratios["Decentralized"].extend(result["decentralized"]["last_ratios"])
    mean_distances["Centralized"].extend(result["centralized"]["mean_distances"])
    mean_distances["Decentralized"].extend(result["decentralized"]["mean_distances"])
    total_apples["Centralized"].extend(result["centralized"]["total_apples"])
    total_apples["Decentralized"].extend(result["decentralized"]["total_apples"])
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], ratios,
                       'Ratio of Apples Picked Per Agent',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions", 'Ratio')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], mean_distances, 'Mean Distance Between Agents',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions", 'Distance')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], total_apples, 'Total Apples Created',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions", 'Apples')


if __name__ == '__main__':
    base = {
        "length": 5,
        "num_agents": 4,
        "hidden_dimensions": 16,
        "dimensions": 4,
    }

    sweep = {
        "width": [1, 2]
    }

    run(base, sweep)





