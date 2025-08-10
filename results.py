import math

from evaluate_policies import evaluate_factory
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

COLOURS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']


INPUT_DIM = {
    2: 74,
    4: 164,
    7: 290,
    10: 452
}


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


def convert_to_weights(hidden_dimensions, num_agents):
    return {
        # "Decentralized": [(hidden_dim * INPUT_DIM[num_agents] + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in hidden_dimensions[:len(hidden_dimensions) - 1]],
        "Decentralized": [(hidden_dim * INPUT_DIM[num_agents] + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in hidden_dimensions],
        "Centralized": [hidden_dim * (INPUT_DIM[num_agents] - 2) + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1 for hidden_dim in hidden_dimensions],
        "Decentralized (local view = 5)": [(hidden_dim * 27 + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in hidden_dimensions],
        # "Decentralized (local view = 5)": [(hidden_dim * 27 + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in  hidden_dimensions[:len(hidden_dimensions) - 1]],
        "Centralized (local view = 5)": [hidden_dim * 27 + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1 for hidden_dim in hidden_dimensions],
    }


def plot_apple_picking(widths, num_agents, series_dict, title, label_x, label_y):
    """
    Plots apple-picking ratios for different approaches.
    NaNs in data are interpolated to connect lines.
    """
    plt.figure()

    # series_dict["Decentralized"] = series_dict["Decentralized"][:len(series_dict["Decentralized"]) - 1]

    # series_dict["Decentralized (local view = 5)"] = series_dict["Decentralized (local view = 5)"][:len(series_dict["Decentralized (local view = 5)"]) - 1]

    # scale_x = convert_to_weights(widths, num_agents)
    #
    # x_all = []
    #
    # for label in scale_x.keys():
    #     x_all.extend(scale_x[label])
    #
    # x_min = min(x_all)
    # # x_max = max(x_all)
    # x_max = scale_x["Centralized"][-1]
    #
    # scale_x["Random"] = np.linspace(x_min, x_max, len(series_dict["Random"])).tolist()

    for label, ratios in series_dict.items():
        if len(series_dict[label]) > 0:
            if type(series_dict[label][1]) is float:
                # y_interp = interpolate_nans(scale_x[label], ratios)
                # if "Decentralized" in label:
                #     y_interp = interpolate_nans(widths[:len(widths) - 1], ratios)
                # else:
                y_interp = interpolate_nans(widths, ratios)
                if 'Centralized' == label:
                    color = 'blue'
                elif 'Decentralized' == label:
                    color = 'orange'
                elif 'Random' == label:
                    color = 'red'
                elif "Decentralized (local view = 5)" == label:
                    color = "green"
                else:
                    color = "purple"
                # plt.plot(scale_x[label], y_interp, marker='o', label=label, color=color)
                # if "Decentralized" in label:
                #     plt.plot(widths[:len(widths) - 1], y_interp, marker='o', label=label, color=color)
                # else:
                plt.plot(widths, y_interp, marker='o', label=label, color=color)


    # log_min = math.floor(math.log10(x_min))
    # log_max = math.ceil(math.log10(x_max))
    # tick_values = [10**i for i in range(log_min, log_max + 1)]
    # plt.xticks(tick_values)
    # plt.xscale('log')  # optional: apply log scale for better spacing

    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xticks(widths)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_series(changing_param, series_dict, title, label_x, label_y):

    x_local = [0 + 100000 * i for i in range(11)]
    for label, series in series_dict.items():
        if label != "Random":
            x = x_local
            plt.figure()
            for i in range(len(series)):
                if series[i]:
                    y_interp = interpolate_nans(x, series[i])
                    plt.plot(x, y_interp, marker='o', label=label + f" ({changing_param[i]} hidden dim.)")
        y_interp = interpolate_nans([0, 200000, 400000, 600000, 800000, 1000000], series_dict["Random"][0])
        plt.plot([0, 200000, 400000, 600000, 800000, 1000000], y_interp, marker='o', label="Random", color="red")
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)
        plt.legend()
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

    log_dir = Path(f"logs/{length}_by_{width}")

    if architecture == "Centralized":
        pattern = (
            f"Centralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<.*?>"
            f"-discount-<.*?>-hidden_dimensions-<{hidden_dimensions}>-dimensions-<{dimensions}>-vision-<0>-epsilon-<.*?>-batch_size-<.*?>.log"
        )
    elif architecture == "Decentralized":
        pattern = (
            f"Decentralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{hidden_dimensions}>-<{dimensions}>-vision-<0>-batch_size-<.*?>.log"
        )
    elif architecture == "Decentralized (local view = 5)":
        pattern = (
            f"Decentralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{hidden_dimensions}>-<{dimensions}>-vision-<5>-batch_size-<.*?>.log"
        )
    elif architecture == "Centralized (local view = 5)":
        pattern = (
            f"Centralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<.*?>"
            f"-discount-<.*?>-hidden_dimensions-<{hidden_dimensions}>-dimensions-<{dimensions}>-vision-<5>-batch_size-<.*?>.log"
        )
    else:
        pattern = (
            f"ActorCritic-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<128>-<.*?>-vision-<0>-batch_size-<.*?>-actor_alpha-<.*?>-actor_hidden-"
            f"<{hidden_dimensions}>-actor_layers-<{dimensions}>.log"
        )

    for log_file in log_dir.glob("*.log"):
        if re.fullmatch(pattern, log_file.name):
            # Found the matching file
            last_ratio = None
            mean_distance = None
            total_apples = None
            picked_per_agents = []
            total_picked = None
            last_picked = None

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
                            picked_per_agents.append(float(match.group(1)))
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
        "Centralized": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Decentralized": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Decentralized (local view = 5)": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Centralized (local view = 5)": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Actor Critic": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        }
    }

    # Identify which parameter is being swept (we assume 1 for simplicity)
    # assert len(sweep_params) == 1, "Only one sweep parameter is supported for now"
    sweep_key = list(sweep_params.keys())[0]
    sweep_values = sweep_params[sweep_key]

    for val in sweep_values:
        config = {**base_config, sweep_key: val}

        for arch in ["Centralized", "Decentralized", "Decentralized (local view = 5)", "Centralized (local view = 5)", "Actor Critic"]:
            result = parse_log_metrics(
                architecture=arch,
                num_agents=config["num_agents"],
                length=config["length"],
                width=config["width"],
                hidden_dimensions=config["hidden_dimensions"],
                dimensions=config["dimensions"]
            )

            results[arch]["sweep_values"].append(val)
            results[arch]["last_ratios"].append(result[0] if result else None)
            results[arch]["mean_distances"].append(result[1] if result else None)
            results[arch]["total_apples"].append(result[2] if result else None)
            results[arch]["picked_per_agent"].append(result[3][-1] if result else None)
            results[arch]["total_picked"].append(result[4] if result else None)
            results[arch]["picked_over_time"].append(result[3] if result else None)

    return results


def init_dicts():
    return {
        "Decentralized": [],
        "Centralized": [],
        "Decentralized (local view = 5)": [],
        "Centralized (local view = 5)": [],
        "Random": [],
        "Actor Critic": []
    }


def add_random(ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series):
    ratios["Random"].append(random_res["ratio_per_agent"])
    total_apples["Random"].append(random_res["total_apples"])
    mean_distances["Random"].append(random_res["mean_distance"])
    picked_per_agents["Random"].append(random_res["picked_per_agent"])
    total_picked["Random"].append(random_res["total_picked"])
    picked_s = [random_res["picked_per_agent"] for _ in range(6)]
    picked_series["Random"].append(picked_s)


def run(base_config: dict, sweep_params: dict):
    # init the dictionaries
    ratios = init_dicts()
    total_apples = init_dicts()
    mean_distances = init_dicts()
    total_picked = init_dicts()
    picked_per_agents = init_dicts()
    picked_series = init_dicts()

    # first, get the results for random
    if "width" not in sweep_params or "dimensions" in sweep_params:
        random_res = evaluate_factory(base_config["length"], base_config["width"], base_config["num_agents"])
        for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
            add_random(ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series)
    else:
        for width in sweep_params["width"]:
            random_res = evaluate_factory(base_config["length"], width, base_config["num_agents"])
            add_random(ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series)
    result = sweep_logs(base_config, sweep_params)

    for arch_name in init_dicts().keys():
        if arch_name != "Random":
            ratios[arch_name].extend(result[arch_name]["last_ratios"])
            total_apples[arch_name].extend(result[arch_name]["total_apples"])
            mean_distances[arch_name].extend(result[arch_name]["mean_distances"])
            picked_per_agents[arch_name].extend(result[arch_name]["picked_per_agent"])
            total_picked[arch_name].extend(result[arch_name]["total_picked"])
            picked_series[arch_name].extend(result[arch_name]["picked_over_time"])

    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], ratios,
                       'Ratio of Apples Picked Per Agent',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions" if "hidden_dimensions" == list(sweep_params.keys())[0] else "Width", 'Ratio')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], mean_distances, 'Mean Distance Between Agents',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions" if "hidden_dimensions" == list(sweep_params.keys())[0] else "Width", 'Distance')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], total_apples, 'Total Apples Created',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions" if "hidden_dimensions" == list(sweep_params.keys())[0] else "Width", 'Apples')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], total_picked, 'Total Apples Picked',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions" if "hidden_dimensions" == list(sweep_params.keys())[0] else "Width", 'Apples')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], picked_per_agents, 'Total Apples Picked Per Agent',
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Hidden Dimensions" if "hidden_dimensions" == list(sweep_params.keys())[0] else "Width", 'Apples')
    plot_series(sweep_params[list(sweep_params.keys())[0]], picked_series, 'Total Apples Picked per Agent over Time', "Training Steps", "Apples Picked per Agent")


def percent_gains(res):
    centralized = res["Centralized"]
    decentralized = res["Decentralized"]
    return [((d - c) / c) * 100 for c, d in zip(centralized, decentralized)]


def difference(res):
    centralized = res["Centralized"]
    decentralized = res["Decentralized"]
    return [(d - c) for c, d in zip(centralized, decentralized)]


if __name__ == '__main__':
    base = {
        "length": 15,
        "num_agents": 10,
        "width": 15,
        "dimensions": 4
    }

    sweep = {
        "hidden_dimensions": [4, 16, 32, 64, 128, 256, 512, 1024]
    }

    # Example usage
    # res = {
    #     "Centralized": [1174.0, 907.6666666666666, 610.0, 418.2857142857143, 313.9],
    #     "Decentralized": [1184.0, 975.5, 871.0, 835.6190476190476, 760.3666666666667]
    # }
    #
    #
    # plot_apple_picking([1, 2, 4, 7, 10], 0, res, "Apples Picked Per Agent For Different Agent Counts",  "Number of Agents", "Apples Picked Per Agent")
    #
    # gains = percent_gains(res)
    #
    # agents = [1, 2, 4, 7, 10]
    # # Plot
    # plt.figure(figsize=(6, 4))
    # plt.plot(agents, gains, marker='o', linestyle='-', linewidth=2)
    # plt.xlabel("Number of Agents")
    # plt.ylabel("Decentralized Gain over Centralized (%)")
    # plt.title("Decentralized vs Centralized Performance Gains")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(6, 4))
    # plt.plot(agents, difference(res), marker='o', linestyle='-', linewidth=2)
    # plt.xlabel("Number of Agents")
    # plt.ylabel("Apples Picked per Agent")
    # plt.title("Decentralized vs Centralized Performance Difference")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()
    run(base, sweep)
