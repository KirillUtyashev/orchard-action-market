import math

from evaluate_policies import evaluate_factory, evaluate_network
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
# log_min = math.floor(math.log10(x_min))
# log_max = math.ceil(math.log10(x_max))
# tick_values = [10**i for i in range(log_min, log_max + 1)]
# plt.xticks(tick_values)
# plt.xscale('log')  # optional: apply log scale for better spacing


def plot_apple_picking(widths, num_agents, series_dict, title, label_x, label_y):
    """
    Plots apple-picking ratios for different approaches.
    NaNs in data are interpolated to connect lines.
    """
    plt.figure()
    for label, ratios in series_dict.items():
        if len(series_dict[label]) > 0:
            if type(series_dict[label][1]) is float:
                y_interp = interpolate_nans(widths, ratios)
                if 'Centralized' in label:
                    color = 'blue'
                elif 'Decentralized Personal Q-Values' in label:
                    color = 'green'
                elif 'Decentralized' in label:
                    color = 'orange'
                elif 'Random' == label:
                    color = 'red'
                elif "Decentralized (local view = 5)" == label:
                    color = "green"
                elif "Rates" in label and "W/o" in label:
                    color = "goldenrod"
                elif "Rates" in label and "W" in label:
                    color = "green"
                elif "Beta" in label and "Rates" not in label:
                    color = "darkblue"
                else:
                    color = "purple"
                plt.plot(widths, y_interp, marker='o', label=label, color=color, linestyle=':')

    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xticks(widths)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def _to_float_array(vals):
    """Convert list with None/str to float array with NaNs."""
    return np.array(vals, dtype=np.float64)

def _is_power_of_two(x):
    # treat near-integers as integers
    xi = int(round(x))
    return xi > 0 and (xi & (xi - 1)) == 0 and abs(x - xi) < 1e-9

def min_width_for_threshold_pow2_only(widths, performances, threshold):
    """
    Return the smallest power-of-two width whose (power-of-two-only) performance
    >= threshold. Only interpolates across *power-of-two* widths.
    """
    w = np.array(widths, dtype=np.float64)
    y_raw = np.array(performances, dtype=np.float64)  # None -> NaN

    # keep only power-of-two widths
    mask_pow2 = np.array([_is_power_of_two(x) for x in w])
    if not np.any(mask_pow2):
        return np.nan

    w2 = w[mask_pow2]
    y2 = y_raw[mask_pow2]

    # sort by width (just in case)
    order = np.argsort(w2)
    w2 = w2[order]
    y2 = y2[order]

    # interpolate ONLY within power-of-two points (fill internal NaNs)
    y2_interp = interpolate_nans(w2, y2)

    if np.all(np.isnan(y2_interp)) or np.nanmax(y2_interp) < threshold:
        return np.nan

    # pick the smallest power-of-two width meeting the threshold
    for wi, yi in zip(w2, y2_interp):
        if not np.isnan(yi) and yi >= threshold:
            return wi

    return np.nan


def plot_fixed_thresholds(widths, series_dict, thresholds):
    """
    Plot min hidden dimensions needed to reach fixed absolute performance thresholds.
    thresholds: e.g., [600, 700, 800, 900]
    """
    plt.figure()
    thresholds = list(thresholds)

    for label, ratios in series_dict.items():
        if label.lower() == "random":
            continue
        if not ratios:
            continue
        y = _to_float_array(ratios)
        if np.all(np.isnan(y)):
            continue

        min_widths = [min_width_for_threshold_pow2_only(widths, ratios, t) for t in thresholds]
        plt.plot(thresholds, min_widths, marker='o', linestyle=':', label=label)

    plt.title("Hidden Dimensions Needed for Fixed Performance Levels")
    plt.xlabel("Performance (Apples Per Agent)")
    plt.ylabel("Min Hidden Dimensions")
    plt.grid(True, ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_percentage_of_max(widths, series_dict, percentages):
    """
    Plot min hidden dimensions needed to reach a % of the GLOBAL max performance.
    percentages: e.g., [0.5, 0.6, 0.7, 0.8, 0.9]
    """
    # global max across all approaches/widths
    all_perf = []
    for ratios in series_dict.values():
        if ratios:
            all_perf.extend(_to_float_array(ratios))
    global_max = np.nanmax(all_perf) if len(all_perf) else np.nan

    plt.figure()
    x_pct = [p * 100 for p in percentages]

    for label, ratios in series_dict.items():
        if label.lower() == "random":
            continue
        if not ratios:
            continue
        y = _to_float_array(ratios)
        if np.all(np.isnan(y)):
            continue

        min_widths = [min_width_for_threshold_pow2_only(widths, ratios, global_max * p) for p in percentages]
        plt.plot(x_pct, min_widths, marker='o', linestyle=':', label=label)

    plt.title("Hidden Dimensions Needed for % of Global Max Performance")
    plt.xlabel("Target Performance (% of Max)")
    plt.ylabel("Min Hidden Dimensions")
    plt.grid(True, ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def convert_to_weights(hidden_dimensions, num_agents):
    return {
        # "Decentralized": [(hidden_dim * INPUT_DIM[num_agents] + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in hidden_dimensions[:len(hidden_dimensions) - 1]],
        "Decentralized": [(hidden_dim * INPUT_DIM[num_agents] + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in hidden_dimensions],
        "Centralized": [hidden_dim * (INPUT_DIM[num_agents] - 2) + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1 for hidden_dim in hidden_dimensions],
        "Decentralized (local view = 5)": [(hidden_dim * 27 + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in hidden_dimensions],
        # "Decentralized (local view = 5)": [(hidden_dim * 27 + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1) * num_agents for hidden_dim in  hidden_dimensions[:len(hidden_dimensions) - 1]],
        "Centralized (local view = 5)": [hidden_dim * 27 + 2 * (hidden_dim ** 2) + 4 * hidden_dim + 1 for hidden_dim in hidden_dimensions],
    }


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


def parse_log_metrics(architecture: str, orchard: str, num_agents: int, length: int, width: int, hidden_dimensions: int, alpha: float, dimensions: int, critic_dimensions=None, budget=None, beta=0.0) -> tuple | None:
    """
    Parse the log file for a specific experimental setup and return:
    (last_ratio_picked, mean_distance, total_apples)

    Parameters:
        architecture (str): "Centralized" or "Decentralized"
        orchard (str)
        num_agents (int): number of agents
        length (int)
        width (int)
        alpha (float)
        hidden_dimensions (int)
        dimensions (int)
        critic_dimensions
        budget

    Returns:
        tuple: (last_ratio_picked, mean_distance, total_apples) or None if not found
    """

    log_dir = Path(f"logs/{orchard}")

    if architecture == "Centralized":
        pattern = (
            f"Centralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{alpha}>"
            f"-discount-<.*?>-hidden_dimensions-<{hidden_dimensions}>-dimensions-<{dimensions}>-vision-<0>-epsilon-<.*?>-batch_size-<.*?>-env_cls-<{orchard}>.log"
        )
    elif architecture == "Decentralized":
        pattern = (
            f"Decentralized-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<{alpha}>"
            f"-apple_mean_lifetime-<.*?>-<{hidden_dimensions}>-<{dimensions}>-vision-<0>-batch_size-<.*?>-env-<{orchard}>.log"
        )
    elif architecture == "Decentralized Personal Q-Values":
        pattern = (
            f"DecentralizedPersonal-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<{alpha}>"
            f"-apple_mean_lifetime-<.*?>-<{hidden_dimensions}>-<{dimensions}>-vision-<0>-batch_size-<.*?>-env-<{orchard}>.log"
        )
    elif architecture == "Actor Critic":
        pattern = (
            f"ActorCritic-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{critic_dimensions}>-<.*?>-vision-<0>-batch_size-<.*?>-actor_alpha-<{alpha}>-actor_hidden-"
            f"<{hidden_dimensions}>-actor_layers-<{dimensions}>.log"
        )
    elif architecture == "Actor Critic Rates W/ Beta":
        pattern = (
            f"ActorCriticRates-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{critic_dimensions}>-<.*?>-vision-<0>-batch_size-<.*?>-actor_alpha-<{alpha}>-actor_hidden-"
            f"<{hidden_dimensions}>-actor_layers-<{dimensions}>-beta-<{beta}>-budget-<{budget}>.log"
        )
    elif architecture == "Actor Critic Rates W/o Beta":
        pattern = (
            f"ActorCriticRates-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{critic_dimensions}>-<.*?>-vision-<0>-batch_size-<.*?>-actor_alpha-<{alpha}>-actor_hidden-"
            f"<{hidden_dimensions}>-actor_layers-<{dimensions}>-beta-<{beta}>-budget-<{budget}>.log"
        )
    else:
        pattern = (
            f"ActorCriticBeta-<{num_agents}>_agents-_length-<{length}>_width-<{width}>_s_target-<.*?>-alpha-<.*?>"
            f"-apple_mean_lifetime-<.*?>-<{critic_dimensions}>-<.*?>-vision-<0>-batch_size-<.*?>-actor_alpha-<{alpha}>-actor_hidden-"
            f"<{hidden_dimensions}>-actor_layers-<{dimensions}>-beta-<.*?>.log"
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
            return last_ratio, total_apples, mean_distance, picked_per_agents[-1], total_picked,  picked_per_agents

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
        "Decentralized Personal Q-Values": {
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
        },
        "Actor Critic Rates W/ Beta": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Actor Critic Rates W/o Beta": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Actor Critic Beta": {
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

        for arch in ["Centralized", "Decentralized", "Decentralized Personal Q-Values"]:
            result = parse_log_metrics(
                architecture=arch,
                orchard=config["orchard"],
                num_agents=config["num_agents"],
                length=config["length"],
                width=config["width"],
                hidden_dimensions=config["hidden_dimensions"],
                alpha=config["alpha"],
                dimensions=config["dimensions"]
            )

            results[arch]["sweep_values"].append(val)
            results[arch]["last_ratios"].append(result[0] if result else None)
            results[arch]["mean_distances"].append(result[1] if result else None)
            results[arch]["total_apples"].append(result[2] if result else None)
            results[arch]["picked_per_agent"].append(result[3] if result else None)
            results[arch]["total_picked"].append(result[4] if result else None)
            results[arch]["picked_over_time"].append(result[5] if result else None)

    return results


def init_dicts(critic_dimensions=None, actor_dimensions=None):
    if not critic_dimensions:
        return {
            "Decentralized": [],
            "Centralized": [],
            "Decentralized Personal Q-Values": [],
            "Random": [],
            "Actor Critic": []
        }
    if not actor_dimensions:
        return {
            f"Decentralized, {critic_dimensions} hidden dimensions": [],
            f"Centralized, {critic_dimensions} hidden dimensions": [],
            "Decentralized (local view = 5)": [],
            "Centralized (local view = 5)": [],
            "Random": [],
            "Actor Critic": []
        }
    return {
        f"Actor Critic": [],
        f"Actor Critic Beta": [],
        "Random": [],
        "Actor Critic Rates W/ Beta": [],
        "Actor Critic Rates W/o Beta": []
    }


def add_random(arch, ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series):
    ratios[arch].append(random_res[0])
    total_apples[arch].append(random_res[1])
    mean_distances[arch].append(random_res[2])
    picked_per_agents[arch].append(random_res[3])
    total_picked[arch].append(random_res[4])
    picked_s = [random_res[3] for _ in range(6)]
    picked_series[arch].append(picked_s)


def run(base_config: dict, sweep_params: dict):
    # init the dictionaries
    ratios = init_dicts()
    total_apples = init_dicts()
    mean_distances = init_dicts()
    total_picked = init_dicts()
    picked_per_agents = init_dicts()
    picked_series = init_dicts()

    # first, get the results for random
    if "width" not in sweep_params or "dimensions" in sweep_params or "alpha" in sweep_params:
        random_res = evaluate_factory(base_config["length"], base_config["width"], base_config["num_agents"], base_config["orchard"])
        for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
            add_random("Random", ratios, total_apples, mean_distances, picked_per_agents, total_picked, (random_res["ratio_per_agent"],
                                                                                                         random_res["total_apples"],
                                                                                                         random_res["mean_distance"],
                                                                                                         random_res["picked_per_agent"],
                                                                                                         random_res["total_picked"]), picked_series)
    elif "width" in sweep_params:
        for width in sweep_params["width"]:
            random_res = evaluate_factory(base_config["length"], width, base_config["num_agents"])
            add_random(ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series)
    else:
        pass
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

    # Example thresholds for Plot 1
    thresholds = [400, 500, 600, 700, 800, 850, 900, 950, 1000]
    plot_fixed_thresholds(sweep_params[list(sweep_params.keys())[0]], picked_per_agents, thresholds)

    # Example percentages for Plot 2
    percentages = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plot_percentage_of_max(sweep_params[list(sweep_params.keys())[0]], picked_per_agents, percentages)


def run_actor_critic(base_config: dict, sweep_params: dict, critic_dimensions: int):
    ratios = init_dicts(critic_dimensions)
    total_apples = init_dicts(critic_dimensions)
    mean_distances = init_dicts(critic_dimensions)
    total_picked = init_dicts(critic_dimensions)
    picked_per_agents = init_dicts(critic_dimensions)
    picked_series = init_dicts(critic_dimensions)

    # first, get the results for random
    if "width" not in sweep_params or "dimensions" in sweep_params or "alpha" in sweep_params:
        random_res = evaluate_factory(base_config["length"], base_config["width"], base_config["num_agents"])
        for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
            add_random("Random", ratios, total_apples, mean_distances, picked_per_agents, total_picked, (random_res["ratio_per_agent"],
                                                                                                         random_res["total_apples"],
                                                                                                         random_res["mean_distance"],
                                                                                                         random_res["picked_per_agent"],
                                                                                                         random_res["total_picked"]), picked_series)
    else:
        for width in sweep_params["width"]:
            random_res = evaluate_factory(base_config["length"], width, base_config["num_agents"])
            add_random("Random", ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series)

    # next, we get the results for DC and C at given dimensions
    c_res = parse_log_metrics("Centralized", base_config["num_agents"], base_config["length"], base_config["width"], critic_dimensions, 0.000275, 4)
    for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
        add_random(f"Centralized, {critic_dimensions} hidden dimensions", ratios, total_apples, mean_distances, picked_per_agents, total_picked, c_res, picked_series)
    dc_res = parse_log_metrics("Decentralized", base_config["num_agents"], base_config["length"], base_config["width"], critic_dimensions, 0.000275, 4)
    for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
        add_random(f"Decentralized, {critic_dimensions} hidden dimensions", ratios, total_apples, mean_distances, picked_per_agents, total_picked, dc_res, picked_series)

    results = {
        "Actor Critic": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
    }

    sweep_key = list(sweep_params.keys())[0]
    sweep_values = sweep_params[sweep_key]

    for val in sweep_values:
        config = {**base_config, sweep_key: val}

        for arch in ["Actor Critic"]:
            result = parse_log_metrics(
                architecture=arch,
                num_agents=config["num_agents"],
                length=config["length"],
                width=config["width"],
                hidden_dimensions=config["hidden_dimensions"],
                alpha=config["alpha"],
                dimensions=config["dimensions"],
                critic_dimensions=critic_dimensions
            )

            results[arch]["sweep_values"].append(val)
            results[arch]["last_ratios"].append(result[0] if result else None)
            results[arch]["mean_distances"].append(result[1] if result else None)
            results[arch]["total_apples"].append(result[2] if result else None)
            results[arch]["picked_per_agent"].append(result[3] if result else None)
            results[arch]["total_picked"].append(result[4] if result else None)
            results[arch]["picked_over_time"].append(result[5] if result else None)

    for arch_name in ["Actor Critic"]:
        ratios[arch_name].extend(results[arch_name]["last_ratios"])
        total_apples[arch_name].extend(results[arch_name]["total_apples"])
        mean_distances[arch_name].extend(results[arch_name]["mean_distances"])
        picked_per_agents[arch_name].extend(results[arch_name]["picked_per_agent"])
        total_picked[arch_name].extend(results[arch_name]["total_picked"])
        picked_series[arch_name].extend(results[arch_name]["picked_over_time"])

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
                       "Linear Layers" if "dimensions" == list(sweep_params.keys())[0] else "Actor Network Hidden Dimensions" if "hidden_dimensions" == list(sweep_params.keys())[0] else "Width", 'Apples')
    plot_series(sweep_params[list(sweep_params.keys())[0]], {k: picked_series[k] for k in ["Actor Critic", "Random"]}, 'Total Apples Picked per Agent over Time', "Training Steps", "Apples Picked per Agent")


def run_budget(base_config: dict, sweep_params: dict, critic_dimensions: int):
    ratios = init_dicts(critic_dimensions, base_config["hidden_dimensions"])
    total_apples = init_dicts(critic_dimensions, base_config["hidden_dimensions"])
    mean_distances = init_dicts(critic_dimensions, base_config["hidden_dimensions"])
    total_picked = init_dicts(critic_dimensions, base_config["hidden_dimensions"])
    picked_per_agents = init_dicts(critic_dimensions, base_config["hidden_dimensions"])
    picked_series = init_dicts(critic_dimensions, base_config["hidden_dimensions"])

    # first, get the results for random
    if "width" not in sweep_params or "dimensions" in sweep_params or "alpha" in sweep_params:
        random_res = evaluate_factory(base_config["length"], base_config["width"], base_config["num_agents"])
        for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
            add_random("Random", ratios, total_apples, mean_distances, picked_per_agents, total_picked, (random_res["ratio_per_agent"],
                                                                                                         random_res["total_apples"],
                                                                                                         random_res["mean_distance"],
                                                                                                         random_res["picked_per_agent"],
                                                                                                         random_res["total_picked"]), picked_series)
    else:
        for width in sweep_params["width"]:
            random_res = evaluate_factory(base_config["length"], width, base_config["num_agents"])
            add_random("Random", ratios, total_apples, mean_distances, picked_per_agents, total_picked, random_res, picked_series)

    # next, we get the results for AC and AC Beta (if present) at given dimensions
    ac_res = parse_log_metrics("Actor Critic", base_config["num_agents"], base_config["length"], base_config["width"], base_config["hidden_dimensions"], base_config["alpha"], 4, critic_dimensions)
    for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
        add_random(f"Actor Critic", ratios, total_apples, mean_distances, picked_per_agents, total_picked, ac_res, picked_series)
    ac_res = parse_log_metrics("Actor Critic Beta", base_config["num_agents"], base_config["length"], base_config["width"], base_config["hidden_dimensions"], base_config["alpha"], 4, critic_dimensions)
    for _ in range(len(sweep_params[list(sweep_params.keys())[0]])):
        add_random(f"Actor Critic Beta", ratios, total_apples, mean_distances, picked_per_agents, total_picked, ac_res, picked_series)

    results = {
        "Actor Critic Rates W/ Beta": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
        "Actor Critic Rates W/o Beta": {
            "sweep_values": [],
            "mean_distances": [],
            "total_apples": [],
            "last_ratios": [],
            "total_picked": [],
            "picked_per_agent": [],
            "picked_over_time": []
        },
    }

    sweep_key = list(sweep_params.keys())[0]
    sweep_values = sweep_params[sweep_key]

    for val in sweep_values:
        config = {**base_config, sweep_key: val}

        for arch in ["Actor Critic Rates W/ Beta", "Actor Critic Rates W/o Beta"]:
            result = parse_log_metrics(
                architecture=arch,
                num_agents=config["num_agents"],
                length=config["length"],
                width=config["width"],
                hidden_dimensions=config["hidden_dimensions"],
                alpha=config["alpha"],
                dimensions=config["dimensions"],
                critic_dimensions=critic_dimensions,
                budget=config["budget"],
                beta=config["beta"] if arch == "Actor Critic Rates W/ Beta" else 0.0
            )

            results[arch]["sweep_values"].append(val)
            results[arch]["last_ratios"].append(result[0] if result else None)
            results[arch]["mean_distances"].append(result[1] if result else None)
            results[arch]["total_apples"].append(result[2] if result else None)
            results[arch]["picked_per_agent"].append(result[3] if result else None)
            results[arch]["total_picked"].append(result[4] if result else None)
            results[arch]["picked_over_time"].append(result[5] if result else None)

    for arch_name in ["Actor Critic Rates W/ Beta", "Actor Critic Rates W/o Beta"]:
        ratios[arch_name].extend(results[arch_name]["last_ratios"])
        total_apples[arch_name].extend(results[arch_name]["total_apples"])
        mean_distances[arch_name].extend(results[arch_name]["mean_distances"])
        picked_per_agents[arch_name].extend(results[arch_name]["picked_per_agent"])
        total_picked[arch_name].extend(results[arch_name]["total_picked"])
        picked_series[arch_name].extend(results[arch_name]["picked_over_time"])

    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], ratios,
                       'Ratio of Apples Picked Per Agent', "Budget", 'Ratio')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], mean_distances, 'Mean Distance Between Agents',
                       "Budget", 'Distance')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], total_apples, 'Total Apples Created',
                       "Budget", 'Apples')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], total_picked, 'Total Apples Picked',
                       "Budget", 'Apples')
    plot_apple_picking(sweep_params[list(sweep_params.keys())[0]], base_config["num_agents"], picked_per_agents, 'Total Apples Picked Per Agent',
                       "Budget", 'Apples')
    plot_series(sweep_params[list(sweep_params.keys())[0]], {k: picked_series[k] for k in ["Actor Critic Rates W/ Beta", "Actor Critic Rates W/o Beta", "Random"]}, 'Total Apples Picked per Agent over Time', "Training Steps", "Apples Picked per Agent")


def read_performance_log(filepath):
    """Return {'Centralized': [...], 'Decentralized': [...]} from a log file."""
    out = {"Centralized": [], "Decentralized": []}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, val = line.split(":", 1)
            label = label.strip()
            try:
                v = float(val.strip())
            except ValueError:
                continue
            if label in out:
                out[label].append(v)
    return out


def reduce_log_to_scalar(log_dict, reducer="mean"):
    """
    Reduce {'Centralized': [..], 'Decentralized': [..]} to scalars per key.
    reducer: 'mean' | 'last' | 'max' | 'min'
    """
    scalars = {}
    for k, vals in log_dict.items():
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            scalars[k] = np.nan
            continue
        if reducer == "last":
            scalars[k] = arr[~np.isnan(arr)][-1]
        elif reducer == "max":
            scalars[k] = np.nanmax(arr)
        elif reducer == "min":
            scalars[k] = np.nanmin(arr)
        else:
            scalars[k] = np.nanmean(arr)
    return scalars


def build_series_by_hidden_dims(filepaths, hidden_dims, reducer="mean"):
    """
    Given lists of filepaths and corresponding hidden_dims (same length),
    return series_dict aligned to hidden_dims:
      {'Centralized': [...], 'Decentralized': [...]}
    """
    if len(filepaths) != len(hidden_dims):
        raise ValueError("filepaths and hidden_dims must have the same length.")

    # Sort by hidden_dims to keep x monotone
    order = np.argsort(hidden_dims)
    hidden_dims_sorted = list(np.array(hidden_dims)[order])
    files_sorted = list(np.array(filepaths)[order])

    series = {"Centralized": [], "Decentralized": []}
    for fp in files_sorted:
        log = read_performance_log(fp)
        scalars = reduce_log_to_scalar(log, reducer=reducer)
        series["Centralized"].append(scalars.get("Centralized", np.nan))
        series["Decentralized"].append(scalars.get("Decentralized", np.nan))

    return hidden_dims_sorted, series

# --- simple metrics if you don't already have them ---


def percent_gains(series_dict):
    c = np.array(series_dict["Centralized"], dtype=np.float64)
    d = np.array(series_dict["Decentralized"], dtype=np.float64)
    return 100.0 * (d - c) / c


def difference(series_dict):
    c = np.array(series_dict["Centralized"], dtype=np.float64)
    d = np.array(series_dict["Decentralized"], dtype=np.float64)
    return d - c


def read_log_pairs(filepath):
    """Read Centralized/Decentralized pairs from a log file."""
    centralized = []
    decentralized = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, val = line.split(":", 1)
            try:
                val = float(val.strip())
            except ValueError:
                continue
            if label.strip() == "Centralized":
                centralized.append(val)
            elif label.strip() == "Decentralized":
                decentralized.append(val)

    # Ensure same length
    n = min(len(centralized), len(decentralized))
    return np.array(centralized[:n]), np.array(decentralized[:n])


def plot_percent_diffs(filepaths, agent_counts):
    """
    For each file (corresponding to an agent count), read centralized/decentralized
    pairs, compute % difference, and plot all points with that agent count on x-axis.
    """

    plt.figure(figsize=(6, 4))

    for filepath, agents in zip(filepaths, agent_counts):
        match = re.search(r'(\d+)', filepath)
        if match:
            number = int(match.group(1))
        c_vals, d_vals = read_log_pairs(filepath)
        pct_diffs = 100.0 * (d_vals - c_vals) / c_vals
        if len(c_vals) == 5:
            pct_diffs = np.append(pct_diffs, None)
        plt.plot(agent_counts, pct_diffs, marker='o', label=f"{number} hidden dimensions", alpha=0.7, linestyle=":")

    plt.xlabel("Number of Agents")
    plt.ylabel("Decentralized Gain over Centralized (%)")
    plt.title("Performance % Differences Across Hidden Dimensions")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_diffs(filepaths, agent_counts):
    """
    For each file (corresponding to an agent count), read centralized/decentralized
    pairs, compute % difference, and plot all points with that agent count on x-axis.
    """

    plt.figure(figsize=(6, 4))
    random_res = [238 for _ in range(6)]

    for filepath, agents in zip(filepaths, agent_counts):
        match = re.search(r'(\d+)', filepath)
        if match:
            number = int(match.group(1))
        c_vals, d_vals = read_log_pairs(filepath)
        plt.plot(agent_counts, c_vals, marker='o', label=f"Centralized, {number} hidden dimensions", alpha=0.7, linestyle=":", color="blue")
        plt.plot(agent_counts, d_vals, marker='o', label=f"Decentralized, {number} hidden dimensions", alpha=0.7, linestyle=":", color="orange")
    plt.plot(agent_counts, random_res, marker='o', label="Random", alpha=0.7, linestyle=":", color="red")
    plt.xlabel("Number of Agents")
    plt.ylabel("Apples Picked Per Agent For Different Agent Counts")
    plt.title("Apples Picked Per Agent")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base = {
        "length": 9,
        "num_agents": 4,
        "width": 9,
        "dimensions": 4,
        "alpha": 0.000275,
        "orchard": "OrchardEuclideanRewards"
    }

    # sweep = {
    #     "alpha": [0.00005, 0.0000316, 0.0001, 0.000275, 0.000316, 0.00063, 0.001]
    # }
    sweep = {
        "hidden_dimensions": [8, 64]
    }
    run(base, sweep)

