import os
import random
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

from actor_critic_perfect_info import ActorCriticPerfect
from agents.actor_critic_agent import ACAgent
from agents.simple_agent import SimpleAgent
from config import CHECKPOINT_DIR, DEVICE
from main import plot_agents_heatmap_alpha, run_environment_1d
from metrics.metrics import plot_agents_trajectories
from models.actor_dc_1d import ActorNetwork
from models.value_function import VNetwork
from policies.random_policy import random_policy
from policies.nearest import nearest_policy
from configs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from orchard.algorithms import spawn_apple, despawn_apple
from value_function_learning.controllers import AgentControllerDecentralized, \
    ViewController
from value_function_learning.train_value_function import \
    CentralizedValueFunction, DecentralizedValueFunction, evaluate_policy, \
    make_baseline_factory
from run_experiments import parse_args
from agents.communicating_agent import CommAgent


def load_weights_only(name, networks, agents_list, path: str):
    """
    Load only the network weights from a saved checkpoint into the algo's agents.
    Does NOT restore RNG state, agent positions, or apples.

    Parameters
    ----------
    algo : Algorithm
        An initialized Algorithm object with agents_list and network objects created.
    path : str
        Path to the directory containing the saved checkpoint file (without filename).
    agents_list: List
    networks: List
    name: str
    """
    ckpt_path = os.path.join(path, f"{name}_ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Map critics (skip RNG/env state)
    crit_blobs = ckpt.get("critics", [])
    for (name_now, obj_now), saved in zip(networks, crit_blobs):
        obj_now.import_net_state(saved["blob"])

    # Map actors (aligned with agents_list)
    act_blobs = ckpt.get("actors", [])
    for agent, blob in zip(agents_list, act_blobs):
        if blob and hasattr(agent, "policy_network") and agent.policy_network is not None:
            agent.policy_network.import_net_state(blob, device=DEVICE)


def evaluate_factory(length, width, num_agents):
    random.seed(42069)
    np.random.seed(42069)
    torch.manual_seed(42069)

    env_config = EnvironmentConfig(
        s_target=0.16,
        apple_mean_lifetime=5,
        length=length,
        width=width,
        spawn_algo=spawn_apple,
        despawn_algo=despawn_apple
    )

    return evaluate_policy(
        env_config,
        num_agents=num_agents,
        agent_factory=make_baseline_factory(random_policy),
        timesteps=10000,
        seed=42069
    )


def evaluate_network(length, width, num_agents, hidden_dim, algorithm, old):
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)


    env_config = EnvironmentConfig(
        s_target=0.16,
        apple_mean_lifetime=5.0,
        length=length,
        width=width,
    )

    train_config = TrainingConfig(
        batch_size=num_agents,
        num_agents=num_agents,
        hidden_dimensions=hidden_dim,
        num_layers=4,
    )

    exp_config = ExperimentConfig(
        env_config=env_config,
        train_config=train_config,
    )

    if algorithm == "Centralized":
        algo = CentralizedValueFunction(exp_config)
        agents_list = []
        if not train_config.alt_input:
            network = VNetwork(env_config.width * env_config.length, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
        else:
            if env_config.width != 1:
                network = VNetwork(train_config.vision ** 2 + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            else:
                network = VNetwork(train_config.vision + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
        if old:
            network.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_cen_" + ".pt"))
        else:
            load_weights_only(algo.name, [network], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))
        for nummer in range(train_config.num_agents):
            agent = SimpleAgent("value_function", nummer)
            agent.policy_value = network
            agents_list.append(agent)
    elif algorithm == "Decentralized":
        agents_list = []
        algo = DecentralizedValueFunction(exp_config)
        # Initialize networks and agents
        for nummer in range(train_config.num_agents):
            agent = CommAgent("value_function", train_config.num_agents, nummer)
            if train_config.alt_input:
                if env_config.width != 1:
                    network = VNetwork(train_config.vision ** 2 + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
                else:
                    network = VNetwork(train_config.vision + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            else:
                network = VNetwork(env_config.length * env_config.width + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            if old:
                network.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_decen_" + str(nummer) + ".pt"))
            agent.policy_value = network
            agents_list.append(agent)

        if not old:
            load_weights_only(algo.name, [("1", agent.policy_value) for agent in agents_list], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))
    else:
        agents_list = []
        algo = ActorCriticPerfect(exp_config)
        # Initialize networks and agents
        for nummer in range(train_config.num_agents):
            agent = ACAgent("learned_policy", nummer)
            if train_config.alt_input:
                if env_config.width != 1:
                    policy_value = VNetwork(train_config.vision ** 2 + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
                else:
                    policy_value = VNetwork(train_config.vision + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            else:
                policy_value = VNetwork(env_config.length * env_config.width + 1, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            if old:
                policy_value.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_critic_network_AC_" + str(nummer) + ".pt"))
            agent.policy_value = policy_value
            network = ActorNetwork(env_config.length * env_config.width + 1, 5, train_config.actor_alpha, train_config.discount, train_config.hidden_dimensions_actor, train_config.num_layers_actor)

            if old:
                network.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_actor_network_AC_" + str(nummer) + ".pt"))
            agent.policy_network = network
            agents_list.append(agent)

        if not old:
            load_weights_only(algo.name, [("1", agent.policy_value) for agent in agents_list], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))
    total_apples, total_picked, picked_per_agent, per_agent, mean_dist, apples_per_sec, same_actions, idle_actions = \
        run_environment_1d(
            num_agents,
            env_config.length,
            env_config.width,
            None, None,
            algo.name,
            agents_list=agents_list,
            spawn_algo=env_config.spawn_algo,
            despawn_algo=env_config.despawn_algo,
            timesteps=5000,
            vision=train_config.vision,
            s_target=env_config.s_target,
            apple_mean_lifetime=env_config.apple_mean_lifetime,
            epsilon=train_config.epsilon
        )

    # for agent_id in range(train_config.num_agents):
    #     plt.figure(figsize=(10, 4))
    #     arr = algo.alpha_ema[agent_id]
    #     for other_agent in range(algo.train_config.num_agents):
    #         series = arr[:, other_agent]  # <-- column j, not row j
    #         if series.size > 0:
    #             plt.plot(series, label=f"Q-value from agent {other_agent}")
    #         plt.plot(agents_list[agent_id].agent_alphas[other_agent])
    #     plt.legend()
    #     plt.title(f"Observed Q-values for Agent {agent_id}")
    #     plt.xlabel("Training Step")
    #     plt.ylabel("Q-value")
    #     plt.show()

    # positions = np.load(f"positions/{algo.name}_pos.npy")      # shape (T, N, 2)
    # fig, ax = plt.subplots(figsize=(6, 5))
    # plot_agents_heatmap_alpha(
    #     positions,
    #     agent_ids=[0, 1, 2, 3],                      # pick any subset
    #     colors=[
    #         "royalblue",   # 0
    #         "crimson",     # 1
    #         "gold",        # 2
    #         "purple",      # 3
    #         # "forestgreen", # 4
    #         # "orange",      # 5
    #         # "deeppink"     # 6
    #     ],
    #     ax=ax
    # )
    # plt.show()

    # plot_agents_trajectories(positions, agent_ids=[0, 7], colors=["royalblue", "purple"])
    # plt.show()
    #
    # return {
    #     "total_apples": int(total_apples),
    #     "total_picked": int(total_picked),
    #     "picked_per_agent": float(picked_per_agent),
    #     "ratio_per_agent": float(per_agent),
    #     "mean_distance": float(mean_dist),
    #     "apples_per_sec": float(apples_per_sec)
    # }


if __name__ == "__main__":
    pass
    # widths = [6]
    # for width in widths:
    #     print(evaluate_factory(6, 6, 2))
    # evaluate_network(sys.argv[1:])

    evaluate_network(20, 1, 4, 128, "Decentralized", False)

