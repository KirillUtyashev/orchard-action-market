import os
import random
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

from agents.simple_agent import SimpleAgent
from config import CHECKPOINT_DIR
from main import plot_agents_heatmap_alpha, run_environment_1d
from metrics.metrics import plot_agents_trajectories
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


def evaluate_network(args):
    args = parse_args(args)
    env_config = EnvironmentConfig(
        s_target=args.s_target,
        apple_mean_lifetime=args.apple_life,
        length=args.length,
        width=args.width,
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        alpha=args.alpha,
        timesteps=args.timesteps,
        num_agents=args.num_agents,
        hidden_dimensions=args.hidden_dim,
        num_layers=args.num_layers,
        alt_input=True if args.alt_vision == 0 else False,
        vision=args.vision,
        skip=True if args.skip == 0 else False,
        epsilon=args.epsilon
    )

    exp_config = ExperimentConfig(
        env_config=env_config,
        train_config=train_config,
        debug=args.debug
    )

    if args.algorithm == "Centralized":
        algo = CentralizedValueFunction(exp_config)
        agents_list = []
        if not train_config.alt_input:
            network = VNetwork(env_config.width * env_config.length, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
        else:
            if env_config.width != 1:
                network = VNetwork(train_config.vision ** 2 + 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            else:
                network = VNetwork(train_config.vision + 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)

        network.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_cen_" + ".pt"))
        for _ in range(train_config.num_agents):
            agent = SimpleAgent(policy="value_function")
            agent.policy_value = network
            agents_list.append(agent)
    else:
        agents_list = []
        algo = DecentralizedValueFunction(exp_config)
        # Initialize networks and agents
        for nummer in range(train_config.num_agents):
            agent = CommAgent("value_function", nummer)
            if train_config.alt_input:
                if env_config.width != 1:
                    network = VNetwork(train_config.vision ** 2 + 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
                else:
                    network = VNetwork(train_config.vision + 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            else:
                network = VNetwork(env_config.length * env_config.width + 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            network.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_decen_" + str(nummer) + ".pt"))
            agent.policy_value = network
            agents_list.append(agent)

    positions = np.load(f"positions/Eval-{env_config.length}x{env_config.width}_pos.npy")      # shape (T, N, 2)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_agents_heatmap_alpha(
        positions,
        agent_ids=[0, 1, 2, 3],                      # pick any subset
        colors=["royalblue", "crimson", "gold", "purple"],  # one hue per agent
        ax=ax
    )
    plt.show()

    plot_agents_trajectories(positions, agent_ids=[0], colors=["royalblue"])
    plt.show()

    # total_apples, total_picked, picked_per_agent, per_agent, mean_dist, apples_per_sec, same_actions, idle_actions = \
    #     run_environment_1d(
    #         args.num_agents,
    #         env_config.length,
    #         env_config.width,
    #         None, None,
    #         f"Eval-{env_config.length}x{env_config.width}",
    #         agents_list=agents_list,
    #         spawn_algo=env_config.spawn_algo,
    #         despawn_algo=env_config.despawn_algo,
    #         timesteps=10000,
    #         vision=train_config.vision,
    #         s_target=env_config.s_target,
    #         apple_mean_lifetime=env_config.apple_mean_lifetime,
    #         epsilon=train_config.epsilon
    #     )


if __name__ == "__main__":

    # widths = [6]
    # for width in widths:
    #     print(evaluate_factory(6, 6, 2))
    evaluate_network(sys.argv[1:])

