import glob
import os
import random
import re
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

from actor_critic.actor_critic_following_rates import ActorCriticRatesFixed
from actor_critic.actor_critic_perfect_info import ActorCriticPerfect
from actor_critic.actor_imperfect_critic_perfect import \
    ActorImperfectCriticPerfect
from agents.actor_critic_agent import ACAgent, ACAgentRatesFixed
from agents.agent import AgentInfo
from agents.simple_agent import SimpleAgent
from config import CHECKPOINT_DIR, DEVICE
from helpers.controllers import ViewController, ViewControllerOrchardSelfless
from helpers.helpers import create_env
from main import eval_performance
from metrics.metrics import plot_agent_heatmap_alpha, plot_agents_trajectories
from models.actor_network import ActorNetwork
from models.value_function import VNetwork
from orchard.environment import OrchardBasic
from policies.random_policy import random_policy
from configs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from orchard.algorithms import spawn_apple, despawn_apple
from run_experiments import parse_args, set_config
from value_function_learning.train_value_function import \
    CentralizedValueFunction, DecentralizedValueFunction, \
    DecentralizedValueFunctionPersonal
from algorithm import ENV_MAP


def evaluate_policy(env_config,
                    num_agents,
                    orchard,
                    timesteps=10000,
                    seed=42069):
    """
    Runs `eval_performance` with agents created by `agent_factory` and
    returns a dict of metrics.

    agent_factory: fn(i: int) -> Agent
    """
    # seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agents_list = [SimpleAgent(AgentInfo(policy=random_policy)) for _ in range(num_agents)]

    env = create_env(env_config, num_agents, None, None, agents_list, ENV_MAP[orchard])

    # run the environment
    total_apples, total_picked, picked_per_agent, per_agent, mean_dist, apples_per_sec, same_actions, idle_actions = \
        eval_performance(
            num_agents,
            None,
            env,
            "Random",
            timesteps,
            agents_list
        )

    return {
        "total_apples": int(total_apples),
        "total_picked": int(total_picked),
        "picked_per_agent": float(picked_per_agent),
        "ratio_per_agent": float(per_agent),
        "mean_distance": float(mean_dist),
        "apples_per_sec": float(apples_per_sec)
    }


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
    path = os.path.join(CHECKPOINT_DIR, name)

    # 1) Find the newest ckpt_{number}.pt by numeric suffix
    candidates = glob.glob(os.path.join(path, "*_ckpt_*.pt"))

    latest_step = None
    latest_path = None

    for f in candidates:
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(f))
        if m:
            step = int(m.group(1))
            if latest_step is None or step > latest_step:
                latest_step, latest_path = step, f

    # Fallback to legacy file if no step-tagged snapshot is present
    if latest_path is None:
        latest_path = os.path.join(path, f"{name}_ckpt.pt")

    ckpt = torch.load(latest_path, map_location="cpu")

    # 2) Set global 'times' to the detected step (or ckpt['step'] if present)
    step_in_ckpt = ckpt.get("step")
    final_step = step_in_ckpt if isinstance(step_in_ckpt, int) else (latest_step or 0)
    global times
    times = final_step

    # Map critics (skip RNG/env state)
    crit_blobs = ckpt.get("critics", [])
    for (name_now, obj_now), saved in zip(networks, crit_blobs):
        obj_now.import_net_state(saved["blob"])

    # Map actors (aligned with agents_list)
    act_blobs = ckpt.get("actors", [])
    for agent, blob in zip(agents_list, act_blobs):
        if blob and hasattr(agent, "policy_network") and agent.policy_network is not None:
            agent.policy_network.import_net_state(blob, device=DEVICE)


def evaluate_factory(length, width, num_agents, orchard):
    random.seed(42069)
    np.random.seed(42069)
    torch.manual_seed(42069)

    env_config = EnvironmentConfig(
        s_target=0.16,
        apple_mean_lifetime=5,
        length=length,
        width=width
    )

    return evaluate_policy(
        env_config,
        num_agents=num_agents,
        orchard=orchard,
        timesteps=10000,
        seed=42069
    )


def evaluate_network(args):
    args = parse_args(args)
    env_config, train_config = set_config(args)

    exp_config = ExperimentConfig(
        env_config=env_config,
        train_config=train_config,
    )

    if args.algorithm == "Centralized":
        algo = CentralizedValueFunction(exp_config)
    elif args.algorithm == "Decentralized":
        algo = DecentralizedValueFunction(exp_config)
    elif args.algorithm == "DecentralizedPersonal":
        algo = DecentralizedValueFunctionPersonal(exp_config)
    elif args.algorithm == "ActorCritic":
        algo = ActorImperfectCriticPerfect(exp_config)
    elif args.algorithm == "ActorCriticRates":
        algo = ActorCriticRatesFixed(exp_config)
    else:
        algo = ActorImperfectCriticPerfect(exp_config)
    algo.build_experiment(view_controller_cls=ViewController if algo.env_cls is OrchardBasic else ViewControllerOrchardSelfless)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    agents, controller = algo.init_agents_for_eval()
    env = algo.create_env(None, None, agents_list=agents, env_cls=algo.env_cls)
    for agent in agents:
        agent.personal_q_value = 0.0
        agent.apples_picked = [[0] for _ in range(args.num_agents)]

    personal_q_values, agent_distance_hist = eval_performance(args.num_agents, controller, env, algo.name, 10000, agents, train_config.epsilon, True)

    def average_every_n(arr, n=500):
        """Average consecutive n points in 1D array."""
        length = len(arr) // n * n  # trim remainder
        arr = arr[:length]
        return arr.reshape(-1, n).mean(axis=1)

    plt.figure()
    for agent_id in range(args.num_agents):
        arr = personal_q_values[agent_id]
        smoothed = average_every_n(np.array(arr), n=150)
        x_axis = np.arange(len(smoothed)) * 50  # step numbers
        plt.plot(x_axis, smoothed, label=f"Agent {agent_id}")

    plt.xlabel("Step")
    plt.ylabel("Q-value")
    plt.legend()
    plt.show()

    for agent_id in range(args.num_agents):
        plt.figure()
        for agent_id_2 in range(args.num_agents):
            arr = agents[agent_id].apples_picked[agent_id_2]
            smoothed = average_every_n(np.array(arr), n=150)
            x_axis = np.arange(len(smoothed)) * 50  # step numbers
            plt.plot(x_axis, smoothed, label=f"Agent {agent_id_2}")

        plt.xlabel("Step")
        plt.ylabel("Apples Picked By Agent ID")
        plt.legend()
        plt.show()

    positions = np.load(f"positions/{algo.name}_pos.npy")      # shape (T, N, 2)

    for agent_id in range(args.num_agents):
        plot_agent_heatmap_alpha(positions, agent_ids=[agent_id], colors=["red"])
        plt.show()

    for agent_id in range(args.num_agents):
        plot_agents_trajectories(positions, agent_ids=[agent_id], colors=["royalblue"])
        plt.show()


if __name__ == "__main__":
    evaluate_network(sys.argv[1:])
