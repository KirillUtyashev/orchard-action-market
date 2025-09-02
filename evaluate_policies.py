import glob
import os
import random
import re

import numpy as np
import torch

from actor_critic.actor_critic_following_rates import ActorCriticRatesFixed
from actor_critic.actor_critic_perfect_info import ActorCriticPerfect
from actor_critic.actor_imperfect_critic_perfect import \
    ActorImperfectCriticPerfect
from agents.actor_critic_agent import ACAgent, ACAgentRatesFixed
from agents.simple_agent import SimpleAgent
from config import CHECKPOINT_DIR, DEVICE
from main import eval_performance
from models.actor_network import ActorNetwork
from models.value_function import VNetwork
from policies.random_policy import random_policy
from configs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from orchard.algorithms import spawn_apple, despawn_apple
from value_function_learning.train_value_function import \
    CentralizedValueFunction, DecentralizedValueFunction, \
    DecentralizedValueFunctionPersonal
from agents.communicating_agent import CommAgent, CommAgentPersonal


def evaluate_policy(env_config,
                    num_agents,
                    agent_factory,
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

    # create agents
    agents = [agent_factory(i) for i in range(num_agents)]

    # run the environment
    total_apples, total_picked, picked_per_agent, per_agent, mean_dist, apples_per_sec, same_actions, idle_actions = \
        eval_performance(
            num_agents,
            env_config.length,
            env_config.width,
            None, None,
            f"Eval-{env_config.length}x{env_config.width}",
            agents_list=agents,
            spawn_algo=env_config.spawn_algo,
            despawn_algo=env_config.despawn_algo,
            timesteps=timesteps,
            s_target=env_config.s_target,
            apple_mean_lifetime=env_config.apple_mean_lifetime
        )

    return {
        "total_apples": int(total_apples),
        "total_picked": int(total_picked),
        "picked_per_agent": float(picked_per_agent),
        "ratio_per_agent": float(per_agent),
        "mean_distance": float(mean_dist),
        "apples_per_sec": float(apples_per_sec)
    }


def make_baseline_factory(policy_name):
    """Returns an agent_factory for SimpleAgent(policy=policy_name)."""
    def factory(i):
        return SimpleAgent(policy_name, i)
    return factory


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
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)


    env_config = EnvironmentConfig(
        s_target=0.16,
        apple_mean_lifetime=5.0,
        length=length,
        width=width,
    )

    train_config = TrainingConfig(
        num_agents=num_agents,
        hidden_dimensions=hidden_dim,
        num_layers=4,
        batch_size=num_agents
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
            agent = CommAgent("value_function", nummer, train_config.num_agents)
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

            agent.personal_q_value = 0.0

        if not old:
            load_weights_only(algo.name, [("1", agent.policy_value) for agent in agents_list], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))
    elif algorithm == "DecentralizedPersonal":
        agents_list = []
        algo = DecentralizedValueFunctionPersonal(exp_config)
        # Initialize networks and agents
        for nummer in range(train_config.num_agents):
            agent = CommAgentPersonal("value_function", nummer, train_config.num_agents)
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

            agent.personal_q_value = 0.0

        if not old:
            load_weights_only(algo.name, [("1", agent.policy_value) for agent in agents_list], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))

    elif algorithm == "ActorCritic":
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
    elif algorithm == "ActorCriticRates":
        agents_list = []
        algo = ActorCriticRatesFixed(exp_config)
        # Initialize networks and agents
        for nummer in range(train_config.num_agents):
            agent = ACAgentRatesFixed("learned_policy", train_config.num_agents, nummer, 0)
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
            agent.agent_alphas = np.zeros(num_agents)
            agent.rate = 0.05
            agent.personal_q_value = 0.0

            if old:
                network.function.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, algo.name) + "/" + algo.name + "_actor_network_AC_" + str(nummer) + ".pt"))
            agent.policy_network = network
            agents_list.append(agent)

        if not old:
            load_weights_only(algo.name, [("1", agent.policy_value) for agent in agents_list], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))

    else:
        agents_list = []
        algo = ActorImperfectCriticPerfect(exp_config)
        # Initialize networks and agents
        for nummer in range(train_config.num_agents):
            agent = ACAgent("learned_policy",  nummer)
            if train_config.critic_vision != 0:
                if env_config.width != 1:
                    critic_input_dim = train_config.critic_vision ** 2 + 1
                else:
                    critic_input_dim = train_config.critic_vision + 1
            else:
                critic_input_dim = env_config.length * env_config.width + 1

            # Get actor network vision
            if train_config.actor_vision != 0:
                if env_config.width != 1:
                    actor_input_dim = train_config.actor_vision ** 2 + 1
                else:
                    actor_input_dim = train_config.actor_vision + 1
            else:
                actor_input_dim = env_config.length * env_config.width + 1

            network = ActorNetwork(actor_input_dim, 5 if env_config.width > 1 else 3, train_config.actor_alpha, train_config.discount, train_config.hidden_dimensions_actor, train_config.num_layers_actor)
            policy_value = VNetwork(critic_input_dim, 1, train_config.alpha, train_config.discount, train_config.hidden_dimensions, train_config.num_layers)
            agent.policy_value = policy_value
            agent.agent_alphas = np.zeros(num_agents)
            agent.rate = 0.05
            agent.personal_q_value = 0.0
            agent.policy_network = network
            agents_list.append(agent)

        if not old:
            load_weights_only(algo.name, [("1", agent.policy_value) for agent in agents_list], agents_list, str(os.path.join(CHECKPOINT_DIR, algo.name)))

    algo.agents_list = agents_list
    algo.agent_controller.agents_list = algo.agents_list

    total_apples, total_picked, picked_per_agent, per_agent, mean_dist, apples_per_sec, same_actions, idle_actions = \
        eval_performance(
            num_agents,
            env_config.length,
            env_config.width,
            algo.agent_controller, algo.name,
            agents_list=agents_list,
            spawn_algo=env_config.spawn_algo,
            despawn_algo=env_config.despawn_algo,
            timesteps=10000,
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

    evaluate_network(12, 12, 7, 64, "ActorImperfectCriticPerfect", False)
