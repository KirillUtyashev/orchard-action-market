import os

import numpy as np
from matplotlib import pyplot as plt

from actor_critic_kirill import train_ac_value
from agents.actor_critic_agent import ACAgent
from agents.actor_critic_alloc_agent import ACAgent as ACAAgent
from config import CHECKPOINT_DIR
from models.value_function import VNetwork
from policies.random_policy import random_policy_1d
from models.actor_dc_1d import ActorNetwork
from value_function_learning.train_value_function import training_loop_d
from config import get_config
import torch

torch.set_default_dtype(torch.float64)

"""
The main training loop for *Policy Iteration*. Includes provisions for training in the middle of an iteration (i.e. after Q training, but before AC training).
"""


def policy_iteration(approach, alpha):
    num_agents = get_config()["num_agents"]
    orchard_size = get_config()["orchard_length"]
    name = "AC-" + str(num_agents) + "_" + str(orchard_size)

    agents_list = []
    # if approach == "rate":
    #     for i in range(num_agents):
    #         agents_list.append(ACAAgent(policy=random_policy_1d, num=i, num_agents=num_agents))
    # else:
    for i in range(num_agents):
        agents_list.append(ACAgent(policy="learned_policy", num=i, num_agents=num_agents))
    path = os.path.join(CHECKPOINT_DIR, name + "_" + approach)
    # We want to combine actor and critic into 1 step

    # Понедельник
    # Create a loop that does all the same things as value function learning but also incorporate actor network and change policy actions:
    # 1. Use softmax + temperature (decay too) for choosing actions
    # 2. Tweak Actor Network to make it similar to value function one

    # Вторник
    # Возможно придется посидеть на 4-20, чтобы понять как улучшить обучение:
    # Выписать параметры, которые могу менять, поменять их, сделать определенные гипотезы

    for nummer, agn in enumerate(agents_list):
        agn.policy_network = ActorNetwork(orchard_size + 1, agents_list, alpha, get_config()["discount"])
        agn.policy_value = VNetwork(get_config()["orchard_length"] + 1, 0.0005, get_config()["discount"])
        agn.policy = "learned_policy"

    # Perform actor-critic training
    if approach == "value":
        return train_ac_value(orchard_size, num_agents, agents_list, name + "_" + approach, 0.99, 25000)
    # elif approach == "beta":
    #     train_ac_beta(orchard_size, num_agents, agents_list, name + "_" + approach, get_config()["discount"], 600000)
    # elif approach == "rate":
    #     train_ac_rate(orchard_size, num_agents, agents_list, name + "_" + approach, get_config()["discount"], 600000)
    # elif approach == "binary":
    #     train_ac_binary(orchard_size, num_agents, agents_list, name + "_" + approach, get_config()["discount"], 600000)


def generate_learning_rates(start=0.004, end=0.000004, num=8):
    """
    Return `num` values geometrically spaced between `start` and `end`.
    """
    return np.geomspace(start, end, num=num)


def plot_learning_rate_vs_reward(results):
    """
    Plot learning rate vs. final reward ratio.

    Args:
      results: dict mapping lr -> reward_ratio
    """
    lrs = sorted(results.keys())
    reward_ratios = [results[lr] for lr in lrs]

    plt.figure()
    plt.xscale('log')
    plt.plot(lrs, reward_ratios, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Reward Ratio')
    plt.title('Learning Rate vs Final Reward Ratio')
    plt.grid(True, which="both", ls="--")
    plt.show()


def sweep_learning_rates_and_plot(
        start=0.01,
        end=0.0004,
        num_rates=4
):
    """
    Runs `train_fn` across a geometric grid of learning rates, collects the final
    reward ratio for each, and plots the results.

    Args:
      train_fn: function(learning_rate=X, **train_kwargs) -> final reward ratio (float)
      train_kwargs: dict of other fixed kwargs for train_fn (e.g. num_episodes, env)
      start, end: endpoints for LR grid
      num_rates: how many rates to try

    Returns:
      results dict mapping lr -> final reward ratio
    """
    lrs = generate_learning_rates(start, end, num_rates)
    results = {}

    for lr in lrs:
        print(f"Training with lr = {lr:.5g}")
        reward_ratio = policy_iteration("value", lr)
        print(reward_ratio)
        results[lr] = reward_ratio

    plot_learning_rate_vs_reward(results)
    return results

# Example usage:
# def train_agent(learning_rate, num_episodes, env, agent_class, **kwargs):
#     agent = agent_class(lr=learning_rate, **kwargs)
#     return agent.train(num_episodes, env)  # should return final reward ratio
#
# results = sweep_learning_rates_and_plot(
#     train_fn=train_agent,
#     train_kwargs={
#         'num_episodes': 1000,
#         'env': my_env,
#         'agent_class': MyRLAgent
#     },
#     start=0.05,
#     end=0.0004,
#     num_rates=10
# )


if __name__ == "__main__":
    """
    Call the policy iteration function (from main_pi.py).
    Commences training.
    """
    import random
    import numpy as np
    random.seed(10)
    np.random.seed(10)

    policy_iteration("value", 0.00005)

    # sweep_learning_rates_and_plot()


