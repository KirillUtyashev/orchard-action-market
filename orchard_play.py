import argparse
import logging
import time
import numpy as np
import torch
import sys

from agents.actor_critic_agent import ACAgent
from agents.simple_agent import SimpleAgent
from agents.communicating_agent import CommAgent
from main import step
from models.actor_dc_1d import ActorNetwork
from models.value_function import VNetwork
from orchard.algorithms import single_apple_despawn, single_apple_spawn, spawn_apple_same_pos_once_every_4_steps
from orchard.environment import Orchard
from policies.random_policy import random_policy_1d

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    done = False

    returns = 0

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        _, reward = step(env.agents_list, env)
        returns += reward

        if render:
            env.render()
            print("agents: ", env.agents.tolist())
            print("apples: ", env.apples.tolist())
            time.sleep(0.5)

    print("Returns: ", returns)


def main(episodes, num_ag, orch_length, is_1d, render=False):
    env = initialize_game(num_ag, orch_length, is_1d)

    for episode in range(episodes):
        _game_loop(env, render)


def initialize_game(num_ag, orch_length, is_1d):
    if is_1d:
        height = 1
    else:
        height = orch_length
    S = np.zeros((orch_length, height))
    for j in range(orch_length):
        for i in range(height):
            S[j][i] = 0.04
    phi = 0.2
    agents_list = []
    for i in range(num_ag):
        trained_agent = ACAgent(policy=random_policy_1d, num="f")
        trained_agent.policy_value = VNetwork(orch_length + 1, 0.0002, 0.99)
        trained_agent.policy_value.function.load_state_dict(torch.load(f"/Users/utya.kirill/Desktop/orchard-action-market/policyitchk/DC-RANDOM_2_10/DC-RANDOM_2_10_decen_{i}_it_99.pt"))
        trained_agent.policy_value = ActorNetwork(orch_length + 1, None, 0.00005, None)
        trained_agent.policy_value.function.load_state_dict(torch.load(f"/Users/utya.kirill/Desktop/orchard-action-market/policyitchk/AC-2_10_value/AC-2_10_value_{i}_it_0.pt"))
        agents_list.append(trained_agent)
    env = Orchard(orch_length, num_ag, S, phi, one=is_1d, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    env.initialize(agents_list)
    return env


if __name__ == "__main__":
    # num_agents = sys.argv[1]
    # orchard_length = sys.argv[2]
    # main(1, int(num_agents), int(orchard_length), True)

    main(1, 2, 10, True, True)
