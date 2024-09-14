from main import run_environment_1d
from models.simple_connected_multiple_blind import SCMBNetwork
from models.simple_connected_multiple import SCMNetwork
from orchard.environment import *
import numpy as np
import matplotlib.pyplot as plt
import orchard.environment_one
import random
from models.simple_connected_multiple_dc import SCMNetwork, SimpleConnectedMultiple
from models.actor_dc_1d import ActorNetwork
# from models.simple_connected_multiple_dc_altinput import SCMNetwork, SimpleConnectedMultiple
# from models.actor_dc_1d_altinput import ActorNetwork
from orchard.algorithms import single_apple_spawn, single_apple_despawn


import torch
torch.set_default_dtype(torch.float64)
"""
Find the "Alpha" and "Beta" 
"""

def find_ab(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000, iteration=99):
    for each_agent in agents_list:
        each_agent.alphas = np.zeros(len(agents_list))
        each_agent.alpha_agents = np.zeros(len(agents_list))
        each_agent.beta = 0
        each_agent.times = 0
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env
    print("Finding Alpha/Beta for", name)
    v_network_list = []

    for agn in range(len(agents_list)):
        if iteration == 99:
            network1 = SCMNetwork(orchard_length, alpha, discount)
            agents_list[agn].policy_value = network1
            network1.function.load_state_dict(torch.load(name + "_" + str(agn) + ".pt"))
            v_network_list.append(network1)
        else:
            assert agents_list[agn].policy_value is not None
            v_network_list.append(agents_list[agn].policy_value)

    total_reward = 0

    for i in range(timesteps):

        agent = random.randint(0, env.n - 1)  # Choose random agent
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        action = agents_list[agent].get_action(state, discount, agents_list=agents_list)

        reward, new_position = env.main_step(agents_list[agent].position.copy(),
                                             action)  # Take the action. Observe reward, and get new position
        agents_list[agent].position = new_position.copy()  # Save new position.
        total_reward += reward  # Add to reward.

        new_state = env.get_state()

        beta_sum = 0
        for each_agent in agents_list:
            valued = discount * each_agent.get_value_function(new_state["agents"].copy(), new_state["apples"].copy())[0]
            if each_agent.num == agent:
                valued += reward
                each_agent.times += 1
            each_agent.alpha_agents[agent] += 1
            each_agent.alphas[agent] += valued
            beta_sum += valued
        beta_sum = beta_sum
        agents_list[agent].beta += beta_sum
        if i % 20000 == 0:
            print("At timestep:", i)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)

    total_alpha = 0
    total_beta = 0
    alphas = []
    betas = []
    for each_agent in agents_list:
        print("Agent", str(each_agent.num) + "; Alphas / Alpha / Beta")
        print(list(np.divide(each_agent.alphas, each_agent.alpha_agents)), np.mean(each_agent.alphas) / timesteps, each_agent.beta / each_agent.times)
        total_alpha += np.sum(each_agent.alphas)
        total_beta += each_agent.beta
        betas.append(each_agent.beta / each_agent.times)
        alphas.append(np.divide(np.sum(each_agent.alphas), np.sum(each_agent.alpha_agents)))

    print("Total Alpha:", total_alpha)
    print("Total Beta:", total_beta)

    return alphas, betas

def find_ab_bin(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000, iteration=99):
    for each_agent in agents_list:
        each_agent.alphas = np.zeros(len(agents_list))
        each_agent.alpha_agents = np.zeros(len(agents_list))
        each_agent.beta = 0
        each_agent.times = 0
    print(orchard_length)
    print(len(agents_list))
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env
    print("Experiment", name, "Binary")
    v_network_list = []
    max_alphas = [0, 0, 0, 0]
    min_alphas = [900, 900, 900, 900]


    for agn in range(len(agents_list)):
        if iteration == 99:
            network1 = SCMNetwork(orchard_length, alpha, discount)
            agents_list[agn].policy_value = network1
            network1.function.load_state_dict(torch.load(name + "_" + str(agn) + ".pt"))
            v_network_list.append(network1)
        else:
            assert agents_list[agn].policy_value is not None
            v_network_list.append(agents_list[agn].policy_value)

    total_reward = 0

    for i in range(timesteps):

        agent = random.randint(0, env.n - 1)  # Choose random agent
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        action = agents_list[agent].get_action(state, discount, agents_list=agents_list)

        reward, new_position = env.main_step(agents_list[agent].position.copy(),
                                             action)  # Take the action. Observe reward, and get new position
        agents_list[agent].position = new_position.copy()  # Save new position.
        total_reward += reward  # Add to reward.

        new_state = env.get_state()

        beta_sum = 0
        for numnow, each_agent in enumerate(agents_list):
            if each_agent.avg_alpha is not None:
                valued = discount * each_agent.get_value_function_bin(new_state["agents"].copy(), new_state["apples"].copy())[0]
            else:
                valued = discount * each_agent.get_value_function(new_state["agents"].copy(), new_state["apples"].copy())[0]
            if each_agent.num == agent:
                valued += reward
                each_agent.times += 1
            # valued *= each_agent.agent_rates[agent]
            each_agent.alpha_agents[agent] += 1
            each_agent.alphas[agent] += valued
            beta_sum += valued

            max_alphas[numnow] = max(valued, max_alphas[numnow])
            min_alphas[numnow] = min(valued, min_alphas[numnow])

        agents_list[agent].beta += beta_sum
        if i % 20000 == 0:
            print("At timestep", i)
        # if i == 60000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.0001
        # if i == 100000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00003
        # if i == 150000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00001
        # if i == 200000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.000003
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)

    total_alpha = 0
    total_beta = 0
    alphas = []
    avg_alphas = []
    betas = []
    for each_agent in agents_list:
        print("Agent", str(each_agent.num) + "; Alphas / Alpha / Beta")
        print(list(np.divide(each_agent.alphas, each_agent.alpha_agents)), np.mean(each_agent.alphas) / timesteps, each_agent.beta / each_agent.times)
        total_alpha += np.sum(each_agent.alphas)
        total_beta += each_agent.beta
        betas.append(each_agent.beta / each_agent.times)
        alphas.append(np.divide(each_agent.alphas, each_agent.alpha_agents))
        avg_alphas.append(np.sum(each_agent.alphas) / np.sum(each_agent.alpha_agents))
    print("Max Alphas", max_alphas)
    print("Min Alphas", min_alphas)
    print("Avg Alphas", avg_alphas)

    print("Total Alpha:", total_alpha)
    print("Total Beta:", total_beta)

    return alphas, betas