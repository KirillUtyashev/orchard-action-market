from ac_utilities import find_ab, find_ab_bin
from alloc.allocation import find_allocs
from orchard.environment import *
import numpy as np
import random
from policies.random_policy import random_policy_1d
from plots import graph_plots, add_to_plots, setup_plots, init_plots
from orchard.algorithms import single_apple_spawn, single_apple_despawn
from helpers import convert_position, env_step, generate_sample_states, \
    eval_network

import torch
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def training_loop(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000, iteration=99, has_beta=False, betas_list=None, altname=None):
    print(orchard_length)
    print(len(agents_list))
    print("Using Beta Value:", has_beta)

    if has_beta:
        assert betas_list is not None

    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env
    print("Experiment", name)
    v_network_list = []
    p_network_list = []

    for agn in range(len(agents_list)):
        assert agents_list[agn].policy_value is not None
        v_network_list.append(agents_list[agn].policy_value)

        assert agents_list[agn].policy_network is not None
        p_network_list.append(agents_list[agn].policy_network)

    total_reward = 0

    """ Plotting Setup """
    one_plot, two_plot, loss_plot, loss_plot1, loss_plot2, ratio_plot = init_plots()
    setup_plots(p_network_list[0].function.state_dict(), one_plot)
    maxi = 0
    """"""
    sample_state, sample_state5, sample_state6 = generate_sample_states(orchard_length, len(agents_list))

    for i in range(timesteps):
        s, new_s, r, old_pos, agent = env_step(agents_list, env, i, timesteps, "AC")
        total_reward += r  # Add to reward.
        for each_agent in range(len(agents_list)):
            if each_agent == agent:
                agents_list[each_agent].policy_value.add_experience(np.concatenate([s, convert_position(old_pos)], axis=0),
                                                                    np.concatenate([new_s, convert_position(agents_list[each_agent].position)], axis=0),
                                                                    None,
                                                                    r)
            else:
                agents_list[each_agent].policy_value.add_experience(np.concatenate([s, convert_position(agents_list[each_agent].position)], axis=0),
                                                                    np.concatenate([new_s, convert_position(agents_list[each_agent].position)], axis=0),
                                                                    None,
                                                                    0)





        p_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)
        if i % 1000 == 0:
            # Train every 1k steps for stability
            add_to_plots(p_network_list[0].function.state_dict(), i, one_plot)
            v_value = agents_list[1].policy_network.get_function_output(sample_state["agents"], sample_state["apples"], pos=sample_state["poses"][0])
            v_value1 = agents_list[1].policy_network.get_function_output(sample_state5["agents"], sample_state5["apples"],
                                                                         pos=sample_state5["poses"][1])
            v_value2 = agents_list[0].policy_network.get_function_output(sample_state6["agents"],
                                                                         sample_state6["apples"],
                                                                         pos=sample_state6["poses"][0])
            if i % 20000 == 0:
                print("Sample Value:", v_value)
            loss_plot.append(v_value[0])
            loss_plot1.append(v_value1[0])
            loss_plot2.append(v_value2[0])
            for ntwk in p_network_list:
                ntwk.train_multiple(agents_list)

        if i % 20000 == 0 and i != 0:
            print("At timestep: ", i)
        if i == 50000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.001
        if i == 100000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0005
        if i == 200000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001
        if i == 300000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if i == 740000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if i == 860000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00002
        if i == 1000000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00001
        if (i % 50000 == 0 and i != 0) or i == timesteps - 1:
            print("=====Eval at", i, "steps======")
            fname = name
            if altname is not None:
                fname = altname
            maxi = eval_network(fname, maxi, network_list=p_network_list, iteration=iteration, num_agents=len(agents_list), side_length=orchard_length)
            print("=====Completed Evaluation=====")
    graph_plots(p_network_list[0].function.state_dict(), name, one_plot, loss_plot, loss_plot1, loss_plot2)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)



from agents.actor_critic_agent import ACAgent

def train_ac_value(side_length, num_agents, agents_list, name, discount, timesteps, iteration=0):
    training_loop(agents_list, side_length, None, None, 0.0013, name, discount, timesteps=timesteps, iteration=iteration)


def train_ac_beta(side_length, num_agents, agents_list, name, discount, timesteps, iteration=0):
    alphas, betas = find_ab(agents_list, side_length, None, None, 0.0003, name, discount=discount, timesteps=20000, iteration=iteration)

    training_loop(agents_list, side_length, None, None, 0.0013, name, betas_list=betas, has_beta=True, discount=discount,
                  timesteps=timesteps, iteration=iteration)


def train_ac_rate(side_length, num_agents, agents_list, name, discount, timesteps, iteration=0):
    # step 1 - find normal A/B
    alphas1, betas1 = find_ab(agents_list, side_length, None, None, 0.0003, name, discount=0.99, timesteps=20000, iteration=iteration)

    # step 2 - get allocs and regenerate
    allocs = []
    for i in range(num_agents):
        allocs.append(find_allocs(alphas1[i]))
        allocsf = (1 - np.exp(-allocs[i]))
        agents_list[i].set_alloc_rates(allocsf, 1)

    alphas1, betas1 = find_ab(agents_list, side_length, None, None, 0.0003, name, discount=0.99, timesteps=20000, iteration=iteration) # regen with rates

    print(betas1)

    training_loop(agents_list, side_length, None, None, 0.0013, name, betas_list=betas1, has_beta=True, discount=discount,
                  timesteps=timesteps, iteration=iteration)


def train_ac_binary(side_length, num_agents, agents_list, name, discount, timesteps, iteration=0):
    avg_alphas, betas = find_ab(agents_list, side_length, None, None, 0.0003, name, discount=discount, timesteps=20000, iteration=iteration)
    print(avg_alphas)
    for ida, agent in enumerate(agents_list):
        agent.avg_alpha = avg_alphas[ida]
    alphas, betas = find_ab_bin(agents_list, side_length, None, None, 0.0003, name, discount=discount, timesteps=20000,
                                iteration=iteration)
    print(betas)
    training_loop(agents_list, side_length, None, None, 0.0013, name, betas_list=betas, has_beta=True,
                  discount=discount,
                  timesteps=timesteps, iteration=iteration)
