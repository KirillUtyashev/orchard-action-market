from orchard.environment import *
import random
from models.value_function import VNetwork
from orchard.algorithms import single_apple_spawn, single_apple_despawn
from plots import graph_plots, add_to_plots, setup_plots, init_plots
from helpers import generate_sample_states, convert_position, env_step
from eval_network import eval_network

import torch


torch.set_default_dtype(torch.float64)


def training_loop_c(agents_list, orchard_length, S, phi, discount=0.99, timesteps=100000, iteration=99):
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env

    name = f"C-RANDOM_{len(agents_list)}_{orchard_length}"

    batch_size = 64

    print("Experiment", name)
    total_reward = 0
    network = agents_list[0].policy_value

    sample_state, sample_state5, sample_state6 = generate_sample_states(env.length, len(agents_list))

    """ Plotting Setup """
    one_plot, two_plot, loss_plot, loss_plot1, loss_plot2, ratio_plot = init_plots()
    setup_plots(network.function.state_dict(), one_plot)
    maxi = 0
    """"""
    # --- collect into Python lists, not np.array objects! ---

    for i in range(timesteps):
        batch_states = []
        batch_next_states = []
        batch_rewards = []

        for _ in range(batch_size):
            s, new_s, r = env_step(agents_list, env, i, timesteps, "C")
            batch_states.append(s)
            batch_next_states.append(new_s)
            batch_rewards.append(r)
        # --- now convert once to numpy (or torch) arrays ---
        batch_states = np.stack(batch_states, axis=0).squeeze()   # shape (batch_size, state_dim)
        batch_next_states = np.stack(batch_next_states, axis=0)
        network.train(batch_states, batch_next_states, batch_rewards)

        """ For Plotting """
        if i % 1000 == 0:
            add_to_plots(network.function.state_dict(), i, one_plot)
            v_value = agents_list[0].get_value_function(np.concatenate([sample_state["agents"], sample_state["apples"]], axis=0).T)
            v_value1 = agents_list[0].get_value_function(np.concatenate([sample_state5["agents"], sample_state5["apples"]], axis=0).T)
            v_value2 = agents_list[0].get_value_function(np.concatenate([sample_state6["agents"], sample_state6["apples"]], axis=0).T)
            print("P", v_value)
            loss_plot.append(v_value.item())
            loss_plot1.append(v_value1[0])
            loss_plot2.append(v_value2[0])

        # For centralized learning, the learning rates should be lower because we are learning a more complex function
        # and hence the observations are noiser --> mitigate this with a lower learning rate
        if i == 782:
            for g in network.optimizer.param_groups:
                g['lr'] = 0.0008
        if i == 1562:
            for g in network.optimizer.param_groups:
                g['lr'] = 0.0002
        if i == 3125:
            for g in network.optimizer.param_groups:
                g['lr'] = 0.00005
        if i == timesteps - 1:
            print("=====Eval at", i, "steps======")
            fname = name
            maxi, ratio = eval_network(fname, maxi, len(agents_list), [network], side_length=orchard_length, iteration=iteration)
            ratio_plot.append(ratio)
            print("=====Completed Evaluation=====")
    graph_plots(network.function.state_dict(), name, one_plot, loss_plot, loss_plot1, loss_plot2, ratio_plot)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)


"""
Decentralized Q-value function training.
"""


def training_loop_d(agents_list, orchard_length, S, phi, alpha, discount=0.99, timesteps=100000, iteration=99):
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env

    name = f"DC-RANDOM_{len(agents_list)}_{orchard_length}"

    print("Experiment", name)
    network_list = []
    for agn in range(len(agents_list)):
        network = VNetwork(orchard_length + 1, alpha, discount)
        agents_list[agn].policy_value = network
        network_list.append(network)
    total_reward = 0

    batch_size = 64

    sample_state, sample_state5, sample_state6 = generate_sample_states(env.length, len(agents_list))

    """ Plotting Setup """
    one_plot, two_plot, loss_plot, loss_plot1, loss_plot2, ratio_plot = init_plots()
    setup_plots(network_list[0].function.state_dict(), one_plot)
    """"""
    maxi = 0
    for i in range(timesteps):
        for _ in range(batch_size):
            s, new_s, r, old_pos, agent = env_step(agents_list, env, i, timesteps, "DC")
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
        for each_agent in range(len(agents_list)):
            agents_list[each_agent].policy_value.train()

        """ For Plotting """
        if i % 500 == 0 and i != 0:
            add_to_plots(network_list[0].function.state_dict(), i, one_plot)
            v_value = agents_list[0].get_comm_value_function(sample_state["agents"], sample_state["apples"], agents_list, agent_poses=sample_state["poses"], debug=True)
            v_value1 = agents_list[0].get_comm_value_function(sample_state5["agents"], sample_state5["apples"], agents_list, agent_poses=sample_state5["poses"], debug=True)
            v_value2 = agents_list[0].get_comm_value_function(sample_state6["agents"], sample_state6["apples"], agents_list, agent_poses=sample_state6["poses"], debug=True)
            print("P", v_value)
            loss_plot.append(v_value[0])
            loss_plot1.append(v_value1[0])
            loss_plot2.append(v_value2[0])
        if i == 782:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0005
        if i == 1562:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001
        if i == 3125:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if (i % 2500 == 0 and i != 0) or i == timesteps - 1:
            print("=====Eval at", i, "steps======")
            fname = name
            maxi, reward = eval_network(fname, maxi, len(agents_list), network_list, side_length=orchard_length, iteration=iteration)
            print("=====Completed Evaluation=====")
    graph_plots(network_list[0].function.state_dict(), name, one_plot, loss_plot, loss_plot1, loss_plot2, ratio_plot)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)
