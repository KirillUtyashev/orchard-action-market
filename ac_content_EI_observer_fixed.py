import random
import numpy as np
# random.seed(35279038)
random.seed(35279038)
# np.random.seed(389043)
np.random.seed(389045)
import time
import torch.nn.functional as F
from ac_content_utilities import find_ab, find_ab_bin, find_ab_content_infl
from actor_critic import eval_network
from alloc.allocation import find_allocs, roundabout_find_allocs, roundabout_find_allocs_with_b0, \
    roundabout_find_allocs_with_b0_full_vec
from main import run_environment_1d
from models.simple_connected_multiple import SCMNetwork
from orchard.environment import *
import matplotlib.pyplot as plt

from policies.random_policy import random_policy_1d, random_policy
from models.simple_connected_multiple_dc import SCMNetwork, SimpleConnectedMultiple
# from models.actor_dc_1d import ActorNetwork
from models.content_actor_1d import ActorNetwork
from models.content_observer_1d import ObserverNetwork, ten

forever_discount = 0.75
onemdiscount = 1 - forever_discount

#from models.actor_dc_1d_altinput import ActorNetwork # USING ALTINPUT
#from models.simple_connected_multiple_dc_altinput import SCMNetwork, SimpleConnectedMultiple
from orchard.algorithms import single_apple_spawn, single_apple_despawn

import torch

torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True


"""
Some variables / parameters for the *content market*.
"""
def load_adjs(agents_list, filename):
    # adjs = np.load(filename)
    adjs = [[79, 35, 53, 54, 38], [5, 46, 34, 70, 3], [12, 64, 80, 78, 82], [1, 21, 90, 75, 87], [51, 10, 53, 44, 13], [1, 91, 61, 18, 42], [38, 67, 63, 87, 79], [79, 42, 19, 33, 71], [46, 51, 26, 53, 95], [90, 99, 41, 30, 20], [4, 72, 14, 22, 13], [64, 29, 96, 90, 94], [2, 77, 29, 56, 53], [4, 10, 75, 21, 98], [10, 22, 17, 61, 89], [23, 57, 84, 27, 85], [60, 66, 94, 85, 79], [14, 41, 71, 29, 53], [5, 55, 62, 64, 24], [7, 49, 46, 61, 25], [9, 93, 41, 80, 36], [3, 13, 92, 60, 49], [10, 14, 85, 52, 51], [15, 91, 35, 36, 72], [18, 96, 85, 38, 77], [19, 44, 80, 42, 35], [8, 58, 63, 77, 95], [15, 88, 47, 98, 43], [91, 87, 76, 36, 74], [11, 12, 17, 34, 70], [9, 94, 35, 62, 84], [72, 69, 68, 73, 44], [70, 82, 73, 61, 41], [7, 38, 91, 74, 96], [1, 29, 44, 63, 50], [0, 23, 25, 30, 43], [20, 23, 28, 69, 50], [49, 70, 68, 76, 38], [0, 6, 24, 33, 37], [76, 93, 88, 70, 80], [62, 60, 66, 42, 64], [9, 17, 20, 32, 49], [5, 7, 25, 40, 74], [27, 35, 82, 62, 78], [4, 25, 31, 34, 74], [99, 76, 57, 79, 67], [1, 8, 19, 99, 71], [27, 75, 82, 49, 99], [91, 67, 92, 99, 81], [19, 21, 37, 41, 47], [34, 36, 71, 68, 64], [4, 8, 22, 72, 76], [22, 88, 68, 81, 92], [0, 4, 8, 12, 17], [0, 61, 74, 75, 95], [18, 75, 85, 60, 88], [12, 58, 62, 71, 89], [15, 45, 98, 77, 80], [26, 56, 63, 95, 68], [73, 67, 77, 92, 90], [16, 21, 40, 55, 95], [5, 14, 19, 32, 54], [18, 30, 40, 43, 56], [6, 26, 34, 58, 73], [2, 11, 18, 40, 50], [89, 82, 86, 87, 73], [16, 40, 81, 94, 67], [6, 45, 48, 59, 66], [31, 37, 50, 52, 58], [31, 36, 93, 78, 90], [1, 29, 32, 37, 39], [7, 17, 46, 50, 56], [10, 23, 31, 51, 84], [31, 32, 59, 63, 65], [28, 33, 42, 44, 54], [3, 13, 47, 54, 55], [28, 37, 39, 45, 51], [12, 24, 26, 57, 59], [2, 43, 69, 83, 97], [0, 6, 7, 16, 45], [2, 20, 25, 39, 57], [48, 52, 66, 98, 94], [2, 32, 43, 47, 65], [78, 86, 88, 84, 92], [15, 30, 72, 83, 86], [15, 16, 22, 24, 55], [65, 83, 84, 98, 97], [3, 6, 28, 65, 89], [27, 39, 52, 55, 83], [14, 56, 65, 87, 93], [3, 9, 11, 59, 69], [5, 23, 28, 33, 48], [21, 48, 52, 59, 83], [20, 39, 69, 89, 96], [11, 16, 30, 66, 81], [8, 26, 54, 58, 60], [11, 24, 33, 93, 97], [78, 86, 96], [13, 27, 57, 81, 86], [9, 45, 46, 47, 48]]

    for agnum, agent in enumerate(agents_list):
        agent.adjs = adjs[agnum]
        print(agnum, list(adjs[agnum]))
    #create_symm_adjs(agents_list)
    check_connected(agents_list)

def create_symm_adjs(agents_list):
    for agnum, agent in enumerate(agents_list):
        agent.adjs = []
        for ij in range(-3, 4):
            nadj = agent.num + ij
            if nadj < 0:
                nadj = len(agents_list) + nadj
            if nadj > len(agents_list) - 1:
                nadj = nadj - len(agents_list)
            if nadj != agent.num:
                agent.adjs.append(nadj)
    print("Symmetric Adjacencies")

def check_connected(agents_list):
    visited = []
    connected = [0]
    curr = 0
    for adj in agents_list[curr].adjs:
        visited.append(adj)
    while len(visited) != 0:
        next = visited[0]
        visited.pop(0)
        connected.append(next)
        for adj in agents_list[next].adjs:
            if adj not in visited and adj not in connected:
                visited.append(adj)

    if len(connected) == 100:
        print("connected")
    else:
        print("nope")


def create_adjs(agents_list):
    neighbours = 5
    lister = list(range(0, num_agents))
    time1 = time.time()
    adjacencies = []
    for agent in agents_list:
        while len(agent.adjs) != neighbours:
            if len(lister) == 1 and lister[0] == agent.num:
                break
            tar = np.random.choice(lister)
            if tar != agent.num and tar not in agent.adjs and len(agents_list[tar].adjs) != neighbours:
                agent.adjs.append(tar)
                agents_list[tar].adjs.append(agent.num)
                if len(agent.adjs) == neighbours:
                    lister.remove(agent.num)
                if len(agents_list[tar].adjs) == neighbours:
                    lister.remove(tar)
            cur = time.time()
            if cur - time1 > 10:
                print("failed")
                break
    print("done")
    for agent in agents_list:
        adjacencies.append(agent.adjs)
    print(adjacencies)
    np.save("adj.npy", np.array(adjacencies))
    print("saved")

def get_discounted_value(old, new, discount):
    if old < 0:
        return new
    return old * (1 - discount) + new * discount

def training_loop(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000,
                  has_beta=False, betas_list=None, altname=None, scenario=0, in_thres=0.00005, folder=''):

    load_adjs(agents_list, "adj.npy")
    #create_adjs(agents_list) # If needed; load from adj.npy instead

    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env
    num_agents = len(agents_list)
    print("Experiment", name)

    p_network_list = []  # Producer Networks
    o_network_list = []  # Observer Networks
    i_network_list = []  # Influencer Networks
    for agn in range(len(agents_list)):
        p_network_list.append(agents_list[agn].policy_network)
        o_network_list.append(agents_list[agn].follower_network)
        i_network_list.append(agents_list[agn].influencer_network)

    total_reward = 0

    """Construct Sample States"""
    sample_state0 = {
        "agents": np.array([[0]] * side_length),
        "apples": np.array([[0]] * side_length),
        "pos": [np.array([4, 0])] * num_agents,
    }
    sample_state1 = {
        "agents": np.array([[0]] * side_length),
        "apples": np.array([[0]] * side_length),
        "pos": [np.array([5, 0])] * num_agents,
    }
    sample_state2 = {
        "agents": np.array([[0]] * side_length),
        "apples": np.array([[0]] * side_length),
        "pos": [np.array([6, 0])] * num_agents,
    }

    """ Plotting Setup """
    singulars = []
    one_singulars = []

    """
    Plotting setup for Emergent Influencer
    """
    follow_plots = []
    follow_indices = []
    follow_setup = False

    sw_plot = []
    att_plot = []
    rep_plots = []
    agent_sw_plots = []
    real_rep_plots = []
    public_rep_plots = []
    external_plots = []


    indirect_plots = []
    indirect_plots2 = []
    direct_plots2 = []
    direct_plots = []
    peragrep_plots = []
    peragrep_plots2 = []
    peragprod_plots = []

    followrates_plots = []

    rep_estimate_plots = []

    ext_plots_x = []
    ext_plots_y = []
    for ll in agents_list:
        external_plots.append([])
        ext_plots_x.append([])
        ext_plots_y.append([])
        rep_plots.append([])
        agent_sw_plots.append([])
        real_rep_plots.append([])
        public_rep_plots.append([])
        rep_estimate_plots.append([])
        indirect_plots.append([])
        indirect_plots2.append([])
        direct_plots2.append([])
        peragrep_plots.append([])
        peragrep_plots2.append([])
        peragprod_plots.append([])
        direct_plots.append([])

        followrates_plots.append([])
        for rr in range(len(agents_list)):
            indirect_plots[ll.num].append([])
            indirect_plots2[ll.num].append([])
            direct_plots2[ll.num].append([])
            direct_plots[ll.num].append([])
            peragrep_plots[ll.num].append([])
            peragrep_plots2[ll.num].append([])
            followrates_plots[ll.num].append([])
    """
    End Plotting Setup
    """
    gossip_timestep = 1000
    round_actions = np.zeros(num_agents)
    lkdown = False

    for agent1 in agents_list:

        for ag_num in range(len(agents_list)):
            agent1.alphas_asinfl[ag_num] = -0.00001 #0
            agent1.alphas_asinfl_raw[ag_num] = -0.00001 #0
            agent1.alphas[ag_num] = -0.00001 #0
            agent1.alphas_raw[ag_num] = -0.00001 #0
            agent1.indirect_alphas[ag_num] = 0 #0
            agent1.indirect_alphas_raw[ag_num] = 0 #0
        if agent1.num != 50:
            agent1.target_influencer = 50
            agents_list[50].followers.append(agent1.num)

    for i in range(timesteps):
        # agent = random.randint(0, env.n - 1)  # Choose random agent
        agent = i % num_agents
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        action = agents_list[agent].get_action(state, discount)  # Get action.

        round_actions[agent] = action
        peragprod_plots[agent].append(action)

        new_state = env.get_state()

        sp_state = {
            "agents": state["agents"].copy(),
            "apples": state["apples"].copy(),
            "pos": old_pos.copy()
        }
        sp_new_state = {
            "agents": new_state["agents"].copy(),
            "apples": new_state["apples"].copy(),
            "pos": agents_list[agent].position.copy()
        }
        reward = 0
        """
        Alpha/Beta in-place calculation
        """
        beta_sum = 0

        # Influencer Retweet

        for numnow, each_agent in enumerate(agents_list):
            """
            This is the INFLUENCER receiving feedback section.
            - If you have more than 0 followers, you "share" the content to your followers based on your "each_agent.agent_rates[agent]" rate.
            - Then, this becomes your alphas_asinfl (for this timestep).
            """
            if len(each_agent.followers) > 0:
                for agent_num in range(len(agents_list)): # alpha for agent_num while each_agent is an INFL
                    if agent_num != agent:
                        each_agent.alphas_asinfl[agent_num] = get_discounted_value(each_agent.alphas_asinfl[agent_num], 0, each_agent.discount_factor)
                        each_agent.alphas_asinfl_raw[agent_num] = get_discounted_value(each_agent.alphas_asinfl_raw[agent_num],0, each_agent.discount_factor)

                tot_val = 0
                tot_val_raw = 0
                for foll_agent in agents_list:
                    if foll_agent.target_influencer == each_agent.num:
                        bval = each_agent.agent_rates[agent] * foll_agent.get_util_pq_raw(action,
                                                                                              agents_list[agent])
                        tot_val += bval * foll_agent.infl_rate
                        tot_val_raw += bval

                beta_sum += tot_val
            else:
                """
                This is the FOLLOWER receiving UTILITY section.
                - If you have 0 followers, then this is just how much stuff you get from directly following this agent.
                """
                valued = each_agent.get_util_pq(action, agents_list[agent])
                raw_value = each_agent.get_util_pq_raw(action, agents_list[agent])
                if each_agent.num == agent:
                    valued = 0
                    each_agent.times += 1
                # valued *= each_agent.agent_rates[agent]
                each_agent.alpha_agents[agent] += 1
                # each_agent.alphas[agent] += valued

                for agent_num in range(len(agents_list)):
                    if agent_num != agent:
                        each_agent.alphas[agent_num] = get_discounted_value(each_agent.alphas[agent_num], 0,
                                                                               each_agent.discount_factor)
                        each_agent.alphas_raw[agent_num] = get_discounted_value(each_agent.alphas_raw[agent_num],
                                                                                   0,
                                                                                   each_agent.discount_factor)

                each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], valued, each_agent.discount_factor)
                each_agent.alphas_raw[agent] = get_discounted_value(each_agent.alphas_raw[agent], raw_value,
                                                                each_agent.discount_factor)

                beta_sum += valued

            # Here, we use valued to build a "sharing" / indirect alpha system
            # Each agent keeps track of ALPHA && INDIRECT ALPHA for EVERY AGENT
            """
            This is the FOLLOWER getting util from the INFLUENCER section.
            - If you have more than 0 followers, you "share" the content to your followers based on your "each_agent.agent_rates[agent]" rate.
            - Then, the utility you provide THEM is added to their indirect_alphas.
            """
            if len(each_agent.followers) > 0:
                tot_val = 0
                tot_val_raw = 0
                for indirect_num, indirect_agent in enumerate(agents_list):
                    if indirect_num == agent:
                        indirect_agent.indirect_alphas[numnow] *= (1 - indirect_agent.discount_factor)
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)

                    if indirect_num == numnow or agents_list[indirect_num].target_influencer != each_agent.num:
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)

                    elif indirect_agent.target_influencer == each_agent.num:
                        bval = each_agent.agent_rates[agent] * indirect_agent.get_util_pq_raw(action,
                                                                                              agents_list[agent])
                        #inflrate = max(min(1, indirect_agent.infl_rate), 0.02 * len(each_agent.followers) / 100)
                        inflrate = indirect_agent.infl_rate
                        #inflrate = ((each_agent.base_budget - each_agent.raw_b0_rate) / each_agent.base_budget) * 0.5 + 0.5
                        # if inflrate > -1:
                        #     inflrate = 1
                        indirect_agent.indirect_alphas[numnow] = get_discounted_value(indirect_agent.indirect_alphas[numnow], bval * inflrate, indirect_agent.discount_factor)
                        indirect_agent.indirect_alphas_raw[numnow] = get_discounted_value(
                            indirect_agent.indirect_alphas_raw[numnow], bval,
                            indirect_agent.discount_factor)
                        tot_val += bval * inflrate
                        tot_val_raw += bval

                    else:
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)
                each_agent.alphas_asinfl[agent] = get_discounted_value(each_agent.alphas_asinfl[agent], tot_val, each_agent.discount_factor)
                each_agent.alphas_asinfl_raw[agent] = get_discounted_value(each_agent.alphas_asinfl_raw[agent], tot_val_raw,
                                                                      each_agent.discount_factor)
            else:
                """
                If you have no followers clearly you aren't sharing anything
                """
                for indirect_num, indirect_agent in enumerate(agents_list):
                    indirect_agent.indirect_alphas[numnow] *= 0 #(1 - indirect_agent.discount_factor)
                    indirect_agent.indirect_alphas_raw[numnow] *= 0#(1 - indirect_agent.discount_factor)


        agents_list[agent].bqueue.append(beta_sum)
        agents_list[agent].beta = np.mean(agents_list[agent].bqueue)
        for agent3 in agents_list:
            if agent3.num != agent:
                agent3.beta *= (1 - agents_list[agent].discount_factor)

        p_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)
        o_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)
        i_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)

        if i > gossip_timestep:
            if i % 100 == 0 and i > gossip_timestep + 4:
                print("@ Timestep", i)
                for agent1 in agents_list:
                    #agent1.generate_agent_rates_static_const_ext(state["agents"], state["apples"], i)
                    agent1.set_functional_rates(sp_state["agents"], sp_state["apples"], sp_state["pos"])

            for agent1 in agents_list:
                # if len(agent1.followers) == 99:
                #     if not lkdown:
                #         print("locked")
                #     lkdown = True
                if agent1.trigger_change(i, True) or np.abs(len(agent1.followers) - agent1.LRF) >= agent1.n0:
                    if np.abs(len(agent1.followers) - agent1.LRF) >= agent1.n0:
                        print(agent1.num, "momentum trigger")
                    ext_plots_x[agent1.num].append(i-1)
                    ext_plots_y[agent1.num].append(agent1.raw_b0_rate)

                    #agent1.generate_agent_rates_static(state["agents"], state["apples"], agents_list)
                    agent1.generate_rates_only(state["agents"], state["apples"], const_ext=False)

                    ext_plots_x[agent1.num].append(i)
                    ext_plots_y[agent1.num].append(agent1.raw_b0_rate)
                # if agent1.trigger_change(i, False) or (len(agent1.followers) == 0 and agent1.target_influencer == -1) or (agent1.target_influencer == -2 and i > gossip_timestep + 500):
                #     if len(agent1.followers) == 0 and agent1.target_influencer == -1:
                #         agent1.LRF = 0
                #     if len(agent1.followers) == 0 and lkdown == False:
                #         agent1.identify_influencer(agents_list)

            # for ag in agents_list:
            #     ag.shed_influencer(agents_list)

        p_network_list[agent].train_multiple(agents_list)
        # if i != 0:
        #     if i % 20 == 0:
        #         for ntwk in p_network_list:
        #             ntwk.train_multiple(agents_list)
        if i > gossip_timestep:
            if i % 50 == 0:
                for ag in agents_list:
                    if len(ag.followers) > 0:
                        i_network_list[ag.num].update(agents_list)
                    else:
                        o_network_list[ag.num].update(agents_list)

        if i % 100 == 0 or i == timesteps - 1:
            if i != 0:
                # if i % 10 == 0:
                #     for ntwk in p_network_list:
                #         ntwk.train_multiple(agents_list)
                # for ntwk in o_network_list:
                #     ntwk.train_multiple(agents_list)
                if i > gossip_timestep:
                    if i % 100 == 0:
                        if not follow_setup:
                            follow_setup = True
                            for ag in agents_list:
                                #if len(ag.followers) > 0:
                                follow_indices.append(ag.num)
                                follow_plots.append([])
                        if follow_setup:
                            for index, infl_num in enumerate(follow_indices):
                                if len(agents_list[infl_num].followers) > agents_list[infl_num].max_followers:
                                    agents_list[infl_num].max_followers = len(agents_list[infl_num].followers)
                                follow_plots[index].append(len(agents_list[infl_num].followers))
                if i % 100 == 0:
                    if i < gossip_timestep:
                        for agent1 in agents_list:
                            #agent1.generate_agent_rates(state["agents"], state["apples"])
                            agent1.generate_agent_rates_static(state["agents"], state["apples"], agents_list)
                # for infl in agents_list[0].influencers:
                #     infl.generate_agent_rates(state["agents"], state["apples"])
                if i % 200 == 0:
                    for agent1 in agents_list:
                        if len(agent1.followers) > 0:
                            infl_rep = 0
                            for sagn in agents_list:
                                infl_rep += sagn.R[agent1.num]
                            print("Agent " + str(agent1.num) + ":", len(agent1.followers), "followers", "/",
                                  agent1.raw_b0_rate, "/", infl_rep, "/", list(agent1.followers))
        if i % 100 == 0 or i == timesteps - 1:
            # Sum up things
            if i > gossip_timestep + 1:
                total_sw = 0
                for agent1 in agents_list:
                    total_sw += np.sum(agent1.alphas)
                    total_sw += np.sum(agent1.indirect_alphas)
                sw_plot.append(total_sw)

                total_att = 0
                rep = np.zeros(num_agents)
                true_rep = np.zeros(num_agents)
                pub_rep = np.zeros(num_agents)

                indirect = np.zeros(num_agents)
                for agent1 in agents_list:
                    total_att += agent1.budget - agent1.raw_b0_rate
                    rep += agent1.R
                    true_rep += agent1.PR
                    pub_rep += agent1.PB

                    indirect += agent1.indirect_alphas
                for num1 in range(len(agents_list)):

                        rep_plots[num1].append(rep[num1])
                        agent_sw_plots[num1].append(sum(agents_list[num1].alphas) + sum(agents_list[num1].alphas_asinfl))
                        real_rep_plots[num1].append(true_rep[num1])
                        public_rep_plots[num1].append(pub_rep[num1])
                for agent1 in agents_list:
                    for num1 in range(len(agents_list)):

                        followrates_plots[num1][agent1.num].append(agent1.raw_agent_rates[num1])
                        indirect_plots[num1][agent1.num].append(agent1.indirect_alphas[num1])
                        direct_plots[num1][agent1.num].append(agent1.alphas[num1])

                att_plot.append(total_att)

                for lnum, agent1 in enumerate(agents_list):
                    external_plots[lnum].append(agent1.raw_b0_rate)

            if (i % 10000 == 0 or i == timesteps - 1) and i != 0:

                # q_rates = np.array(roundabout_find_allocs_with_b0_full_vec(agents_list[1].alphas_raw,
                #                                                            agents_list[1].infl_alphas_raw,
                #                                                            budget=agents_list[1].budget,
                #                                                            b0=agents_list[1].b0))

                # q_rates = 1 - np.exp(-q_rates)
                print("Sample Reps for Agent 30:", agents_list[30].R)
        if i % 100 == 0:
            if i > gossip_timestep + 1:
                for agent1 in agents_list:
                    for num1 in range(len(agents_list)):
                        indirect_plots2[num1][agent1.num].append(agent1.indirect_alphas[num1])
                        direct_plots2[num1][agent1.num].append(agent1.alphas[num1] + agent1.alphas_asinfl[num1])
                        #peragrep_plots[num1][fagent1.num].append(agent1.R[num1])
                        #if num1 == agent:
        if i > gossip_timestep + 1:
            for agent1 in agents_list:
                peragrep_plots[agent][agent1.num].append(agent1.R[agent])
        if i == gossip_timestep:
            for ag in agents_list:
                # DIV10
                # ag.PR = np.copy(ag.alphas) / 4 + np.copy(ag.indirect_alphas)
                ag.PR = np.zeros(len(agents_list))
                for ag1 in range(len(agents_list)):

                    ag.PR[ag1] = ag.alphas[ag1] + ag.indirect_alphas[ag1]

                ag.PB = np.copy(ag.PR)
                ag.R = np.copy(ag.PR)
        if i > gossip_timestep:

            """
            Gossip Algorithm
            """

            for ag in agents_list:
                # DIV10
                #ag.PR = np.copy(ag.alphas) / 10 + np.copy(ag.indirect_alphas)
                #ag.PR = np.copy(ag.alphas) / 4 + np.copy(ag.indirect_alphas)
                ag.PR = np.zeros(len(agents_list))
                for ag1 in range(len(agents_list)):
                    ag.PR[ag1] = ag.alphas[ag1] + ag.indirect_alphas[ag1]
                # for ag1 in range(len(agents_list)):
                #     if ag.PR[ag1] < 0:
                #         print("Less than zero PR", ag.num, "/", ag1, "/", ag.PR[ag1])
                ag.has_gossiped = np.zeros(num_agents)

            # PR Update
            for ag in agents_list:
                for agnum in range(0, 100):
                    d = ag.PR[agnum] - ag.PB[agnum]
                    ag.PB[agnum] = ag.PR[agnum]
                    ag.R[agnum] += d


            for ag in agents_list:
                for adj_ind in range(len(ag.adjs)):
                #for jkjk in range(4):
                    #adj_ind = random.randint(0, len(ag.adjs)-1)
                    fr_ind = ag.adjs[adj_ind]
                    fr_ind2 = adj_ind
                    #fr_ind2 = adj_ind
                    # if ag.has_gossiped[fr_ind2] == 0:
                    #     adj_ind2 = agents_list[fr_ind].adjs.index(ag.num)
                    #     #if len(agents_list[fr_ind].followers) > 0:
                    #     #    continue
                    #
                    #     ag.has_gossiped[fr_ind2] = 1
                    #
                    #     agents_list[fr_ind].has_gossiped[adj_ind2] = 1
                    for jkjk in range(len(agents_list)):
                        if jkjk == agent or len(agents_list[jkjk].followers) > 0:
                            together = np.mean([agents_list[fr_ind].R[jkjk], ag.R[jkjk]])
                            agents_list[fr_ind].R[jkjk] = together
                            ag.R[jkjk] = together

                    # together = np.add(agents_list[fr_ind].R, ag.R) / 2
                    # agents_list[fr_ind].R = together
                    # ag.R = together
                    # together = (agents_list[fr_ind].R[agent] + ag.R[agent]) / 2
                    # agents_list[fr_ind].R[agent] = together
                    # ag.R[agent] = together

        if i % 1000 == 0 and i != 0:
            print("At timestep: ", i)
        if i % 500 == 0 and i > gossip_timestep:
            frontrn = np.argmax(agents_list[30].R)
            print("Frontrunner Follow Rates:", list(agents_list[frontrn].agent_rates))
            print("Frontrunner Alphas:", np.sum(agents_list[frontrn].alphas_asinfl))

            print("Agent 38 Follow Rates:", list(agents_list[38].agent_rates))
            print("Agent 38 Direct Alphas:", np.sum(agents_list[38].alphas))
            print("Agent 38 Indir Alpha:", agents_list[38].indirect_alphas[agents_list[38].target_influencer])
            print("Agent 38 TI", agents_list[38].target_influencer, "INFL RATE", agents_list[38].infl_rate,
                  "b0 RATE", agents_list[38].raw_b0_rate)
            print("Frontrunner (for agent 30):", agents_list[30].R[frontrn], "agent:", frontrn)
            print("Sample Reputations for Frontrunner:", agents_list[1].R[frontrn], agents_list[4].R[frontrn],
                  agents_list[12].R[frontrn],
                  agents_list[53].R[frontrn])
            print("Sample Indirect Alphas for Frontrunner", agents_list[1].indirect_alphas[frontrn],
                  agents_list[4].indirect_alphas[frontrn], agents_list[12].indirect_alphas[frontrn],
                  agents_list[30].indirect_alphas[frontrn],
                  agents_list[53].indirect_alphas[frontrn])
        if i > gossip_timestep + 99:
            for agent1 in agents_list:
                peragrep_plots2[agent][agent1.num].append(agent1.R[agent])
        if i > gossip_timestep + num_agents:

            if i % 100 == num_agents - 1:
                if len(round_actions) == 0:
                    print("roundactions empty")
                    continue
                for agent1 in agents_list:
                    est_sum = 0
                    if len(agent1.followers) > 0:
                        for agnum, topic in enumerate(round_actions):
                            est_sum += agent1.get_follower_feedback(agents_list[agnum], topic, agents_list) * \
                                       agent1.agent_rates[agnum]
                    for agnum, agent2 in enumerate(agents_list):
                        est_sum += agent2.get_util_pq(round_actions[agent1.num], agent1)
                    rep_estimate_plots[agent1.num].append(est_sum)

    """
    ###
    Graphing Section
    
    
    ###
    """

    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)
    follow = []
    for age1 in agents_list:
        follow.append(age1.max_followers)
    top1 = follow.index(max(follow))
    follow[top1] = -1
    top2 = follow.index(max(follow))
    plt.figure("Followers" + name)
    plt.title("Followers Over Time")
    for ih, follow_plot in enumerate(follow_plots):
        if ih == top1:
            plt.plot(follow_plot, label="Agent " + str(follow_indices[ih]) + " R", color='red')
        elif ih == top2:
            plt.plot(follow_plot, label="Agent " + str(follow_indices[ih]) + " R", color='blue')
        else:
            plt.plot(follow_plot)
    plt.legend()

    if folder == '':
        folder = "placeholder"
    prefix0 = "graphs_EI/" + folder
    prefix1 = prefix0 + "/"
    #prefix1 = "graphs_EI/K3/"
    if not os.path.exists(prefix0):
        os.mkdir(prefix0)

    plt.savefig(prefix1 + name + "_followers.png")
    plt.close()



    plt.figure("B0fr" + name)
    exts = []
    for agnum in range(len(agents_list)):
        exts.append(min(external_plots[agnum]))
    top1e = exts.index(min(exts))
    exts[top1] = 500

    top2e = exts.index(min(exts))
    for ih, plot in enumerate(external_plots):
        if ih == top1:
            #plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " R", color='red')
            plt.plot(ext_plots_x[ih], ext_plots_y[ih], label="Agent " + str(follow_indices[ih]) + " Followers", color='red')
        elif ih == top2:
            #plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " R", color='blue')
            plt.plot(ext_plots_x[ih], ext_plots_y[ih], label="Agent " + str(follow_indices[ih]) + " Followers", color='blue')
        else:
            #plt.plot(plot)
            plt.plot(ext_plots_x[ih], ext_plots_y[ih])
    print("LOWEST EXT: AGENT", top1e, agents_list[top1e].raw_b0_rate, list(agents_list[top1e].alphas_asinfl))
    print("LOWEST EXT: AGENT", top1e, agents_list[top1e].raw_b0_rate, list(agents_list[top1e].agent_rates))
    print("SECOND LOWEST EXT: AGENT", top2e, agents_list[top2e].raw_b0_rate, list(agents_list[top2e].alphas))
    plt.legend()
    plt.title("External Rate of Eventual Influencer")
    plt.savefig(prefix1 + name + "_ext.png")
    plt.close()

    plt.figure("Ext_Old" + name)
    for ih, plot in enumerate(external_plots):
        if ih == top1:
            plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " Rate", color='red')
        elif ih == top2:
            plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " Rate", color='blue')
        else:
            plt.plot(plot)
    plt.legend()
    plt.title("External Rates")
    plt.savefig(prefix1 + name + "_ext_old.png")
    plt.close()

    repp2 = np.copy(rep_plots[top1])

    plt.figure("Reputations" + name)
    for ih, plot in enumerate(rep_plots):
        if ih == top1:
            plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " R", color='red')
        elif ih == top2:
            plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " R", color='blue')
        else:
            plt.plot(plot)
    plt.legend()
    plt.title("Reputations")
    plt.savefig(prefix1 + name + "_reps.png")
    plt.close()

    plt.figure("SelfUtility" + name)
    for ih, plot in enumerate(agent_sw_plots):
        if ih == top1:
            plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " a_i + a_d", color='red')
        elif ih == top2:
            plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " a_i + a_d", color='blue')
        else:
            plt.plot(plot)
        # for val in plot:
        #     if val < 0:
        #         print(val, ih)
        #     break
    plt.legend()
    plt.title("Utility (Received) of Agents")
    plt.savefig(prefix1 + name + "_selfutil.png")
    plt.close()

    plt.figure("TopAgentRep" + name)
    for ih, plot in enumerate(peragrep_plots[top1]):
        plt.plot(plot)
    #plt.plot(rep_plots[top1])
    plt.legend()
    plt.title("Reputations of Agent " + str(top1))
    plt.savefig(prefix1 + name + "_topagentrep.png")
    plt.close()

    if not os.path.exists(prefix1 + "/per_agent"):
        os.makedirs(prefix1 + "/per_agent")

        if not os.path.exists(prefix1 + "/per_agent_rates"):
            os.makedirs(prefix1 + "/per_agent_rates")

    if not os.path.exists(prefix1 + "/per_agent_production"):
        os.makedirs(prefix1 + "/per_agent_production")

    if not os.path.exists(prefix1 + "/per_agent_indiralphs"):
        os.makedirs(prefix1 + "/per_agent_indiralphs")

    if not os.path.exists(prefix1 + "/per_agent_diralphs"):
        os.makedirs(prefix1 + "/per_agent_diralphs")

    for agent in agents_list:
        plt.figure("Reputations" + str(agent.num))
        for num in range(len(agents_list)):
            if num == top1:
                plt.plot(peragrep_plots[num][agent.num], label="Agent " + str(num) + " R", color='red')
            elif num == top2:
                plt.plot(peragrep_plots[num][agent.num], label="Agent " + str(num) + " R", color='blue')
            else:
                plt.plot(peragrep_plots[num][agent.num])
        plt.legend()
        plt.title("Per-Agent Reputations for All, Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent/" + name + "_" + str(agent.num) + "_repalls.png")
        plt.close()

        plt.figure("Reputations33" + str(agent.num))
        for num in range(len(agents_list)):
            if num == top1:
                plt.plot(np.array(peragrep_plots[num][agent.num])[1::5], label="Agent " + str(num) + " R", color='red')
            elif num == top2:
                plt.plot(np.array(peragrep_plots[num][agent.num])[1::5], label="Agent " + str(num) + " R", color='blue')
            else:
                plt.plot(np.array(peragrep_plots[num][agent.num])[1::5])
        plt.legend()
        plt.title("Per-Agent Reputations for All, Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent/" + name + "_" + str(agent.num) + "_repallsWEAKER.png")
        plt.close()

        plt.figure("Reputations424" + str(agent.num))
        for num in range(len(agents_list)):
            if num == top1:
                plt.plot(np.array(peragrep_plots2[num][agent.num]), label="Agent " + str(num) + " R", color='red')
            elif num == top2:
                plt.plot(np.array(peragrep_plots2[num][agent.num]), label="Agent " + str(num) + " R", color='blue')
            else:
                plt.plot(np.array(peragrep_plots2[num][agent.num]))
        plt.legend()
        plt.title("Per-Agent Reputations (Marked after Agent Action), Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent/" + name + "_" + str(agent.num) + "_repallsON_POINT.png")
        plt.close()

    for agent in agents_list:
        plt.figure("IndirectAlphas" + str(agent.num))
        for num in range(len(agents_list)):
            if num == top1:
                plt.plot(indirect_plots2[num][agent.num], label="Agent " + str(num) + " IA", color='red')
            elif num == top2:
                plt.plot(indirect_plots2[num][agent.num], label="Agent " + str(num) + " IA", color='blue')
            else:
                plt.plot(indirect_plots2[num][agent.num])
        plt.legend()
        plt.title("Per-Agent I Alphas for All, Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent_indiralphs/" + name + "_" + str(agent.num) + "_allindirs.png")
        plt.close()

    if not os.path.exists(prefix1 + "/per_agent_stats"):
        os.makedirs(prefix1 + "/per_agent_stats")

    for agent in agents_list:
        plt.figure("Stats2" + str(agent.num))
        plt.plot(external_plots[agent.num], label="Agent " + str(agent.num) + " External", color='red')
        plt.plot(follow_plots[agent.num], label="Agent " + str(agent.num) + " Followers", color='blue')
        plt.legend()
        plt.title("Followers and External Rate, Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent_stats/" + name + "_" + str(agent.num) + "_stats.png")
        plt.close()

    for agent in agents_list:
        plt.figure("DirectAlphas" + str(agent.num))
        for num in range(len(agents_list)):
            if num == top1:
                plt.plot(direct_plots2[num][agent.num], label="Agent " + str(num) + " DR", color='red')
            elif num == top2:
                plt.plot(direct_plots2[num][agent.num], label="Agent " + str(num) + " DR", color='blue')
            else:
                plt.plot(direct_plots2[num][agent.num])
        plt.legend()
        plt.title("Per-Agent D Alphas for All, Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent_diralphs/" + name + "_" + str(agent.num) + "_alldirs.png")
        plt.close()

    for agent in agents_list:
        plt.figure("Productions" + str(agent.num))
        plt.plot(peragprod_plots[agent.num])
        plt.legend()
        plt.title("Agent Productions for Agent " + str(agent.num))
        plt.savefig(prefix1 + "/per_agent_production/" + name + "_" + str(agent.num) + "_prodalls.png")
        plt.close()

    for agnum in range(num_agents):
        plt.figure("Agent Rates" + str(agnum))
        for num in range(len(agents_list)):
            plt.plot(followrates_plots[num][agnum])
        plt.legend()
        plt.title("Agent Following Rates, Agent " + str(agnum))
        plt.savefig(prefix1 + "/per_agent_rates/" + name + "_" + str(agnum) + "_RATES.png")
        plt.close()

    plt.figure("INDIRALS" + name)
    for ih, plot in enumerate(indirect_plots[top1]):
        # if ih == top1 or ih == top2:
        #     plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " R")
        # else:
        plt.plot(plot)
    plt.legend()
    plt.title("Indirect Alphas")
    plt.savefig(prefix1 + name + "_indiralphs.png")
    plt.close()

    plt.figure("INDIRALS total" + name)
    plt.plot(np.sum(indirect_plots[top1], axis=0), label="Indirect Alpha")
    plt.plot(rep_plots[top1], label="Reputation")
    plt.plot(real_rep_plots[top1], label="Private Rep")
    plt.plot()
    plt.plot()
    plt.legend()
    plt.title("Indirect Alphas Total for " + str(top1))
    plt.savefig(prefix1 + name + "_indiralphs_total.png")
    plt.close()

    plt.figure("DIRALS" + name)
    for ih, plot in enumerate(direct_plots[top1]):
        # if ih == top1 or ih == top2:
        #     plt.plot(plot, label="Agent " + str(follow_indices[ih]) + " R")
        # else:
        plt.plot(plot)
    plt.legend()
    plt.title("Direct Alphas")
    plt.savefig(prefix1 + name + "_diralphs.png")
    plt.close()

    plt.figure("Estimates" + name)
    plt.plot(repp2, label="Agent " + str(top1) + " R", color="red")
    # plt.plot(rep_plots[top2], label="Agent " + str(top2) + " R", color="green")
    plt.plot(np.array(rep_estimate_plots[top1]) / num_agents, label="Agent " + str(top1) + " Estimate", color="orange",
             linestyle="-.")
    # plt.plot(real_rep_plots[top2], label="Agent " + str(top2) + " PR", color="blue", linestyle="-.")
    # plt.plot(public_rep_plots[top1], label="Agent " + str(top1) + " PB", color="red", linestyle="dotted")
    # plt.plot(public_rep_plots[top2], label="Agent " + str(top2) + " PB", color="green", linestyle="dotted")
    plt.legend()
    plt.title("Reputation vs. Estimate")
    plt.savefig(prefix1 + name + "_reps_est.png")
    plt.close()

    plt.figure("Total Social Welfare" + name)
    plt.plot(sw_plot)
    plt.legend()
    plt.title("Total Social Welfare")
    plt.savefig(prefix1 + name + "_socwel.png")
    plt.close()

    plt.figure("Total Attention In Community" + name)
    plt.plot(att_plot)
    plt.title("Total Attention In Community")
    plt.savefig(prefix1 + name + "_att.png")
    plt.close()


    if not os.path.exists("saved_item/" + name):
        os.makedirs("saved_item/" + name)
    np.save("saved_item/" + name + "/socwell.npy", sw_plot)
    np.save("saved_item/" + name + "/ext.npy", np.array(external_plots))
    np.save("saved_item/" + name + "/rep.npy", np.array(rep_plots))
    np.save("saved_item/" + name + "/rep_est.npy", np.array(rep_estimate_plots))
    np.save("saved_item/" + name + "/followers.npy", np.array(follow_plots))
    np.save("saved_item/" + name + "/direct.npy", np.array(direct_plots))
    np.save("saved_item/" + name + "/indirect.npy", np.array(indirect_plots))


def train_ac_content(side_length, num_agents, agents_list, name, discount, timesteps, iteration=0, scenario=0,
                     in_thres=None, folder=''):
    # A edited version of AC Rate. Binary Projection is automatically packaged because of the p/q functions.
    # Further edited to only be a single iteration.

    # Agent set-up
    for agent in agents_list:
        agent.agent_rates = np.array([1] * num_agents)
        agent.infl_rate = 1  # Assume only one influencer at the moment
        agent.b0_rate = 1

    # Influencer set up
    if len(agents_list[0].influencers) > 0:
        agents_list[0].influencers[0].follow_rates = np.array([1] * num_agents)

    # step 1 - find normal A/B
    # No step one, since agents have to learn on their own
    # alphas1, betas1 = find_ab_content_infl(agents_list, side_length, None, None, 0.0003, name, discount=0.99, timesteps=3000 * num_agents, iteration=iteration)

    """
    
    No Alloc. Generation
    
    # step 2 - get allocs and regenerate
    allocs = []
    print("Allocing")
    for i in range(num_agents):
        arates, irates, b0_rate = roundabout_find_allocs_with_b0(agents_list[i].alphas, agents_list[i].infl_alphas, budget=agents_list[i].budget, b0=agents_list[i].b0 * agents_list[i].b0_rate)

        agents_list[i].raw_agent_rates = arates
        agents_list[i].raw_infl_rates = irates
        agents_list[i].raw_b0_rate = b0_rate

        a_allocs = (1 - np.exp(-arates))
        i_allocs = (1 - np.exp(-irates))
        agents_list[i].agent_rates = a_allocs
        agents_list[i].infl_rates = i_allocs
        agents_list[i].b0_rate = (1-np.exp(-1 * b0_rate))
        print(a_allocs, i_allocs, agents_list[i].b0_rate)

    # influencer
    for i in range(len(influencers)):
        arates = find_allocs(influencers[i].agent_alphas, budget=influencers[i].budget)
        print((1-np.exp(-arates)))
        influencers[i].raw_follow_rates = arates
        influencers[i].follow_rates = (1-np.exp(-arates))

    alphas1, betas1 = find_ab_content_infl(agents_list, side_length, None, None, 0.0003, name, discount=0.99, timesteps=3000 * num_agents, iteration=iteration) # regen with rates
    """

    alpha = 0.001
    alpha = 0.002
    for nummer, agn in enumerate(agents_list):
        agn.policy_network = ActorNetwork(side_length, alpha, discount, num=nummer)
        agn.policy = "learned_policy"
        agn.observer_network = ObserverNetwork(side_length, num_agents, 0.0001, discount, num=nummer)
        agn.follower_network = agn.observer_network
        agn.influencer_network = ObserverNetwork(side_length, num_agents, 0.0001, discount, num=nummer,
                                                 infl_net=True)

    #if len(agents_list[0].influencers) > 0:
    #    agents_list[0].influencers[0].observer_network = ObserverNetwork(side_length, num_agents, alpha * 6, discount,
    #                                                                     num=nummer, infl_net=True)

    training_loop(agents_list, side_length, None, None, 0.0005, name, has_beta=True, discount=discount,
                  timesteps=timesteps, scenario=scenario, in_thres=in_thres, folder=folder)


def experiment(name, base_a=0.01, alpha_decay=(0.7 / 10000), kappa_decay=5, threshold=5, ts=5000, folder=''):
    """

    :param name:
    :param base_a: base rate
    :param alpha_decay: alpha rate of the rate allocation decay
    :param threshold: the threshold to make the influencer realloc.
    :param ts: total timesteps
    :return:

    Note that e^(-0.7) ~= 0.5
    For 10,000 steps -> e^(0.7), we have alpha = 0.7 / 10,000
    """
    random.seed(35279038)
    np.random.seed(389045)
    num_agents = 100
    side_length = num_agents * 2
    discount = 0.99
    agents_list = []
    gen_budget = 25
    scenario = 7
    lin_dec = False
    nb0 = 40
    #b0 = b0 / num_agents

    # influencers = [GhostInfluencer(num_agents, agents_list, discount, budget=infl_budget)]
    influencers = []
    # for ida, agent in enumerate(agents_list):
    #     agent.influencers = influencers

    packs = {
        "base_a": base_a,
        "alpha": alpha_decay,
        "kappa": kappa_decay,
        "n0": threshold
    }
    for i in range(num_agents):
        main_int = 2 * i + 1
        agents_list.append(
            ContentAgent(policy="random", id=i, num_agents=num_agents, influencers=influencers, main_interest=main_int,
                         budget=gen_budget, b0=nb0, topics=side_length, nonfixedpackage=packs))

    train_ac_content(side_length, num_agents, agents_list, name, discount, ts, iteration=0,
                     scenario=scenario, folder=folder)


from agents.content_agent import ContentAgent

num_agents = 100
side_length = num_agents * 2
discount = 0.99
agents_list = []

infl_budget = 15 * (num_agents / 10)
gen_budget = 25  #5
scenario = "both"  # 1 is to save AGENT-demand too

lin_dec = False
b0 = 0.01  #0.4
b0 = b0 / num_agents

influencers = []

for i in range(num_agents):
    main_int = 2 * i + 1
    agents_list.append(
        ContentAgent(policy="random", id=i, num_agents=num_agents, influencers=influencers, main_interest=main_int,
                     budget=gen_budget, b0=b0, topics=side_length))

name = str(gen_budget) + "-" + str(infl_budget)
approach = ""
if b0 == 0:
    approach += "NoExt"
elif b0 >= 1:
    approach += "ExtPerfect"
else:
    approach += "Ext" + str(b0)[2:]

if lin_dec:
    approach += "_Linear"

prefix = "policyitchk/" + name + "_" + approach + "/"
prefix_path = "policyitchk/" + name + "_" + approach

import os

if not os.path.isdir(prefix_path):
    os.mkdir(prefix_path)

single = True
if single:
    for itee in range(1):
        print("ITERATION ", itee)
        experiment("feb19_fixed2", base_a=0.01, kappa_decay=5, ts=50000,
                   folder="feb19_fixed2")

else:
    thres1 = [0, 0.00001, 0.00003, 0.00005, 0.0001, 0.0003]
    thres2 = [0, 0.00001, 0.00005, 0.0001, 0.00015, 0.00025, 0.0003, 0.00035, 0.0005]  #already done 0.00001 & 0.00005
    thres11 = ["0", "1", "3", "5", "10", "30"]  # ["0", "1", "3", "5", "10", "15"]
    thres21 = ["0", "1", "5", "10", "15", "25", "30", "35", "50"]  #already done 1

    def_thres1 = 0.00003
    def_thres2 = 0.0001
    for itee in range(0, len(thres1)):
        name = "nfirst_" + thres11[itee]
        experiment(name, thres1[itee], def_thres2, ts=8000)

    for itee in range(0, len(thres2)):
        name = "nsecond_" + thres21[itee]
        experiment(name, def_thres1, thres2[itee], ts=8000)