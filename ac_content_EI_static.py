import random
import numpy as np
random.seed(35279038)
np.random.seed(389043)
import time
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
num_topics = 10

def load_adjs(agents_list, filename):
    adjs = np.load(filename)
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
    np.save("adj.npy", np.array(adjacencies))
    print("saved")

debug2 = False
def get_discounted_value(old, new, discount):
    return old * (1 - discount) + new * discount

def training_loop(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000,
                  has_beta=False, betas_list=None, altname=None, scenario=0, in_thres=0.00005, folder=''):


    print(orchard_length)
    print(len(agents_list))
    print("Using Beta Value:", has_beta)

    load_adjs(agents_list, "adj.npy")
    #create_adjs(agents_list)

    # if has_beta:
    #     assert betas_list is not None

    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env
    print("Experiment", name)
    v_network_list = []
    p_network_list = []
    o_network_list = []
    num_agents = len(agents_list)
    for agn in range(len(agents_list)):

        p_network_list.append(agents_list[agn].policy_network)
        o_network_list.append(agents_list[agn].observer_network)

        if debug2:
            assert agents_list[agn].policy_network is not None or agents_list[agn].policy == "random"
            #p_network_list.append(agents_list[agn].policy_network)

            network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
            agents_list[agn].policy_network = network2
            p_network_list.append(network2)
            agents_list[agn].policy = "learned_policy"

            # v_network_list.append(agents_list[agn].policy_value)

        # network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
        # agents_list[agn].policy_network = network2
        # p_network_list.append(network2)

    total_reward = 0

    """Construct Sample States"""
    sample_state = {
        "agents": np.array([[0]] * side_length),
        "apples": np.array([[0]] * side_length),
        "pos": [np.array([4, 0])] * num_agents,
    }
    sample_state5 = {
        "agents": np.array([[0]] * side_length),
        "apples": np.array([[0]] * side_length),
        "pos": [np.array([5, 0])] * num_agents,
    }
    sample_state6 = {
        "agents": np.array([[0]] * side_length),
        "apples": np.array([[0]] * side_length),
        "pos": [np.array([6, 0])] * num_agents,
    }

    """ Plotting Setup """
    """setup_plots(p_network_list[0].function.state_dict(), one_plot)
    global loss_plot
    loss_plot = []
    maxi = 0"""
    """"""
    infls = agents_list[0].influencers
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

    gossip_timestep = 3000
    round_actions = np.zeros(num_agents)
    lkdown = False

    for agent1 in agents_list:
        for ag_num in range(len(agents_list)):
            agent1.alphas_asinfl[ag_num] = 0
            agent1.alphas_asinfl_raw[ag_num] = 0
            agent1.alphas[ag_num] = 0
            agent1.alphas_raw[ag_num] = 0
            agent1.indirect_alphas[ag_num] = 0
            agent1.indirect_alphas_raw[ag_num] = 0

    for i in range(timesteps):

        agent = random.randint(0, env.n - 1)  # Choose random agent
        agent = i % num_agents
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        #action = agents_list[agent].get_action(state, discount)

        epsilon = 0
        chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        if chance < epsilon:
            action = random_policy_1d(state, old_pos)
        else:
            action = agents_list[agent].get_action(state, discount)  # Get action.
            #action = agent * 2

        round_actions[agent] = action

        peragprod_plots[agent].append(action)

        #reward, new_position = env.main_step(agents_list[agent].position.copy(),
        #                                     3)  # Take the action. Observe reward, and get new position
        #agents_list[agent].position = new_position.copy()  # Save new position.
        #total_reward += reward  # Add to reward.

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
        infl_sum = 0
        for infl in infls:
            infl_sum += infl.give_producer_feedback(agents_list[agent], action)
        #agents_list[agent].influencers[0].agent_alphas[agent] += infl_sum

        # Influencer Retweet
        if len(infls) > 0:
            random_val = random.random()
            if infls[0].follow_rates[agent] > random_val:
                singulars[len(agents_list)][action] = 1
                if scenario == "both":
                    one_singulars[len(agents_list)][action] = 1

        # # #

        for numnow, each_agent in enumerate(agents_list):
            """
            This is the INFLUENCER receiving feedback section.
            - If you have more than 0 followers, you "share" the content to your followers based on your "each_agent.agent_rates[agent]" rate.
            - Then, this becomes your alphas_asinfl (for this timestep).
            """
            if len(each_agent.followers) > 0:
                #print("Agent " + str(numnow) + ":", len(each_agent.followers), "followers")
                #raw_value = each_agent.get_follower_feedback(agents_list[agent], action, agents_list, prev=True)
                #raw_value = raw_value
                #valued = each_agent.get_follower_feedback(agents_list[agent], action, agents_list) * \
                #         each_agent.agent_rates[agent]
                #infl_sum += valued
                #asinfl = np.zeros(num_agents)
                #asinfl_raw = np.zeros(num_agents)
#
                #asinfl[agent] = valued
                #asinfl_raw[agent] = raw_value
#
                #each_agent.alphas_asinfl = get_discounted_value(each_agent.alphas_asinfl, asinfl, each_agent.discount_factor)
                #each_agent.alphas_asinfl_raw = get_discounted_value(each_agent.alphas_asinfl_raw, asinfl_raw,
                #                                                        each_agent.discount_factor)
                for agent_num in range(len(agents_list)):
                    if agent_num != agent:
                        each_agent.alphas_asinfl[agent_num] = get_discounted_value(each_agent.alphas_asinfl[agent_num], 0,
                                                                               each_agent.discount_factor)
                        each_agent.alphas_asinfl_raw[agent_num] = get_discounted_value(each_agent.alphas_asinfl_raw[agent_num],
                                                                                   0,
                                                                                   each_agent.discount_factor)

                tot_val = 0
                tot_val_raw = 0
                for foll_agent in agents_list:
                    if foll_agent.target_influencer == each_agent.num:
                        bval = each_agent.agent_rates[agent] * foll_agent.get_util_pq_raw(action,
                                                                                              agents_list[agent])
                        tot_val += bval * foll_agent.infl_rate
                        tot_val_raw += bval

                beta_sum += tot_val
#
#
                #each_agent.alphas_asinfl[agent] = get_discounted_value(each_agent.alphas_asinfl[agent], tot_val, each_agent.discount_factor)
                #each_agent.alphas_asinfl_raw[agent] = get_discounted_value(each_agent.alphas_asinfl_raw[agent], tot_val_raw,
                #                                                       each_agent.discount_factor)

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

                #asconsum = np.zeros(num_agents)
                #asconsum_raw = np.zeros(num_agents)
                #asconsum[agent] = valued
                #asconsum_raw[agent] = raw_value
                each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], valued, each_agent.discount_factor)
                each_agent.alphas_raw[agent] = get_discounted_value(each_agent.alphas_raw[agent], raw_value,
                                                                each_agent.discount_factor)

                beta_sum += valued

            # Here, we use valued to build a "sharing" / indirect alpha system
            # Each agent keeps track of ALPHA && INDIRECT ALPHA for EVERY AGENT
            #for indirect_num, indirect_agent in enumerate(agents_list):
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

                        # indirect_agent.indirect_alphas[numnow] *= (1 - indirect_agent.discount_factor)
                        # indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)

                    elif indirect_agent.target_influencer == each_agent.num:
                        bval = each_agent.agent_rates[agent] * indirect_agent.get_util_pq_raw(action,
                                                                                              agents_list[agent])
                        inflrate = min(1, indirect_agent.infl_rate + 0.1 + 0.4 * len(each_agent.followers) / 100)
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

            # Scenario for tracking demand (making it 1)
        agents_list[agent].bqueue.append(beta_sum)
        agents_list[agent].beta = np.mean(agents_list[agent].bqueue)
        #agents_list[agent].beta = get_discounted_value(agents_list[agent].beta, beta_sum, agents_list[agent].discount_factor)
        for agent3 in agents_list:
            if agent3.num != agent:
                agent3.beta *= (1 - agents_list[agent].discount_factor)

        #p_network_list[agent].train(sp_state, sp_new_state, reward, action, agents_list)
        p_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)
        o_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)

        if i > gossip_timestep:
            if i % 10 == 0 and i > gossip_timestep + 4:
                print("@ Timestep", i)
                for agent1 in agents_list:
                    agent1.generate_agent_rates_static_const_ext(state["agents"], state["apples"], i)

            for agent1 in agents_list:
                if len(agent1.followers) == 99:
                    if not lkdown:
                        print("locked")
                    lkdown = True
                if agent1.trigger_change(i, True) or np.abs(len(agent1.followers) - agent1.LRF) >= agent1.n0:
                    if np.abs(len(agent1.followers) - agent1.LRF) >= agent1.n0:
                        print(agent1.num, "momentum trigger")
                    ext_plots_x[agent1.num].append(i-1)
                    ext_plots_y[agent1.num].append(agent1.raw_b0_rate)
                    agent1.generate_agent_rates_static(state["agents"], state["apples"], agents_list)
                    ext_plots_x[agent1.num].append(i)
                    ext_plots_y[agent1.num].append(agent1.raw_b0_rate)
                if agent1.trigger_change(i, False) or (len(agent1.followers) == 0 and agent1.target_influencer == -1) or (agent1.target_influencer == -2 and i > gossip_timestep + 500):
                    if len(agent1.followers) == 0 and agent1.target_influencer == -1:
                        agent1.LRF = 0
                    if len(agent1.followers) == 0 and lkdown == False:
                        agent1.identify_influencer(agents_list)

            for ag in agents_list:
                ag.shed_influencer(agents_list)

        p_network_list[agent].train_multiple(agents_list)
        # if i != 0:
        #     if i % 20 == 0:
        #         for ntwk in p_network_list:
        #             ntwk.train_multiple(agents_list)
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
        if i % 20 == 0:
            if i > gossip_timestep + 1:
                for agent1 in agents_list:
                    for num1 in range(len(agents_list)):
                        indirect_plots2[num1][agent1.num].append(agent1.indirect_alphas[num1])
                        direct_plots2[num1][agent1.num].append(agent1.alphas[num1] + agent1.alphas_asinfl[num1])
                        peragrep_plots[num1][agent1.num].append(agent1.R[num1])
                        #if num1 == agent:

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
        for val in plot:
            if val < 0:
                print(val, ih)
            break
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
        agn.observer_network = ObserverNetwork(side_length, num_agents, alpha * 4, discount, num=nummer)

    #if len(agents_list[0].influencers) > 0:
    #    agents_list[0].influencers[0].observer_network = ObserverNetwork(side_length, num_agents, alpha * 6, discount,
    #                                                                     num=nummer, infl_net=True)

    training_loop(agents_list, side_length, None, None, 0.0013, name, has_beta=True, discount=discount,
                  timesteps=timesteps, scenario=scenario, in_thres=in_thres, folder=folder)


def experiment(name, base_a=0.01, alpha_decay=(0.7 / 6000), kappa_decay=5, threshold=5, ts=5000, folder=''):
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
    num_agents = 100
    side_length = num_agents * 2
    discount = 0.99
    agents_list = []
    gen_budget = 25
    scenario = 7
    lin_dec = False
    b0 = 100
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
                         budget=gen_budget, b0=b0, topics=side_length, nonfixedpackage=packs))

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
    # if i == 6:
    #     agents_list.append(
    #         ContentAgent(policy="random", id=i, num_agents=num_agents, influencers=influencers, main_interest=main_int,
    #                      budget=gen_budget, b0=1000, topics=side_length))
    #     continue
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
        #experiment("AAB_test_5_kappa3", kappa_decay=5, ts=10000)
        #experiment("december_kappa2", kappa_decay=2, ts=5000 + 2000)
        # experiment("dec4_k2", kappa_decay=2, ts=4000 + 2000)
        # experiment("dec4_k3", kappa_decay=2, ts=5000 + 2000)
        experiment("dec18_K5_sa_fin", kappa_decay=5, ts=7000, folder="dec22_K5_sa_final2")
        # experiment("dec18_K2_ma", kappa_decay=2, ts=7000, folder="dec22_K2_ma")
        # experiment("dec18_k5_ma", kappa_decay=5, ts=8000, folder="dec22_K5_ma")
#
        # experiment("dec22_ndk5", kappa_decay=5, ts=7000, alpha_decay=0, folder="dec22_nodecay_sa_K5")
        # experiment("dec22_ndk3", kappa_decay=3, ts=7000, alpha_decay=0, folder="dec22_nodecay_sa_K3")
        # experiment("dec22_ndk2", kappa_decay=2, ts=7000, alpha_decay=0, folder="dec22_nodecay_sa_K2")
#
#
        #experiment("dec16_mdk3", kappa_decay=3, ts=7000, alpha_decay=1.5/5000, folder="dec16_moredecay_K3")


        #experiment("dec18_a20", base_a=0.02, kappa_decay=3, ts=7000, folder="dec18_sa_K3A20")
        #experiment("dec18_a05", base_a=0.005, kappa_decay=3, ts=10000, folder="dec18_sa_K3A5")

        # experiment("dec4_a20_k5", base_a=0.02, kappa_decay=5, ts=5000 + 2000)
        # experiment("dec4_a05_k5", base_a=0.005, kappa_decay=5, ts=5000 + 2000)
        # experiment("dec4_a20_k3", base_a=0.02, kappa_decay=3, ts=5000 + 2000)
        # experiment("dec4_a05_k3", base_a=0.005, kappa_decay=3, ts=5000 + 2000)
        #experiment("dec4_k5", kappa_decay=5, ts=5000 + 2000)
        # experiment("december_kappa5", kappa_decay=5, ts=5000 + 2000)

else:
    #experiment("new", 0.00003, 0, ts=10000)
    thres1 = [0, 0.00001, 0.00003, 0.00005, 0.0001, 0.0003]
    thres2 = [0, 0.00001, 0.00005, 0.0001, 0.00015, 0.00025, 0.0003, 0.00035, 0.0005]  #already done 0.00001 & 0.00005

    #thres1 = [0, 0.0001, 0.00001, 0.00004]

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

    #experiment("ffa", 0, 0, ts=10000)
