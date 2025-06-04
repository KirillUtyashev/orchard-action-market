import random
import numpy as np

random.seed(35279038)
np.random.seed(389043)
import time
from actor_critic import eval_network
from main import run_environment_1d
from orchard.environment import *
import matplotlib.pyplot as plt

from policies.random_policy import random_policy_1d, random_policy
from models.jan_action_actor import ActorNetwork
from models.content_observer_1d import ObserverNetwork, ten
from models.jan_value_actor import ValueNetwork

forever_discount = 0.75
onemdiscount = 1 - forever_discount

# from models.actor_dc_1d_altinput import ActorNetwork # USING ALTINPUT
# from models.simple_connected_multiple_dc_altinput import SCMNetwork, SimpleConnectedMultiple
from orchard.algorithms import single_apple_spawn, single_apple_despawn

import torch

torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

"""
Some variables / parameters for the *content market*.
"""


def all_gossip(agents_list):
    for agnum, agent in enumerate(agents_list):
        shortls = []
        for i in range(len(agents_list)):
            if i != agent.num:
                shortls.append(i)
        agent.adjs = shortls


def load_adjs(agents_list, filename):
    adjs = np.load(filename)
    for agnum, agent in enumerate(agents_list):
        agent.adjs = adjs[agnum]
        print(agnum, list(adjs[agnum]))
    # create_symm_adjs(agents_list)
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
        print("Network is Connected")
    else:
        print("(Error) Network is Not Connected")


def create_adjs(agents_list):
    neighbours = 3
    lister = list(range(0, len(agents_list)))
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
                print("Create Adjacencies Failed (Infinite Loop)")
                break
    print("Create Adjacencies Done")
    for agent in agents_list:
        adjacencies.append(agent.adjs)
    np.save("adj3.npy", np.array(adjacencies))
    print("Saved Adjacencies")


def get_discounted_value(old, new, discount):
    if old < 0:
        return new
    return old * (1 - discount) + new * discount


def training_loop(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000,
                  has_beta=False, betas_list=None, altname=None, scenario=0, in_thres=0.00005, folder=''):
    maxi = 0
    # load_adjs(agents_list, "adj3.npy")
    # create_adjs(agents_list) # If needed; load from adj.npy instead
    all_gossip(agents_list)

    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    initial_positions = []
    for i in range(len(agents_list)):
        initial_positions.append([i * 2, 0])
    env.initialize(agents_list, initial_positions)  # attach agents to env
    num_agents = len(agents_list)
    print("Experiment", name)

    p_network_list = []  # Producer Networks
    o_network_list = []  # Observer Networks
    i_network_list = []  # Influencer Networks

    v_network_list = []
    for agn in range(len(agents_list)):
        p_network_list.append(agents_list[agn].policy_network)
        o_network_list.append(agents_list[agn].follower_network)
        i_network_list.append(agents_list[agn].influencer_network)
        v_network_list.append(agents_list[agn].value_network)

    total_reward = 0

    """Construct Sample States"""
    # samp_state = {
    #     "agents": np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    #     "apples": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    #     "pos": np.array([4, 0])
    # }
    # samp_state = {
    #     "agents": np.array([3, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
    #     "apples": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    #     "pos": np.array([2, 0])
    # }
    samp_state = {
        "agents": np.array([8, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "apples": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "pos": np.array([2, 0])
    }

    sample_state0 = {
        "agents": np.array([[0]] * orchard_length),
        "apples": np.array([[0]] * orchard_length),
        "pos": [np.array([4, 0])] * num_agents,
    }
    sample_state1 = {
        "agents": np.array([[0]] * orchard_length),
        "apples": np.array([[0]] * orchard_length),
        "pos": [np.array([5, 0])] * num_agents,
    }
    sample_state2 = {
        "agents": np.array([[0]] * orchard_length),
        "apples": np.array([[0]] * orchard_length),
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

    apple_pos_x = []
    apple_pos_y = []

    sw_plot = []
    att_plot = []
    rep_plots = []
    agent_sw_plots = []
    real_rep_plots = []
    public_rep_plots = []
    external_plots = []

    reward_plot = []
    indirect_plots = []
    indirect_plots2 = []
    direct_plots2 = []
    direct_plots = []
    peragrep_plots = []
    peragrep_plots2 = []
    peragprod_plots = []
    peragval_plots = []

    peragpos_plots = []

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
        peragpos_plots.append([])
        peragrep_plots2.append([])
        peragprod_plots.append([[], [], []])
        peragval_plots.append([])
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
    # Theoretically, the gossiping period lasts around 10,000 - 20,000 steps at the most (far less than needed to even "learn" the Orchard - this may need to be increased / delayed).
    gossip_timestep = 16000
    training_timestep = 10000

    round_actions = np.zeros(num_agents)
    lkdown = False

    for agent1 in agents_list:
        agent1.beta = -0.00001
        for ag_num in range(len(agents_list)):
            agent1.alphas_asinfl[ag_num] = -0.00001  # 0
            agent1.alphas_asinfl_raw[ag_num] = -0.00001  # 0
            agent1.alphas[ag_num] = -0.00001  # 0
            agent1.alphas_raw[ag_num] = -0.00001  # 0
            agent1.indirect_alphas[ag_num] = 0  # 0
            agent1.indirect_alphas_raw[ag_num] = 0  # 0

        agent1.PR = np.zeros(num_agents)
        agent1.PB = np.zeros(num_agents)
        agent1.R = np.zeros(num_agents)
    total_actions = 0

    for agent1 in agents_list:
        agent1.raw_acting_rate = 0.5 * agent1.base_budget
        basic_agent_raw_rate = (0.5 * agent1.base_budget) / num_agents
        agent1.acting_rate = 1 - np.exp(-agent1.raw_acting_rate)
        agent1.agent_rates = np.ones(num_agents) * (1 - np.exp(-basic_agent_raw_rate))
        agent1.agent_rates[agent1.num] = 0

    round_reward = 0
    agent = 0
    th = False
    for i in range(timesteps):
        if i > gossip_timestep+20000 and th is False:
            th = False
            for agent1 in agents_list:
                if len(agent1.followers) == num_agents - 1:
                    th = True
            if th is True:
                gossip_timestep = 99999999
        acted = False
        # agent = random.randint(0, env.n - 1)  # Choose random agent
        # agent += 1
        # if agent >= num_agents:
        #     agent = 0
        agent = i % num_agents
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)
        state["agents"][agents_list[0].position[0]] -= 1
        # print(state)

        for ap in range(orchard_length):
            if state["apples"][ap] == 1:
                apple_pos_x.append(i % 1000)
                apple_pos_y.append(ap)

        # action = agents_list[agent].get_action(state, discount)  # Get action.
        # epsilon_val = random.random()
        # if i < 50000:
        #     action = random.choice([0, 1, 4])
        # if epsilon_val < 0.05 and i < 20000:
        #     action = random.choice([0, 1, 4])
        # elif epsilon_val < 0.02:
        #     action = random.choice([0, 1, 4])

        epsilon = 0.02
        chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        if chance < epsilon:
            action = random_policy_1d(state, old_pos)
        else:
            action = agents_list[agent].get_action(state, discount)  # Get action.

        action_val = random.random()
        # if action_val > agents_list[agent].acting_rate:
        #     agent += 1
        # if agent == 0:
        #     agent += 1
        if action_val < agents_list[agent].acting_rate:
            reward, new_position = env.main_step(agents_list[agent].position.copy(), action)
            acted = True
            total_actions += 1
        else:
            reward, new_position = env.main_step_without_action(agents_list[agent].position.copy())
        total_reward += reward
        round_reward += reward
        agents_list[agent].position = new_position

        round_actions[agent] = action
        # peragprod_plots[agent].append(action)

        new_state = env.get_state()
        new_state["agents"][agents_list[agents_list[agent].target_influencer].position[0]] -= 1

        """ Placeholder States """
        sp_state = {
            "agents": samp_state["agents"].copy(),
            "apples": samp_state["apples"].copy(),
            "pos": [0, 0]
        }
        sp_new_state = {
            "agents": samp_state["agents"].copy(),
            "apples": samp_state["apples"].copy(),
            "pos": [0, 0]
        }
        train_state = {
            "agents": state["agents"].copy(),
            "apples": state["apples"].copy(),
            "pos": old_pos.copy()
        }
        train_new_state = {
            "agents": new_state["agents"].copy(),
            "apples": new_state["apples"].copy(),
            "pos": agents_list[agent].position.copy()
        }
        """
        Alpha/Beta in-place calculation
        """
        beta_sum = 0
        state_a = new_state["agents"]
        state_b = new_state["apples"]

        action_utils = np.zeros(num_agents)
        action_utils_raw = np.zeros(
            num_agents)  # the value function value observed for each agent (to prevent recalculations)
        action_utils_infl = np.zeros(num_agents)
        """ Utility Observations - Observers """
        for numnow, each_agent in enumerate(agents_list):
            if len(each_agent.followers) == 0:
                raw_value = 0
                if acted:
                    raw_value = each_agent.get_util_learned(state_a, state_b,
                                                            each_agent.position)  # singular value function call per step
                valued = raw_value * each_agent.agent_rates[agent]
                action_utils[numnow] = valued
                action_utils_raw[numnow] = raw_value
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

                each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], valued,
                                                                each_agent.discount_factor)
                each_agent.alphas_raw[agent] = get_discounted_value(each_agent.alphas_raw[agent], raw_value,
                                                                    each_agent.discount_factor)
        """ Utility Observations - Influencers"""
        for numnow, each_agent in enumerate(agents_list):
            if len(each_agent.followers) > 0:
                # infl_sum = 0
                for agent_num in range(len(agents_list)):  # alpha for agent_num while each_agent is an INFL
                    if agent_num != agent:
                        """ Evaluating Agents that didn't act """
                        each_agent.alphas_asinfl[agent_num] = get_discounted_value(each_agent.alphas_asinfl[agent_num],
                                                                                   0, each_agent.discount_factor)
                        each_agent.alphas_asinfl_raw[agent_num] = get_discounted_value(
                            each_agent.alphas_asinfl_raw[agent_num], 0, each_agent.discount_factor)

                # for following_agent in agents_list:
                #     if following_agent.target_influencer == each_agent.num:
                #         bval = each_agent.agent_rates[agent] * action_utils_raw[following_agent.num] * following_agent.infl_rate
                #         infl_sum += bval
                # action_utils_infl[numnow] = infl_sum

        """ Utility Observations - Indirect Following & Influencer Totals"""
        # Influencer Retweet
        for numnow, each_agent in enumerate(agents_list):
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
                    # if indirect_num == agent:
                    #     indirect_agent.indirect_alphas[numnow] *= (1 - indirect_agent.discount_factor)
                    #     indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)

                    if indirect_num == numnow or agents_list[indirect_num].target_influencer != each_agent.num:
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)

                    elif indirect_agent.target_influencer == each_agent.num:
                        bval = each_agent.agent_rates[agent] * action_utils_raw[indirect_num]
                        inflrate = indirect_agent.infl_rate
                        indirect_agent.indirect_alphas[numnow] = get_discounted_value(
                            indirect_agent.indirect_alphas[numnow], bval * inflrate, indirect_agent.discount_factor)
                        indirect_agent.indirect_alphas_raw[numnow] = get_discounted_value(
                            indirect_agent.indirect_alphas_raw[numnow], bval,
                            indirect_agent.discount_factor)
                        tot_val += bval * inflrate
                        tot_val_raw += bval

                    else:
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)
                action_utils_infl[numnow] = tot_val
                each_agent.alphas_asinfl[agent] = get_discounted_value(each_agent.alphas_asinfl[agent], tot_val,
                                                                       each_agent.discount_factor)
                each_agent.alphas_asinfl_raw[agent] = get_discounted_value(each_agent.alphas_asinfl_raw[agent],
                                                                           tot_val_raw,
                                                                           each_agent.discount_factor)
            else:
                """
                If you have no followers clearly you aren't sharing anything
                """
                for indirect_num, indirect_agent in enumerate(agents_list):
                    indirect_agent.indirect_alphas[numnow] *= 0  # (1 - indirect_agent.discount_factor)
                    indirect_agent.indirect_alphas_raw[numnow] *= 0  # (1 - indirect_agent.discount_factor)

        # agents_list[agent].bqueue.append(beta_sum)

        # p_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)
        o_network_list[agent].add_experience(sp_state, sp_new_state, reward, action, agents_list)
        i_network_list[agent].add_experience(sp_state, sp_new_state, reward, action, agents_list)
        # v_network_list[agent].addexp(state, new_state, reward, action, agents_list)
        if i % 500 == 0 and th:
            for agent1 in agents_list:
                agent1.generate_rates_only(samp_state["agents"], samp_state["apples"], const_ext=False)
        if i % 200 == 0 and th:
            for agent1 in agents_list:
                agent1.set_functional_rates(sp_state["agents"], sp_state["apples"], sp_state["pos"])
                for ja in range(num_agents):
                    if agent1.agent_rates[ja] < 1e-5 and ja != agent1.num:
                        agent1.agent_rates[ja] = 0.01
        if i % 200 == 0 and i < gossip_timestep:
            for agent2 in agents_list:
                outpt = agent2.get_learned_action_record(samp_state)
                peragprod_plots[agent2.num][0].append(outpt[0])
                peragprod_plots[agent2.num][1].append(outpt[1])
                peragprod_plots[agent2.num][2].append(outpt[2])
                if agent2.num == 1:
                    print("OUT:", outpt)
                peragval_plots[agent2.num].append(
                    agent2.get_sum_value(samp_state["agents"], samp_state["apples"], samp_state["pos"]))
        if i > gossip_timestep:
            if i % 200 == 0 and i > gossip_timestep:
                print("@ Timestep", i)
                for agent2 in agents_list:
                    outpt = agent2.get_learned_action_record(samp_state)
                    peragprod_plots[agent2.num][0].append(outpt[0])
                    peragprod_plots[agent2.num][1].append(outpt[1])
                    peragprod_plots[agent2.num][2].append(outpt[2])
                    if agent2.num == 1:
                        print("OUT:", outpt)
                    peragval_plots[agent2.num].append(
                        agent2.get_sum_value(samp_state["agents"], samp_state["apples"], samp_state["pos"]))
            if i % 10 == 0 and i < gossip_timestep + 30000 and i > gossip_timestep:

                for agent1 in agents_list:
                    #agent1.generate_agent_rates_static_const_ext(state["agents"], state["apples"], i)
                    agent1.set_functional_rates(sp_state["agents"], sp_state["apples"], sp_state["pos"])
                    for ja in range(num_agents):
                        if agent1.agent_rates[ja] < 1e-5 and ja != agent1.num:
                            agent1.agent_rates[ja] = 0.01
            elif i % 100 == 0 and i > gossip_timestep + 30000:
                for agent1 in agents_list:
                    #agent1.generate_agent_rates_static_const_ext(state["agents"], state["apples"], i)
                    agent1.set_functional_rates(sp_state["agents"], sp_state["apples"], sp_state["pos"])
            for agent1 in agents_list:
                # if len(agent1.followers) == 99:
                #     if not lkdown:
                #         print("locked")
                #     lkdown = True
                if agent1.trigger_change(i-gossip_timestep, True) or np.abs(len(agent1.followers) - agent1.LRF) >= agent1.n0:
                    if np.abs(len(agent1.followers) - agent1.LRF) >= agent1.n0:
                        print(agent1.num, "momentum trigger")
                    ext_plots_x[agent1.num].append(i-1)
                    ext_plots_y[agent1.num].append(agent1.raw_b0_rate)

                    #agent1.generate_agent_rates_static(state["agents"], state["apples"], agents_list)
                    agent1.generate_rates_only(samp_state["agents"], samp_state["apples"], const_ext=False)

                    ext_plots_x[agent1.num].append(i)
                    ext_plots_y[agent1.num].append(agent1.raw_b0_rate)
                if agent1.trigger_change(i-gossip_timestep, False) or (len(agent1.followers) == 0 and agent1.target_influencer == -1) or (agent1.target_influencer == -2 and i > gossip_timestep + 500):
                    if len(agent1.followers) == 0 and agent1.target_influencer == -1:
                        agent1.LRF = 0
                    if len(agent1.followers) == 0 and lkdown == False:
                        agent1.identify_influencer(agents_list)

            for ag in agents_list:
                ag.shed_influencer(agents_list)
        # print state, action, feedback, reward, agents_list
        feedback = np.sum(action_utils) + np.sum(action_utils_infl) + agents_list[agent].get_util_learned(state_a,
                                                                                                          state_b,
                                                                                                          agents_list[
                                                                                                              agent].position)
        if acted: # and i < training_timestep + (num_agents * 100):
            for agnum, ntwk in enumerate(v_network_list):
                if agnum == agent:
                    v_network_list[agnum].train(state, new_state, reward, old_pos, agents_list[agnum].position)
                else:
                    v_network_list[agnum].train(state, new_state, 0, agents_list[agnum].position,
                                                agents_list[agnum].position)
        if acted and i > training_timestep:
            p_network_list[agent].add_experience(train_state, train_new_state, reward, action, agents_list, feedback,
                                                 action_utils_raw)
            # for agnum in range(len(agents_list)):
            #     if agnum == agent:
            #         v_network_list[agnum].train(state, new_state, reward, old_pos, agents_list[agnum].position)
            #     else:
            #         v_network_list[agnum].train(state, new_state, 0, agents_list[agnum].position, agents_list[agnum].position)
        # if acted and i > 5000:
        #
        #     # p_network_list[agent].train_with_feedback(state, old_pos, action, feedback, reward, agents_list)
        #     p_network_list[agent].train_with_v_value(state, old_pos, action, feedback, reward, agents_list)
        if i > training_timestep + (num_agents * 100):
            if i % (num_agents * 100) == 1:
                for a in range(0, len(agents_list)):
                    p_network_list[a].train_multiple_with_beta(agents_list)
        """ Perform Beta Updates AFTER p """
        # agents_list[agent].beta = get_discounted_value(agents_list[agent].beta,
        #                                                feedback,
        #                                                agents_list[agent].beta_discount_factor)
        # for agent3 in agents_list:
        #     if agent3.num != agent:
        #         agent3.beta *= (1 - agents_list[agent].discount_factor)

        # if i != 0:
        #     if i % 20 == 0:
        #         for ntwk in p_network_list:
        #             ntwk.train_multiple(agents_list)

        # if i != 0 and acted:
        #     #if i % 20 == 0:
        #     for agnum, ntwk in enumerate(v_network_list):
        #         if agnum == agent:
        #             v_network_list[agnum].train(state, new_state, reward, old_pos, agents_list[agnum].position)
        #         else:
        #             v_network_list[agnum].train(state, new_state, 0, agents_list[agnum].position, agents_list[agnum].position)

        if i > gossip_timestep:
            if i % 20 == 0:
                for ag in agents_list:
                    if len(ag.followers) > 0:
                        i_network_list[ag.num].update(agents_list)
                    else:
                        o_network_list[ag.num].update(agents_list)

        if i % 100 == 0 or i == timesteps - 1:
            """ Should all be Plotting / Recording """
            if i != 0:
                if i > gossip_timestep:
                    if i % 100 == 0:
                        if not follow_setup:
                            follow_setup = True
                            for ag in agents_list:
                                # if len(ag.followers) > 0:
                                follow_indices.append(ag.num)
                                follow_plots.append([])
                        if follow_setup:
                            for index, infl_num in enumerate(follow_indices):
                                if len(agents_list[infl_num].followers) > agents_list[infl_num].max_followers:
                                    agents_list[infl_num].max_followers = len(agents_list[infl_num].followers)
                                follow_plots[index].append(len(agents_list[infl_num].followers))
                # if i % 100 == 0:
                #     if i < gossip_timestep:
                #         for agent1 in agents_list:
                #             #agent1.generate_agent_rates(state["agents"], state["apples"])
                #             agent1.generate_agent_rates_static(state["agents"], state["apples"], agents_list)
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
                            print("Agent " + str(agent) + ":", agents_list[agent].beta, "beta /",
                                  np.sum(action_utils) + np.sum(action_utils_infl), "this action /", reward, "reward")
            # Sum up things
            reward_plot.append(total_reward)
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

            # if (i % 10000 == 0 or i == timesteps - 1) and i != 0:

            # q_rates = np.array(roundabout_find_allocs_with_b0_full_vec(agents_list[1].alphas_raw,
            #                                                            agents_list[1].infl_alphas_raw,
            #                                                            budget=agents_list[1].budget,
            #                                                            b0=agents_list[1].b0))

            # q_rates = 1 - np.exp(-q_rates)
            # print("Sample Reps for Agent 30:", agents_list[30].R)
        if i % 100 == 0:
            if i > gossip_timestep + 1:
                for agent1 in agents_list:
                    for num1 in range(len(agents_list)):
                        indirect_plots2[num1][agent1.num].append(agent1.indirect_alphas[num1])
                        direct_plots2[num1][agent1.num].append(agent1.alphas[num1] + agent1.alphas_asinfl[num1])
                        # peragrep_plots[num1][fagent1.num].append(agent1.R[num1])
                        # if num1 == agent:
        # if i > gossip_timestep + 1:
        for agent1 in agents_list:
            # peragrep_plots[agent][agent1.num].append(agent1.R[agent])
            peragpos_plots[agent1.num].append(agent1.position[0])
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
                # ag.PR = np.copy(ag.alphas) / 10 + np.copy(ag.indirect_alphas)
                # ag.PR = np.copy(ag.alphas) / 4 + np.copy(ag.indirect_alphas)
                ag.PR = np.zeros(len(agents_list))
                for ag1 in range(len(agents_list)):
                    ag.PR[ag1] = ag.alphas[ag1] + ag.indirect_alphas[ag1]
                # for ag1 in range(len(agents_list)):
                #     if ag.PR[ag1] < 0:
                #         print("Less than zero PR", ag.num, "/", ag1, "/", ag.PR[ag1])
                ag.has_gossiped = np.zeros(num_agents)

            # PR Update
            for ag in agents_list:
                for agnum in range(0, num_agents):
                    d = ag.PR[agnum] - ag.PB[agnum]
                    ag.PB[agnum] = ag.PR[agnum]
                    ag.R[agnum] += d

            for ag in agents_list:
                for adj_ind in range(len(ag.adjs)):
                    # for jkjk in range(4):
                    # adj_ind = random.randint(0, len(ag.adjs)-1)
                    fr_ind = ag.adjs[adj_ind]
                    fr_ind2 = adj_ind
                    # fr_ind2 = adj_ind
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
            runner = 0
            strunner = str(runner)
            frontrn = np.argmax(agents_list[runner].R)
            print("Frontrunner Follow Rates:", list(agents_list[frontrn].agent_rates))
            print("Frontrunner Alphas:", np.sum(agents_list[frontrn].alphas_asinfl))

            print("Agent " + strunner + " Follow Rates:", list(agents_list[runner].agent_rates))
            print("Agent " + strunner + " Direct Alphas:", np.sum(agents_list[runner].alphas))
            print("Agent " + strunner + " Indir Alpha:",
                  agents_list[runner].indirect_alphas[agents_list[runner].target_influencer])
            print("Agent " + strunner + " TI", agents_list[runner].target_influencer, "INFL RATE",
                  agents_list[runner].infl_rate,
                  "b0 RATE", agents_list[runner].raw_b0_rate)
            print("Frontrunner (for agent " + strunner + "):", agents_list[runner].R[frontrn], "agent:", frontrn)
            # print("Sample Reputations for Frontrunner:", agents_list[1].R[frontrn], agents_list[4].R[frontrn],
            #       agents_list[12].R[frontrn],
            #       agents_list[53].R[frontrn])
            # print("Sample Indirect Alphas for Frontrunner", agents_list[1].indirect_alphas[frontrn],
            #       agents_list[4].indirect_alphas[frontrn], agents_list[12].indirect_alphas[frontrn],
            #       agents_list[30].indirect_alphas[frontrn],
            #       agents_list[53].indirect_alphas[frontrn])
        if i > gossip_timestep + 99:
            for agent1 in agents_list:
                peragrep_plots2[agent][agent1.num].append(agent1.R[agent])

        if (i % 10000 == 0 and i != 0) or i == timesteps - 1:
            print("Reward for the last 10000 steps:", round_reward)
            round_reward = 0

            plt.figure("Positions" + name)
            for ih, plot in enumerate(peragpos_plots):
                plt.plot(plot[-1000:-1], label="Agent " + str(ih))
            plt.plot(apple_pos_x, apple_pos_y, label="Apple Position", marker="o", linestyle="")
            plt.legend()
            plt.title("Positions in Last 1000 Steps, Timestep " + str(i))
            plt.savefig(name + "_positions.png")
            plt.close()

            for agent in agents_list:
                plt.figure("Productions" + str(agent.num) + "temp")
                plt.plot(peragprod_plots[agent.num][0], label="Left")
                plt.plot(peragprod_plots[agent.num][1], label="Right")
                plt.plot(peragprod_plots[agent.num][2], label="Stay")
                plt.plot(peragval_plots[agent.num], label="Value Function")
                plt.legend()
                plt.title("Agent Productions for Agent " + str(agent.num))
                plt.savefig(name + "_" + str(agent.num) + "_prodalls.png")
                plt.close()

            plt.figure("Ext_Old" + name)
            for ih, plot in enumerate(external_plots):
                plt.plot(plot, label="Agent " + str(ih))
            plt.legend()
            plt.title("External Rates")
            plt.savefig(name + "_ext_old.png")
            plt.close()

            plt.figure("Total Social Welfare" + name)
            plt.plot(sw_plot)
            plt.legend()
            plt.title("Total Social Welfare")
            plt.savefig(name + "_socwel.png")
            plt.close()

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
                    plt.plot(follow_plot, label="Agent " + str(ih) + " R", color='red')
                elif ih == top2:
                    plt.plot(follow_plot, label="Agent " + str(ih) + " R", color='blue')
                else:
                    plt.plot(follow_plot)
            plt.legend()

            print("=====Eval at", i, "steps======")
            fname = name
            if altname is not None:
                fname = altname
            for numbering, network in enumerate(p_network_list):
                torch.save(network.function.state_dict(), fname + "_Actor_" + str(numbering) + ".pt")
                torch.save(network.optimizer.state_dict(), fname + "_Actor_" + str(numbering) + "_optimizer.pt")
            for numbering, network in enumerate(v_network_list):
                torch.save(network.function.state_dict(), fname + "_Value_" + str(numbering) + ".pt")
                torch.save(network.optimizer.state_dict(), fname + "_Value_" + str(numbering) + "_optimizer.pt")
            # maxi = eval_network(fname, discount, agents_list[0].base_budget, maxi, p_network_list, v_network_list,
            #                     iteration=0, num_agents=len(agents_list), side_length=orchard_length)
            print("=====Completed Evaluation=====")
        if i % 1000 == 0:
            apple_pos_x = []
            apple_pos_y = []
        # if i == num_agents * 10000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.0005
        # if i == num_agents * 20000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.0002
        # if i > gossip_timestep + num_agents:

        #     if i % 100 == num_agents - 1:
        #         if len(round_actions) == 0:
        #             print("roundactions empty")
        #             continue
        #         for agent1 in agents_list:
        #             est_sum = 0
        #             if len(agent1.followers) > 0:
        #                 for agnum, topic in enumerate(round_actions):
        #                     est_sum += agent1.get_follower_feedback(agents_list[agnum], topic, agents_list) * \
        #                                agent1.agent_rates[agnum]
        #             for agnum, agent2 in enumerate(agents_list):
        #                 est_sum += agent2.get_util_pq(round_actions[agent1.num], agent1)
        #             rep_estimate_plots[agent1.num].append(est_sum)

    """
    ### End of Loop ^
    Graphing Section


    ###
    """

    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)
    print("Actions:", total_actions, "/", timesteps)
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
            plt.plot(follow_plot, label="Agent " + str(ih) + " R", color='red')
        elif ih == top2:
            plt.plot(follow_plot, label="Agent " + str(ih) + " R", color='blue')
        else:
            plt.plot(follow_plot)
    plt.legend()

    if folder == '':
        folder = "placeholder"
    prefix0 = "graphs_EI_MARL/" + folder
    prefix1 = prefix0 + "/"
    # prefix1 = "graphs_EI/K3/"
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

    plt.figure("Positions" + name)
    for ih, plot in enumerate(peragpos_plots):
        plt.plot(plot[-1000:-1], label="Agent " + str(ih) + " Rate")
    plt.legend()
    plt.title("Positions in Last 1000 Steps")
    plt.savefig(prefix1 + name + "_positions.png")
    plt.close()

    repp2 = np.copy(rep_plots[top1])
    if not os.path.exists(prefix1 + "/per_agent_production"):
        os.makedirs(prefix1 + "/per_agent_production")
    if not os.path.exists(prefix1 + "/per_agent_rates"):
        os.makedirs(prefix1 + "/per_agent_rates")

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
        plt.plot(peragprod_plots[agent.num][0], label="Policy Network 0")
        plt.plot(peragprod_plots[agent.num][1], label="Policy Network 1")
        plt.plot(peragprod_plots[agent.num][2], label="Policy Network 2")
        plt.plot(peragval_plots[agent.num], label="Value Function")
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

    plt.figure("Total Reward Over Time" + name)
    plt.plot(reward_plot)
    plt.legend()
    plt.title("Total Reward Over Time")
    plt.savefig(prefix1 + name + "_reward.png")
    plt.close()

    # plt.figure("Total Attention In Community" + name)
    # plt.plot(att_plot)
    # plt.title("Total Attention In Community")
    # plt.savefig(prefix1 + name + "_att.png")
    # plt.close()

    if not os.path.exists("saved_item/" + name):
        os.makedirs("saved_item/" + name)
    # np.save("saved_item/" + name + "/socwell.npy", sw_plot)
    # np.save("saved_item/" + name + "/ext.npy", np.array(external_plots))
    # np.save("saved_item/" + name + "/rep.npy", np.array(rep_plots))
    # np.save("saved_item/" + name + "/rep_est.npy", np.array(rep_estimate_plots))
    # np.save("saved_item/" + name + "/followers.npy", np.array(follow_plots))
    # np.save("saved_item/" + name + "/direct.npy", np.array(direct_plots))
    # np.save("saved_item/" + name + "/indirect.npy", np.array(indirect_plots))


def eval_network(name, discount, gen_budget, maxi, p_network_list, v_network_list, num_agents=4, side_length=10,
                 iteration=99):
    network_list = []
    a_list = []
    # for ii in range(num_agents):
    #     # print("A")
    #     network = ActorNetwork(side_length, 0.0012, discount)
    #     network.function.load_state_dict(torch.load(name + "_Actor3_" + str(ii) + ".pt"))
    #     #network.function.load_state_dict(torch.load("../" + name + "_Actor_BETA_ALPHA_" + str(i) + ".pt"))
    #     # network.function.load_state_dict(torch.load("../" + experiment_name + "_Actor_BETA_ALPHA_" + str(i) + ".pt"))
    #     # for param in network.function.parameters():
    #     #     print(param.data)
    #     network_list.append(network)

    packs = {
        "base_a": 0,
        "alpha": 0,
        "kappa": 0,
        "n0": 0
    }

    for ii in range(num_agents):
        trained_agent = OrchardAgent(policy="learned_policy", id=ii, num_agents=num_agents, influencers=[],
                                     main_interest=ii,
                                     budget=gen_budget, b0=5, topics=side_length, nonfixedpackage=packs)
        # print(trained_agent.num)
        trained_agent.policy_network = p_network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        val = run_environment_1d(num_agents, random_policy_1d, side_length, None, None, "MARL", "test1",
                                 agents_list=a_list,
                                 spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=10000)
    # if val > maxi and iteration != 99:
    #     print("saving best")
    #     for nummer, netwk in enumerate(p_network_list):
    #         torch.save(netwk.function.state_dict(),
    #                    "policyitchk/" + name + "/" + name + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")
    #     for nummer, netwk in enumerate(v_network_list):
    #         torch.save(netwk.function.state_dict(),
    #                    "policyitchk/" + name + "/" + name + "_value_" + str(nummer) + "_it_" + str(iteration) + ".pt")

    maxi = max(maxi, val)
    return maxi


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

    # alpha = 0.000005 * 100 / 2
    alpha = 0.001
    for nummer, agn in enumerate(agents_list):
        agn.policy_network = ActorNetwork(side_length, alpha, discount, num=nummer)
        agn.policy = "learned_policy"
        agn.observer_network = ObserverNetwork(side_length, num_agents, alpha, discount, num=nummer)
        agn.follower_network = agn.observer_network
        agn.influencer_network = ObserverNetwork(side_length, num_agents, alpha, discount, num=nummer,
                                                 infl_net=True)
        agn.value_network = ValueNetwork(side_length, 0.0002, discount)

    # if len(agents_list[0].influencers) > 0:
    #    agents_list[0].influencers[0].observer_network = ObserverNetwork(side_length, num_agents, alpha * 6, discount,
    #                                                                     num=nummer, infl_net=True)

    training_loop(agents_list, side_length, None, None, 0.00002, name, has_beta=True, discount=discount,
                  timesteps=timesteps, scenario=scenario, in_thres=in_thres, folder=folder)


def experiment(name, base_a=0.01, alpha_decay=(0.7 / 40000), kappa_decay=5, threshold=5, ts=5000, folder=''):
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
    num_agents = 4
    side_length = 10
    discount = 0.99
    agents_list = []
    gen_budget = 10
    scenario = 7
    lin_dec = False
    nb0 = 1
    # b0 = b0 / num_agents

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
        main_int = 1
        agents_list.append(
            OrchardAgent(policy="learned_policy", id=i, num_agents=num_agents, influencers=influencers,
                         main_interest=main_int,
                         budget=gen_budget, b0=nb0, topics=side_length, nonfixedpackage=packs))

    train_ac_content(side_length, num_agents, agents_list, name, discount, ts, iteration=0,
                     scenario=scenario, folder=folder)


from agents.jan_marl_agent import OrchardAgent

import os

for itee in range(1):
    print("ITERATION ", itee)
    experiment("emergent_influencer_10_5", base_a=0.01, kappa_decay=5, ts=500000,
               folder="emergent_influencer_10_5")
