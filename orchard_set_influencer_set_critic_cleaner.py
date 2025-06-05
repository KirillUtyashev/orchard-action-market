import random
import numpy as np
random.seed(35279038)
np.random.seed(389043)
from actor_critic import eval_network
from main import run_environment_1d_acting_rate
from orchard.environment import *
import matplotlib.pyplot as plt

from policies.random_policy import random_policy_1d
from models.jan_action_actor import ActorNetwork
from models.content_observer_1d import ObserverNetwork
from models.jan_value_actor import ValueNetwork

from orchard.algorithms import single_apple_spawn, single_apple_despawn

import torch

torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True


def get_discounted_value(old, new, discount):
    if old < 0:
        return new
    return old * (1 - discount) + new * discount

def construct_sample_state(o_length, n_agents):
    samp_state1 = np.zeros((o_length, 1))
    samp_state2 = np.zeros((o_length, 1))
    samp_state1[0] = n_agents - 1
    samp_state1[2] = 1
    samp_state2[3] = 1
    samp_state3 = np.array([2, 0])

    samp_state = {
        "agents": samp_state1,
        "apples": samp_state2,
        "pos": samp_state3
    }
    return samp_state

plotting = True

def training_loop(agents_list, orchard_length, S, phi, name, discount=0.99, timesteps=100000, altname=None, folder=''):
    maxi = 0
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

    v_network_list = []  # Since we trained the value functions earlier, we just use this list for loading the state dicts of the value networks
    for agn in range(len(agents_list)):
        p_network_list.append(agents_list[agn].policy_network)
        o_network_list.append(agents_list[agn].follower_network)
        i_network_list.append(agents_list[agn].influencer_network)
        v_network_list.append(agents_list[agn].value_network)

        if agn >= 0:
            # Load the state
            print("Loading State Dict for Agent", agn)
            agents_list[agn].value_network.function.load_state_dict(torch.load("policyitchk/" + "DC-RANDOM_2_10/" + f"DC-RANDOM_2_10_decen_{agn}_it_99.pt"))
            v_network_list[agn].function.load_state_dict(torch.load("policyitchk/" + "DC-RANDOM_2_10/" + f"DC-RANDOM_2_10_decen_{agn}_it_99.pt"))

    total_reward = 0

    """Construct Sample States"""
    samp_state = construct_sample_state(orchard_length, num_agents)

    """ Plotting Setup """

    """
    Plotting setup for Emergent Influencer
    """
    if plotting:
        follow_plots = []
        follow_indices = []
        follow_setup = False

        apple_pos_x = []
        apple_pos_y = []

        sw_plot = []
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
    round_actions = np.zeros(num_agents)

    for agent1 in agents_list:
        agent1.beta = -0.00001
        for ag_num in range(len(agents_list)):
            agent1.alphas_asinfl[ag_num] = -0.00001 #0
            agent1.alphas_asinfl_raw[ag_num] = -0.00001 #0
            agent1.alphas[ag_num] = -0.00001 #0
            agent1.alphas_raw[ag_num] = -0.00001 #0
            agent1.indirect_alphas[ag_num] = 0 #0
            agent1.indirect_alphas_raw[ag_num] = 0 #0
    total_actions = 0


    """
    We initialize following rates. We assume that agent 1 is the influencer.
    """
    for agent1 in agents_list:
        if agent1.num != 0:
            agents_list[0].followers.append(agent1.num)
            agent1.target_influencer = 0
            agent1.raw_acting_rate = 0.75 * agent1.base_budget
            agent1.acting_rate = 1 - np.exp(-agent1.raw_acting_rate)
            agent1.raw_agent_rates = np.ones(num_agents) / agent1.base_budget
            agent1.agent_rates = 1 - np.exp(-agent1.raw_agent_rates)
            agent1.agent_rates[0] = 0
            agent1.agent_rates[agent1.num] = 0
            agent1.infl_rate = 0.6
        else:
            agent1.raw_acting_rate = 0
            agent1.acting_rate = 0
            agent1.agent_rates = np.ones(num_agents) * 0.6
            agent1.agent_rates[0] = 0


    round_reward = 0
    agent = 0

    for i in range(timesteps):
        agent = (agent + 1) % num_agents

        action_val = random.random()
        if action_val > agents_list[agent].acting_rate:
            agent = (agent + 1) % num_agents
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        if plotting:
            for ap in range(orchard_length):
                if state["apples"][ap] == 1:
                    apple_pos_x.append(i % 1000)
                    apple_pos_y.append(ap)

        epsilon = 0.02
        chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        if chance < epsilon:
            action = random_policy_1d(state, old_pos)
        else:
            action = agents_list[agent].get_action(state, discount)  # Get action.

        reward, new_position = env.main_step(agents_list[agent].position.copy(), action)
        acted = True
        total_actions += 1
        total_reward += reward
        round_reward += reward
        agents_list[agent].position = new_position

        round_actions[agent] = action

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
        """
        Alpha/Beta in-place calculation
        """
        state_a = new_state["agents"]
        state_b = new_state["apples"]

        action_utils = np.zeros(num_agents)
        action_utils_raw = np.zeros(num_agents)  # the value function value observed for each agent (to prevent recalculations)
        action_utils_infl = np.zeros(num_agents)
        """ Utility Observations - Observers """
        for numnow, each_agent in enumerate(agents_list):
            if len(each_agent.followers) == 0:
                raw_value = 0
                if acted:
                    raw_value = each_agent.get_utility(state_a, state_b, each_agent.position)  # singular value function call per step
                valued = raw_value * each_agent.agent_rates[agent]
                action_utils[numnow] = valued
                action_utils_raw[numnow] = raw_value
                if each_agent.num == agent:
                    valued = 0
                    each_agent.times += 1
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
                        each_agent.alphas_asinfl[agent_num] = get_discounted_value(each_agent.alphas_asinfl[agent_num],0, each_agent.discount_factor)
                        each_agent.alphas_asinfl_raw[agent_num] = get_discounted_value(each_agent.alphas_asinfl_raw[agent_num], 0, each_agent.discount_factor)

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
                    # You don't provide indirect alpha to yourself or if they don't have you as a target influencer
                    if indirect_num == numnow or agents_list[indirect_num].target_influencer != each_agent.num:
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)
                    # You provide indirect if they have you as a target influencer
                    elif indirect_agent.target_influencer == each_agent.num:
                        bval = each_agent.agent_rates[agent] * action_utils_raw[indirect_num]
                        inflrate = indirect_agent.infl_rate
                        indirect_agent.indirect_alphas[numnow] = get_discounted_value(indirect_agent.indirect_alphas[numnow], bval * inflrate, indirect_agent.discount_factor)
                        indirect_agent.indirect_alphas_raw[numnow] = get_discounted_value(
                            indirect_agent.indirect_alphas_raw[numnow], bval,
                            indirect_agent.discount_factor)
                        tot_val += bval * inflrate
                        tot_val_raw += bval

                    else:
                        indirect_agent.indirect_alphas[numnow] *= 0
                        indirect_agent.indirect_alphas_raw[numnow] *= (1 - indirect_agent.discount_factor)
                action_utils_infl[numnow] = tot_val
                each_agent.alphas_asinfl[agent] = get_discounted_value(each_agent.alphas_asinfl[agent], tot_val, each_agent.discount_factor)
                each_agent.alphas_asinfl_raw[agent] = get_discounted_value(each_agent.alphas_asinfl_raw[agent], tot_val_raw,
                                                                      each_agent.discount_factor)
            else:
                """
                If you have no followers clearly you aren't sharing anything
                """
                for indirect_num, indirect_agent in enumerate(agents_list):
                    indirect_agent.indirect_alphas[numnow] *= 0
                    indirect_agent.indirect_alphas_raw[numnow] *= 0
        for ag in agents_list:
            if len(ag.followers) > 0:
                i_network_list[ag.num].addexp(sp_state, sp_new_state, reward, action, agents_list)
            else:
                o_network_list[ag.num].addexp(sp_state, sp_new_state, reward, action, agents_list)



        if i % 200 == 0:
            for agent2 in agents_list:
                outpt = agent2.get_learned_action_record(samp_state)
                if agent2.debug:
                    if agent2.num == 1:
                        print("OUT:", outpt)
                if plotting:
                    peragprod_plots[agent2.num][0].append(outpt[0])
                    peragprod_plots[agent2.num][1].append(outpt[1])
                    peragprod_plots[agent2.num][2].append(outpt[2])
                    peragval_plots[agent2.num].append(
                        agent2.get_value_function(samp_state["agents"], samp_state["apples"], samp_state["pos"]))

        feedback = np.sum(action_utils) + np.sum(action_utils_infl) + agents_list[agent].get_utility(state_a,
                                                                                                          state_b,
                                                                                                          agents_list[
                                                                                                              agent].position)
        p_network_list[agent].train_with_feedback(train_state, train_state["pos"], action, feedback, reward,
                                                  agents_list)
        agents_list[agent].beta = get_discounted_value(agents_list[agent].beta, feedback + reward, agents_list[agent].beta_discount_factor)
        """ Perform Beta Updates AFTER p """

        if i % 100 == 0:
            for ag in agents_list:
                if len(ag.followers) > 0:
                    i_network_list[ag.num].update(agents_list)
                else:
                    o_network_list[ag.num].update(agents_list)
        if i % 100 == 0:
            for agent1 in agents_list:
                agent1.set_functional_rates(sp_state["agents"], sp_state["apples"], sp_state["pos"])
        if i % 1000 == 0:
            for agent1 in agents_list:
                agent1.generate_rates_only(state["agents"], state["apples"], const_ext=False)


        if plotting:
            if i % 100 == 0 or i == timesteps - 1:
                """ Should all be Plotting / Recording """
                if i != 0:
                    if i % 200 == 1:
                        for agent1 in agents_list:
                            if len(agent1.followers) > 0:
                                infl_rep = 0
                                for sagn in agents_list:
                                    infl_rep += sagn.R[agent1.num]
                                print("Agent " + str(agent1.num) + ":", len(agent1.followers), "followers", "/",
                                      agent1.raw_b0_rate, "/", infl_rep, "/", list(agent1.followers))
                                print("Agent " + str(agent) + ":", agents_list[agent].beta, "beta /", np.sum(action_utils) + np.sum(action_utils_infl), "this action /", reward, "reward")
                # Sum up things
                reward_plot.append(total_reward)
            for agent1 in agents_list:
                peragpos_plots[agent1.num].append(agent1.position[0])

        if i % 1000 == 0 and i != 0:
            print("At timestep: ", i)
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

            for agent_y in agents_list:
                plt.figure("Productions" + str(agent_y.num) + "temp")
                plt.plot(peragprod_plots[agent_y.num][0], label="Left")
                plt.plot(peragprod_plots[agent_y.num][1], label="Right")
                plt.plot(peragprod_plots[agent_y.num][2], label="Stay")
                plt.plot(peragval_plots[agent_y.num], label="Value Function")
                plt.legend()
                plt.title("Agent Productions for Agent " + str(agent_y.num))
                plt.savefig(name + "_" + str(agent_y.num) + "_prodalls.png")
                plt.close()

            print("=====Eval at", i, "steps======")
            fname = name
            if altname is not None:
                fname = altname
            for numbering, network in enumerate(p_network_list):
                torch.save(network.function.state_dict(), fname + "_SetInfl2_" + str(numbering) + ".pt")
                torch.save(network.optimizer.state_dict(), fname + "_SetInfl2_" + str(numbering) + "_optimizer.pt")
            maxi = eval_network(fname, discount, agents_list[0].base_budget, maxi, p_network_list, v_network_list, iteration=0, num_agents=num_agents, side_length=orchard_length)
            print("=====Completed Evaluation=====")
        if i % 1000 == 0:
            apple_pos_x = []
            apple_pos_y = []
        base_alpha = 0.0005
        if i == num_agents * 10000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = base_alpha / 2
        if i == num_agents * 20000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = base_alpha / 4
        if i == num_agents * 30000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = base_alpha / 8
        if i == num_agents * 40000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = base_alpha / 16

    """
    ### End of Loop ^
    Graphing Section
    
    
    ###
    """

    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)
    print("Actions:", total_actions, "/", timesteps)
    if plotting:
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
        if not os.path.exists(prefix0):
            os.mkdir(prefix0)

        plt.savefig(prefix1 + name + "_followers.png")
        plt.close()

        plt.figure("Positions" + name)
        for ih, plot in enumerate(peragpos_plots):
            plt.plot(plot[-1000:-1], label="Agent " + str(ih) + " Rate")
        plt.legend()
        plt.title("Positions in Last 1000 Steps")
        plt.savefig(prefix1 + name + "_positions.png")
        plt.close()

        if not os.path.exists(prefix1 + "/per_agent_production"):
            os.makedirs(prefix1 + "/per_agent_production")
        if not os.path.exists(prefix1 + "/per_agent_rates"):
            os.makedirs(prefix1 + "/per_agent_rates")

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

def eval_network(name, discount, gen_budget, maxi, p_network_list, v_network_list, num_agents=4, side_length=10, iteration=99):
    a_list = []

    num_agents2 = num_agents - 1
    for ii in range(num_agents2):
        trained_agent = OrchardAgent(policy="learned_policy", id=ii, num_agents=num_agents, budget=gen_budget)
        trained_agent.policy_network = p_network_list[ii+1]
        a_list.append(trained_agent)
        trained_agent.acting_rate = 1
    with torch.no_grad():
        val = run_environment_1d_acting_rate(num_agents2, random_policy_1d, side_length, None, None, "MARL", name, agents_list=a_list,
                           spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=10000)
    if val > maxi:
        print("saving best")
        if not os.path.exists("policyitchk/" + name):
            os.mkdir("policyitchk/" + name)
        for nummer, netwk in enumerate(p_network_list):
            torch.save(netwk.function.state_dict(),
                       "policyitchk/" + name + "/" + name + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")

    maxi = max(maxi, val)
    return maxi


def train_ac_content(side_length, num_agents, agents_list, name, discount, timesteps, folder='', S=None):
    # A edited version of AC Rate. Binary Projection is automatically packaged because of the p/q functions.
    # Further edited to only be a single iteration.

    base_alpha = 0.0005
    alpha = base_alpha
    for nummer, agn in enumerate(agents_list):
        agn.policy_network = ActorNetwork(side_length, alpha, discount, num=nummer)
        agn.policy = random_policy_1d
        agn.observer_network = ObserverNetwork(side_length, num_agents, alpha, discount, num=nummer)
        agn.follower_network = agn.observer_network
        agn.influencer_network = ObserverNetwork(side_length, num_agents, alpha, discount, num=nummer,
                                                 infl_net=True)
        agn.value_network = ValueNetwork(side_length, 0.0003, discount)

    training_loop(agents_list, side_length, S, None, name, discount,
                  timesteps=timesteps, folder=folder)


def call_experiment(num_agents, side_length, timesteps, S=None):
    discount = 0.99
    agents_list = []
    gen_budget = 5
    name = "Orchard_set_influencer_" + str(num_agents) + "_" + str(side_length)
    ts = timesteps
    print("Starting Experiment", name)

    for i in range(num_agents):
        agents_list.append(
            OrchardAgent(policy="learned_policy", id=i, num_agents=num_agents, budget=gen_budget))

    train_ac_content(side_length, num_agents, agents_list, name, discount, ts, folder=name, S=S)


from agents.marl_agent import OrchardAgent
import os


if __name__ == "__main__":

    call_experiment(2, 10, 1000000)
    # call_experiment(14, 70, 1000000)
