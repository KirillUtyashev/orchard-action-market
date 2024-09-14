import time

from ac_utilities import find_ab, find_ab_bin
from alloc.allocation import find_allocs
from main import run_environment_1d
from models.simple_connected_multiple import SCMNetwork
from orchard.environment import *
import numpy as np
import matplotlib.pyplot as plt
import random
from policies.random_policy import random_policy_1d, random_policy
from models.simple_connected_multiple_dc import SCMNetwork, SimpleConnectedMultiple
from models.actor_dc_1d import ActorNetwork

#from models.actor_dc_1d_altinput import ActorNetwork # USING ALTINPUT
#from models.simple_connected_multiple_dc_altinput import SCMNetwork, SimpleConnectedMultiple
from orchard.algorithms import single_apple_spawn, single_apple_despawn


import torch
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

one_plot = {}
two_plot = {}
def setup_plots(dictn, plot):
    for param_tensor in dictn:
        #print(dictn[param_tensor].size())
        plot[param_tensor] = []
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor: # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)] = []
def add_to_plots(dictn, timestep, plot):
    for param_tensor in dictn:
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor: # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)].append(dictn[param_tensor].cpu().flatten()[id])

graph = 0
loss_plot = []
loss_plot1 = []
loss_plot2 = []
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def graph_plots(dictn, name, plot):
    global graph
    graph += 1
    plt.figure("plots" + str(graph) + name, figsize=(10, 5))
    num = 0
    for param_tensor in dictn:
        num += 1
        if num == 10:
            break
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor: # and "1" not in param_tensor:
            for id in range(5):
                if id == 0:
                    plt.plot(plot[param_tensor + str(id)], color=colours[num], label="Tensor " + str(num))
                else:
                    plt.plot(plot[param_tensor + str(id)], color=colours[num])
        # else:
        #     plt.plot(plot[param_tensor], label=str(num))
    plt.legend()
    plt.title("Model Parameters during Training, iteration " + str(graph))
    plt.savefig("ParamsActor_" + name + str(graph) + ".png")

    plt.figure("loss" + str(graph), figsize=(10, 5))
    plt.plot(loss_plot)
    plt.plot(loss_plot1)
    plt.plot(loss_plot2)
    #print(loss_plot2)
    plt.title("Value Function for Sample State, iteration " + str(graph))
    plt.savefig("ValActor_" + name + str(graph) + ".png")

# sample_state = {
#     "agents": np.array([[1], [0], [0], [0], [3], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([0, 0]), np.array([4, 0]), np.array([4, 0])]
# }
# sample_state5 = {
#     "agents": np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([2, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }
# sample_state6 = {
#     "agents": np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }

sample_state = {
    "agents": np.array([[0], [2], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0]]),
    "pos": [np.array([1, 0]), np.array([1, 0])]
}
sample_state5 = {
    "agents": np.array([[0], [0], [2], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0]]),
    "pos": [np.array([2, 0]), np.array([2, 0])]
}
sample_state6 = {
    "agents": np.array([[0], [0], [0], [0], [2]]),
    "apples": np.array([[1], [0], [0], [0], [0]]),
    "pos": [np.array([4, 0]), np.array([4, 0])]
}

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
        if iteration == 99:
            print("creating SCMs")
            network1 = SCMNetwork(orchard_length, alpha, discount)
            agents_list[agn].policy_value = network1
            network1.function.load_state_dict(torch.load(name + "_" + str(agn) + ".pt"))
            v_network_list.append(network1)
            network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
            agents_list[agn].policy_network = network2
            p_network_list.append(network2)
        else:
            assert agents_list[agn].policy_value is not None
            v_network_list.append(agents_list[agn].policy_value)

            if iteration == 0:
                network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
                agents_list[agn].policy_network = network2
                p_network_list.append(network2)
                agents_list[agn].policy = "learned_policy"
            else:
                assert agents_list[agn].policy_network is not None
                p_network_list.append(agents_list[agn].policy_network)

        # network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
        # agents_list[agn].policy_network = network2
        # p_network_list.append(network2)

    total_reward = 0

    """ Plotting Setup """
    setup_plots(p_network_list[0].function.state_dict(), one_plot)
    global loss_plot
    loss_plot = []
    maxi = 0
    """"""
    for i in range(timesteps):

        agent = random.randint(0, env.n - 1)  # Choose random agent
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        #action = agents_list[agent].get_action(state, discount)

        epsilon = 0
        chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        if chance < epsilon:
            action = random_policy_1d(state, old_pos)
        else:
            action = agents_list[agent].get_action(state, discount)  # Get action.

        reward, new_position = env.main_step(agents_list[agent].position.copy(),
                                             action)  # Take the action. Observe reward, and get new position
        agents_list[agent].position = new_position.copy()  # Save new position.
        total_reward += reward  # Add to reward.

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

        #p_network_list[agent].train(sp_state, sp_new_state, reward, action, agents_list)
        p_network_list[agent].addexp(sp_state, sp_new_state, reward, action, agents_list)
        if i % 1000 == 0:
            add_to_plots(p_network_list[0].function.state_dict(), i, one_plot)
            v_value = agents_list[1].policy_network.get_function_output(sample_state["agents"], sample_state["apples"], pos=sample_state["pos"][0])
            v_value1 = agents_list[1].policy_network.get_function_output(sample_state5["agents"], sample_state5["apples"],
                                                                        pos=sample_state5["pos"][1])
            v_value2 = agents_list[0].policy_network.get_function_output(sample_state6["agents"],
                                                                         sample_state6["apples"],
                                                                         pos=sample_state6["pos"][0])
            if i % 20000 == 0:
                print("Sample Value:", v_value)
            loss_plot.append(v_value[0])
            loss_plot1.append(v_value1[0])
            loss_plot2.append(v_value2[0])
            for ntwk in p_network_list:
                ntwk.train_multiple(agents_list)

        if i % 20000 == 0 and i != 0:
            print("At timestep: ", i)
        # was: 300000
        if i == 50000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.001
        if i == 100000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0005
        # was: 500000
        if i == 200000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001
        # was: 700000
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
        if i % 40000 == 0 and i != 0:
            print("=====Eval at", i, "steps======")
            fname = name
            if altname is not None:
                fname = altname
            #   for numbering, network in enumerate(p_network_list):
            #       torch.save(network.function.state_dict(), fname + "_Actor3_" + str(numbering) + ".pt")
            #       torch.save(network.optimizer.state_dict(), fname + "_Actor3_" + str(numbering) + "_optimizer.pt")
            maxi = eval_network(fname, discount, maxi, network_list=p_network_list, iteration=iteration, num_agents=len(agents_list), side_length=orchard_length)
            print("=====Completed Evaluation=====")
    fname = name
    if altname is not None:
        fname = altname
    for numbering, network in enumerate(p_network_list):
        torch.save(network.function.state_dict(), fname + "_Actor3_" + str(numbering) + ".pt")
    graph_plots(p_network_list[0].function.state_dict(), fname, one_plot)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)



from agents.actor_critic_agent import ACAgent

def eval_network(name, discount, maxi, network_list, num_agents=4, side_length=10, iteration=99):
    network_list = []
    a_list = []

    for ii in range(num_agents):
        trained_agent = ACAgent(policy="learned_policy", num=ii, num_agents=num_agents)
        # print(trained_agent.num)
        trained_agent.policy_network = network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        val = run_environment_1d(num_agents, random_policy_1d, side_length, None, None, "AC", "test", agents_list=a_list,
                           spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)
    if val > maxi and iteration != 99:
        print("saving best")
        for nummer, netwk in enumerate(network_list):
            if nummer == 0:
                print("policyitchk/" + name + "/" + name + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")
            torch.save(netwk.function.state_dict(),
                       "policyitchk/" + name + "/" + name + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")

    maxi = max(maxi, val)
    return maxi

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


if __name__ == "__main__":
    side_length = 10
    num_agents = 4

    S = None
    phi = None

    agents_list = []
    for i in range(num_agents):
        agents_list.append(ACAgent(policy=random_policy_1d, num=i, num_agents=num_agents))
        #agents_list[i].policy = "learned_policy"

    for i in range(1):
        print("loop", i)

        training_loop(agents_list, side_length, S, phi, 0.0013, "D-RANDOM_4_10", discount=0.99, timesteps=1500000, altname="AltD-RANDOM_4_10")

        if i > 0:
            print("Total Same Actions:", agents_list[0].same_actions)
            agents_list[0].same_actions = 0










