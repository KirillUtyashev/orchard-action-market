from agents.communicating_agent import CommAgent
from agents.jan_marl_agent import OrchardAgent
from agents.simple_agent import SimpleAgent
from orchard.environment import *
import numpy as np
import random
from policies.nearest_uniform import replace_agents_1d
from policies.random_policy import random_policy_1d, random_policy
from policies.nearest import nearest_1d, nearest
from metrics.metrics import append_metrics, plot_metrics, append_positional_metrics, plot_agent_specific_metrics
from controllers import AgentController, AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController
import time
same_actions = 0


def step(agents_list, environment: Orchard, agent_controller):
    agent = random.randint(0, environment.n-1)
    state = environment.get_state()
    if agents_list[agent].policy == "value_function":
        action = agent_controller.get_best_action(state, agent, environment.available_actions)
    else:
        action = agents_list[agent].policy(environment.available_actions)
    global same_actions
    same_actions += (action == nearest(state, agents_list[agent].position))
    reward, new_position = environment.main_step(agents_list[agent].position.copy(), action)
    agents_list[agent].position = new_position
    # print(new_position)
    return agent, reward


def run_environment_1d_acting_rate(num_agents, policy, side_length, S, phi, name="Default", experiment="Default", timesteps=5000, agents_list=None, action_algo=None, spawn_algo=None, despawn_algo=None):
    metrics = []
    agent_metrics = []
    for j in range(5):
        metrics.append([])
    for j in range(num_agents):
        agent_metrics.append([])

    # if agents_list is None:
    #     agents_list = []
    #     for _ in range(num_agents):
    #         agents_list.append(Agent(policy=policy))

    env = Orchard(side_length, num_agents, S, phi, agents_list=agents_list, action_algo=action_algo, spawn_algo=spawn_algo, despawn_algo=despawn_algo)
    env.initialize(agents_list) #, agent_pos=[np.array([1, 0]), np.array([3, 0])]) #, agent_pos=[np.array([2, 0]), np.array([5, 0]), np.array([8, 0])])
    reward = 0
    for i in range(timesteps):
        old_state = env.get_state()
        apples = old_state["apples"]
        agent = i % num_agents  # random.randint(0, env.n - 1)

        state = env.get_state()
        # state["agents"][agents_list[0].position[0]] -= 1
        action = agents_list[agent].get_action(state, agents_list=agents_list)
        val = random.random()
        if val < agents_list[agent].acting_rate:
            i_reward, new_position = env.main_step(agents_list[agent].position.copy(), action)
            acted = True
        else:
            i_reward, new_position = env.main_step_without_action(agents_list[agent].position.copy())
        agents_list[agent].position = new_position
        # print(new_position)
        reward += i_reward
        if name != "test" and experiment != "test":
            metrics = append_metrics(metrics, env.get_state(), reward, i)
            agent_metrics = append_positional_metrics(agent_metrics, agents_list)
        new_state = env.get_state()
        # if np.sum(apples) == 0:
        #     assert np.array_equal(old_state["agents"], new_state["agents"])
        # else:
        #     assert not np.array_equal(old_state["agents"], new_state["agents"]) or not np.array_equal(old_state["apples"], new_state["apples"])

    if name != "test" and experiment != "test":
        print("Same Actions:", same_actions)
    print("Results for", name)
    print("Reward: ", reward)
    print("Total Apples: ", env.total_apples)
    print("Average Reward: ", reward / timesteps)
    print("Apple Ratio: ", reward / env.total_apples)
    if name != "test" and experiment != "test":
        plot_agent_specific_metrics(agent_metrics, experiment, name)
        # plot_metrics(metrics, name, experiment)
    return reward


def run_environment_1d(num_agents, side_length, width, S, phi, name="Default", experiment="Default", timesteps=5000, agents_list=None, action_algo=None, spawn_algo=None, despawn_algo=None, vision=None):
    metrics = []
    agent_metrics = []
    for j in range(5):
        metrics.append([])
    for j in range(num_agents):
        agent_metrics.append([])
    env = Orchard(side_length, width, num_agents, S, phi, agents_list=agents_list, action_algo=action_algo, spawn_algo=spawn_algo, despawn_algo=despawn_algo)
    env.initialize(agents_list) #, agent_pos=[np.array([1, 0]), np.array([3, 0])]) #, agent_pos=[np.array([2, 0]), np.array([5, 0]), np.array([8, 0])])
    reward = 0
    if type(agents_list[0]) is CommAgent:
        agent_controller = AgentControllerDecentralized(agents_list, ViewController(vision))
    else:
        agent_controller = AgentControllerCentralized(agents_list, ViewController(vision))
    for i in range(timesteps):
        # env.render()
        # time.sleep(0.05)
        if i % 1000 == 0:
            print(i)
        agent, i_reward = step(agents_list, env, agent_controller)
        reward += i_reward
        if name != "test" and experiment != "test":
            agent_metrics = append_positional_metrics(agent_metrics, agents_list)
    if name != "test" and experiment != "test":
        print("Same Actions:", same_actions)
    print("Results for", name)
    print("Reward: ", reward)
    print("Total Apples: ", env.total_apples)
    print("Average Reward: ", reward / env.total_apples)
    if name != "test" and experiment != "test":
        plot_agent_specific_metrics(agent_metrics, experiment, name)
        # plot_metrics(metrics, name, experiment)
    return reward, reward / env.total_apples


def all_three_1d(num_agents, length, S, phi, experiment, time=5000):
    run_environment_1d(num_agents, nearest, length, S, phi, "Nearest", experiment, time)
    run_environment_1d(num_agents, nearest, length, S, phi, "Nearest-Uniform", experiment, time, action_algo=replace_agents_1d)
    run_environment_1d(num_agents, random_policy_1d, length, S, phi, "Random", experiment, time)


if __name__ == "__main__":
    side_length = 40
    num_agents = int(side_length * 0.2)

    from orchard.algorithms import single_apple_spawn, single_apple_despawn
    #
    agents_list = []
    for i in range(num_agents):
        main_int = 1
        agents_list.append(
            SimpleAgent(policy=random_policy_1d, debug=False, num=num_agents))
    # run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest", "Single_Apple", spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    # run_environment_1d(num_agents, nearest, side_length, S3, phi, "Nearest-Uniform", "Single_Apple", action_algo=replace_agents_1d, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
    run_environment_1d(num_agents, random_policy_1d, side_length, None, None, "Random", "Single_Apple", spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=40000, agents_list=agents_list)
