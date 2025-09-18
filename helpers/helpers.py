import random

import numpy as np
import torch
from orchard.environment import Orchard, OrchardBasic, OrchardBasicNewDynamic, \
    OrchardEuclideanNegativeRewardsNewDynamic, OrchardEuclideanRewardsNewDynamic
from policies.random_policy import random_policy

same_cell_no_reward = 0
count = 0


def create_env(env_config, num_agents, agent_pos, apples, agents_list, env_cls=OrchardBasic, debug=False):
    env = env_cls(env_config.length, env_config.width, num_agents, agents_list, s_target=env_config.s_target, apple_mean_lifetime=env_config.apple_mean_lifetime, debug=debug)
    env.initialize(agents_list, agent_pos=agent_pos, apples=apples)
    return env


def convert_position(pos):
    if pos is not None:
        temp = np.zeros((len(pos), 1), dtype=int)
        for i in range(len(pos)):
            temp[i] = np.array(pos[i])
        return temp
    return None


def unwrap_state(state):
    return state["agents"].copy(), state["apples"].copy()


def get_empty_fields(env_length, env_width):
    return {
        "agents": np.zeros((env_width, env_length), dtype=int),
        "apples": np.zeros((env_width, env_length), dtype=int),
        "poses": np.zeros((env_width, env_length), dtype=int)
    }


def generate_sample_states(env_length, env_width, num_agents, alt_vision=False):
    """
    Create three sample states. In state i (i=0,1,2), all agents are placed at (row=0, col=i+1),
    and a single apple is placed at (0, 0). Returns a tuple of three state dicts.

    Assumes get_empty_fields(L, W) -> dict with keys:
      - "agents": 2D array-like [L x W] (agent count per cell)
      - "apples": 2D array-like [L x W]
      - "poses":  array of shape [num_agents, 2] (row, col) for each agent
    """
    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    # Need columns 1,2,3 => require width >= 4 (since we index i+1)
    if env_width < 4:
        raise ValueError("env_width must be at least 4 (to place agents at columns 1..3)")
    if env_length < 1:
        raise ValueError("env_length must be >= 1")

    states = []
    for i in range(3):
        s = get_empty_fields(env_length, env_width)

        # Place all agents at (0, i+1)
        col = i + 1
        # Use numpy-style indexing if arrays, but remain compatible with lists
        s["agents"][0][col] = num_agents
        s["poses"] = np.repeat([[0, col]], repeats=num_agents, axis=0) if num_agents > 0 else np.empty((0, 2), dtype=int)

        # One apple at (0, 0)
        s["apples"][0][0] = 1

        states.append(s)

    return tuple(states)


def generate_alt_states(env_length, num_agents):
    res = [get_empty_fields(env_length), get_empty_fields(env_length), get_empty_fields(env_length)]
    for i in range(len(res)):
        res[i]["agents"][i + 1] = np.array(num_agents)


def ten(c, device):
    return torch.from_numpy(c).to(device).double()


def get_discounted_value(old, new, discount_factor=0.05):
    return old * (1 - discount_factor) + new * discount_factor


def step(agents_list, environment: Orchard, agent_controller, epsilon, inference=False):
    agent = random.randint(0, environment.n - 1)
    if agents_list[agent].policy is not random_policy:
        action = agent_controller.agent_get_action(environment, agent, epsilon)
    else:
        action = random_policy(environment.available_actions)
    environment.process_action_eval(agent, agents_list[agent].position.copy(), action)

    if isinstance(environment, OrchardBasicNewDynamic) or isinstance(environment, OrchardEuclideanRewardsNewDynamic) or isinstance(environment, OrchardEuclideanNegativeRewardsNewDynamic):
        environment.remove_apple(agents_list[agent].position.copy())
    # if inference:
    #     # Update personal Q-value from given action
    #     for agent_num in range(len(agents_list)):
    #         if agent_num == agent:
    #             reward = action_result.picker_reward
    #         elif (agent_num + 1) == action_result.owner_id:
    #             reward = action_result.owner_reward
    #         else:
    #             reward = 0
    #
    #         q_value = reward + get_config()["discount"] * agents_list[agent_num].get_value_function(agent_controller.critic_view_controller.process_state(environment.get_state(), agents_list[agent_num].position, agent_num + 1))
    #         agents_list[agent_num].personal_q_value = q_value


def step_reward_learning_decentralized(agents_list, environment, agent_controller, epsilon,
                                       inference=False, tol=1e-1):
    agent_idx = random.randint(0, environment.n - 1)
    action = agent_controller.agent_get_action(environment, agent_idx, None)

    if isinstance(environment, OrchardBasicNewDynamic):
        # Step env and labels
        result = environment.process_action(agent_idx, agents_list[agent_idx].position.copy(), action)
        labels = result.reward_vector

    reward_predictions = []
    for ag in agents_list:
        s = agent_controller.critic_view_controller.process_state(environment.get_state(), ag.position, ag)
        reward_predictions.append(float(ag.reward_network.get_value_function(s)))

    if not isinstance(environment, OrchardBasicNewDynamic):
        # Step env and labels
        result = environment.process_action(agent_idx, agents_list[agent_idx].position.copy(), action)
        labels = result.reward_vector

    # Tolerance-based correctness
    correct_predictions = [1 if abs(p - y) <= tol else 0 for p, y in zip(reward_predictions, labels)]

    # Update counters
    for ag, c in zip(agents_list, correct_predictions):
        ag.correct_predictions += c
        ag.total_predictions += 1
        # Predictions by reward
        if str(labels[ag.id]) in ag.correct_predictions_by_reward.keys():
            ag.correct_predictions_by_reward[str(labels[ag.id])] += c
            ag.total_predictions_by_reward[str(labels[ag.id])] += 1
        else:
            ag.correct_predictions_by_reward["other"] += c
            ag.total_predictions_by_reward["other"] += 1

    global same_cell_no_reward
    for agent in agents_list:
        if environment.apples[agent.position[0]][agent.position[1]] == 1 and labels[agent.id] == 0:
            same_cell_no_reward += 1

    global count
    count += 1

    if count == 10000 * len(agents_list):
        count = 0
        same_cell_no_reward = 0
        print(f"Total picked, env side: {environment.total_picked}")

    # if count % 1000 == 0:
    #     print(same_cell_no_reward)

    if isinstance(environment, OrchardBasicNewDynamic):
        environment.remove_apple(agents_list[agent_idx].position.copy())


def step_reward_learning_centralized(agents_list, environment, agent_controller, epsilon,
                                       inference=False, tol=1e-1):
    agent_idx = random.randint(0, environment.n - 1)
    action = agent_controller.agent_get_action(environment, agent_idx, None)

    if isinstance(environment, OrchardBasicNewDynamic):
        # Step env and labels
        result = environment.process_action(agent_idx, agents_list[agent_idx].position.copy(), action)
        labels = result.reward_vector

    s = agent_controller.critic_view_controller.process_state(environment.get_state(), None, None)
    reward_prediction = agents_list[0].reward_network.get_value_function(s)

    if not isinstance(environment, OrchardBasicNewDynamic):
        # Step env and labels
        result = environment.process_action(agent_idx, agents_list[agent_idx].position.copy(), action)
        labels = result.reward_vector

    # Tolerance-based correctness
    correct_prediction = 1 if abs(reward_prediction - np.sum(labels)) <= tol else 0
    agents_list[0].correct_predictions += correct_prediction
    agents_list[0].total_predictions += 1

    key = float(int(np.round(np.sum(labels))))
    agents_list[0].correct_predictions_by_reward[str(key)] += correct_prediction
    agents_list[0].total_predictions_by_reward[str(key)] += 1

    if (isinstance(environment, OrchardBasicNewDynamic)) and np.sum(result.reward_vector) > 0:
        environment.remove_apple(agents_list[agent_idx].position.copy())
