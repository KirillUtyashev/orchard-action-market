import random

import numpy as np
import torch
from orchard.environment import Orchard, OrchardBasic, OrchardBasicNewDynamic
from policies.random_policy import random_policy


def create_env(env_config, num_agents, agent_pos, apples, agents_list, env_cls=OrchardBasic):
    env = env_cls(env_config.length, env_config.width, num_agents, agents_list, s_target=env_config.s_target, apple_mean_lifetime=env_config.apple_mean_lifetime)
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
    # if not alt_vision:
    res = [get_empty_fields(env_length, env_width), get_empty_fields(env_length, env_width), get_empty_fields(env_length, env_width)]
    for i in range(len(res)):
        res[i]["agents"][0][i + 1] = num_agents
        res[i]["poses"] = np.array([[0, i + 1] for _ in range(num_agents)])
        res[i]["apples"][0][0] = 1
    return tuple(res)


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


def step_reward_learning(agents_list, environment, agent_controller, epsilon,
                         inference=False, tol=1e-2):
    agent_idx = random.randint(0, environment.n - 1)
    action = random_policy(environment.available_actions)

    # Step env and labels
    result = environment.process_action(agent_idx, agents_list[agent_idx].position.copy(), action)
    labels = result.reward_vector

    reward_predictions = []
    for ag in agents_list:
        s = agent_controller.critic_view_controller.process_state(environment.get_state(), ag.position, ag)
        reward_predictions.append(float(ag.reward_network.get_value_function(s)))

    # Tolerance-based correctness
    correct_predictions = [1 if abs(p - y) <= tol else 0 for p, y in zip(reward_predictions, labels)]

    # Update counters
    for ag, c in zip(agents_list, correct_predictions):
        ag.correct_predictions += c
        ag.total_predictions += 1

    if (isinstance(environment, OrchardBasicNewDynamic)) and np.sum(result.reward_vector) > 0:
        environment.remove_apple(agents_list[agent_idx].position.copy())
