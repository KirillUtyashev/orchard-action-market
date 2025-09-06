import numpy as np
import torch
from orchard.environment import OrchardBasic


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



