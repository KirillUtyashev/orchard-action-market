import math
import random
import numpy as np
import torch

from config import get_config

def convert_position(pos):
    if pos is not None:
        temp = np.zeros((len(pos), 1), dtype=int)
        for i in range(len(pos)):
            temp[i] = np.array(pos[i])
        return temp


def get_closest_left_right_1d(mat, agent_pos):
    mat = list(mat)
    left = -1
    right = -1
    pos, count = agent_pos, mat[agent_pos]
    while pos > -1:
        if count > 0:
            left = agent_pos - pos
            break
        else:
            pos -= 1
            count = mat[pos]
    pos, count = agent_pos, mat[agent_pos]
    while pos < len(mat):
        if count > 0:
            right = pos - agent_pos
            break
        else:
            pos += 1
            if pos >= len(mat):
                break
            count = mat[pos]
    return left, right


def unwrap_state(state):
    return state["agents"].copy(), state["apples"].copy()


def convert_input(state, agent_pos):
    a, b = unwrap_state(state)
    #print(list(a.flatten()), list(b.flatten()), agent_pos)
    fill = np.array([[-1] for _ in range(get_config()["vision"])])
    ap = np.concatenate((fill, a, fill))
    bp = np.concatenate((fill, b, fill))

    leftmost = agent_pos[0] - (get_config()["vision"] // 2)
    rightmost = agent_pos[0] + (get_config()["vision"] // 2)
    true_a = ap[leftmost + get_config()["vision"]: rightmost + 1 + get_config()["vision"]]
    true_b = bp[leftmost + get_config()["vision"]: rightmost + 1 + get_config()["vision"]]

    return {"agents": true_a, "apples": true_b}


# def convert_input(a, b, agent_pos):
#     #print(list(a.flatten()), list(b.flatten()), agent_pos)
#
#     a[agent_pos[0], agent_pos[1]] -= 1
#     a = a.flatten()
#     b = b.flatten()
#     #print(list(a), list(b), agent_pos)
#     left1, right1 = get_closest_left_right_1d(b, agent_pos[0])
#     left2, right2 = get_closest_left_right_1d(a, agent_pos[0])
#     arr = [left1, right1]
#     arr1 = [left2, right2]
#     #print(arr, arr1)
#     return [np.array(arr), np.array(arr1)]

def get_possible_states(state, agent_pos):
    apples = state["apples"].copy()
    agents = state["agents"].copy()
    length = agents.size

    apples1 = apples.copy()
    agents1 = agents.copy()
    new_pos1 = [np.clip(agent_pos[0] - 1, 0, length-1), agent_pos[1]]
    agents1[agent_pos[0], agent_pos[1]] -= 1
    agents1[new_pos1[0], new_pos1[1]] += 1
    if apples1[new_pos1[0], new_pos1[1]] > 0:
        apples1[new_pos1[0], new_pos1[1]] -= 1
    state1 = {
        "apples": apples1,
        "agents": agents1
    }

    apples2 = apples.copy()
    agents2 = agents.copy()
    new_pos2 = [np.clip(agent_pos[0] + 1, 0, length - 1), agent_pos[1]]
    agents2[agent_pos[0], agent_pos[1]] -= 1
    agents2[new_pos2[0], new_pos2[1]] += 1
    if apples2[new_pos2[0], new_pos2[1]] > 0:
        apples2[new_pos2[0], new_pos2[1]] -= 1
    state2 = {
        "apples": apples2,
        "agents": agents2
    }

    if apples[agent_pos[0], agent_pos[1]] > 0:
        apples[agent_pos[0], agent_pos[1]] -= 1

    state3 = {
        "apples": apples,
        "agents": agents
    }
    return state1, state2, state3


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


def debug_before_loop(env, agents_list):
    print(env.get_state())
    for i in agents_list:
        print(i.position)


def ten(c, device):
    return torch.from_numpy(c).to(device).double()


def get_epsilon(step, total_steps,
                eps_start=1.0, eps_end=0.01, decay_frac=0.2):
    """
    ε = 1.0                     for step <= warmup_steps
      linearly from 1.0 → 0.01 for warmup_steps < step <= warmup_steps + decay_steps
      = 0.01                    thereafter

    where decay_steps = int(decay_frac * total_steps)
    """
    # 1) Pure exploration phase
    warmup_steps = 0.4 * total_steps
    if step <= warmup_steps:
        return eps_start
    # 2) Decay phase
    decay_steps = int(decay_frac * total_steps)
    step_in_decay = step - warmup_steps
    if step_in_decay < decay_steps:
        frac = step_in_decay / decay_steps
        return eps_start + frac * (eps_end - eps_start)

    # 3) Final phase
    return eps_end


def staged_linear_tau(step, total_steps,
                      initial_tau=2.0, final_tau=0.5):
    """
    Keeps tau = initial_tau for the first 20% of total_steps,
    then linearly decays from initial_tau → final_tau over the
    next 40% of total_steps,
    then holds tau = final_tau for the remaining 40%.

    Args:
        step (int): current timestep (0-based or 1-based).
        total_steps (int): total timesteps planned.
        initial_tau (float): starting temperature.
        final_tau (float): ending temperature.
    Returns:
        float: current temperature tau.
    """
    # compute breakpoints
    t0 = 0.2 * total_steps          # end of “hold initial” phase
    t1 = t0 + 0.4 * total_steps     # end of “decay” phase

    if step <= t0:
        return initial_tau
    elif step <= t1:
        # fraction of decay completed
        frac = (step - t0) / (t1 - t0)
        # linear interpolate
        return initial_tau + frac * (final_tau - initial_tau)
    else:
        return final_tau


def env_step(agents_list, env, step, timesteps, type_):
    # --- interact with the env and get one transition ---
    agent_idx = random.randint(0, env.n - 1)
    positions = []
    for i in range(len(agents_list)):
        positions.append(agents_list[i].position)
    state = env.get_state()  # this is assumed to be a dict with "agents" and "apples"
    old_pos = agents_list[agent_idx].position
    if type_ == "C" or type_ == "DC":
        action = random_policy_1d(state, agents_list[agent_idx].position)
    else:
        action = agents_list[agent_idx].get_action(state, agents_list)
    reward, new_pos = env.main_step(agents_list[agent_idx].position.copy(), action)
    agents_list[agent_idx].position = new_pos.copy()
    if type_ == "C":
        return state, env.get_state(), reward
    elif type_ == "DC":
        return state, env.get_state(), reward, old_pos, agent_idx
    else:
        return state, env.get_state(), reward, agent_idx, positions, action


def get_discounted_value(old, new, discount_factor=0.05):
    return old * (1 - discount_factor) + new * discount_factor
