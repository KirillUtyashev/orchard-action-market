import random
import numpy as np
import torch
from policies.random_policy import random_policy_1d


def convert_position(pos):
    temp = np.zeros((2, 1), dtype=int)
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

    a[agent_pos[0], agent_pos[1]] -= 1
    a = a.flatten()
    b = b.flatten()
    #print(list(a), list(b), agent_pos)
    left1, right1 = get_closest_left_right_1d(b, agent_pos[0])
    left2, right2 = get_closest_left_right_1d(a, agent_pos[0])
    arr = [left1, right1]
    arr1 = [left2, right2]
    #print(arr, arr1)
    return [np.array(arr), np.array(arr1)]


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


def get_empty_fields(env_length):
    return {
        "agents": np.zeros((env_length, 1), dtype=int),
        "apples": np.zeros((env_length, 1), dtype=int),
        "poses": np.zeros((env_length, 1), dtype=int)
    }


def generate_sample_states(env_length, num_agents):
    res = [get_empty_fields(env_length), get_empty_fields(env_length), get_empty_fields(env_length)]
    for i in range(len(res)):
        res[i]["agents"][i + 1] = np.array(num_agents)
        res[i]["poses"] = np.array([[i + 1, 0] for _ in range(num_agents)])
        res[i]["apples"][0] = np.array(1)

    return tuple(res)


def debug_before_loop(env, agents_list):
    print(env.get_state())
    for i in agents_list:
        print(i.position)


def ten(c, device):
    return torch.from_numpy(c).to(device).double()


def get_epsilon(step, total_steps,
                eps_start=1.0, eps_end=0.01, decay_frac=0.5):
    """
    Linearly decay ε from eps_start → eps_end over the first
    decay_frac*total_steps steps, then keep it at eps_end.
    """
    decay_steps = int(decay_frac * total_steps)
    if step >= decay_steps:
        return eps_end
    # fraction through the decay window [0..1]
    frac = step / decay_steps
    # linear interp: at frac=0 → eps_start; at frac=1 → eps_end
    res = eps_start + frac * (eps_end - eps_start)
    return res


def env_step(agents_list, env, step, timesteps, type_):
    # --- interact with the env and get one transition ---
    agent_idx = random.randint(0, env.n - 1)
    state = env.get_state()  # this is assumed to be a dict with "agents" and "apples"
    old_pos = agents_list[agent_idx].position
    if get_epsilon(step, timesteps) > random.random():
        action = random_policy_1d(state, agents_list[agent_idx].position)
    else:
        action = agents_list[agent_idx].get_action(state, agents_list=agents_list)
    reward, new_pos = env.main_step(agents_list[agent_idx].position.copy(), action)
    agents_list[agent_idx].position = new_pos.copy()

    # --- build your flat state vector ---
    state_vec = np.concatenate([state["agents"], state["apples"]], axis=0)
    new_state_vec = np.concatenate([env.get_state()["agents"], env.get_state()["apples"]], axis=0)
    if type_ == "C":
        return state_vec, new_state_vec, reward
    else:
        return state_vec, new_state_vec, reward, old_pos, agent_idx
