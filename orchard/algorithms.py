import math

import numpy as np
import random
# time_constant = 10 # number of timesteps before change
apples = 1

import numpy as np
import math


def distance(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2))

mean_distances = []


def get_nearest_agent_apple_distance(apple_pos, agents):
    min_distance = float('inf')
    for agent_ in agents:
        dist = distance(agent_.position, apple_pos)
        if dist < min_distance:
            min_distance = dist
    return min_distance


def _density(env):
    """Agent density r."""
    return env.n / (env.length * env.width)


def _lifetime_seconds(r, is_2d):
    """Deterministic T from the write-up."""
    if is_2d:
        return math.ceil(1 / (2 * math.sqrt(r))) + 1
    else:  # 1-D
        return 1 / (2 * r) + 1


def despawn_apple(env, q_despawn):
    """
    One 'second-boundary' update:
      • despawn apples with geom-hazard q = 1/T
      • spawn new apples in empty cells with p_cell = r · s_target
    Call exactly once after every n micro-ticks.
    """
    H, L = env.apples.shape            # ★ rows = width, cols = length

    # ---- despawn ----
    rand_mat = np.random.rand(H, L)    # ★ same shape as env.apples
    mask_apples = (env.apples != 0)
    removal_mask = mask_apples & (rand_mat < q_despawn)
    total_removed = removal_mask.sum()
    env.apples[removal_mask] -= 1
    return total_removed


def spawn_apple(env, p_cell):
    H, L = env.apples.shape  # ★ rows = width, cols = length

    rand_mat = np.random.rand(H, L)  # ★ new RNG draw, same shape
    spawn_mask = rand_mat < p_cell
    positions = np.argwhere(spawn_mask)
    total_spawned = spawn_mask.sum()
    if total_spawned > 0:
        for position in positions:
            mean_distances.append(get_nearest_agent_apple_distance(position, env.agents_list))
    env.apples[spawn_mask] += 1
    return total_spawned


def single_apple_spawn_new(env, timestep):
    time_constant = 3
    # # time_constant = int((env.length * env.width / 2) * math.sqrt(env.n / (env.length * env.width))) + 2
    #
    res = 0
    if timestep % time_constant == 0:
        position = np.random.randint(0, [env.width, env.length])
        mean_distances.append(get_nearest_agent_apple_distance(position, env.agents_list))
        env.apples[position[0], position[1]] += apples
        res += 1
    return res


def single_apple_despawn_new(env, timestep):
    # time_constant = int(env.length / 2) + 1
    time_constant = 3
    # time_constant = int((env.length * env.width / 2) * math.sqrt(env.n / (env.length * env.width))) + 2
    res = 0
    if timestep % time_constant == 0:
        if np.any(env.apples == 1):
            res = 1
        env.apples = np.zeros((env.width, env.length), dtype=int)
    return res


def single_apple_spawn(env):
    global time
    time += 1
    time_constant = int(env.length / 2) + 1
    if time % time_constant == 0:
        position = np.random.randint(0, [env.width, env.length])
        env.apples[position[0], position[1]] += apples
        return apples
    return 0


def single_apple_despawn(env):
    time_constant = int(env.length / 2) + 1
    res = 0
    if time % time_constant == time_constant - 1:
        if np.any(env.apples == 1):
            res = 1
        env.apples = np.zeros((env.width, env.length), dtype=int)
    return res


def spawn_apple_same_pos_once_every_4_steps(env):
    global time
    time += 1
    time_constant = int(env.length / 2) + 5
    if time % time_constant == 0:
        position = env.length - 1
        env.apples[position, 0] += apples
        return apples
    return 0

def find_farthest_position_1d(env):
    best_pos = 0
    farthest_dist = 0
    for i in range(env.length):
        dists = []
        for agent in env.agents_list:
            dists.append(np.abs(agent.position[0]-i))
        cur_min_dist = min(dists)
        if cur_min_dist > farthest_dist:
            best_pos = i
            farthest_dist = cur_min_dist
    return np.array([int(best_pos), 0])


def single_apple_spawn_malicious(env):
    global time
    time += 1
    time_constant = int(env.length / 2) + 1
    if time % time_constant == 1:
        position = find_farthest_position_1d(env)
        env.apples[position[0], position[1]] += apples
        return apples
    return 0
