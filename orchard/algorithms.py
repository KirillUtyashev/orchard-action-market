import math

import numpy as np
import random

time = 0
# time_constant = 10 # number of timesteps before change
apples = 1

Q = 0.1

"""
Singular Apple Spawning Algorithm

Spawns a single apple in the current orchard. Takes the environment as an argument, and changes it in-place.
Note that the spawned apple functions return the number of apples that spawned (which in this case is 1).
"""


def apple_despawn(env, apple_ages):
    length, width = env.length, env.width
    r = env.n / (length * width)
    total_removed = 0

    probability_spawn = Q * r * 4

    for i in range(width):
        for j in range(length):
            if env.apples[i, j] > 0:
                if random.random() < probability_spawn:
                    env.apples[i, j] -= 1
                    total_removed += 1

    # for i in range(width):
    #     for j in range(length):
    #         survivors = []
    #         for age in apple_ages[i][j]:
    #             # compute per-apple death probability
    #             prob = 1 - math.exp(-probability_spawn * age)
    #             if random.random() < prob:
    #                 # apple dies
    #                 env.apples[i, j] -= 1
    #                 total_removed += 1
    #             else:
    #                 # bump age and keep it
    #                 survivors.append(age + 1)
    #         # replace with only the survivors
    #         apple_ages[i][j] = survivors

    return total_removed


def apple_spawn(env, apple_ages):
    length, width = env.length, env.width

    r = env.n / (env.length * env.width)

    prob = Q * r

    sum_ = 0
    for i in range(width):
        for j in range(length):
            if random.random() < prob:
                env.apples[i, j] += 1
                apple_ages[i][j].append(0)
                sum_ += 1
    return sum_


def single_apple_spawn(env, apple_ages):
    global time
    time += 1
    time_constant = int(env.length / 2) + 1
    if time % time_constant == 0:
        position = np.random.randint(0, [env.width, env.length])
        env.apples[position[0], position[1]] += apples
        return apples
    return 0


def single_apple_despawn(env, apple_ages):
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
