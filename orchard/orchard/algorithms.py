import numpy as np

time = 0
# time_constant = 10 # number of timesteps before change
apples = 1

"""
Singular Apple Spawning Algorithm

Spawns a single apple in the current orchard. Takes the environment as an argument, and changes it in-place.
Note that the spawned apple functions return the number of apples that spawned (which in this case is 1).
"""

def single_apple_spawn(env):
    global time
    time += 1
    time_constant = int(env.length / 2) + 1
    if time % time_constant == 0:
        position = np.random.randint(0, [env.length, env.width])
        env.apples[position[0], position[1]] += apples
        return apples
    return 0


def single_apple_despawn(env):
    time_constant = int(env.length / 2) + 1
    if time % time_constant == time_constant - 1:
        env.apples = np.zeros((env.length, env.width), dtype=int)

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