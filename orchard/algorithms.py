import numpy as np

time = 0
time_constant = 20 # number of timesteps before change
apples = 1

def single_apple_spawn(env):
    global time
    time += 1
    if time % time_constant == 1:
        position = np.random.randint(0, [env.length, env.width])
        env.apples[position[0], position[1]] += apples
        return apples
    return 0


def single_apple_despawn(env):
    if time % time_constant == 0: #or time % time_constant == 0:
        env.apples = np.zeros((env.length, env.width), dtype=int)
