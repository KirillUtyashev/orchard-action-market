from orchard.environment import Orchard
import numpy as np
"""
Additional algorithms for the NEAREST-UNIFORM baseline policy. This uses the NEAREST policy, but uses replace_agents_1d to send the agents back to their "optimal" position periodically.
Expected performance is above Nearest.

Sample execution: run_environment_1d(num_agents, nearest, length, S, phi, "Nearest-Uniform", experiment, time, action_algo=replace_agents_1d)
"""

time = 0


def calculate_time_1d(length, num_agents):
    """
    Calculate how long to allow before resetting. Should be max distance agent needs to go multiplied by num agents.

    :param length: length of 1d orchard.
    :param num_agents:
    :return:
    """
    max_distance = np.ceil(length / (2 * num_agents))

    return int(2 * max_distance * num_agents)

def calc_positions_1d(length, num_agents):
    max_distance = np.ceil(length / (2 * num_agents))

    poses = []
    marker = max_distance
    for i in range(num_agents):
        if marker >= length:
            marker = length-1
        poses.append([int(marker), 0])
        marker += 2 * max_distance

    return poses


def replace_agents_1d(env: Orchard, position, action):
    global time
    time += 1

    max_time = calculate_time_1d(env.length, env.n)
    reward, new_pos = env.process_action(env, position, action)
    if time % max_time == 0:
        env.set_positions(calc_positions_1d(env.length, env.n))

    return reward, new_pos
