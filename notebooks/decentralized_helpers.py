import random
import numpy as np
from pathlib import Path
import pickle
import sys
import os
from orchard.environment import calc_distance
import json


def pick_agent_uniformly(num):
    return random.randint(0, num - 1)


def get_agent_positions(agents_grid, picker_pos, index):
    """Returns a list of agent positions ensuring that index index is the index of the picker.

    Args:
        agents_grid: grid of agents
        picker_pos: position of the picker agent
        index: target index for the picker agent

    Returns:
        list of agent positions where index 0 is the position of the picker agent
    """
    positions = np.argwhere(agents_grid != 0).tolist()
    picker_pos = list(picker_pos)

    if picker_pos in positions:
        i = positions.index(picker_pos)
        # only swap if not already at the target index
        if i != index and index < len(positions):
            positions[i], positions[index] = positions[index], positions[i]
    else:
        # this should not happen
        raise Exception("Picker position not in agents grid")
        if index < len(positions):
            positions[index] = picker_pos
        else:
            positions.append(picker_pos)

    return np.array(positions)


def generate_reward_vector(picker_id, positions_list, apple_pos, picked):
    res = np.zeros(len(positions_list))
    if not picked:
        return res
    else:
        for agent_num in range(len(positions_list)):
            res[agent_num] = calc_distance(positions_list[agent_num], apple_pos)
        res = res / np.sum(res)
        res = 2 * res
        res[picker_id] = -1
        return res
