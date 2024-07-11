import numpy as np
import random

action_vectors_1d = [
            -1,
            1,
            0
        ]

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

def nearest(state, agent_pos):
    apples = state["apples"]
    agent_pos = np.array(agent_pos)
    target = (-1, -1)
    smallest_distance = np.linalg.norm(apples.size)
    for point, count in np.ndenumerate(apples):
        if np.linalg.norm(point - agent_pos) < smallest_distance and count > 0:
            target = point
            smallest_distance = np.linalg.norm(point - agent_pos)

    if np.array_equal(target, agent_pos) or target[0] == -1:
        return 4 # no movement
    elif target[0] > agent_pos[0]: # move up / down first before moving left / right
        return 1
    elif target[0] < agent_pos[0]:
        return 0
    elif target[1] > agent_pos[1]:
        return 2
    else:
        return 3

def nearest_1d(state, agent_pos):
    """
    Nearest algorithm, but only working under the assumption that the orchard is a 1d field.
    :param state:
    :param agent_pos:
    :return:
    """
    apples = state["apples"]

    target = -1 # we use -1 here to represent 'no apple found'
    smallest_distance = apples.size
    for point, count in np.ndenumerate(apples):
        point = point[0]
        if np.abs(point - agent_pos) < smallest_distance and count > 0:
            target = point
            smallest_distance = np.abs(point - agent_pos)

    if target == agent_pos or target == -1: # don't move
        return 2
    elif target > agent_pos:
        return 1 # move right
    elif target < agent_pos:
        return 0 # move left


def tests():
    apples = np.array([1, 0, 0, 0, 0])
    agent_pos = 3
    state = {"apples": apples}
    assert nearest_1d(state, agent_pos) == 0

    apples = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    agent_pos = 6
    state = {"apples": apples}
    assert nearest_1d(state, agent_pos) == 1

    apples = np.array([0, 0, 1, 0, 1])
    agent_pos = 2
    state = {"apples": apples}
    assert nearest_1d(state, agent_pos) == 2

    apples = np.array([0, 0, 0, 0, 0])
    agent_pos = 2
    state = {"apples": apples}
    assert nearest_1d(state, agent_pos) == 2

    apples = np.array([[0, 1], [1, 0]])
    agent_pos = [0, 1]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 4

    apples = np.array([[0, 0],
                       [0, 1]])
    agent_pos = [0, 0]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 1

    apples = np.array([[0, 0],
                       [0, 1]])
    agent_pos = [1, 0]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 2

    apples = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]])
    agent_pos = [2, 0]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 0

    apples = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]])
    agent_pos = [1, 1]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 0

    apples = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [1, 0, 1]])
    agent_pos = [0, 1]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 2

    apples = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 0]])
    agent_pos = [0, 1]
    state = {"apples": apples}
    assert nearest(state, agent_pos) == 2

tests()