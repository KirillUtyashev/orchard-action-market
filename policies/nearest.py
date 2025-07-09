import numpy as np
import random

"""
The NEAREST baseline algorithm. This algorithm functions by looking at all apples in the field, and then choosing to move toward the apple that has the closest
euclidean distance.

Synonymous with the "Greedy" algorithm.
"""


def nearest_policy(state, agent_pos):
    """
    Nearest algorithm, but only working under the assumption that the orchard is a 1d field.
    :param state:
    :param agent_pos:
    :return:
    """
    apples = state["apples"]
    if not np.any(apples):
        return 2
    nearest = None
    min_dist = np.inf
    agent_row, agent_col = agent_pos  # assume a tuple (r, c)
    for (r, c), count in np.ndenumerate(apples):
        if count > 0:
            # Euclidean distance in 2D
            dist = np.hypot(r - agent_row, c - agent_col)
            if dist < min_dist:
                min_dist = dist
                nearest = np.array([r, c])
    if (nearest == agent_pos).all():  # don't move
        return 2
    elif agent_pos[1] < nearest[1]:
        return 1
    elif agent_pos[1] > nearest[1]:
        return 0
    elif agent_pos[0] > nearest[0]:
        return 3
    elif agent_pos[0] < nearest[0]:
        return 4
    else:
        return None


def tests():
    apples = np.array([1, 0, 0, 0, 0])
    agent_pos = 3
    state = {"apples": apples}
    assert nearest_policy(state, agent_pos) == 0

    apples = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    agent_pos = 6
    state = {"apples": apples}
    assert nearest_policy(state, agent_pos) == 1

    apples = np.array([0, 0, 1, 0, 1])
    agent_pos = 2
    state = {"apples": apples}
    assert nearest_policy(state, agent_pos) == 2

    apples = np.array([0, 0, 0, 0, 0])
    agent_pos = 2
    state = {"apples": apples}
    assert nearest_policy(state, agent_pos) == 2

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
