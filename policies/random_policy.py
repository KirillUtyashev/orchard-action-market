import numpy as np
import random

"""
The RANDOM baseline policy. Every action is totally random.
"""


def random_policy(available_actions):
    return random.randint(0, len(available_actions) - 1)
