import numpy as np
from typing import Callable, Dict
from tadd_helpers.env_functions import State, init_empty_state


def get_v_start_constant(
    num_agents: int,
    num_apples: int,
    grid_cells: int,
    reward_func: Callable,
    gamma: float,
) -> float:
    """
    Calculates V_start. Since the next state is random, this is a CONSTANT
    representing the discounted average value of the universe.
    Formula: V_start = (gamma * R_avg) / (1 - gamma^2)
    """
    if gamma == 0:
        return 0.0

    # Calculate R_avg (Expected reward of a single random jump)
    p_hit = num_apples / grid_cells

    # We probe the reward function to get values for "Self Hit" vs "Other Hit"
    # Create dummy state
    dummy_s = init_empty_state(1, 1, num_agents)  # Size irrelevant, just logic
    dummy_s.apples[0, 0] = 1

    # Case 1: Self Hits
    dummy_s.set_agent_position(0, np.array([0, 0]))
    r_self = reward_func(dummy_s, acting_agent_idx=0)[0]

    # Case 2: Other Hits (if exists)
    r_other = 0.0
    if num_agents > 1:
        dummy_s.set_agent_position(1, np.array([0, 0]))
        r_other = reward_func(dummy_s, acting_agent_idx=1)[0]

    # Weighted Average
    term_self = (1.0 / num_agents) * p_hit * r_self
    term_other = ((num_agents - 1.0) / num_agents) * p_hit * r_other
    r_avg = term_self + term_other

    return (gamma * r_avg) / (1.0 - gamma**2)


def get_v_mid_exact(
    state: State,
    acting_idx: int,
    self_idx: int,
    reward_func: Callable,
    v_start_constant: float,
) -> float:
    """
    Calculates V(S_mid).
    Formula: V(S_mid) = R_immediate(S_mid) + gamma * V_start
    """
    # 1. Get Immediate Reward
    rewards = reward_func(state, acting_idx)
    r_immediate = rewards[self_idx]

    # 2. Add Discounted Future Constant
    # Note: The transition is Mid -> Start -> Mid...
    # The immediate reward happens, then we are effectively at Start for the next cycle.
    # So we discount V_start once.
    # (In the derivation V_mid = R + gamma * V_start)

    # However, V_start already includes the 'gamma' from the Move phase.
    # Let's stick to the derivation: V_mid = R + gamma * V_start

    return (
        r_immediate + 0.99 * v_start_constant
    )  # Assuming gamma passed in context, using arg for safety
