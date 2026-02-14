import numpy as np
from typing import Callable, Dict
from tadd_helpers.env_functions import State, init_empty_state


def calculate_stream_values(rho: float, gamma: float, num_agents: int, p_hit: float):
    """
    Calculates v_RAND, v_OFF, v_ON for a given reward magnitude rho.
    Implements the algebraic solution from Section 1.5.
    """
    assert 0 <= gamma < 1, "Gamma must be in [0, 1)"

    # 1.5 Algebraic Solution for v_RAND
    # v_RAND = (p_hit * alpha * rho) / (1 - gamma)
    alpha = 1.0 / num_agents
    v_rand = (p_hit * alpha * rho) / (1.0 - gamma)

    # 1.3/1.5 Solve v_OFF
    # v_OFF = (gamma * alpha * v_RAND) / (1 - gamma * beta)
    beta = (num_agents - 1.0) / num_agents
    denom_beta = 1.0 - (gamma * beta)
    v_off = (gamma * alpha * v_rand) / denom_beta

    # 1.3/1.5 Solve v_ON
    # v_ON = (alpha * rho + gamma * alpha * v_RAND) / (1 - gamma * beta)
    v_on = (alpha * rho + gamma * alpha * v_rand) / denom_beta

    return v_rand, v_off, v_on


def get_exact_value(
    state: State,
    acting_agent_idx: int,
    self_agent_idx: int,
    reward_func: Callable[[State, int], Dict[int, float]],
    gamma: float,
) -> float:
    """
    Calculates Exact V(s) using the expansion from Section 1.6.
    V(s) = R_t + gamma * [ v_rand(actor) + sum(v_static(others)) ]

    Args:
        state (State): Current environment state.
        acting_agent_idx (int): Index of the agent that was teleported.
        self_agent_idx (int): Index of the "self" agent for whom we compute V(s).
        reward_func (Callable): Function to compute rewards given state and acting agent. First parameter is the state, second is acting_agent_idx.
        gamma (float): Discount factor.

    Returns:
        float: Exact value V(s) for the self agent.
    """
    # --- 1. Immediate Reward R_t ---
    rewards = reward_func(state, acting_agent_idx)
    r_immediate = rewards[self_agent_idx]

    # --- 2. Calculate Constants (Streams) ---
    num_agents = len(state._agents)
    total_cells = state.H * state.L
    p_hit = np.sum(state.apples) / total_cells

    # Probe Reward Func for Self (k=i) -> rho_self
    # We create a dummy state where self hits apple
    dummy: State = init_empty_state(1, 1, num_agents)
    dummy.apples[0, 0] = 1
    dummy.set_agent_position(self_agent_idx, np.array([0, 0]))
    rho_self = reward_func(dummy, self_agent_idx)[self_agent_idx]

    # Probe Reward Func for Other (k!=i) -> rho_other
    rho_other = 0.0
    if num_agents > 1:
        other_idx = (
            self_agent_idx + 1
        ) % num_agents  # just get an agent that is not self
        dummy.set_agent_position(other_idx, np.array([0, 0]))
        rho_other = reward_func(dummy, other_idx)[self_agent_idx]

    # Get Stream Components
    v_rand_s, v_off_s, v_on_s = calculate_stream_values(
        rho_self, gamma, num_agents, p_hit
    )
    v_rand_o, v_off_o, v_on_o = calculate_stream_values(
        rho_other, gamma, num_agents, p_hit
    )

    # --- 3. Sum Future Value E[V(s_next)] ---
    future_value = 0.0

    for k in range(num_agents):
        # Determine which set of constants to use (Is k Self or Other?)
        if k == self_agent_idx:
            c_rand, c_off, c_on = v_rand_s, v_off_s, v_on_s
        else:
            c_rand, c_off, c_on = v_rand_o, v_off_o, v_on_o

        if k == acting_agent_idx:
            # k is the Actor -> Becomes RAND
            future_value += c_rand
        else:
            # k is Static -> Check Position
            pos = state.agent_position(k)
            if state.apples[pos[0], pos[1]] > 0:
                future_value += c_on  # ON
            else:
                future_value += c_off  # OFF

    return r_immediate + gamma * future_value


def get_exact_value_centralized(
    state: State,
    acting_agent_idx: int,
    reward_func: Callable[[State, int], float],
    gamma: float,
) -> float:
    """
    Calculates Exact V(s) for the Centralized Team Reward case.
    Here, 'rho' is the same for everyone.
    """
    # 1. Immediate Reward
    r_immediate = reward_func(state, acting_agent_idx)

    # 2. Calculate Constants
    num_agents = len(state._agents)
    total_cells = state.H * state.L
    p_hit = np.sum(state.apples) / total_cells

    # Probe Reward Func for "Team Hit" (rho)
    dummy: State = init_empty_state(1, 1, num_agents)
    dummy.apples[0, 0] = 1
    dummy.set_agent_position(0, np.array([0, 0]))
    rho = reward_func(dummy, 0)

    # Calculate streams (everyone shares these)
    v_rand, v_off, v_on = calculate_stream_values(rho, gamma, num_agents, p_hit)

    # 3. Sum Future Value
    future_value = 0.0
    for k in range(num_agents):
        if k == acting_agent_idx:
            future_value += v_rand
        else:
            pos = state.agent_position(k)
            if state.apples[pos[0], pos[1]] > 0:
                future_value += v_on
            else:
                future_value += v_off

    return r_immediate + gamma * future_value
