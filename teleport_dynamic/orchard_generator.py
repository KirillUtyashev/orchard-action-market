import numpy as np
from tadd_helpers.env_functions import State, init_empty_state


def init_fixed_apples_state(
    width: int, height: int, num_agents: int, num_apples: int, seed: int
) -> State:
    """
    Creates a State with fixed apples and random agents.
    This serves as the 'Base' state for the experiment.
    """
    rng = np.random.RandomState(seed)
    s = init_empty_state(height, width, num_agents)
    s.apples[:] = 0  # Clear random spawn

    possible_indices = np.arange(height * width)
    apple_indices = rng.choice(possible_indices, size=num_apples, replace=False)

    for idx in apple_indices:
        r, c = divmod(idx, width)
        s.apples[r, c] = 1

    return s


def teleport_agents_and_get_actor(state: State, rng: np.random.RandomState) -> int:
    """
    Modifies the state IN-PLACE by teleporting agents.
    Returns the acting_agent_idx.
    """
    H, W = state.H, state.L
    num_agents = len(state._agents)

    for i in range(num_agents):
        r = rng.randint(0, H)
        c = rng.randint(0, W)
        state.set_agent_position(i, np.array([r, c]))

    acting_idx = rng.randint(0, num_agents)
    return acting_idx
