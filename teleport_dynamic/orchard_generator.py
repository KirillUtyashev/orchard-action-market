import numpy as np
from tadd_helpers.env_functions import State, init_empty_state


def init_fixed_apples(
    width: int, height: int, num_agents: int, num_apples: int
) -> State:
    """
    Creates a State with fixed apples.
    """
    # Note: Relies on np.random.seed() being set in the notebook
    s = init_empty_state(height, width, num_agents)
    s.apples[:] = 0

    possible_indices = np.arange(height * width)
    apple_indices = np.random.choice(possible_indices, size=num_apples, replace=False)

    for idx in apple_indices:
        r, c = divmod(idx, width)
        s.apples[r, c] = 1

    return s


def teleport_agent(state: State, acting_idx: int) -> None:
    """
    Teleports agent c to a random location.
    Modifies state IN-PLACE.
    """
    H, W = state.H, state.L

    # 2. Teleport
    r = np.random.randint(0, H)
    c = np.random.randint(0, W)
    state.set_agent_position(acting_idx, np.array([r, c]))
