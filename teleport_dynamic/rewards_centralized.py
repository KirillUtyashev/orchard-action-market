from tadd_helpers.env_functions import State


def get_reward_centralized(state: State, acting_agent_idx: int) -> float:
    """
    Returns 1.0 if the acting agent is on an apple. 0.0 otherwise.
    """
    pos = state.agent_position(acting_agent_idx)
    # Check apple map at agent's location
    if state.apples[pos[0], pos[1]] > 0:
        return 1.0
    return 0.0
