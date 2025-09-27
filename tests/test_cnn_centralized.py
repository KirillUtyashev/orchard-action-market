import pytest
import numpy as np
from models.reward_cnn import RewardCNNCentralized


def test_cnn_centralized_input():
    """Given a 3 by 3 raw state, test that cnn centralized converts it correctly"""
    apples = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ]
    )
    agents = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    raw_state = {"agents": agents, "apples": apples}

    model = RewardCNNCentralized(3, 3, 0.0001)
    converted_state = model.raw_state_to_nn_input(raw_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
