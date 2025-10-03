import pytest
import numpy as np
from models.reward_cnn import RewardCNNDecentralized


def test_cnn_decentralized_input_0_1():
    """Given a 4 by 4 raw state, test that cnn decentralized converts it correctly"""
    apples = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    agents = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    agent_pos = np.array([1, 2])
    raw_state = {"agents": agents, "apples": apples}

    model = RewardCNNDecentralized(4, 4, 0.0001)
    converted_state = model.raw_state_to_nn_input(raw_state, agent_pos=agent_pos)

    expected_state = np.array(
        [
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )

    assert converted_state.shape == expected_state.shape
    assert np.array_equal(converted_state, expected_state)


def test_cnn_centralized_input_0():
    """Given a 4 by 4 raw state, test that cnn centralized converts it correctly with all 0s"""
    apples = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    agents = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    raw_state = {"agents": agents, "apples": apples}
    agent_pos = np.array([0, 1])

    model = RewardCNNDecentralized(4, 4, 0.0001)
    converted_state = model.raw_state_to_nn_input(raw_state, agent_pos=agent_pos)

    expected_state = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )

    assert converted_state.shape == expected_state.shape
    assert np.array_equal(converted_state, expected_state)


def test_cnn_centralized_input_biggers_numbers1():
    """Here acting agent is part of the "big" group"""
    apples = np.array(
        [
            [0, 10, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    agents = np.array(
        [
            [0, 13, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    raw_state = {"agents": agents, "apples": apples}
    agent_pos = np.array([0, 1])
    model = RewardCNNDecentralized(4, 4, 0.0001)
    converted_state = model.raw_state_to_nn_input(raw_state, agent_pos=agent_pos)

    expected_state = np.array(
        [
            [
                [0, 10, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 12, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )

    assert converted_state.shape == expected_state.shape
    assert np.array_equal(converted_state, expected_state)


def test_cnn_centralized_input_biggers_numbers2():
    """Here acting agent is part of the "big" group"""
    apples = np.array(
        [
            [0, 10, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    agents = np.array(
        [
            [0, 13, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    agent_pos = np.array([1, 2])

    raw_state = {"agents": agents, "apples": apples}
    model = RewardCNNDecentralized(4, 4, 0.0001)
    converted_state = model.raw_state_to_nn_input(raw_state, agent_pos=agent_pos)

    expected_state = np.array(
        [
            [
                [0, 10, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 13, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )

    assert converted_state.shape == expected_state.shape
    assert np.array_equal(converted_state, expected_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
