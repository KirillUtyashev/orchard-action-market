import numpy as np
from models.cnn import CNNDecentralized


def test_decentralized_cnn_input_channels():
    """
    Tests if the raw_state_to_nn_input method for the decentralized
    CNN correctly separates the 'self' agent, 'other' agents, and apples.
    """
    # 1. Arrange: Create a simple raw state
    H, W = 4, 4
    raw_state = {
        "agents": np.zeros((H, W), dtype=int),
        "apples": np.zeros((H, W), dtype=int),
    }
    raw_state["agents"][1, 1] = 1  # Agent 0
    raw_state["agents"][2, 2] = 1  # Agent 1
    raw_state["apples"][0, 3] = 1  # An apple

    # The network instance is only needed to call the method
    cnn = CNNDecentralized(height=H, width=W, alpha=0.001)

    # --- Test for Agent 0 ---
    # 2. Act: Process the state from the perspective of Agent 0
    agent_0_pos = np.array([1, 1])
    nn_input_0 = cnn.raw_state_to_nn_input(raw_state, agent_pos=agent_0_pos)

    # 3. Assert
    assert nn_input_0.shape == (3, H, W), "Output shape is incorrect for Agent 0"

    # Channel 0: Apples
    expected_apples = np.zeros((H, W))
    expected_apples[0, 3] = 1
    np.testing.assert_array_equal(
        nn_input_0[0], expected_apples, "Apple channel is incorrect for Agent 0"
    )

    # Channel 1: Other Agents (should only contain Agent 1)
    expected_others_0 = np.zeros((H, W))
    expected_others_0[2, 2] = 1
    np.testing.assert_array_equal(
        nn_input_0[1],
        expected_others_0,
        "Other agents channel is incorrect for Agent 0",
    )

    # Channel 2: Self Agent (should be a one-hot map of Agent 0's position)
    expected_self_0 = np.zeros((H, W))
    expected_self_0[1, 1] = 1
    np.testing.assert_array_equal(
        nn_input_0[2], expected_self_0, "Self agent channel is incorrect for Agent 0"
    )

    # --- Test for Agent 1 ---
    # 2. Act: Process the state from the perspective of Agent 1
    agent_1_pos = np.array([2, 2])
    nn_input_1 = cnn.raw_state_to_nn_input(raw_state, agent_pos=agent_1_pos)

    # 3. Assert
    assert nn_input_1.shape == (3, H, W), "Output shape is incorrect for Agent 1"

    # Channel 1: Other Agents (should only contain Agent 0)
    expected_others_1 = np.zeros((H, W))
    expected_others_1[1, 1] = 1
    np.testing.assert_array_equal(
        nn_input_1[1],
        expected_others_1,
        "Other agents channel is incorrect for Agent 1",
    )

    # Channel 2: Self Agent (should be a one-hot map of Agent 1's position)
    expected_self_1 = np.zeros((H, W))
    expected_self_1[2, 2] = 1
    np.testing.assert_array_equal(
        nn_input_1[2], expected_self_1, "Self agent channel is incorrect for Agent 1"
    )
