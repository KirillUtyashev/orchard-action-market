# File: tests/test_orchard/test_algorithms.py

import numpy as np
from orchard.environment import OrchardBasic
from orchard.algorithms import spawn_apple

class MockAgent:
    def __init__(self, pos):
        self.position = np.array(pos)

def test_spawn_apple_avoids_agents():
    """
    Verify that the spawn_apple function respects the rule of not
    spawning apples in cells occupied by agents.
    """
    # 1. Arrange: Create a mock environment and place agents
    H, W = 5, 5
    agents_list = [MockAgent([1, 1]), MockAgent([3, 3])]
    
    # Manually create the agents map
    agents_map = np.zeros((H, W), dtype=int)
    agents_map[1, 1] = 1
    agents_map[3, 3] = 1

    # Mock the necessary parts of the Orchard environment class
    class MockEnv:
        def __init__(self):
            self.apples = np.zeros((H, W), dtype=int)
            self.agents = agents_map
            self.agents_list = agents_list

    env = MockEnv()
    
    # 2. Act: Call the spawn algorithm with a high probability (p=1.0)
    # This guarantees that an apple *should* try to spawn in every empty cell.
    spawn_apple(env, p_cell=1.0)

    # 3. Assert: Check the results
    # Assert that no apples were spawned where agents are located
    assert env.apples[1, 1] == 0, "Apple spawned on top of an agent at [1, 1]"
    assert env.apples[3, 3] == 0, "Apple spawned on top of an agent at [3, 3]"

    # Assert that apples were spawned in all other cells
    expected_apples = np.ones((H, W), dtype=int)
    expected_apples[1, 1] = 0
    expected_apples[3, 3] = 0
    
    np.testing.assert_array_equal(env.apples, expected_apples, "Apples map does not match expected output")