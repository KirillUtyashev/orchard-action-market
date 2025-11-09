import pytest
import numpy as np
from orchard.environment import MoveAction
from value_function_learning.train_value_function_cnn import (
    DecentralizedValueFunctionCNNAlgorithm,
)
from agents.communicating_agent import CommAgent
from models.value_cnn import ValueCNNDecentralized
from policies.cnn_controllers import AgentControllerDecentralizedCNN


def test_decentralized_cnn_build_experiment(default_config):
    """
    Tests if the build_experiment method correctly initializes all components
    for the decentralized CNN algorithm.
    """
    # 1. Arrange
    algo = DecentralizedValueFunctionCNNAlgorithm(default_config)
    num_agents = default_config.train_config.num_agents

    # 2. Act
    algo.build_experiment()

    # 3. Assert
    # Check agents list
    assert len(algo.agents_list) == num_agents
    assert all(isinstance(agent, CommAgent) for agent in algo.agents_list)

    # Check networks (critically, they must be different instances)
    agent_networks = [agent.policy_value for agent in algo.agents_list]
    assert all(isinstance(net, ValueCNNDecentralized) for net in agent_networks)
    assert (
        len(set(id(net) for net in agent_networks)) == num_agents
    ), "Agents are sharing the same network instance!"

    # Check controller
    assert isinstance(algo.agent_controller, AgentControllerDecentralizedCNN)

    # Check environment
    assert algo.env is not None
    assert algo.env.n == num_agents


def test_decentralized_cnn_single_step_stores_correct_experience(default_config):
    """
    Verifies that after one agent takes one action, the correct experience
    (old_state, new_state, reward) is added to EACH agent's network buffer.
    This test is deterministic and checks the content of the buffers.
    """
    # 1. Arrange
    algo = DecentralizedValueFunctionCNNAlgorithm(default_config)
    algo.build_experiment()

    # --- Manually set a deterministic initial state ---
    # Place Agent 0 at [1,1] and Agent 1 at [3,3]
    # Place an apple at [1,2], in the path of Agent 0
    agent_0_pos = np.array([1, 1])
    agent_1_pos = np.array([3, 3])
    apple_pos = np.array([1, 2])

    algo.env.agents.fill(0)
    algo.env.apples.fill(0)

    algo.agents_list[0].position = agent_0_pos
    algo.agents_list[1].position = agent_1_pos
    algo.env.agents[agent_0_pos[0], agent_0_pos[1]] = 1
    algo.env.agents[agent_1_pos[0], agent_1_pos[1]] = 1
    algo.env.apples[apple_pos[0], apple_pos[1]] = 1

    # Mock the controller to force Agent 0 to move right (action=1)
    class MockController(AgentControllerDecentralizedCNN):
        def agent_get_action(self, env, agent_id, epsilon=None):
            return MoveAction.RIGHT.idx  # Always move RIGHT

    algo.agent_controller = MockController(algo.agents_list)

    # 2. Act
    # We call the core step function directly for a *single* tick, forcing agent 0 to act.
    # This bypasses the randomness of step_and_collect_observation.
    env_step_result = algo.single_agent_env_step(tick=0, agent_id=0)

    # Now, manually replicate the inner loop of `step_and_collect_observation`
    # to test the data processing logic in isolation.
    for i, agent in enumerate(algo.agents_list):
        assert isinstance(agent, CommAgent)
        assert isinstance(agent.policy_value, ValueCNNDecentralized)
        network: ValueCNNDecentralized = agent.policy_value
        reward = env_step_result.reward_vector[i]

        processed_state = network.raw_state_to_nn_input(
            env_step_result.old_state,
            agent_pos=env_step_result.old_positions[i],
        )
        processed_new_state = network.raw_state_to_nn_input(
            env_step_result.new_state, agent_pos=agent.position
        )
        network.add_experience(processed_state, processed_new_state, reward)

    # 3. Assert
    # --- Assert the environment transition was correct ---
    assert np.array_equal(algo.agents_list[0].position, [1, 2])  # Agent 0 moved right
    assert np.array_equal(algo.agents_list[1].position, [3, 3])  # Agent 1 did not move
    assert env_step_result.new_state["apples"][1, 2] == 0, "Apple was not picked"
    assert np.array_equal(env_step_result.reward_vector, [1.0, 0.0])

    # --- Assert the buffer content for EACH agent ---
    for i, agent in enumerate(algo.agents_list):
        assert isinstance(agent, CommAgent)
        assert isinstance(agent.policy_value, ValueCNNDecentralized)
        net = agent.policy_value
        assert len(net.batch_states) == 1, f"Agent {i} buffer should have 1 state"
        assert len(net.batch_rewards) == 1, f"Agent {i} buffer should have 1 reward"

        # Check the reward value
        expected_reward = 1.0 if i == 0 else 0.0
        assert (
            net.batch_rewards[0] == expected_reward
        ), f"Incorrect reward for Agent {i}"

        # Validate the content of the stored 'new_state' for Agent 0
        if i == 0:
            stored_new_state = net.batch_new_states[0]  # Shape (3, H, W)
            # Channel 0 (Apples): Should be all zero
            assert np.sum(stored_new_state[0]) == 0
            # Channel 1 (Others): Should contain Agent 1 at [3,3]
            expected_others = np.zeros_like(stored_new_state[1])
            expected_others[3, 3] = 1
            np.testing.assert_array_equal(stored_new_state[1], expected_others)
            # Channel 2 (Self): Should contain Agent 0 at its new position [1,2]
            expected_self = np.zeros_like(stored_new_state[2])
            expected_self[1, 2] = 1
            np.testing.assert_array_equal(stored_new_state[2], expected_self)


def test_training_step_triggers_training_after_full_batch(default_config):
    """
    Verifies that the agent networks are trained only after the batch is full.
    This test now correctly accounts for the loop in `step_and_collect_observation`.
    """
    # 1. Arrange
    # Batch size is 4. num_agents is 2.
    # After one call to training_step, buffers will have size 2.
    # After two calls, buffers will have size 4 and should train.
    default_config.train_config.batch_size = 4
    algo = DecentralizedValueFunctionCNNAlgorithm(default_config)
    algo.build_experiment()

    agent_0_net = algo.agents_list[0].policy_value

    # 2. Act (First training step)
    algo.training_step(step=0)

    # 3. Assert (After first step)
    # The `step_and_collect` method runs `num_agents` (2) times, so 2 experiences are added.
    assert len(agent_0_net.batch_states) == 2
    assert (
        len(agent_0_net.loss_history) == 0
    ), "Network should not have trained yet (batch not full)"

    # 2. Act (Second training step)
    algo.training_step(step=1)

    # 3. Assert (After second step)
    # Buffers now contain 2+2=4 experiences, so training should have occurred.
    assert (
        len(agent_0_net.loss_history) == 1
    ), "Network should have been trained exactly once"
    assert len(agent_0_net.batch_states) == 0, "Buffer should be cleared after training"


def test_training_step_triggers_training(default_config):
    """
    Verifies that agent networks are trained only after the batch is full.
    """
    # 1. Arrange
    # Use a batch size of 2*num_agents for this test
    default_config.train_config.batch_size = 4
    algo = DecentralizedValueFunctionCNNAlgorithm(default_config)
    algo.build_experiment()

    agent_0_net = algo.agents_list[0].policy_value

    # 2. Act (Step 1)
    algo.training_step(step=0)

    # 3. Assert (After Step 1)
    # Buffers should contain num_agents=2 experiences, but batch size is 4, so no training yet.
    assert len(agent_0_net.batch_states) == 2
    assert len(agent_0_net.loss_history) == 0, "Network should not have trained yet"

    # 2. Act (Step 2)
    algo.training_step(step=1)

    # 3. Assert (After Step 2)
    # Buffers are now full (2+2=4), so training should have occurred.
    assert len(agent_0_net.loss_history) == 1, "Network should have been trained once"
    assert len(agent_0_net.batch_states) == 0, "Buffer should be cleared after training"
