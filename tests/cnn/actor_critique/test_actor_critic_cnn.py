# In tests/actor_critic/test_actor_critic_cnn.py

import pytest
import numpy as np
import torch

from models.actor_cnn import ActorCNN
from policies.cnn_controllers import AgentControllerActorCriticCNN
from actor_critic.actor_critic_cnn import ActorCriticCNNAlgorithm
from configs.config import ExperimentConfig, EnvironmentConfig, TrainingConfig
from orchard.environment import Action2D


@pytest.fixture
def ac_cnn_config():
    """Provides a default config for Actor-Critic CNN tests."""
    return ExperimentConfig(
        env_config=EnvironmentConfig(
            length=5, width=5, env_cls="OrchardEuclideanNegativeRewards"
        ),
        train_config=TrainingConfig(num_agents=2, batch_size=2),
    )


def test_actor_cnn_output_shape(ac_cnn_config):
    """Tests if the ActorCNN outputs a correctly shaped probability distribution."""
    # Arrange
    cfg = ac_cnn_config
    num_actions = 5  # For a 2D environment
    actor_net = ActorCNN(
        height=cfg.env_config.width,
        width=cfg.env_config.length,
        num_actions=num_actions,
        alpha=0.001,
        mlp_hidden_features=16,
        mlp_hidden_layers=1,
    )
    # A dummy processed state (Channels, Height, Width)
    dummy_state = np.random.rand(3, cfg.env_config.width, cfg.env_config.length).astype(
        np.float32
    )

    # Act
    probs = actor_net.get_action_probabilities(dummy_state)

    # Assert
    assert probs.shape == (
        num_actions,
    ), f"Output shape should be ({num_actions},), but was {probs.shape}"
    assert np.isclose(np.sum(probs), 1.0), "Probabilities should sum to 1"


def test_agent_controller_actor_critic_cnn(ac_cnn_config):
    """Tests if the controller returns a valid action index."""
    # Arrange
    algo = ActorCriticCNNAlgorithm(ac_cnn_config)
    algo.build_experiment()  # This creates the agents and controller
    controller = algo.agent_controller
    env = algo.env
    agent_id = 0

    # Act
    action_idx = controller.agent_get_action(env, agent_id)

    # Assert
    assert isinstance(action_idx, int)
    assert 0 <= action_idx < len(env.available_actions)


def test_ac_cnn_step_and_collect_observation(ac_cnn_config):
    """
    CRITICAL TEST: Verifies that after one step, the correct experiences
    (including advantage) are added to the correct buffers.
    """
    # Arrange
    algo = ActorCriticCNNAlgorithm(ac_cnn_config)
    algo.build_experiment()

    # --- Create a deterministic scenario ---
    # Agent 0 will move RIGHT onto an apple
    acting_agent_id = 0
    other_agent_id = 1

    # Set agent positions
    algo.agents_list[acting_agent_id].position = np.array([2, 1])
    algo.agents_list[other_agent_id].position = np.array([4, 4])

    # Clear and set the environment grids
    algo.env.agents.fill(0)
    algo.env.apples.fill(0)
    algo.env.agents[2, 1] = 1
    algo.env.agents[4, 4] = 1
    algo.env.apples[2, 2] = 1  # Apple is to the right of agent 0

    # Mock the controller to force the action
    algo.agent_controller.agent_get_action = (
        lambda env, agent_id, epsilon: Action2D.RIGHT.idx
    )

    # 1. Simulate a single agent step
    env_step_result = algo.single_agent_env_step(tick=0, agent_id=acting_agent_id)

    # 2. Manually run the experience collection logic from your algorithm
    acting_agent = algo._agents_list[acting_agent_id]
    reward = env_step_result.reward_vector[acting_agent_id]

    critic_net = acting_agent.policy_value
    actor_net = acting_agent.policy_network

    processed_old_state_critic = critic_net.raw_state_to_nn_input(
        env_step_result.old_state,
        agent_pos=env_step_result.old_positions[acting_agent_id],
    )
    processed_new_state_critic = critic_net.raw_state_to_nn_input(
        env_step_result.new_state, agent_pos=acting_agent.position
    )

    with torch.no_grad():
        v_old = critic_net.get_value_function(processed_old_state_critic)
        v_new = critic_net.get_value_function(processed_new_state_critic)

    advantage = reward + algo.train_config.discount * v_new - v_old

    # Add experiences just as the algorithm does
    for i, agent in enumerate(algo._agents_list):
        r_i = env_step_result.reward_vector[i]
        p_old = agent.policy_value.raw_state_to_nn_input(
            env_step_result.old_state, agent_pos=env_step_result.old_positions[i]
        )
        p_new = agent.policy_value.raw_state_to_nn_input(
            env_step_result.new_state, agent_pos=agent.position
        )
        agent.policy_value.add_experience(p_old, p_new, r_i)

    actor_processed_state = actor_net.raw_state_to_nn_input(
        env_step_result.old_state,
        agent_pos=env_step_result.old_positions[acting_agent_id],
    )
    actor_net.add_experience(actor_processed_state, env_step_result.action, advantage)

    # Assert
    acting_agent = algo.agents_list[acting_agent_id]
    other_agent = algo.agents_list[other_agent_id]

    # --- 1. Check the Critic Buffers ---
    # Both critics should have one experience.
    assert len(acting_agent.policy_value.batch_states) == 1
    assert len(other_agent.policy_value.batch_states) == 1

    # Check rewards for the EuclideanNegativeRewards env
    # Reward for picker is -1. Reward for other is 2 * (dist_other / (dist_picker + dist_other)).
    # Here, dist_picker is 0, dist_other is sqrt((4-2)^2 + (4-2)^2) = sqrt(8)
    # So, other_reward is 2. The vector should be [-1, 2].
    assert acting_agent.policy_value.batch_rewards[0] == -1.0
    assert np.isclose(other_agent.policy_value.batch_rewards[0], 2.0)

    # --- 2. Check the Actor Buffers ---
    # ONLY the acting agent's actor should have an experience.
    assert len(acting_agent.policy_network.batch_states) == 1
    assert len(other_agent.policy_network.batch_states) == 0

    # Check the content for the acting agent's actor
    assert acting_agent.policy_network.batch_actions[0] == Action2D.RIGHT.idx

    # Check that the advantage was calculated and is a float
    advantage = acting_agent.policy_network.batch_advantages[0]
    assert isinstance(advantage, float)
