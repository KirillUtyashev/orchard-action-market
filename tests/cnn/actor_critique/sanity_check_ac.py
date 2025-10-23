# In actor_critic/sanity_check_ac_cnn.py

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from actor_critic.actor_critic_cnn import ActorCriticCNNAlgorithm
from configs.config import ExperimentConfig, EnvironmentConfig, TrainingConfig


def run_ac_cnn_sanity_check():
    """
    A minimal, deterministic test to verify that the Actor-Critic CNN algorithm
    is capable of learning a simple, single-agent task.
    """
    print("--- Running Sanity Check for Actor-Critic CNN ---")

    # --- Hyperparameters to Tune ---
    CRITIC_LR = 0.001
    ACTOR_LR = 0.0001
    # -----------------------------

    config = ExperimentConfig(
        env_config=EnvironmentConfig(
            length=5, width=5, s_target=0.0, apple_mean_lifetime=10000
        ),
        train_config=TrainingConfig(
            num_agents=1,  # CRITICAL: Only one agent for simplicity
            timesteps=3000,
            batch_size=32,
            alpha=CRITIC_LR,
            actor_alpha=ACTOR_LR,
            epsilon=0.1,  # Keep some exploration
            discount=0.99,
        ),
    )

    # 2. Initialize the Algorithm
    algo = ActorCriticCNNAlgorithm(config)
    algo.build_experiment()

    # 3. Manually Set the Deterministic State
    agent = algo.agents_list[0]
    env = algo.env
    start_pos = np.array([0, 0])
    apple_pos = np.array([0, 4])

    # 4. Run a Short Training Loop
    critic_loss_history = []
    actor_loss_history = []
    start_state_value_history = []  # Track the value of the starting state

    for step in range(config.train_config.timesteps):
        # Reset to the same state every time to focus the learning
        agent.position = start_pos
        env.agents.fill(0)
        env.agents[start_pos] = 1
        env.apples.fill(0)
        env.apples[apple_pos] = 1

        # Log the value of the starting state before the step
        critic_net = agent.policy_value
        start_state_processed = critic_net.raw_state_to_nn_input(
            env.get_state(), agent_pos=start_pos
        )
        start_state_value = critic_net.get_value_function(start_state_processed)
        start_state_value_history.append(start_state_value)

        # Use the algorithm's training step
        algo.training_step(step)

        # Collect losses for plotting
        if algo.critic_loss:
            critic_loss_history.extend(algo.critic_loss)
            algo.critic_loss.clear()
        if algo.actor_loss:
            actor_loss_history.extend(algo.actor_loss)
            algo.actor_loss.clear()

        if step % 500 == 0:
            print(f"Step {step}, Start State Value: {start_state_value:.4f}")

    # 5. Plot the Results
    print("--- Sanity Check Complete ---")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1.plot(critic_loss_history)
    ax1.set_title("Critic Loss (MSE)")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.set_yscale("log")

    ax2.plot(actor_loss_history)
    ax2.set_title("Actor Loss (Policy Gradient)")
    ax2.set_ylabel("Loss")
    ax2.grid(True)

    ax3.plot(start_state_value_history)
    ax3.set_title("Value of Starting State")
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Predicted Value")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_ac_cnn_sanity_check()
