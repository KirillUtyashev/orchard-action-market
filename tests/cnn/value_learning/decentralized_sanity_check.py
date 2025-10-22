# File: tests/sanity_check_decentralized.py
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
import torch
import numpy as np
import matplotlib.pyplot as plt
from configs.config import ExperimentConfig, EnvironmentConfig, TrainingConfig
from value_function_learning.train_value_function_cnn import (
    DecentralizedValueFunctionCNNAlgorithm,
)


def run_sanity_check():
    """
    A minimal, deterministic test to verify that the decentralized CNN learning
    algorithm is capable of learning a simple task.
    """
    print("--- Running Sanity Check for Decentralized CNN ---")

    # 1. --- Configure a Trivial Problem ---
    # We use the full ExperimentConfig to ensure the algorithm initializes correctly.
    config = ExperimentConfig(
        env_config=EnvironmentConfig(
            length=5,
            width=5,
            s_target=0.0,  # No random apple spawning
            apple_mean_lifetime=10000,  # Apples don't despawn
        ),
        train_config=TrainingConfig(
            num_agents=1,  # CRITICAL: Only one agent
            timesteps=1000,  # Short run
            batch_size=4,  # Train frequently
            alpha=0.001,  # A reasonably fast learning rate for a simple problem
            epsilon=0.1,  # Some exploration is still needed
            discount=0.99,
        ),
    )

    # 2. --- Initialize the Algorithm ---
    algo = DecentralizedValueFunctionCNNAlgorithm(config)
    algo.build_experiment()

    # 3. --- Manually Set the Deterministic State ---
    agent = algo.agents_list[0]
    env = algo.env

    agent.position = np.array([0, 0])
    env.agents.fill(0)
    env.agents[0, 0] = 1

    env.apples.fill(0)
    env.apples[0, 4] = 1  # Apple at the far right

    print(f"Setup: Agent at {agent.position}, Apple at [0, 4]")

    # 4. --- Run a Short Training Loop ---
    loss_history = []
    for step in range(config.train_config.timesteps):
        # We don't use the full `training_step` because we don't want
        # random spawning/despawning. We just want to collect experience and train.
        algo.step_and_collect_observation(step)

        network = agent.policy_value
        if len(network.batch_states) >= config.train_config.batch_size:
            loss = network.train_batch()
            if loss is not None:
                loss_history.append(loss)
                if len(loss_history) % 10 == 0:
                    print(f"Step {step}, Training Batch... Loss: {loss:.6f}")

    # 5. --- Plot the Results ---
    print("--- Sanity Check Complete ---")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title("Sanity Check: Critic Loss During Training")
    plt.xlabel("Training Batch Number")
    plt.ylabel("MSE Loss")
    plt.yscale("log")  # Log scale is best for viewing loss
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_sanity_check()
