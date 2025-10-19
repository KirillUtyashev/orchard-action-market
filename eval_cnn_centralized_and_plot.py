import torch
from models.cnn import CNNCentralized
from tests.cnn.centralized.generate_synthetic_states import (
    generate_synthetic_state_at_most_1_apples,
)
from train_scripts.train_centralized_cnn import MODEL_SAVE_PATH
import matplotlib.pyplot as plt
from config import FINAL_DIR


def plot_cnn_accuracy_simple():
    """Must run train_scripts/train_centralized_cnn.py first to generate the model file."""

    # --- Configuration (should match the trained model) ---
    width = 4
    height = 4
    learning_rate = 0.001  # Needed for initialization, but not used for training
    model_path = MODEL_SAVE_PATH  # The file to load

    num_test_episodes = 500  # Can use more test episodes now
    tols = [0.1, 0.01, 0.001, 0.0001]
    num_agents = 2
    p = 0.9

    model = CNNCentralized(width, height, learning_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    accuracies = []
    for tol in tols:
        # --- Test Loop (no training here!) ---
        num_correct = 0
        for i in range(num_test_episodes):
            state, label = generate_synthetic_state_at_most_1_apples(
                num_agents=num_agents, width=width, height=height, p=p
            )
            # Get the raw float prediction
            prediction = model.get_model_reward_prediction_from_raw(state).item()
            error = abs(prediction - label)
            if error < tol:
                num_correct += 1
        accuracy = num_correct / num_test_episodes
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(tols, accuracies, marker="o")
    plt.xscale("log")
    plt.xlabel("Tolerance")
    plt.ylabel("Accuracy")
    plt.title("CNN Centralized Model Accuracy vs. Tolerance")
    plt.grid(True)
    plt.savefig(f"{FINAL_DIR}/cnn_centralized_accuracy_vs_tolerance_{p}p.png")


plot_cnn_accuracy_simple()
