import sys
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Add the root directory to the Python path to find your modules

from models.reward_cnn import RewardCNNCentralized
from tests.cnn.centralized.generate_synthetic_states import (
    generate_synthetic_state_at_most_1_apples,
)
from config import MODEL_DIR, GRAPHS_DIR

NUM_TRAIN_EPISODES = 1000
BATCH_SIZE = 128
WIDTH = 4
HEIGHT = 4
NUM_AGENTS = 2
P = 0.9
LEARNING_RATES = [0.01]  # Try multiple learning rates
MODEL_SAVE_PATH = MODEL_DIR / "centralized_cnn.pth"  # The output file


def train_and_save_model(lr):
    model = RewardCNNCentralized(WIDTH, HEIGHT, lr)
    # --- Training Loop ---
    for i in tqdm(range(NUM_TRAIN_EPISODES), desc="Training"):
        for _ in range(BATCH_SIZE):
            state, label = generate_synthetic_state_at_most_1_apples(
                num_agents=NUM_AGENTS, width=WIDTH, height=HEIGHT, p=P
            )
            model.add_experience_from_raw(state, label)
        loss = model.train_batch()
    print(f"Final loss after training: {model.loss_history[-1]}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


def find_lr():
    """Trains the CNN on synthetic data and saves the model."""
    print("Starting training...")
    for lr in LEARNING_RATES:
        model = RewardCNNCentralized(WIDTH, HEIGHT, lr)

        # --- Training Loop ---
        for i in tqdm(range(NUM_TRAIN_EPISODES), desc="Training"):
            for _ in range(BATCH_SIZE):
                state, label = generate_synthetic_state_at_most_1_apples(
                    num_agents=NUM_AGENTS, width=WIDTH, height=HEIGHT, p=P
                )
                model.add_experience_from_raw(state, label)
            loss = model.train_batch()

        loss_history = model.loss_history
        PLOT_SAVE_PATH = GRAPHS_DIR / f"centralized_cnn_loss_{lr}lr.png"

        # 2. Create and save the plot
        plt.figure(figsize=(12, 6))
        plt.plot(loss_history)
        plt.title(f"Training Loss (LR={lr})")
        plt.xlabel("Training Batch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.savefig(PLOT_SAVE_PATH)
        print(f"Loss plot saved to {PLOT_SAVE_PATH}")
        print(f"final loss: {loss_history[-1]}")


def main():
    # us lr=0.01
    # find_lr()
    train_and_save_model(0.01)


if __name__ == "__main__":
    main()
