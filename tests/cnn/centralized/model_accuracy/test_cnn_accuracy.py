import torch
from models.reward_cnn import RewardCNNCentralized
from tests.cnn.centralized.generate_synthetic_states import (
    generate_synthetic_state_at_most_1_apples,
)
from train_scripts.train_centralized_cnn import MODEL_SAVE_PATH


def test_cnn_accuracy_simple():
    """Must run train_scripts/train_centralized_cnn.py first to generate the model file."""

    # --- Configuration (should match the trained model) ---
    width = 4
    height = 4
    learning_rate = 0.001  # Needed for initialization, but not used for training
    model_path = MODEL_SAVE_PATH  # The file to load

    num_test_episodes = 500  # Can use more test episodes now
    tol = 0.001
    num_agents = 2
    p = 0.5

    model = RewardCNNCentralized(width, height, learning_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Test Loop (no training here!) ---
    num_correct = 0
    for i in range(num_test_episodes):
        state, label = generate_synthetic_state_at_most_1_apples(
            num_agents=num_agents, width=width, height=height, p=p
        )
        # Get the raw float prediction
        prediction = model.get_model_reward_prediction_from_raw(state).item()

        # Check if the prediction is close to the true label
        if abs(prediction - label) < tol:
            num_correct += 1

    accuracy = num_correct / num_test_episodes
    print(f"Test Accuracy: {accuracy}")
    assert accuracy >= 0.95, f"Accuracy {accuracy} is less than the required 0.95"
