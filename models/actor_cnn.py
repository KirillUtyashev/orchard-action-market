# In a new file: models/actor_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import DEVICE
from models.cnn import CNNDecentralized  # Assuming a decentralized actor
from utils import ten, ten_float


class ActorCNN(CNNDecentralized):
    """A decentralized CNN for learning a policy (Actor)."""

    def __init__(
        self,
        height: int,
        width: int,
        num_actions: int,
        alpha: float,
        mlp_hidden_features: int,
        mlp_hidden_layers: int,
    ):
        # Initialize the parent CNN which sets up the convolutional layers
        super().__init__(height, width, alpha, mlp_hidden_features, mlp_hidden_layers)

        # --- OVERRIDE the final MLP layer ---
        # The parent CNN's mlp_head ends in a single output neuron for value prediction.
        # We need to replace it to output logits for each possible action.

        # Calculate the size of the flattened layer after convolutions
        conv_output_size = self._get_conv_output_size(
            3, height, width
        )  # 3 channels for decentralized

        layers = []
        input_features = conv_output_size
        for _ in range(mlp_hidden_layers):
            layers.append(nn.Linear(input_features, mlp_hidden_features))
            layers.append(nn.ReLU())
            input_features = mlp_hidden_features

        # The final layer outputs a logit for each action
        layers.append(nn.Linear(input_features, num_actions))

        self.mlp_head = nn.Sequential(*layers)

        # Re-initialize the optimizer to include the new mlp_head parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.mlp_head.float()
        self.mlp_head.to(DEVICE)
        # Buffers for actor-specific experiences
        self.batch_actions = []
        self.batch_advantages = []

    def get_action_probabilities(self, processed_state: np.ndarray) -> np.ndarray:
        """Forward pass to get action probabilities."""
        state_tensor = ten_float(processed_state, DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(state_tensor)
            probabilities = F.softmax(logits, dim=1)
        return probabilities.cpu().numpy().squeeze()

    def add_experience(self, state, action, advantage):
        """Adds an experience to the actor's training buffer."""
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.batch_advantages.append(advantage)

    def train_batch(self):
        """Trains the actor using Policy Gradient loss."""
        if not self.batch_states:
            return None

        states = ten_float(np.stack(self.batch_states, axis=0), DEVICE)
        actions = torch.tensor(self.batch_actions, device=DEVICE, dtype=torch.long)
        advantages = ten_float(
            np.array(self.batch_advantages, dtype=np.float32), DEVICE
        )

        # Get action probabilities for the states in the batch
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)

        # Select the log-probabilities of the actions that were actually taken
        # length batch_size and each float in the vector corresponds to log_prob of taken action
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # Calculate policy gradient loss (negative because we if advantage is positive, we want to increase prob
        # of that action)
        loss = -(advantages * action_log_probs).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_advantages.clear()

        return loss.item()
