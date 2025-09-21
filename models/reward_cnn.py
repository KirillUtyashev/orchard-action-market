import sys

from utils import ten

sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import unwrap_state
from config import DEVICE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RewardCNN(nn.Module):
    """Convolutional Neural Network for Reward Prediction

    Attributes:

    """

    batch_states: list[np.ndarray]
    batch_rewards: list[float]
    loss_history: list[float]

    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        alpha: float,
        mlp_hidden_features: int = 128,
    ):
        """Convolutional Neural Network for Reward Prediction

        Args:
            input_channels: Number of input channels (3 using using architecture provided in CNN pdf)
            height: Height of the input images
            width: Width of the input images
            alpha: Learning rate for the optimizer
            hidden_dim: Not used in CNN layers but used in the MLP head after convolutions.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened layer after convolutions and pooling
        # This is a robust way to handle different input sizes
        conv_output_size = self._get_conv_output_size(input_channels, height, width)
        self.mlp_head = nn.Sequential(
            nn.Linear(conv_output_size, mlp_hidden_features),
            nn.ReLU(),
            nn.Linear(mlp_hidden_features, 1),
        )

        # use all parameters of the model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.batch_states = []
        self.batch_rewards = []
        self.loss_history = []

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass of the data through the network."""
        features = self.pool1(F.relu(self.conv1(x)))
        features = self.pool2(F.relu(self.conv2(features)))
        feature_vector = torch.flatten(features, 1)
        prediction = self.mlp_head(feature_vector)
        return prediction

    def _get_conv_output_size(
        self, input_channels: int, height: int, width: int
    ) -> int:
        """Get the number of features after the conv and pooling layers

        Args:
            input_channels: Number of input channels
            height: Height of the input images
            width: Width of the input images

        Returns:
            The number of features after the conv and pooling layers
        """
        # Create a dummy tensor and pass it through the conv layers to find the output size
        with torch.no_grad():
            x: Tensor = torch.zeros(1, input_channels, height, width)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.numel()  # Total number of elements

    def get_model_reward_prediction(self, x) -> np.ndarray:
        # x is a single numpy array of shape [channels, height, width]

        # Convert to a PyTorch tensor and add a batch dimension of 1
        state_input_tensor = ten(x, DEVICE).unsqueeze(
            0
        )  # unsqueeze(0) adds the batch dim, shape becomes [1, C, H, W]

        with torch.no_grad():
            reward_prediction = self.forward(state_input_tensor)
        return reward_prediction.cpu().numpy()

    def train_batch(self):
        """Train the CNN model on the current batch of states and rewards. Note by batch we mean mini-batch training.

        Preconditions:
            - self.batch_states and self.batch_rewards are populated with experiences. This must be done using ViewControllerCNN to ensure correct state shape.

        Returns:
            The loss value for this training step
        """
        if len(self.batch_states) == 0:
            return None

        # self.batch_states is a list of numpy arrays with shape [channels, height, width]
        # Stack them into a single numpy array of shape [batch_size, channels, height, width]
        states_np = np.stack(self.batch_states, axis=0)
        states_tensor = ten(states_np, DEVICE)

        # Targets are the rewards
        targets = ten(np.asarray(self.batch_rewards, dtype=np.float64), DEVICE).view(
            -1
        )  # view is just in case
        preds = self.forward(states_tensor).view(
            -1
        )  # Final predictions, view to ensure shape is [batch_size]
        # --- End of Forward Pass ---

        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(preds, targets)
        loss.backward()
        self.optimizer.step()

        self.batch_states.clear()
        self.batch_rewards.clear()
        loss_val = float(loss.detach().cpu().item())
        self.loss_history.append(loss_val)
        return loss_val

    def _raw_state_to_nn_input(self, state: dict, agent_pos: np.ndarray) -> np.ndarray:
        """
        Transforms state into a (3, height, width) numpy array for a CNN.

        - Channel 0: Apples (1 where apples exist, 0 otherwise)
        - Channel 1: Other agents
        - Channel 2: The current agent (one-hot encoded position)

        Args:
            state: dict with keys "agents" and "apples" with values as 2D numpy arrays.
            agent_pos: The (row, col) position of the agent in the grid.
        """
        agents_map, apples_map = unwrap_state(state)  # Your helper from helpers.py
        height, width = agents_map.shape

        # Channel 0: Apples
        channel_apples = apples_map

        # Create a map for just the current agent
        self_agent_map = np.zeros_like(agents_map, dtype=np.float32)
        if agent_pos is not None:
            self_agent_map[agent_pos[0], agent_pos[1]] = 1.0
        else:
            raise ValueError("agent_pos must be provided for CNN input.")

        # Channel 1: Other agents
        channel_others = agents_map.astype(np.float32) - self_agent_map

        # Channel 2: Current agent
        channel_self_agent = self_agent_map

        # Stack the channels to create the final "image"
        # The shape is (channels, height, width), which PyTorch expects
        cnn_state = np.stack(
            [channel_apples, channel_others, channel_self_agent], axis=0
        )

        return cnn_state

    def add_experience_from_raw(
        self, raw_state: dict, agent_pos: np.ndarray, reward: float
    ):
        """Processes a raw state and adds the experience to the training buffer.

        Args:
            raw_state: state after action was taken. dict with keys "agents" and "apples" with values as 2D numpy arrays.
            agent_pos: The (row, col) position of the agent in the grid.
            reward: The reward received after taking an action.
        """
        processed_state = self._raw_state_to_nn_input(raw_state, agent_pos)
        self.batch_states.append(processed_state)
        self.batch_rewards.append(reward)
