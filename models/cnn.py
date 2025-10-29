import sys
from typing_extensions import override

sys.path.append("../")

from pyparsing import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import ten_float, unwrap_state
from config import DEVICE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    """Convolutional Neural Network used for anything that takes in the orchard state.


    Precondition: height and width must be at least 4 due to two max-pooling layers with kernel size 2.

    Attributes:
        batch_states: List of states for training.
        batch_rewards: List of rewards for training.
        batch_new_states: List of new states for TD-learning (only used in ValueTrainer).

    """

    batch_rewards: list[float]
    batch_states: list[np.ndarray]
    batch_new_states: list[np.ndarray]
    loss_history: list[float]

    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        alpha: float,
        mlp_hidden_features: int = 128,
        num_mlp_hidden_layers: int = 1,
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

        layers = []
        input_features = conv_output_size

        # This loop runs 'num_mlp_hidden_layers' times
        for _ in range(num_mlp_hidden_layers):
            layers.append(nn.Linear(input_features, mlp_hidden_features))
            layers.append(nn.ReLU())
            input_features = mlp_hidden_features

        # This final layer is always added
        layers.append(nn.Linear(input_features, 1))

        self.mlp_head = nn.Sequential(*layers)

        # use all parameters of the model
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=alpha, amsgrad=True)

        self.batch_states = []
        self.batch_rewards = []
        self.loss_history = []
        self.float()
        self.to(DEVICE)

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
            layer_device = self.conv1.weight.device
            layer_dtype = self.conv1.weight.dtype
            x: Tensor = torch.zeros(
                1, input_channels, height, width, device=layer_device, dtype=layer_dtype
            )

            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.numel()  # Total number of elements

    def get_model_reward_prediction_from_proccessed_state(self, x) -> np.ndarray:
        # x is a single numpy array of shape [channels, height, width]

        # Convert to a PyTorch tensor and add a batch dimension of 1
        state_input_tensor = ten_float(x, DEVICE).unsqueeze(
            0
        )  # unsqueeze(0) adds the batch dim, shape becomes [1, C, H, W]

        with torch.no_grad():
            reward_prediction = self.forward(state_input_tensor)
        return reward_prediction.cpu().numpy()

    def get_model_reward_prediction_from_raw(
        self, raw_state: dict, **kwargs
    ) -> np.ndarray:
        """Get the model's reward prediction from a raw state dictionary.

        Args:
            raw_state: dict with keys "agents" and "apples" with values as 2D numpy arrays.
            **kwargs: See raw_state_to_nn_input for details.

        Returns:
            The model's reward prediction as a numpy array.
        """
        processed_state = self.raw_state_to_nn_input(raw_state, **kwargs)
        return self.get_model_reward_prediction_from_proccessed_state(processed_state)

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
        states_tensor = ten_float(states_np, DEVICE)

        # Targets are the rewards
        targets = ten_float(
            np.asarray(self.batch_rewards, dtype=np.float32), DEVICE
        ).view(
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

    @abstractmethod
    def raw_state_to_nn_input(self, state: dict, **kwargs) -> np.ndarray:
        """
        Transforms state into neural net input

        Args:
            state: dict with keys "agents" and "apples" with values as 2D numpy arrays.
            **kwargs: Additional arguments that may be needed by subclasses (e.g., agent_pos for decentralized model)
        """
        raise NotImplementedError(
            "_raw_state_to_nn_input must be implemented in subclasses."
        )

    def add_experience_from_raw(self, raw_state: dict, reward: float, **kwargs):
        """Processes a raw state and adds the experience to the training buffer.

        Args:
            raw_state: state after action was taken. dict with keys "agents" and "apples" with values as 2D numpy arrays.
            reward: The reward received after taking an action.
            **kwargs: See raw_state_to_nn_input for details.
        """
        processed_state = self.raw_state_to_nn_input(raw_state, **kwargs)
        self.batch_states.append(processed_state)
        # turn reward into float32 as well
        self.batch_rewards.append(reward)

    def export_net_state(self):
        """
        Creates a dictionary containing the model's weights and the optimizer's state.
        This makes the CNN compatible with the existing saving and loading mechanism.
        """
        return {
            "weights": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def import_net_state(self, blob, device=DEVICE):
        """
        Loads the model's weights and optimizer state from a dictionary.
        """
        # The main model is a nn.Module, so we can use load_state_dict
        self.load_state_dict(blob["weights"])

        # Also load the optimizer state if it exists
        if blob.get("optimizer") is not None:
            self.optimizer.load_state_dict(blob["optimizer"])
            # The original code has this important loop to move optimizer
            # tensors to the correct device (e.g., GPU), which we should keep.
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)


class CNNDecentralized(CNN):
    @override
    def __init__(
        self,
        height: int,
        width: int,
        alpha: float,
        mlp_hidden_features: int = 256,
        num_mlp_hidden_layers: int = 2,
    ):
        # Centralized model only has 2 channels: apples and all agents (not distinguishing between different agents)
        super().__init__(
            input_channels=3,
            height=height,
            width=width,
            alpha=alpha,
            mlp_hidden_features=mlp_hidden_features,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
        )

    @override
    def raw_state_to_nn_input(self, state: dict, **kwargs) -> np.ndarray:
        """
        Transforms state into a (3, height, width) numpy array for a CNN.

        - Channel 0: Apples (1 where apples exist, 0 otherwise)
        - Channel 1: Other agents
        - Channel 2: The current agent (one-hot encoded position)

        Args:
            state: dict with keys "agents" and "apples" with values as 2D numpy arrays.
            **kwargs:
                agent_pos: (np.ndarray) indicating the position of the current agent.

        Returns:
            A numpy array of shape (3, height, width) suitable for CNN Decentralized input.
        """
        try:
            agent_pos = kwargs["agent_pos"]
        except KeyError:
            # Raise a more descriptive error to help the user.
            raise ValueError(
                "The decentralized model requires 'agent_pos' to be passed as a keyword argument."
            )
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


class CNNCentralized(CNN):
    @override
    def __init__(
        self,
        height: int,
        width: int,
        alpha: float,
        mlp_hidden_features: int = 128,
        num_mlp_hidden_layers: int = 1,
    ):
        # Centralized model only has 2 channels: apples and all agents (not distinguishing between different agents)
        super().__init__(
            input_channels=2,
            height=height,
            width=width,
            alpha=alpha,
            mlp_hidden_features=mlp_hidden_features,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
        )

    @override
    def raw_state_to_nn_input(self, state: dict, **kwargs) -> np.ndarray:
        """
        Converts state dictionary with 2 keys and stacks them into a 2-channel numpy array.
        - Channel 0: Apples
        - Channel 1: All agents (not distinguishing between different agents)
        Args:
            state: dict with keys "agents" and "apples" with values as 2D numpy arrays.
            **kwargs: Not used but included for compatibility with base class.
        Returns:
            A numpy array of shape (2, height, width) suitable for CNN Centralized input

        """
        agents_map, apples_map = unwrap_state(state)  # Your helper from helpers.py
        height, width = agents_map.shape

        # Stack the channels to create the final "image"
        # The shape is (channels, height, width), which PyTorch expects
        cnn_state = np.stack([apples_map, agents_map.astype(np.float32)], axis=0)

        return cnn_state.astype(np.float32)
