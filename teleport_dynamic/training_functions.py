"""
Training Functions for Value Learning (Refactored)

Clean separation of training methods:
- train_supervised: For reward learning and analytical value targets
- train_td0: TD(0) with target network and replay buffer
- train_td_lambda: TD(λ) with trajectory buffer

All functions take explicit inputs rather than being baked into model classes.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from teleport_dynamic.base_value_model import BaseValueModelV2


# =============================================================================
# Supervised Training (Reward Learning / Analytical Value Learning)
# =============================================================================


def train_supervised(
    model: BaseValueModelV2,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float = 10.0,
) -> float:
    """
    Supervised regression training.
    Used for Reward Learning and Analytical Value Learning.

    Args:
        model: The value model (uses model.policy_net and model.optimizer)
        inputs: Batch of state encodings [B, ...]
        targets: Batch of target values [B] or [B, 1]
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Loss value (float)
    """
    model.policy_net.train()

    preds = model.policy_net(inputs).squeeze(-1)
    targets_flat = targets.squeeze(-1) if targets.dim() > 1 else targets

    loss = nn.MSELoss()(preds, targets_flat)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    loss_val = loss.item()
    model.loss_history.append(loss_val)
    return loss_val


# =============================================================================
# TD(0) Training
# =============================================================================


def train_td0(
    model: BaseValueModelV2,
    batch_size: int,
    max_grad_norm: float = 10.0,
) -> Optional[float]:
    """
    TD(0) learning with target network.
    Samples from model's replay buffer.
    Target = r + gamma * target_net(s')

    Args:
        model: The value model (must have policy_net, target_net, optimizer,
               discount, and memory replay buffer)
        batch_size: Number of transitions to sample
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Loss value, or None if buffer has insufficient samples
    """
    if len(model.memory) < batch_size:
        return None

    # Sample batch from replay buffer
    transitions = model.memory.sample(batch_size)

    # Unpack transitions
    from teleport_dynamic.base_value_model import Transition

    batch = Transition(*zip(*transitions))

    # Convert to tensors
    states = torch.tensor(
        np.stack(batch.state), dtype=torch.float32, device=model.device
    )
    next_states = torch.tensor(
        np.stack(batch.next_state), dtype=torch.float32, device=model.device
    )
    rewards = torch.tensor(
        np.array(batch.reward), dtype=torch.float32, device=model.device
    )

    # Forward pass
    model.policy_net.train()
    model.target_net.eval()

    curr_v = model.policy_net(states).squeeze(-1)

    # Compute TD targets
    with torch.no_grad():
        next_v = model.target_net(next_states).squeeze(-1)
        td_targets = rewards + model.discount * next_v

    # Compute loss and update
    loss = nn.MSELoss()(curr_v, td_targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    loss_val = loss.item()
    model.loss_history.append(loss_val)
    return loss_val


def train_td0_batch(
    model: BaseValueModelV2,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    max_grad_norm: float = 10.0,
) -> float:
    """
    TD(0) learning with explicit batch (no replay buffer sampling).
    Useful when you want to control exactly what's in the batch.

    Args:
        model: The value model
        states: Batch of current state encodings [B, ...]
        next_states: Batch of next state encodings [B, ...]
        rewards: Batch of rewards [B]
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Loss value
    """
    model.policy_net.train()
    model.target_net.eval()

    curr_v = model.policy_net(states).squeeze(-1)

    with torch.no_grad():
        next_v = model.target_net(next_states).squeeze(-1)
        td_targets = rewards.squeeze(-1) + model.discount * next_v

    loss = nn.MSELoss()(curr_v, td_targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    loss_val = loss.item()
    model.loss_history.append(loss_val)
    return loss_val


# =============================================================================
# TD(λ) Implementation
# =============================================================================


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""

    state_encoding: np.ndarray
    next_state_encoding: np.ndarray
    reward: float
    done: bool = False


class TrajectoryBuffer:
    """
    Buffer that stores complete trajectories for TD(λ) learning.
    """

    def __init__(self, max_trajectories: int = 100, max_trajectory_length: int = 1000):
        self.max_trajectories = max_trajectories
        self.max_trajectory_length = max_trajectory_length
        self.trajectories: deque = deque(maxlen=max_trajectories)
        self.current_trajectory: List[TrajectoryStep] = []

    def add_step(
        self,
        state_encoding: np.ndarray,
        next_state_encoding: np.ndarray,
        reward: float,
        done: bool = False,
    ):
        """Add a step to the current trajectory."""
        self.current_trajectory.append(
            TrajectoryStep(
                state_encoding=state_encoding,
                next_state_encoding=next_state_encoding,
                reward=reward,
                done=done,
            )
        )

        # End trajectory if max length reached or done
        if len(self.current_trajectory) >= self.max_trajectory_length or done:
            self.end_trajectory()

    def end_trajectory(self):
        """End current trajectory and store it."""
        if len(self.current_trajectory) > 0:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def get_all_trajectories(self) -> List[List[TrajectoryStep]]:
        """Get all stored trajectories."""
        return list(self.trajectories)

    def clear(self):
        """Clear all stored trajectories."""
        self.trajectories.clear()
        self.current_trajectory = []

    def __len__(self):
        return len(self.trajectories)


def compute_lambda_returns(
    trajectory: List[TrajectoryStep],
    model: BaseValueModelV2,
    gamma: float,
    lambda_: float,
) -> List[Tuple[np.ndarray, float]]:
    """
    Compute λ-returns for each step in a trajectory.

    G^λ_t = r_t + γ * ((1-λ) * V(s_{t+1}) + λ * G^λ_{t+1})

    Args:
        trajectory: List of trajectory steps
        model: Value model (uses target_net for bootstrapping)
        gamma: Discount factor
        lambda_: Lambda parameter for eligibility trace decay

    Returns:
        List of (state_encoding, lambda_return) tuples
    """
    T = len(trajectory)
    if T == 0:
        return []

    device = model.device

    # Get value estimates for all states
    state_encodings = np.stack([step.state_encoding for step in trajectory])
    states_tensor = torch.tensor(state_encodings, dtype=torch.float32, device=device)

    with torch.no_grad():
        values = model.target_net(states_tensor).squeeze(-1).cpu().numpy()

    # Handle single-element case
    if np.ndim(values) == 0:
        values = np.array([values])

    # Get bootstrap value for final state
    last_step = trajectory[-1]
    if last_step.done:
        bootstrap_value = 0.0
    else:
        last_next_tensor = torch.tensor(
            last_step.next_state_encoding, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            bootstrap_value = model.target_net(last_next_tensor).item()

    # Compute λ-returns backwards
    lambda_returns = np.zeros(T)

    # Base case: G_{T-1} = r_{T-1} + γ * V(s_T)
    lambda_returns[T - 1] = last_step.reward + gamma * bootstrap_value

    # Backward pass
    for t in range(T - 2, -1, -1):
        r_t = trajectory[t].reward
        v_next = values[t + 1]

        # G^λ_t = r_t + γ * ((1-λ) * V(s_{t+1}) + λ * G^λ_{t+1})
        lambda_returns[t] = r_t + gamma * (
            (1 - lambda_) * v_next + lambda_ * lambda_returns[t + 1]
        )

    return [(trajectory[t].state_encoding, lambda_returns[t]) for t in range(T)]


def train_td_lambda(
    model: BaseValueModelV2,
    trajectory_buffer: TrajectoryBuffer,
    gamma: float,
    lambda_: float,
    batch_size: int,
    max_grad_norm: float = 10.0,
) -> Optional[float]:
    """
    Train using TD(λ) with λ-returns computed from stored trajectories.

    Args:
        model: The value model
        trajectory_buffer: Buffer containing trajectories
        gamma: Discount factor
        lambda_: Lambda parameter
        batch_size: Number of samples to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Loss value, or None if insufficient data
    """
    trajectories = trajectory_buffer.get_all_trajectories()

    if len(trajectories) == 0:
        return None

    # Compute λ-returns for all trajectories
    all_samples = []
    for traj in trajectories:
        samples = compute_lambda_returns(traj, model, gamma, lambda_)
        all_samples.extend(samples)

    if len(all_samples) < batch_size:
        return None

    # Sample batch
    indices = np.random.choice(len(all_samples), size=batch_size, replace=False)
    batch = [all_samples[i] for i in indices]

    states = torch.tensor(
        np.stack([s for s, _ in batch]), dtype=torch.float32, device=model.device
    )
    targets = torch.tensor(
        np.array([g for _, g in batch]), dtype=torch.float32, device=model.device
    )

    # Train
    model.policy_net.train()
    preds = model.policy_net(states).squeeze(-1)
    loss = nn.MSELoss()(preds, targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    loss_val = loss.item()
    model.loss_history.append(loss_val)
    return loss_val


# =============================================================================
# Safety Utilities
# =============================================================================


def check_for_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN values."""
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        print(f"WARNING: NaN detected in {name}")
    return bool(has_nan)


def safe_loss_value(loss: torch.Tensor) -> float:
    """Safely extract loss value, returning inf if NaN."""
    val = loss.item()
    if np.isnan(val) or np.isinf(val):
        return float("inf")
    return val
