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

from teleport_dynamic.base_value_model import BaseValueModelV2, Transition


def train_reward_from_buffer(
    model: BaseValueModelV2, batch_size: int, max_grad_norm: float = 10.0
) -> Optional[float]:
    """Helper to train reward models from their internal replay buffer."""
    if len(model.memory) < batch_size:
        return None

    transitions = model.memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(
        np.stack(batch.state), dtype=torch.float32, device=model.device
    )
    rewards = torch.tensor(
        np.array(batch.reward), dtype=torch.float32, device=model.device
    )

    return train_supervised(model, states, rewards, max_grad_norm)


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


def compute_lambda_returns_vectorized(
    trajectory: List[TrajectoryStep],
    model: BaseValueModelV2,
    gamma: float,
    lambda_: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes G(lambda) entirely on GPU.
    100x faster than moving numpy arrays back and forth.
    """
    device = model.device

    # 1. BATCH PREPARATION
    # Convert list of objects to single tensors immediately (Costly but done once)
    # We stack them to shape [Trajectory_Len, ...]
    states = torch.tensor(
        np.stack([step.state_encoding for step in trajectory]),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.tensor(
        np.array([step.reward for step in trajectory]),
        dtype=torch.float32,
        device=device,
    )

    # Check if last step is terminal
    last_step = trajectory[-1]

    # 2. BATCH INFERENCE
    # Run the network on ALL states in the trajectory at once.
    with torch.no_grad():
        # V(s_t) for all t
        values = model.target_net(states).squeeze(-1)

        # We also need V(s_T+1) (Bootstrap value)
        if last_step.done:
            bootstrap = torch.tensor(0.0, device=device)
        else:
            last_next_s = torch.tensor(
                last_step.next_state_encoding, dtype=torch.float32, device=device
            ).unsqueeze(0)
            bootstrap = model.target_net(last_next_s).squeeze(-1)

    # 3. CREATE NEXT_VALUES VECTOR
    # V_next = [V(s_1), V(s_2), ..., V(s_T), Bootstrap]
    # This allows us to access V(s_{t+1}) by just indexing next_values[t]
    next_values = torch.cat([values[1:], bootstrap.view(1)])

    # 4. RECURSIVE CALCULATION (ON GPU)
    # Even though this is a loop, it's a loop of GPU scalar operations.
    # Because tensors stay on VRAM, this is extremely fast.
    T = len(trajectory)
    lambda_returns = torch.zeros_like(rewards)

    # Initialize G_{t+1} with bootstrap value for the last step
    g_next = bootstrap

    # Loop backwards
    for t in reversed(range(T)):
        # Formula: G_t = r_t + gamma * [ (1-lambda)*V(s_{t+1}) + lambda*G_{t+1} ]
        g_next = rewards[t] + gamma * (
            ((1 - lambda_) * next_values[t]) + (lambda_ * g_next)
        )
        lambda_returns[t] = g_next

    return states, lambda_returns


def train_td_lambda(
    model: BaseValueModelV2,
    trajectory_buffer: TrajectoryBuffer,
    gamma: float,
    lambda_: float,
    batch_size: int,
    max_grad_norm: float = 10.0,
) -> Optional[float]:

    trajectories = trajectory_buffer.get_all_trajectories()
    if len(trajectories) == 0:
        return None

    # --- NEW: Collect Tensors directly ---
    all_states_list = []
    all_targets_list = []

    # Process each trajectory on GPU
    for traj in trajectories:
        s, t = compute_lambda_returns_vectorized(traj, model, gamma, lambda_)
        all_states_list.append(s)
        all_targets_list.append(t)

    # Combine into one massive training batch
    total_samples = sum(len(s) for s in all_states_list)
    if total_samples < batch_size:
        return None

    all_states = torch.cat(all_states_list)
    all_targets = torch.cat(all_targets_list)

    # --- NEW: Shuffle on GPU ---
    # randperm on GPU is very fast
    indices = torch.randperm(total_samples, device=model.device)[:batch_size]

    batch_states = all_states[indices]
    batch_targets = all_targets[indices]

    # Train
    model.policy_net.train()
    preds = model.policy_net(batch_states).squeeze(-1)
    loss = nn.MSELoss()(preds, batch_targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    return loss.item()


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
