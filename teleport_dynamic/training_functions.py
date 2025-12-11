"""
Training Functions for Value Learning (Refactored)

Contains:
- Supervised Training (for Reward Learning)
- TD(0) Training (Standard Replay Buffer)
- TD(lambda) Training (Rolling Trajectory Buffer)
"""

import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass

from teleport_dynamic.base_value_model import BaseValueModelV2, Transition


# =============================================================================
# 1. Supervised Training (Base)
# =============================================================================


def train_supervised(
    model: BaseValueModelV2,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float = 10.0,
) -> float:
    """
    Standard Supervised Regression update.
    Minimizes MSE(prediction, target).
    """
    model.policy_net.train()

    # Forward
    preds = model.policy_net(inputs).squeeze(-1)

    # Ensure targets shape matches preds
    targets_flat = targets.squeeze(-1) if targets.dim() > 1 else targets

    loss = nn.MSELoss()(preds, targets_flat)

    # Backward
    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    return safe_loss_value(loss)


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
# 2. TD(0) Training
# =============================================================================


def train_td0(
    model: BaseValueModelV2,
    batch_size: int,
    max_grad_norm: float = 10.0,
) -> Optional[float]:
    """
    Standard TD(0) update sampling from the model's internal ReplayBuffer.
    Target = r + gamma * V_target(s')
    """
    if len(model.memory) < batch_size:
        return None

    # Sample batch
    transitions = model.memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Prepare Tensors
    states = torch.tensor(
        np.stack(batch.state), dtype=torch.float32, device=model.device
    )
    next_states = torch.tensor(
        np.stack(batch.next_state), dtype=torch.float32, device=model.device
    )
    rewards = torch.tensor(
        np.array(batch.reward), dtype=torch.float32, device=model.device
    )

    # Current Estimates
    model.policy_net.train()
    curr_v = model.policy_net(states).squeeze(-1)

    # Calculate TD Targets (Bootstrap)
    model.target_net.eval()
    with torch.no_grad():
        next_v = model.target_net(next_states).squeeze(-1)
        td_targets = rewards + model.discount * next_v

    # Update
    loss = nn.MSELoss()(curr_v, td_targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.policy_net.parameters(), max_norm=max_grad_norm
    )
    model.optimizer.step()

    return safe_loss_value(loss)


# =============================================================================
# 3. TD(lambda) - The Rolling Buffer Implementation
# =============================================================================


@dataclass
class TrajectoryStep:
    state_encoding: np.ndarray
    next_state_encoding: np.ndarray
    reward: float
    done: bool = False


class TrajectoryBuffer:
    """
    Rolling Buffer for TD(lambda).
    Stores exactly 'max_trajectories' clips.
    When full, adding a new clip automatically drops the oldest one.
    """

    def __init__(self, max_trajectories: int, max_trajectory_length: int):
        # deque with maxlen IS the rolling buffer mechanism
        self.trajectories = deque(maxlen=max_trajectories)
        self.max_trajectory_length = max_trajectory_length
        self.current_trajectory: List[TrajectoryStep] = []

    def add_step(self, s_enc, ns_enc, reward, done=False):
        """Add step to current temporary list. Wraps up if full or done."""
        self.current_trajectory.append(TrajectoryStep(s_enc, ns_enc, reward, done))

        if len(self.current_trajectory) >= self.max_trajectory_length or done:
            self.end_trajectory()

    def end_trajectory(self):
        """Move current list to the main deque. Oldest drops if full."""
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def get_all_trajectories(self) -> List[List[TrajectoryStep]]:
        return list(self.trajectories)

    def clear(self):
        """Manually empty the buffer (optional)."""
        self.trajectories.clear()
        self.current_trajectory = []

    def __len__(self):
        """Returns number of COMPLETED trajectories stored."""
        return len(self.trajectories)


def compute_lambda_returns_vectorized(
    trajectory: List[TrajectoryStep],
    model: BaseValueModelV2,
    gamma: float,
    lambda_: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates lambda-returns recursively for a single trajectory.
    Vectorized on GPU for speed.
    """
    device = model.device

    # Stack into tensors [T, ...]
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

    # Check Bootstrap Condition
    last_step = trajectory[-1]

    with torch.no_grad():
        # Get V(s) for all steps
        values = model.target_net(states).squeeze(-1)

        # Get V(s_T+1) - The Bootstrap
        if last_step.done:
            bootstrap = torch.tensor(0.0, device=device)
        else:
            last_next_s = torch.tensor(
                last_step.next_state_encoding, dtype=torch.float32, device=device
            ).unsqueeze(0)
            bootstrap = model.target_net(last_next_s).squeeze(-1)

    # V_next array for recursion
    # V_next[t] corresponds to V(S_{t+1})
    next_values = torch.cat([values[1:], bootstrap.view(1)])

    # Recursive Lambda Calculation
    # G_t = r + gamma * [ (1-lambda)V(s') + lambda*G_{t+1} ]
    T = len(trajectory)
    lambda_returns = torch.zeros_like(rewards)
    g_next = bootstrap

    for t in reversed(range(T)):
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
    """
    Trains on ALL data currently in the rolling buffer.

    1. Recalculates u_t^lambda for every step in the buffer (keeps targets fresh).
    2. Pools them into one dataset.
    3. Samples 'batch_size' for one gradient update.
    """
    trajectories = trajectory_buffer.get_all_trajectories()
    if len(trajectories) == 0:
        return None

    all_states_list = []
    all_targets_list = []

    # 1. Compute fresh targets for every stored trajectory
    for traj in trajectories:
        s, t = compute_lambda_returns_vectorized(traj, model, gamma, lambda_)
        all_states_list.append(s)
        all_targets_list.append(t)

    # 2. Flatten into one big batch
    total_samples = sum(len(s) for s in all_states_list)
    if total_samples < batch_size:
        return None

    all_states = torch.cat(all_states_list)
    all_targets = torch.cat(all_targets_list)

    # 3. Sample Random Batch
    indices = torch.randperm(total_samples, device=model.device)[:batch_size]
    batch_states = all_states[indices]
    batch_targets = all_targets[indices]

    # 4. Supervised Update
    return train_supervised(model, batch_states, batch_targets, max_grad_norm)


# =============================================================================
# Utilities
# =============================================================================


def safe_loss_value(loss: torch.Tensor) -> float:
    """Safely extract loss value, checking for NaNs."""
    val = loss.item()
    if np.isnan(val) or np.isinf(val):
        return float("inf")
    return val
