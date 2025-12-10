"""
Training Functions for Value Learning

Includes:
- train_batch_regression: For reward learning and analytical value learning
- train_batch_td0: TD(0) learning with replay buffer
- TrajectoryBuffer: Stores trajectories for TD(λ)
- compute_lambda_returns: Computes λ-returns from trajectory
- train_batch_td_lambda: TD(λ) learning with trajectory buffer
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Fixed Import: Use the correct base class from the correct module
from teleport_dynamic.base_value_model import BaseValueModelV2


def train_batch_regression(
    model: BaseValueModelV2,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Supervised regression training.
    Used for Reward Learning and Analytical Value Learning.

    Args:
        model: The value model
        optimizer: Model optimizer
        inputs: Batch of state encodings
        targets: Batch of target values (rewards or exact values)

    Returns:
        Loss value
    """
    model.policy_net.train()

    preds = model.policy_net(inputs).squeeze(1)
    loss = nn.MSELoss()(preds, targets.squeeze())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def train_batch_td0(
    model: BaseValueModelV2,
    inputs: torch.Tensor,
    next_inputs: torch.Tensor,
    rewards: torch.Tensor,
) -> float:
    """
    TD(0) learning with target network.
    Target = r + gamma * target_net(s')

    Args:
        model: The value model (must have policy_net, target_net, optimizer, discount)
        inputs: Batch of current state encodings
        next_inputs: Batch of next state encodings
        rewards: Batch of rewards

    Returns:
        Loss value
    """
    model.policy_net.train()
    model.target_net.eval()

    curr_v = model.policy_net(inputs).squeeze(1)

    with torch.no_grad():
        next_v = model.target_net(next_inputs).squeeze(1)
        td_targets = rewards.squeeze() + model.discount * next_v

    loss = nn.MSELoss()(curr_v, td_targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), max_norm=1.0)
    model.optimizer.step()

    return loss.item()


# =============================================================================
# TD(λ) Implementation
# =============================================================================


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""

    state_encoding: np.ndarray  # Encoded state s_t
    next_state_encoding: np.ndarray  # Encoded state s_{t+1}
    reward: float  # r_t
    done: bool  # Is s_{t+1} terminal?


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
    device: torch.device,
) -> List[Tuple[np.ndarray, float]]:
    """
    Compute λ-returns for each step in a trajectory.
    """
    T = len(trajectory)
    if T == 0:
        return []

    # 1. Get value estimates for all states s_0 ... s_{T-1}
    state_encodings = np.stack([step.state_encoding for step in trajectory])
    states_tensor = torch.tensor(state_encodings, dtype=torch.float32, device=device)

    with torch.no_grad():
        values = model.target_net(states_tensor).squeeze(-1).cpu().numpy()

    # Handle single-element case where squeeze might result in scalar
    if np.ndim(values) == 0:
        values = np.array([values])

    # 2. Get value estimate for the final state s_T (bootstrap)
    last_step = trajectory[-1]
    if last_step.done:
        bootstrap_value = 0.0
    else:
        last_next_state_tensor = torch.tensor(
            last_step.next_state_encoding, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            bootstrap_value = model.target_net(last_next_state_tensor).item()

    # 3. Compute λ-returns backwards
    lambda_returns = np.zeros(T)

    # Base case: Final step T-1
    # G_{T-1} = r_{T-1} + gamma * V(s_T)
    # Note: We use the bootstrap value directly here because we can't look further ahead.
    lambda_returns[T - 1] = last_step.reward + gamma * bootstrap_value

    # Backward pass
    for t in range(T - 2, -1, -1):
        r_t = trajectory[t].reward

        # V(s_{t+1}) is pre-calculated in 'values' at index t+1
        v_next = values[t + 1]

        # G^λ_t = r_t + γ * ((1-λ) * V(s_{t+1}) + λ * G^λ_{t+1})
        lambda_returns[t] = r_t + gamma * (
            (1 - lambda_) * v_next + lambda_ * lambda_returns[t + 1]
        )

    return [(trajectory[t].state_encoding, lambda_returns[t]) for t in range(T)]


def train_batch_td_lambda(
    model: BaseValueModelV2,
    trajectory_buffer: TrajectoryBuffer,
    gamma: float,
    lambda_: float,
    batch_size: int,
    device: torch.device,
) -> Optional[float]:
    """
    Train using TD(λ) with λ-returns computed from stored trajectories.
    """
    trajectories = trajectory_buffer.get_all_trajectories()

    if len(trajectories) == 0:
        return None

    # Compute λ-returns for all trajectories
    all_samples = []
    for traj in trajectories:
        samples = compute_lambda_returns(traj, model, gamma, lambda_, device)
        all_samples.extend(samples)

    if len(all_samples) < batch_size:
        return None

    # Sample batch
    indices = np.random.choice(len(all_samples), size=batch_size, replace=False)
    batch = [all_samples[i] for i in indices]

    states = torch.tensor(
        np.stack([s for s, _ in batch]), dtype=torch.float32, device=device
    )
    targets = torch.tensor(
        np.array([g for _, g in batch]), dtype=torch.float32, device=device
    )

    # Train
    model.policy_net.train()
    preds = model.policy_net(states).squeeze(1)
    loss = nn.MSELoss()(preds, targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), max_norm=1.0)
    model.optimizer.step()

    return loss.item()


# =============================================================================
# Alternative: Online TD(λ) with Eligibility Traces
# =============================================================================


class EligibilityTraceTrainer:
    """
    Online TD(λ) trainer using eligibility traces.
    """

    def __init__(
        self,
        model: BaseValueModelV2,
        gamma: float,
        lambda_: float,
        lr: float,
        device: torch.device,
    ):
        self.model = model
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lr = lr
        self.device = device

        # Initialize eligibility traces (same shape as parameters)
        self.traces = {}
        for name, param in model.policy_net.named_parameters():
            self.traces[name] = torch.zeros_like(param)

    def reset_traces(self):
        """Reset eligibility traces to zero (call at episode start)."""
        for name in self.traces:
            self.traces[name].zero_()

    def update(
        self,
        state_encoding: np.ndarray,
        reward: float,
        next_state_encoding: np.ndarray,
        done: bool = False,
    ) -> float:
        """
        Perform one TD(λ) update.
        """
        state = torch.tensor(
            state_encoding, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        next_state = torch.tensor(
            next_state_encoding, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Compute values
        v = self.model.policy_net(state).squeeze()

        with torch.no_grad():
            v_next = self.model.target_net(next_state).squeeze() if not done else 0.0

        # TD error
        td_error = reward + self.gamma * v_next - v.item()

        # Compute gradient of V(s) w.r.t. parameters
        self.model.optimizer.zero_grad()
        v.backward()

        # Update traces and parameters
        with torch.no_grad():
            for name, param in self.model.policy_net.named_parameters():
                if param.grad is not None:
                    # e = γλe + ∇V(s)
                    self.traces[name] = (
                        self.gamma * self.lambda_ * self.traces[name] + param.grad
                    )
                    # θ += α * δ * e
                    param.add_(self.lr * td_error * self.traces[name])

        return abs(td_error)
