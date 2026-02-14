"""
Training Functions for Value Learning - DEBUG VERSION
"""

import copy
import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass

from teleport_dynamic.base_value_model import BaseValueModelV2, Transition

DEBUG_NAN = True  # Set to False to disable debug prints


def debug_print(msg):
    if DEBUG_NAN:
        print(f"[DEBUG] {msg}")


def check_tensor(t, name):
    """Check tensor for NaN/Inf and print debug info"""
    if torch.isnan(t).any():
        nan_count = torch.isnan(t).sum().item()
        debug_print(f"NaN in {name}! Count: {nan_count}, Shape: {t.shape}, Device: {t.device}")
        return True
    if torch.isinf(t).any():
        debug_print(f"Inf in {name}! Shape: {t.shape}")
        return True
    return False


def check_model_params(model, name="model"):
    """Check all model parameters for NaN"""
    for pname, param in model.named_parameters():
        if torch.isnan(param).any():
            debug_print(f"NaN in {name} param: {pname}")
            return True
        if torch.isinf(param).any():
            debug_print(f"Inf in {name} param: {pname}")
            return True
    return False


# =============================================================================
# 1. Supervised Training (Base)
# =============================================================================


def train_supervised(
    model: BaseValueModelV2,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float = 1.0,
) -> float:
    # Input checks
    if check_tensor(inputs, "inputs"):
        return float("nan")
    if check_tensor(targets, "targets"):
        return float("nan")
    
    # Check model before forward
    if check_model_params(model.policy_net, "policy_net_pre_forward"):
        return float("nan")

    model.policy_net.train()
    preds = model.policy_net(inputs).squeeze(-1)
    
    # Check predictions
    if check_tensor(preds, "predictions"):
        debug_print(f"  Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
        return float("nan")

    targets_flat = targets.squeeze(-1) if targets.dim() > 1 else targets
    
    loss = nn.MSELoss()(preds, targets_flat)
    
    if torch.isnan(loss) or torch.isinf(loss):
        debug_print(f"NaN/Inf loss! Preds: [{preds.min():.4f}, {preds.max():.4f}], Targets: [{targets_flat.min():.4f}, {targets_flat.max():.4f}]")
        return float("nan")

    model.optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_norm = 0.0
    for p in model.policy_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if np.isnan(total_norm) or np.isinf(total_norm):
        debug_print(f"NaN/Inf gradient norm: {total_norm}")
        return float("nan")

    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), max_norm=max_grad_norm)
    model.optimizer.step()

    return safe_loss_value(loss)


def train_reward_from_buffer(
    model: BaseValueModelV2, batch_size: int, max_grad_norm: float = 10.0
) -> Optional[float]:
    if len(model.memory) < batch_size:
        return None

    transitions = model.memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=model.device)
    rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=model.device)

    return train_supervised(model, states, rewards, max_grad_norm)


# =============================================================================
# 2. TD(0) Training
# =============================================================================


def train_td0(
    model: BaseValueModelV2,
    batch_size: int,
    max_grad_norm: float = 1.0,
) -> Optional[float]:
    if len(model.memory) < batch_size:
        return None

    transitions = model.memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=model.device)
    next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=model.device)
    rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=model.device)

    model.policy_net.train()
    curr_v = model.policy_net(states).squeeze(-1)

    model.target_net.eval()
    with torch.no_grad():
        next_v = model.target_net(next_states).squeeze(-1)
        td_targets = rewards + model.discount * next_v

    loss = nn.MSELoss()(curr_v, td_targets)

    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), max_norm=max_grad_norm)
    model.optimizer.step()

    return safe_loss_value(loss)


# =============================================================================
# 3. TD(lambda)
# =============================================================================


@dataclass
class TrajectoryStep:
    state_encoding: np.ndarray
    next_state_encoding: np.ndarray
    reward: float
    done: bool = False


class TrajectoryBuffer:
    def __init__(self, max_trajectories: int, max_trajectory_length: int):
        self.trajectories = deque(maxlen=max_trajectories)
        self.max_trajectory_length = max_trajectory_length
        self.current_trajectory: List[TrajectoryStep] = []

    def add_step(self, s_enc, ns_enc, reward, done=False):
        self.current_trajectory.append(TrajectoryStep(s_enc, ns_enc, reward, done))
        if len(self.current_trajectory) >= self.max_trajectory_length or done:
            self.end_trajectory()

    def end_trajectory(self):
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def get_all_trajectories(self) -> List[List[TrajectoryStep]]:
        return list(self.trajectories)

    def clear(self):
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
    device = model.device
    states = torch.tensor(
        np.stack([step.state_encoding for step in trajectory]),
        dtype=torch.float32, device=device,
    )
    rewards = torch.tensor(
        np.array([step.reward for step in trajectory]),
        dtype=torch.float32, device=device,
    )
    
    # Debug: check inputs
    if check_tensor(states, "lambda_states"):
        debug_print(f"  Trajectory length: {len(trajectory)}")
    if check_tensor(rewards, "lambda_rewards"):
        debug_print(f"  Reward values: {[s.reward for s in trajectory]}")
    last_step = trajectory[-1]
    
    with torch.no_grad():
        # Check target_net params first
        if check_model_params(model.target_net, "target_net"):
            debug_print("Target net has NaN params!")
            return states, torch.zeros_like(rewards)
        
        # Process in chunks to avoid GPU batch bug
        CHUNK_SIZE = 64
        values_list = []
        for i in range(0, len(states), CHUNK_SIZE):
            chunk = states[i:i+CHUNK_SIZE]
            chunk_vals = model.target_net(chunk).squeeze(-1)
            
            # CPU fallback if chunk has NaN
            if torch.isnan(chunk_vals).any():
                import copy
                cpu_net = copy.deepcopy(model.target_net).cpu()
                chunk_vals = cpu_net(chunk.cpu()).squeeze(-1).to(device)
            
            values_list.append(chunk_vals)
        values = torch.cat(values_list)
        
        if check_tensor(values, "target_net_values"):
            debug_print(f"  States range: [{states.min():.4f}, {states.max():.4f}]")
            return states, torch.zeros_like(rewards)
        
        if last_step.done:
            bootstrap = torch.tensor(0.0, device=device)
        else:
            last_next_s = torch.tensor(
                last_step.next_state_encoding, dtype=torch.float32, device=device
            ).unsqueeze(0)
            bootstrap = model.target_net(last_next_s).squeeze(-1)
            
            # GPU fallback: if NaN, try CPU
            if torch.isnan(bootstrap).any():
                debug_print("Bootstrap NaN on GPU, trying CPU...")
                import copy
                cpu_net = copy.deepcopy(model.target_net).cpu()
                cpu_input = last_next_s.cpu()
                bootstrap = cpu_net(cpu_input).squeeze(-1).to(device)
            
            if check_tensor(bootstrap, "bootstrap"):
                bootstrap = torch.tensor(0.0, device=device)
    
    next_values = torch.cat([values[1:], bootstrap.view(1)])
    T = len(trajectory)
    lambda_returns = torch.zeros_like(rewards)
    g_next = bootstrap
    
    for t in reversed(range(T)):
        g_next = rewards[t] + gamma * (((1 - lambda_) * next_values[t]) + (lambda_ * g_next))
        g_next = torch.clamp(g_next, -1000.0, 1000.0)
        lambda_returns[t] = g_next
    
    if check_tensor(lambda_returns, "lambda_returns"):
        debug_print(f"  After recursion, returning zeros")
        return states, torch.zeros_like(rewards)
    
    return states, lambda_returns

def train_td_lambda(
    model: BaseValueModelV2,
    trajectory_buffer: TrajectoryBuffer,
    gamma: float,
    lambda_: float,
    batch_size: int,
    max_grad_norm: float = 1.0,
) -> Optional[float]:
    trajectories = trajectory_buffer.get_all_trajectories()
    if len(trajectories) == 0:
        return None

    all_states_list = []
    all_targets_list = []

    for traj in trajectories:
        s, t = compute_lambda_returns_vectorized(traj, model, gamma, lambda_)
        all_states_list.append(s)
        all_targets_list.append(t)

    total_samples = sum(len(s) for s in all_states_list)
    if total_samples < batch_size:
        return None

    all_states = torch.cat(all_states_list)
    all_targets = torch.cat(all_targets_list)

    # Debug before sampling
    if check_tensor(all_states, "all_states_concat"):
        return float("nan")
    if check_tensor(all_targets, "all_targets_concat"):
        return float("nan")

    indices = torch.randperm(total_samples, device=model.device)[:batch_size]
    batch_states = all_states[indices]
    batch_targets = all_targets[indices]

    return train_supervised(model, batch_states, batch_targets, max_grad_norm)


# =============================================================================
# Utilities
# =============================================================================


def safe_loss_value(loss: torch.Tensor) -> float:
    val = loss.item()
    if np.isnan(val) or np.isinf(val):
        return float("inf")
    return val
