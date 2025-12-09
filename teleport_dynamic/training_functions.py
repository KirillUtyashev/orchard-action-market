import torch
import torch.nn as nn

from models.value_cnn_new import BaseValueModel


def train_batch_regression(
    model: BaseValueModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Used for Reward Learning and Analytical Value Learning.
    Input: Batch of States. Target: Batch of Rewards (or Exact Values).
    """
    model.train()

    preds = model.policy_net(inputs).squeeze(1)
    # Ensure targets shape matches preds (B)
    loss = nn.MSELoss()(preds, targets.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_batch_td0(
    model: BaseValueModel,
    inputs: torch.Tensor,
    next_inputs: torch.Tensor,
    rewards: torch.Tensor,
) -> float:
    """
    Used for TD(0) Learning.
    Target = r + gamma * target_net(s')
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
    # Strict Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), max_norm=1.0)
    model.optimizer.step()

    return loss.item()
