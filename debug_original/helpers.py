"""
helpers.py — Environment, encoding, network, TD(0), MC rollout, evaluation.

Partner's 2-mode orchard logic, written in our code style.
"""

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import psutil
from dataclasses import dataclass
from typing import Dict, Optional

from config import (
    H, W, NUM_AGENTS, GAMMA, GAMMA_STEP, K_NEAREST, INPUT_DIM,
    P_SPAWN, P_DESPAWN, L,
)

# Match partner's float64 default dtype
torch.set_default_dtype(torch.float64)


# =============================================================================
# SEEDING
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# REWARD
# =============================================================================

class Reward:
    def __init__(self, picker_r, num_agents):
        self.picker_r = picker_r
        self.other_r = (1 - picker_r) / (num_agents - 1)
        self.num_agents = num_agents

    def get_reward(self, state, actor_id, actor_pos, mode=1):
        res = np.zeros(self.num_agents)
        if mode == 1:
            if state["apples"][tuple(actor_pos)] >= 1:
                res[actor_id] = self.picker_r
                for a in range(self.num_agents):
                    if a != actor_id:
                        res[a] = self.other_r
        return res


# =============================================================================
# ORCHARD ENVIRONMENT  (partner's 2-mode system)
# =============================================================================

@dataclass
class ProcessAction:
    reward_vector: np.ndarray
    picked: bool


class Orchard:
    def __init__(self, width, length, num_agents, reward_module,
                 p_apple=0.1, d_apple=0.0,
                 start_agents_map=None, start_apples_map=None,
                 start_agent_positions=None):
        self.width = width
        self.length = length
        self.n = num_agents
        self.reward_module = reward_module
        self.p_apple = p_apple
        self.d_apple = d_apple

        self.agent_positions = (
            start_agent_positions.copy() if start_agent_positions is not None
            else np.zeros((num_agents, 2), dtype=int)
        )
        self.agents = (
            start_agents_map.copy() if start_agents_map is not None
            else np.zeros((width, length), dtype=int)
        )
        if start_apples_map is not None:
            self.apples = start_apples_map.copy()
        else:
            self.apples = np.zeros((width, length), dtype=int)
            self.spawn_apples()

    def set_positions(self, agent_pos=None):
        for i in range(self.n):
            pos = (agent_pos[i] if agent_pos is not None
                   else np.random.randint(0, [self.width, self.length]))
            self.agent_positions[i] = pos
            self.agents[tuple(pos)] += 1

    def clear_positions(self):
        self.agents[:] = 0
        self.agent_positions[:] = 0

    def get_state(self):
        return {
            "agents": self.agents.copy(),
            "apples": self.apples.copy(),
            "agent_positions": self.agent_positions.copy(),
        }

    def process_action(self, actor_id, new_pos, mode):
        position = self.agent_positions[actor_id]
        if new_pos is not None:
            self.agents[tuple(new_pos)] += 1
            self.agents[tuple(position)] -= 1
            self.agent_positions[actor_id] = new_pos

        actual_pos = self.agent_positions[actor_id]
        reward_vector = self.reward_module.get_reward(
            self.get_state(), actor_id, actual_pos, mode
        )
        picked = reward_vector.sum() != 0
        if picked and self.apples[tuple(actual_pos)] >= 1:
            self.apples[tuple(actual_pos)] -= 1

        return ProcessAction(reward_vector=reward_vector, picked=picked)

    def spawn_apples(self):
        mask = np.random.rand(self.width, self.length) < self.p_apple
        self.apples[mask] += 1
        return int(mask.sum())

    def despawn_apples(self):
        removed = np.random.binomial(self.apples, self.d_apple)
        self.apples -= removed
        return int(removed.sum())


# =============================================================================
# RANDOM POLICY  (grid-clamped, 5 actions)
# =============================================================================

def random_action(agent_pos, grid_size):
    r, c = int(agent_pos[0]), int(agent_pos[1])
    a = random.randrange(5)  # 0=up 1=down 2=left 3=right 4=stay
    nr, nc = r, c
    if a == 0:   nr = r - 1
    elif a == 1: nr = r + 1
    elif a == 2: nc = c - 1
    elif a == 3: nc = c + 1
    if not (0 <= nr < grid_size and 0 <= nc < grid_size):
        nr, nc = r, c
    return np.array([nr, nc])


# =============================================================================
# 2-MODE TRANSITION  (mode 0 = move, mode 1 = reward check)
# =============================================================================

def transition(step, curr_state, env, actor_idx):
    """
    step == -1 : init only (no env mutation)
    step >= 0  : mode-0 move, then mode-1 reward check
    step == N-1: despawn + spawn after reward check

    Returns (final_state, semi_state, result, next_actor_idx)
    """
    if step == -1:
        if curr_state is None:
            curr_state = dict(env.get_state())
        if actor_idx is None:
            actor_idx = random.randint(0, NUM_AGENTS - 1)
        curr_state["actor_id"] = actor_idx
        curr_state["mode"] = 0
        return curr_state, None, None, actor_idx

    # Mode 0: move
    env.process_action(
        actor_idx,
        random_action(curr_state["agent_positions"][actor_idx], W),
        mode=0,
    )
    semi_state = dict(env.get_state())
    semi_state["actor_id"] = actor_idx
    semi_state["mode"] = 1

    # Mode 1: reward check (no movement)
    res = env.process_action(actor_idx, None, mode=1)

    # After last agent in round: despawn then spawn
    if step == NUM_AGENTS - 1:
        env.despawn_apples()
        env.spawn_apples()

    final_state = dict(env.get_state())
    actor_idx = random.randint(0, NUM_AGENTS - 1)
    final_state["actor_id"] = actor_idx
    final_state["mode"] = 0

    return final_state, semi_state, res, actor_idx


# =============================================================================
# ENCODING  (entity-based, self-centered)
# =============================================================================

def encode_state(state, agent_idx):
    """
    Entity encoding (matches partner's ViewController):
        Scalars (3):  [actor_is_self, mode, apple_under_actor]
        Actor (3):    [dx, dy, dist] relative to self
        Others (3*(N-1)): each agent relative to self
        Apples (4*K): top-K nearest to self, with mask
    Total: INPUT_DIM = 3 + 3 + 3*(N-1) + 4*K
    """
    actor_id = state["actor_id"]
    positions = state["agent_positions"]
    apples = state["apples"]
    grid_h, grid_w = apples.shape

    sr, sc = int(positions[agent_idx][0]), int(positions[agent_idx][1])
    ar, ac = int(positions[actor_id][0]), int(positions[actor_id][1])

    dx_den = max(grid_w - 1, 1)
    dy_den = max(grid_h - 1, 1)
    dmax = float(np.sqrt((grid_w - 1) ** 2 + (grid_h - 1) ** 2)) or 1.0

    def rel(r0, c0, r1, c1):
        dx, dy = c1 - c0, r1 - r0
        return float(dx) / dx_den, float(dy) / dy_den, np.sqrt(dx*dx + dy*dy) / dmax

    features = []

    # Scalars
    features.append(np.array([
        1.0 if actor_id == agent_idx else 0.0,
        float(int(state["mode"])),
        1.0 if apples[ar, ac] > 0 else 0.0,
    ], dtype=np.float32))

    # Actor position relative to self
    features.append(np.array(rel(sr, sc, ar, ac), dtype=np.float32))

    # Other agents relative to self
    for j in range(len(positions)):
        if j == agent_idx:
            continue
        rj, cj = int(positions[j][0]), int(positions[j][1])
        features.append(np.array(rel(sr, sc, rj, cj), dtype=np.float32))

    # Top-K apples nearest to self
    apple_rc = np.argwhere(apples > 0)
    if apple_rc.size == 0:
        topk = np.empty((0, 2), dtype=np.int64)
    else:
        dx = apple_rc[:, 1] - sc
        dy = apple_rc[:, 0] - sr
        d2 = dx * dx + dy * dy
        order = np.lexsort((dy, dx, d2))
        topk = apple_rc[order[:K_NEAREST]]

    for k in range(K_NEAREST):
        if k < len(topk):
            r, c = int(topk[k, 0]), int(topk[k, 1])
            dxn, dyn, dn = rel(sr, sc, r, c)
            features.append(np.array([dxn, dyn, dn, 1.0], dtype=np.float32))
        else:
            features.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    return np.concatenate(features).astype(np.float64)


# =============================================================================
# NETWORK  (LeakyReLU MLP, Xavier init with gain)
# =============================================================================

class ValueNetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_layers=(16, 16, 16, 16),
                 negative_slope=0.01):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights(negative_slope)

    def _init_weights(self, negative_slope):
        gain = nn.init.calculate_gain("leaky_relu", param=negative_slope)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# LR SCHEDULE  (linear decay over 80%, then hold)
# =============================================================================

def lr_factor(step, decay_steps, min_factor):
    decay_steps = max(1, int(decay_steps))
    step = max(0, int(step))
    min_factor = max(0.0, min(1.0, float(min_factor)))
    if step >= decay_steps:
        return min_factor
    t = step / decay_steps
    return 1.0 + t * (min_factor - 1.0)


# =============================================================================
# MC GROUND TRUTH  (matches partner's monte_carlo_full)
# =============================================================================

def _make_env(reward_module, p_apple, d_apple, apples, agents, agent_positions):
    return Orchard(W, W, NUM_AGENTS, reward_module,
                   p_apple=p_apple, d_apple=d_apple,
                   start_apples_map=apples, start_agents_map=agents,
                   start_agent_positions=agent_positions)


def mc_ground_truth(seed, trajectory_length, init_env, init_state,
                    discount_factor, num_trajectories=5, num_rollouts=200, semi_mdp=False):
    """
    Monte Carlo ground truth via repeated rollouts.
    Matches partner's monte_carlo_full exactly.
    """
    T = trajectory_length * NUM_AGENTS * 2
    if semi_mdp:
        discounts = discount_factor ** ((np.arange(T, dtype=np.float64) + 1) // 2)
    else:
        discounts = discount_factor ** np.arange(T, dtype=np.float64)
    base = copy.deepcopy(init_state)

    stored_means = np.zeros((NUM_AGENTS, num_trajectories), dtype=np.float64)
    print(f"semi-mdp={semi_mdp}")
    for i in range(num_trajectories):
        print(f"    MC trajectory {i+1}/{num_trajectories}...")
        returns = np.zeros((NUM_AGENTS, num_rollouts), dtype=np.float64)
        for j in range(num_rollouts):
            set_all_seeds(seed + i * num_rollouts + j)
            cs = copy.deepcopy(base)
            env = _make_env(
                init_env.reward_module, init_env.p_apple, init_env.d_apple,
                cs["apples"].copy(), cs["agents"].copy(),
                cs["agent_positions"].copy(),
            )
            actor = cs["actor_id"]
            rba = np.zeros((NUM_AGENTS, T), dtype=np.float64)
            t = 0
            for _ in range(trajectory_length):
                for step in range(NUM_AGENTS):
                    cs, _, res, actor = transition(step, cs, env, actor)
                    rba[:, t] = 0.0
                    t += 1
                    rba[:, t] = res.reward_vector
                    t += 1
            returns[:, j] = rba @ discounts
        stored_means[:, i] = returns.mean(axis=1)

    return stored_means.mean(axis=1)


def generate_eval_states(num_states, reward_module, seed, discount_factor,
                         p_apple, d_apple,
                         mc_depth=1000, mc_trajectories=5, mc_rollouts=200,
                         cache_dir=None):
    """Generate evaluation states with MC ground truth. Caches to disk."""
    states = []
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(num_states):
        if cache_dir:
            cache_path = Path(cache_dir) / f"eval_state_{idx}.npz"
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=True) as f:
                    states.append(f["state_dict"].item())
                print(f"  Loaded cached eval state {idx}")
                continue

        set_all_seeds(seed + idx)
        orchard = Orchard(W, W, NUM_AGENTS, reward_module,
                          p_apple=0.2, d_apple=d_apple)
        orchard.p_apple = p_apple
        orchard.set_positions()
        st = dict(orchard.get_state())
        st["actor_id"] = random.randint(0, NUM_AGENTS - 1)
        st["mode"] = 0

        t0 = time.time()
        mc = mc_ground_truth(
            seed + idx, mc_depth, orchard, st, discount_factor,
            mc_trajectories, mc_rollouts,
        )
        st["mc"] = mc
        dt = time.time() - t0
        print(f"  Generated eval state {idx}: mc={mc}, ({dt:.1f}s)")

        if cache_dir:
            np.savez_compressed(cache_path, state_dict=np.array(st, dtype=object))
        states.append(st)

    return states


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_on_mc(models, eval_states, device='cpu'):
    """
    Evaluate models on MC eval states.
    Returns dict with per-agent and aggregate metrics (our naming style).
    """
    eps = 1e-8
    preds = np.zeros((len(eval_states), NUM_AGENTS))
    true = np.zeros((len(eval_states), NUM_AGENTS))

    for s_idx, st in enumerate(eval_states):
        true[s_idx] = st["mc"]
        for i in range(NUM_AGENTS):
            enc = encode_state(st, i)
            x = torch.tensor(enc, dtype=torch.float64, device=device).unsqueeze(0)
            with torch.no_grad():
                preds[s_idx, i] = float(models[i](x).cpu().item())

    metrics = {}
    for i in range(NUM_AGENTS):
        err = preds[:, i] - true[:, i]
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        mean_abs_true = float(np.mean(np.abs(true[:, i])))
        pct = (mae / mean_abs_true * 100) if mean_abs_true > eps else float('nan')
        metrics[f'a{i}_mae'] = mae
        metrics[f'a{i}_bias'] = bias
        metrics[f'a{i}_pct_error'] = pct
        metrics[f'a{i}_mean_pred'] = float(np.mean(preds[:, i]))
        metrics[f'a{i}_mean_true'] = float(np.mean(true[:, i]))

    metrics['mae_total'] = np.mean([metrics[f'a{i}_mae'] for i in range(NUM_AGENTS)])
    metrics['pct_err_total'] = np.mean([metrics[f'a{i}_pct_error'] for i in range(NUM_AGENTS)])
    metrics['bias_total'] = np.mean([metrics[f'a{i}_bias'] for i in range(NUM_AGENTS)])
    return metrics


# =============================================================================
# UTILITIES
# =============================================================================

def get_memory_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_weight_stats(model, prev_norms):
    stats = {}
    current = []
    for idx, m in enumerate(model.net):
        if isinstance(m, nn.Linear):
            norm = float(m.weight.data.norm().item())
            stats[f'L{idx}_norm'] = norm
            if prev_norms and idx < len(prev_norms):
                stats[f'L{idx}_delta'] = norm - prev_norms[idx]
            else:
                stats[f'L{idx}_delta'] = 0.0
            current.append(norm)
    return stats, current
