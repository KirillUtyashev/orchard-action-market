"""
helpers.py — State, environment, encoding, network, TD(λ), MC rollout, evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import psutil
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, Optional

from config import (
    H, W, NUM_AGENTS, GAMMA_STEP, K_NEAREST, INPUT_DIM,
    P_SPAWN, P_DESPAWN, ACTION_DELTAS, L,
)


# =============================================================================
# STATE
# =============================================================================

@dataclass
class State:
    agent_positions: Dict[int, Tuple[int, int]]   # {agent_idx: (row, col)}
    apple_positions: Set[Tuple[int, int]]          # {(row, col)}
    actor: int                                     # current actor (round-robin)
    tick_in_round: int                             # T^n: 0..N-1
    ticks_until_spawn: int                         # E^n: 0..L*N-1

    def copy(self):
        return State(
            agent_positions=dict(self.agent_positions),
            apple_positions=set(self.apple_positions),
            actor=self.actor,
            tick_in_round=self.tick_in_round,
            ticks_until_spawn=self.ticks_until_spawn,
        )


# =============================================================================
# ENVIRONMENT
# =============================================================================

def random_action(pos, rng):
    """Random action -> new position. Off-grid moves become stay."""
    r, c = pos
    dr, dc = ACTION_DELTAS[rng.integers(5)]
    nr, nc = r + dr, c + dc
    if 0 <= nr < H and 0 <= nc < W:
        return (nr, nc)
    return (r, c)


def env_step(state, new_pos, r_picker, r_other):
    """
    Atomic move-and-pick for current actor.

    Returns:
        new_state:    final State (actor advanced, counters updated, apple removed if picked)
        rewards:      {agent_idx: float}
        picked:       bool
        arrive_state: State after move, before pick (only if picked; counters NOT advanced)
    """
    actor = state.actor
    moved = state.copy()
    moved.agent_positions[actor] = new_pos

    picked = new_pos in moved.apple_positions
    rewards = {i: 0.0 for i in range(NUM_AGENTS)}
    arrive_state = None

    if picked:
        arrive_state = moved.copy()                  # apple still there, actor same, counters same
        moved.apple_positions.discard(new_pos)       # remove apple
        rewards[actor] = r_picker
        for i in range(NUM_AGENTS):
            if i != actor:
                rewards[i] = r_other

    # Advance counters only in the final state (not arrive)
    moved.actor = (actor + 1) % NUM_AGENTS
    moved.tick_in_round = (state.tick_in_round + 1) % NUM_AGENTS
    moved.ticks_until_spawn = state.ticks_until_spawn - 1

    return moved, rewards, picked, arrive_state


def spawn_despawn(state, rng):
    """Spawn/despawn apples and reset E^n counter."""
    new = state.copy()
    # Despawn existing
    new.apple_positions = {
        pos for pos in new.apple_positions if rng.random() >= P_DESPAWN
    }
    
    agent_locs = set(new.agent_positions.values())
    
    # Spawn on empty cells
    for r in range(H):
        for c in range(W):
            pos = (r, c)
            # Check: Not an apple AND Not under an agent
            if pos not in new.apple_positions and pos not in agent_locs:
                if rng.random() < P_SPAWN:
                    new.apple_positions.add(pos)

    # Reset spawn countdown
    new.ticks_until_spawn = L * NUM_AGENTS - 1
    return new


def random_initial_state(rng):
    """Generate a random initial state with steady-state apple density."""
    p_steady = P_SPAWN / (P_SPAWN + P_DESPAWN)
    agent_positions = {}
    for i in range(NUM_AGENTS):
        agent_positions[i] = (int(rng.integers(H)), int(rng.integers(W)))
    occupied = set(agent_positions.values())
    apple_positions = set()
    for r in range(H):
        for c in range(W):
            if (r, c) not in occupied and rng.random() < p_steady:
                apple_positions.add((r, c))
    return State(
        agent_positions=agent_positions,
        apple_positions=apple_positions,
        actor=0,
        tick_in_round=0,
        ticks_until_spawn=L * NUM_AGENTS - 1,
    )


# =============================================================================
# ENCODING
# =============================================================================

def encode_state(state, agent_idx):
    """
    Encode state from perspective of agent_idx.

    Layout (§3 of spec):
        [is_actor, ticks_until_turn, ticks_until_spawn]           3
        [Δy, Δx, d_man] × (N-1) other agents (fixed index order) 3*(N-1)
        [Δy, Δx, d_man, is_present] × K nearest apples            4*K
    Total: INPUT_DIM = 3 + 3*(N-1) + 4*K = 24
    """
    features = []
    my_r, my_c = state.agent_positions[agent_idx]

    # --- Scalars ---
    features.append(1.0 if agent_idx == state.actor else 0.0)

    ticks_until_turn = (agent_idx - state.actor) % NUM_AGENTS
    features.append(ticks_until_turn / max(NUM_AGENTS - 1, 1))

    max_spawn_ticks = L * NUM_AGENTS - 1
    features.append(state.ticks_until_spawn / max(max_spawn_ticks, 1))

    # --- Other agents (fixed index order, skip self) ---
    for j in range(NUM_AGENTS):
        if j == agent_idx:
            continue
        jr, jc = state.agent_positions[j]
        dy = (jr - my_r) / (W - 1)
        dx = (jc - my_c) / (W - 1)
        d_man = (abs(jr - my_r) + abs(jc - my_c)) / (2 * (W - 1))
        features.extend([dy, dx, d_man])

    # --- K nearest apples by Manhattan distance, tiebreak (Δy, Δx) ---
    apples = list(state.apple_positions)
    apples.sort(key=lambda p: (
        abs(p[0] - my_r) + abs(p[1] - my_c),   # Manhattan distance
        p[0] - my_r,                             # Δy tiebreak
        p[1] - my_c,                             # Δx tiebreak
    ))

    for k in range(K_NEAREST):
        if k < len(apples):
            ar, ac = apples[k]
            dy = (ar - my_r) / (W - 1)
            dx = (ac - my_c) / (W - 1)
            d_man = (abs(ar - my_r) + abs(ac - my_c)) / (2 * (W - 1))
            features.extend([dy, dx, d_man, 1.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


# =============================================================================
# NETWORK
# =============================================================================

class ValueNetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_layers=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.01))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# TD(λ) — VECTORIZED LAMBDA RETURN
# =============================================================================

def compute_lambda_return(rewards, values, gammas, lam):
    """
    Vectorized λ-return: G^λ = v_0 + ⟨c, δ⟩

    Args:
        rewards: (n,) per-transition rewards for one agent
        values:  (n+1,) value estimates for one agent
        gammas:  (n,) per-transition discounts
        lam:     scalar λ
    Returns:
        scalar λ-return for state 0
    """
    n = len(rewards)
    delta = rewards + gammas * values[1:] - values[:-1]
    c = (lam ** np.arange(n)) * np.cumprod(np.concatenate([[1.0], gammas[:n - 1]]))
    return float(values[0] + np.dot(c, delta))


class SlidingWindowBuffer:
    """
    Sliding window buffer for TD(λ). Stores state encodings, per-agent values,
    per-transition rewards and gammas. When num_transitions >= n, the oldest
    state is ready for training.
    """
    def __init__(self, window_size, num_agents):
        self.n = window_size
        self.num_agents = num_agents
        self.encodings = []     # per state: {agent_idx: np.array}
        self.values = []        # per state: {agent_idx: float}
        self.rewards = []       # per transition: {agent_idx: float}
        self.gammas = []        # per transition: float
        self._start = 0

    def add_state(self, enc_dict, val_dict):
        self.encodings.append(enc_dict)
        self.values.append(val_dict)

    def add_transition(self, reward_dict, gamma):
        self.rewards.append(reward_dict)
        self.gammas.append(gamma)

    @property
    def num_transitions(self):
        return len(self.gammas) - self._start

    def can_train(self):
        return self.num_transitions >= self.n

    def pop_oldest(self, lam):
        """Compute λ-returns for oldest state, pop it, return (encoding_dict, targets_dict)."""
        s = self._start
        n = self.n
        gamma_arr = np.array([self.gammas[s + j] for j in range(n)])

        targets = {}
        for i in range(self.num_agents):
            r_i = np.array([self.rewards[s + j][i] for j in range(n)])
            v_i = np.array([self.values[s + j][i] for j in range(n + 1)])
            targets[i] = compute_lambda_return(r_i, v_i, gamma_arr, lam)

        enc = self.encodings[s]
        self._start += 1

        # Periodically compact to free memory
        if self._start > 10000:
            self.encodings = self.encodings[self._start:]
            self.values = self.values[self._start:]
            self.rewards = self.rewards[self._start:]
            self.gammas = self.gammas[self._start:]
            self._start = 0

        return enc, targets


# =============================================================================
# MC ROLLOUT
# =============================================================================

def mc_rollout(initial_state, r_picker, r_other, trajectory_length, rng):
    """
    Run one MC rollout under random policy using the expanded transition model:
    each tick discounts by γ_step for the move, and arrive-on-apple transitions
    contribute rewards at the post-move discount level (γ=1 for the pick itself).

    Returns per-agent discounted returns as dict {agent_idx: float}.
    """
    state = initial_state.copy()
    returns = {i: 0.0 for i in range(NUM_AGENTS)}
    discount = 1.0

    for t in range(trajectory_length):
        actor = state.actor
        new_pos = random_action(state.agent_positions[actor], rng)
        state, rewards, picked, _ = env_step(state, new_pos, r_picker, r_other)

        # Move transition always carries discount γ_step
        discount *= GAMMA_STEP

        if picked:
            # Arrive→pick transition: γ=1, reward = pick rewards
            # Reward is applied at the current (post-move) discount level
            for i in range(NUM_AGENTS):
                returns[i] += discount * rewards[i]
            # γ=1 for arrive→pick, so no additional discount change

        # Spawn/despawn when E^n goes below 0
        if state.ticks_until_spawn < 0:
            state = spawn_despawn(state, rng)

    return returns


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_on_mc(models, mc_states, mc_mean_values, device='cpu'):
    """
    Evaluate models on precomputed MC states.

    Args:
        models:          list of N ValueNetwork
        mc_states:       list of State objects
        mc_mean_values:  np.array shape (num_states, N) — MC ground truth per agent
        device:          torch device

    Returns dict with per-agent and aggregate metrics.
    """
    num_states = len(mc_states)
    preds = np.zeros((num_states, NUM_AGENTS))
    true = mc_mean_values  # (num_states, N)

    for i in range(NUM_AGENTS):
        encs = np.array([encode_state(s, i) for s in mc_states])
        x = torch.tensor(encs, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds[:, i] = models[i](x).cpu().numpy()

    metrics = {}
    for i in range(NUM_AGENTS):
        err = preds[:, i] - true[:, i]
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        mean_abs_true = float(np.mean(np.abs(true[:, i])))
        pct = (mae / mean_abs_true * 100) if mean_abs_true > 1e-10 else float('nan')
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
    """Per-layer weight norms and deltas. Returns (stats_dict, current_norms)."""
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
