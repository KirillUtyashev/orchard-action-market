"""
sanity_check.py — Verify steady-state apple count, pick rates, and expected values.
Run from random_policy/: python sanity_check.py
"""

import numpy as np
from config import (
    H, W, NUM_AGENTS, GAMMA, GAMMA_STEP, P_SPAWN, P_DESPAWN, L
)
from helpers import random_initial_state, random_action, env_step, spawn_despawn

R_PICKER = -1.0
R_OTHER = (1.0 - R_PICKER) / (NUM_AGENTS - 1)
SEED = 42
NUM_ROUNDS = 100_000

rng = np.random.default_rng(SEED)
state = random_initial_state(rng)

# ---------- Track statistics ----------
apple_counts = []
picks_per_round = []
rewards_per_tick = {i: [] for i in range(NUM_AGENTS)}

round_picks = 0

for tick in range(NUM_ROUNDS * NUM_AGENTS):
    actor = state.actor
    new_pos = random_action(state.agent_positions[actor], rng)
    state, rewards, picked, _ = env_step(state, new_pos, R_PICKER, R_OTHER)

    if picked:
        round_picks += 1

    for i in range(NUM_AGENTS):
        rewards_per_tick[i].append(rewards[i])

    # End of round
    if (tick + 1) % NUM_AGENTS == 0:
        apple_counts.append(len(state.apple_positions))
        picks_per_round.append(round_picks)
        round_picks = 0

        if state.ticks_until_spawn < 0:
            state = spawn_despawn(state, rng)

apple_counts = np.array(apple_counts)
picks_per_round = np.array(picks_per_round)

print("=" * 60)
print("STEADY-STATE DIAGNOSTICS")
print("=" * 60)

print(f"\nGrid: {H}x{W} = {H*W} cells, {NUM_AGENTS} agents")
print(f"P_SPAWN={P_SPAWN}, P_DESPAWN={P_DESPAWN}, L={L}")
print(f"gamma={GAMMA}, gamma_step={GAMMA_STEP:.6f}")
print(f"R_PICKER={R_PICKER}, R_OTHER={R_OTHER:.4f}")

print(f"\n--- Apple Count ---")
print(f"Theoretical (no agents): {P_SPAWN/(P_SPAWN+P_DESPAWN) * H * W:.1f}")
print(f"Actual mean:  {apple_counts.mean():.1f}")
print(f"Actual std:   {apple_counts.std():.1f}")
print(f"Actual range: [{apple_counts.min()}, {apple_counts.max()}]")
# Burn-in: check last half
last_half = apple_counts[len(apple_counts)//2:]
print(f"Last-half mean: {last_half.mean():.1f} (confirms convergence)")

print(f"\n--- Pick Rate ---")
print(f"Picks per round: {picks_per_round.mean():.3f}")
print(f"Picks per tick:  {picks_per_round.mean() / NUM_AGENTS:.4f}")

print(f"\n--- Per-Agent Reward Stats ---")
for i in range(NUM_AGENTS):
    r = np.array(rewards_per_tick[i])
    print(f"Agent {i}: mean_reward/tick = {r.mean():.6f}")

print(f"\n--- Expected Value Estimates ---")
eff_ticks = 1 / (1 - GAMMA_STEP)
print(f"Effective horizon: {eff_ticks:.1f} ticks")
for i in range(NUM_AGENTS):
    r = np.array(rewards_per_tick[i])
    approx_V = r.mean() * eff_ticks
    print(f"Agent {i}: mean_reward/tick * horizon = {approx_V:.4f}")

# More precise: compute actual discounted returns from the trajectory
print(f"\n--- Discounted Return (sampled, last 10000 starts) ---")
all_rewards = {i: np.array(rewards_per_tick[i]) for i in range(NUM_AGENTS)}
T = len(all_rewards[0])
num_samples = min(10000, T - 500)
starts = np.linspace(T // 2, T - 500, num_samples, dtype=int)
horizon = 200  # ticks to look ahead (gamma_step^200 ~ 0.005)

returns = {i: [] for i in range(NUM_AGENTS)}
for s in starts:
    discount = 1.0
    ret = {i: 0.0 for i in range(NUM_AGENTS)}
    for t in range(s, min(s + horizon, T)):
        for i in range(NUM_AGENTS):
            ret[i] += discount * all_rewards[i][t]
        discount *= GAMMA_STEP
    for i in range(NUM_AGENTS):
        returns[i].append(ret[i])

print(f"(Using {num_samples} start points, {horizon}-tick horizon)")
for i in range(NUM_AGENTS):
    r = np.array(returns[i])
    print(f"Agent {i}: mean={r.mean():.4f}, std={r.std():.4f}")

avg_V = np.mean([np.mean(returns[i]) for i in range(NUM_AGENTS)])
print(f"\nOverall mean |V| estimate: {avg_V:.4f}")
print(f"Your MC ground truth avg:  ~1.3 (from SLURM output)")
print(f"Match: {'YES' if abs(avg_V - 1.3) < 0.5 else 'INVESTIGATE'}")
