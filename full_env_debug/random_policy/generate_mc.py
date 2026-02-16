"""
generate_mc.py — Monte Carlo ground truth generation.

Usage (single state):
    python generate_mc.py --state_index 0 --r_picker 0.25

SLURM array (100 states):
    sbatch --array=0-99 run_mc.sh

Each job saves an .npz with:
    returns:          (num_trajectories, NUM_AGENTS)  — full distribution
    mean_values:      (NUM_AGENTS,)                   — mean per agent
    agent_positions:  (NUM_AGENTS, 2)                 — initial state
    apple_positions:  (num_apples, 2)                 — initial state
    actor:            int                             — initial actor
    tick_in_round:    int                             — initial tick in round
    ticks_until_spawn: int                            — initial spawn countdown
"""

import argparse
import os
import time
import numpy as np

from config import NUM_AGENTS, GAMMA_STEP
from helpers import random_initial_state, mc_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_index', type=int, required=True)
    parser.add_argument('--num_trajectories', type=int, default=200)
    parser.add_argument('--trajectory_length', type=int, default=20000)
    parser.add_argument('--r_picker', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='mc_data')
    args = parser.parse_args()

    r_other = (1.0 - args.r_picker) / (NUM_AGENTS - 1)

    # Deterministic initial state from (seed + state_index)
    state_rng = np.random.default_rng(args.seed + args.state_index)
    initial_state = random_initial_state(state_rng)

    print(f"State {args.state_index}: {len(initial_state.apple_positions)} apples, "
          f"agents={dict(initial_state.agent_positions)}, actor={initial_state.actor}")
    print(f"tick_in_round={initial_state.tick_in_round}, "
          f"ticks_until_spawn={initial_state.ticks_until_spawn}")
    print(f"Running {args.num_trajectories} trajectories × {args.trajectory_length} steps ...")

    t0 = time.time()
    returns = np.zeros((args.num_trajectories, NUM_AGENTS))

    for traj in range(args.num_trajectories):
        traj_rng = np.random.default_rng(
            args.seed * 100_000 + args.state_index * 1000 + traj
        )
        ret = mc_rollout(initial_state, args.r_picker, r_other,
                         args.trajectory_length, traj_rng)
        for i in range(NUM_AGENTS):
            returns[traj, i] = ret[i]

        if (traj + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {traj + 1}/{args.num_trajectories} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    mean_values = returns.mean(axis=0)
    std_values = returns.std(axis=0)

    print(f"\nDone in {elapsed:.1f}s")
    for i in range(NUM_AGENTS):
        print(f"  Agent {i}: mean={mean_values[i]:.4f}, std={std_values[i]:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    agent_pos_arr = np.array([initial_state.agent_positions[i] for i in range(NUM_AGENTS)])
    apple_pos_arr = (np.array(sorted(initial_state.apple_positions))
                     if initial_state.apple_positions
                     else np.empty((0, 2), dtype=int))

    np.savez(
        os.path.join(args.output_dir, f'state_{args.state_index:04d}.npz'),
        returns=returns,
        mean_values=mean_values,
        agent_positions=agent_pos_arr,
        apple_positions=apple_pos_arr,
        actor=np.array(initial_state.actor),
        tick_in_round=np.array(initial_state.tick_in_round),
        ticks_until_spawn=np.array(initial_state.ticks_until_spawn),
        # Metadata
        r_picker=np.array(args.r_picker),
        trajectory_length=np.array(args.trajectory_length),
        num_trajectories=np.array(args.num_trajectories),
        seed=np.array(args.seed),
    )
    print(f"Saved to {args.output_dir}/state_{args.state_index:04d}.npz")


if __name__ == '__main__':
    main()
