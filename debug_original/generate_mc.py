"""
generate_mc.py — Generate one MC eval state with ground truth values.

Usage:
    python generate_mc.py --state_index 0
    python generate_mc.py --state_index 42 --output_dir my_cache

SLURM array (100 states, 50 at a time):
    sbatch --array=0-99%50 run_mc.sh
"""

import argparse
import os
import time
import numpy as np

from config import NUM_AGENTS, GAMMA_STEP, W, P_SPAWN, P_DESPAWN
from helpers import (
    set_all_seeds, Reward, Orchard, mc_ground_truth,
)
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_index', type=int, required=True)
    parser.add_argument('--r_picker', type=float, default=-1.0)
    parser.add_argument('--seed', type=int, default=42069)
    parser.add_argument('--mc_depth', type=int, default=1000,
                        help='MC rollout depth in rounds')
    parser.add_argument('--num_trajectories', type=int, default=5,
                        help='Outer repetitions')
    parser.add_argument('--num_rollouts', type=int, default=200,
                        help='Rollouts per repetition')
    parser.add_argument('--output_dir', type=str, default='mc_cache')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"eval_state_{args.state_index}.npz")

    if os.path.exists(out_path):
        print(f"State {args.state_index} already exists at {out_path}, skipping.")
        return

    reward_module = Reward(args.r_picker, NUM_AGENTS)
    mc_seed = args.seed + args.state_index

    # Generate initial state (same logic as partner's generate_initial_state_full)
    set_all_seeds(mc_seed)
    orchard = Orchard(W, W, NUM_AGENTS, reward_module, p_apple=0.2, d_apple=P_DESPAWN)
    orchard.p_apple = P_SPAWN
    orchard.set_positions()
    st = dict(orchard.get_state())
    st["actor_id"] = random.randint(0, NUM_AGENTS - 1)
    st["mode"] = 0

    print(f"State {args.state_index}: seed={mc_seed}, "
          f"apples={st['apples'].sum()}, "
          f"actor={st['actor_id']}")
    print(f"Running MC: {args.mc_depth} rounds × "
          f"{args.num_trajectories} traj × {args.num_rollouts} rollouts ...")

    t0 = time.time()
    mc = mc_ground_truth(
        seed=mc_seed,
        trajectory_length=args.mc_depth,
        init_env=orchard,
        init_state=st,
        discount_factor=GAMMA_STEP,
        num_trajectories=args.num_trajectories,
        num_rollouts=args.num_rollouts,
    )
    elapsed = time.time() - t0

    st["mc"] = mc

    print(f"Done in {elapsed:.1f}s")
    for i in range(NUM_AGENTS):
        print(f"  Agent {i}: V = {mc[i]:.6f}")

    np.savez_compressed(out_path, state_dict=np.array(st, dtype=object))
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
