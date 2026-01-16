from typing import Dict, Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
data_dir = Path(__file__).parent.parent / "data"


NUM_AGENTS = 4
W, L = 9, 9
REWARD = -1
PROBABILITY_APPLE = 10 / (W * L)
TRAJECTORY_LENGTH = 100000
NUM_WORKERS = 8
DISCOUNT_FACTOR = 0.99

NUM_SEEDS = 10


def load_monte_carlo_results(
        seed: int,
        num_agents: int,
        width: int,
        length: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Load Monte Carlo simulation results from .npz file.

    Args:
        seed: Seed value used in the simulation
        num_agents: Number of agents in the simulation
        width: Width of the orchard
        length: Length of the orchard
        results_dir: Directory containing results files

    Returns:
        Tuple of (rewards_by_agent array, metadata dict)
    """
    filepath = data_dir / f"monte-carlo/results_seed{seed}_agents{num_agents}_w{width}_l{length}.npz"

    with np.load(filepath, allow_pickle=True) as data:
        rewards_by_agent = data['rewards_by_agent'].copy()

    return rewards_by_agent


def compute_value(reward_by_agent):
    num_agents, T = reward_by_agent.shape
    # gamma^t for t=0..T-1
    discounts = DISCOUNT_FACTOR ** np.arange(T)
    # elementwise multiply then sum over time
    # result shape: (NUM_AGENTS,)
    returns = (reward_by_agent * discounts).sum(axis=1)
    return returns


# 1. load reward stream for each seed
# 2. for each agent idx, compute the value of that state
# 3. store that for that agent => need hash map
# 4. after done iterating all seeds, we have 1000 data points -> they form a distribution
# 5. visualize it per agent


def process():
    res = np.zeros((NUM_AGENTS, NUM_SEEDS), dtype=float)
    for seed in list(range(NUM_SEEDS)):
        res[:, seed] = compute_value(load_monte_carlo_results(seed, NUM_AGENTS, W, L))

    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for agent_id in range(NUM_AGENTS):
        vals = res[agent_id]

        mean = vals.mean()
        std = vals.std(ddof=0)

        print(f"Agent {agent_id}: mean={mean:.3f}, std={std:.3f}")

        plt.figure()
        plt.hist(vals, bins=30, alpha=0.7)
        plt.title(f"Agent {agent_id} returns\nmean={mean:.3f}, std={std:.3f}")
        plt.xlabel("Return")
        plt.ylabel("Count")
        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"agent_{agent_id}_returns.png")
        plt.savefig(out_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    process()
