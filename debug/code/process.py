from typing import Dict, Sequence, Tuple
import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt

from config import (
    NUM_AGENTS,
    REWARD,
    W,
    L,
    DISCOUNT_FACTOR,
    PROBABILITY_APPLE,
    data_dir,
    SEEDS,
)


# ------------- loading -----------------

def load_results(
        kind: str,
        seed: int,
        num_agents: int,
        width: int,
        length: int,
) -> np.ndarray:
    folder = f"{kind}-{PROBABILITY_APPLE:.2f}-{num_agents}-{width}-{REWARD}"
    filepath = data_dir / folder / f"results_seed{seed}.npz"
    with np.load(filepath, allow_pickle=True) as data:
        rewards_by_agent = data["rewards_by_agent"].copy()
    return rewards_by_agent


# ------------- value computation -----------------

def compute_value(reward_by_agent: np.ndarray) -> np.ndarray:
    num_agents, T = reward_by_agent.shape
    discounts = DISCOUNT_FACTOR ** np.arange(T)
    returns = (reward_by_agent * discounts).sum(axis=1)
    return returns


# ------------- closed-form value -----------------

def theoretical_value(
        r_pick: float,
        r_other: float,
        p_apple: float,
        num_agents: int,
        gamma: float,
) -> float:
    """
    Closed-form value V^{(i)} for a single agent, using

        V^{(i)} = E[R_t^{(i)}] / (1 - gamma)

    where
        E[R_t^{(i)}] = r_pick * (1 / N) * P + r_other * (1 - 1 / N) * P.
    """
    expected_reward = (
            r_pick * (1.0 / num_agents) * p_apple
            + r_other * (1.0 - 1.0 / num_agents) * p_apple
    )
    return expected_reward / (1.0 - gamma)


# ------------- plotting -----------------

def plot_distributions(
        values: np.ndarray,
        kind: str,
        r_pick: float,
        r_other: float,
) -> None:
    """
    values: shape (NUM_AGENTS, SEEDS)
    kind: 'MC' or 'IID' (used in titles / filenames)
    """
    exp_name = f"{kind}-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{REWARD}"
    plots_dir = data_dir / "plots" / exp_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    # same theoretical value for every agent in this symmetric setup
    v_theory = theoretical_value(
        r_pick=r_pick,
        r_other=r_other,
        p_apple=PROBABILITY_APPLE,
        num_agents=NUM_AGENTS,
        gamma=DISCOUNT_FACTOR,
    )

    for agent_id in range(NUM_AGENTS):
        vals = values[agent_id]

        mean = vals.mean()
        std = vals.std(ddof=0)

        print(f"[{kind}] Agent {agent_id}: mean={mean:.3f}, std={std:.3f}")

        plt.figure()
        plt.hist(vals, bins=30, alpha=0.7, density=True, color="C0")

        x = np.linspace(vals.min(), vals.max(), 200)
        pdf = norm.pdf(x, loc=mean, scale=std)
        plt.plot(x, pdf, "k-", linewidth=2)

        # red vertical line for closed-form value
        plt.axvline(v_theory, color="red", linestyle="--", linewidth=2)

        plt.title(
            f"{kind} – Agent {agent_id} returns\n"
            f"mean={mean:.3f}, std={std:.3f}, theory={v_theory:.3f}"
        )
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.tight_layout()

        fname = plots_dir / f"agent_{agent_id}_returns.png"
        plt.savefig(fname, dpi=300)
        plt.close()


def load_returns_for_experiment(kind: str, reward: float) -> np.ndarray:
    """
    Load discounted returns across seeds for one (kind, reward).

    Returns: array of shape (NUM_AGENTS, SEEDS)
    """
    folder = f"{kind}-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{reward}"
    returns = np.zeros((NUM_AGENTS, SEEDS), dtype=float)

    for seed in range(SEEDS):
        filepath = data_dir / folder / f"results_seed{seed}.npz"
        with np.load(filepath, allow_pickle=True) as data:
            rewards_by_agent = data["rewards_by_agent"].copy()

        # compute discounted return per agent
        num_agents, T = rewards_by_agent.shape
        discounts = DISCOUNT_FACTOR ** np.arange(T)
        returns[:, seed] = (rewards_by_agent * discounts).sum(axis=1)

    return returns


def compare_mc_iid_by_reward(rewards: Sequence[float]) -> None:
    rewards = list(rewards)
    rewards_sorted = sorted(rewards)

    diff_means = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)
    diff_stds = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)

    for r_idx, r in enumerate(rewards_sorted):
        mc_returns = load_returns_for_experiment("monte-carlo", r)
        iid_returns = load_returns_for_experiment("iid", r)

        mc_mean = mc_returns.mean(axis=1)
        mc_std = mc_returns.std(axis=1, ddof=0)
        iid_mean = iid_returns.mean(axis=1)
        iid_std = iid_returns.std(axis=1, ddof=0)

        diff_means[:, r_idx] = mc_mean - iid_mean
        diff_stds[:, r_idx] = np.sqrt(mc_std ** 2 + iid_std ** 2)

    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]

    # ---- categorical x positions ----
    x_base = np.arange(len(rewards_sorted))

    base_offset = 0.05
    offsets = np.linspace(-base_offset, base_offset, NUM_AGENTS)

    for agent_id in range(NUM_AGENTS):
        x_positions = x_base + offsets[agent_id]

        ax.errorbar(
            x_positions,
            diff_means[agent_id],
            yerr=diff_stds[agent_id],
            fmt="o",
            markersize=6,
            color=colors[agent_id % len(colors)],
            ecolor=colors[agent_id % len(colors)],
            elinewidth=1.5,
            capsize=3,
            linestyle="none",
            label=f"Agent {agent_id}",
            alpha=0.9,
        )

    # ticks labeled by reward values
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{r:.2f}" for r in rewards_sorted])

    ax.axhline(0.0, color="red", linestyle=":", linewidth=1.5)

    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(False, axis="x")

    ax.set_xlabel("Reward")
    ax.set_ylabel("MC − IID value")
    ax.set_title("Difference in value: Monte Carlo vs IID")

    ax.legend(title="Agent", frameon=False, ncol=NUM_AGENTS)
    fig.tight_layout()

    plots_root = data_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    exp_name = f"MC_vs_IID-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}"
    out_path = plots_root / f"{exp_name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ------------- driver -----------------

def process():
    # Monte Carlo values
    mc_vals = np.zeros((NUM_AGENTS, SEEDS), dtype=float)
    for seed in range(SEEDS):
        mc_rewards = load_results("monte-carlo", seed, NUM_AGENTS, W, L)
        mc_vals[:, seed] = compute_value(mc_rewards)

    # IID values
    iid_vals = np.zeros((NUM_AGENTS, SEEDS), dtype=float)
    for seed in range(SEEDS):
        iid_rewards = load_results("iid", seed, NUM_AGENTS, W, L)
        iid_vals[:, seed] = compute_value(iid_rewards)

    # example: plug in your actual r_pick and r_other here
    r_pick = REWARD
    r_other = (1 - r_pick) / (NUM_AGENTS - 1)

    plot_distributions(mc_vals, kind="MC", r_pick=r_pick, r_other=r_other)
    plot_distributions(iid_vals, kind="IID", r_pick=r_pick, r_other=r_other)


if __name__ == "__main__":
    compare_mc_iid_by_reward([-5, -1])
    # process()
