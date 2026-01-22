import argparse
from typing import Dict, Sequence, Tuple
import numpy as np
from collections.abc import Callable, Sequence
from scipy.stats import norm
import matplotlib.pyplot as plt

from config import (
    NUM_AGENTS,
    W,
    L,
    DISCOUNT_FACTOR,
    PROBABILITY_APPLE,
    data_dir,
    SEEDS,
)
from debug.code.monte_carlo import StateType

ACTOR_ID = 0
State = str  # or your Enum type if you have one
ValueGenerator = Callable[[float, State], tuple[np.ndarray, np.ndarray]]
# returns: (mean_by_agent, std_by_agent), each shape (NUM_AGENTS,)



# ------------- loading -----------------
def load_results(
        kind: str,
        seed: int,
        num_agents: int,
        width: int,
        reward,
        state
) -> np.ndarray:
    out_dir = data_dir / state / kind
    folder = f"{PROBABILITY_APPLE:.2f}-{num_agents}-{width}-{reward}"
    filepath = out_dir / folder / f"results_seed{seed}.npz"
    with np.load(filepath, allow_pickle=True) as data:
        rewards_by_agent = data["rewards_by_agent"].copy()
    return rewards_by_agent


# ------------- value computation -----------------

def compute_value(reward_by_agent: np.ndarray, trajectory_length: int | None = None) -> np.ndarray:
    num_agents, T_total = reward_by_agent.shape
    T_eff = T_total if trajectory_length is None else min(int(trajectory_length), T_total)
    discounts = DISCOUNT_FACTOR ** np.arange(T_eff)
    returns = (reward_by_agent[:, :T_eff] * discounts).sum(axis=1)
    return returns


# ------------- closed-form value -----------------

def theoretical_value(
        r_pick: float,
        r_other: float,
        p_apple: float,
        num_agents: int,
        gamma: float,
        state: StateType,
        agent_id: int
) -> float:
    """
    Closed-form value V^{(i)} for a single agent, using

        V^{(i)} = E[R_t^{(i)}] / (1 - gamma)

    where
        E[R_t^{(i)}] = r_pick * (1 / N) * P + r_other * (1 - 1 / N) * P.
    """
    expected_reward_future = (
            r_pick * (1.0 / num_agents) * p_apple
            + r_other * (1.0 - 1.0 / num_agents) * p_apple
    )
    future_value = (gamma * expected_reward_future) / (1.0 - gamma)
    if state == "none_on_apples":
        return future_value
    elif state == "agent_on_apple" and agent_id == ACTOR_ID:
        return r_pick + future_value
    else:
        return r_other + future_value


# ------------- plotting -----------------

def plot_distributions(
        values: np.ndarray,
        kind: str,
        r_pick: float,
        r_other: float,
        state
) -> None:
    """
    values: shape (NUM_AGENTS, SEEDS)
    kind: 'MC' or 'IID' (used in titles / filenames)
    """
    exp_name = f"{kind}-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{r_pick}"
    plots_dir = data_dir / "plots" / state / kind / exp_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    for agent_id in range(NUM_AGENTS):
        v_theory = theoretical_value(
            r_pick=r_pick,
            r_other=r_other,
            p_apple=PROBABILITY_APPLE,
            num_agents=NUM_AGENTS,
            gamma=DISCOUNT_FACTOR,
            state=state,
            agent_id=agent_id
        )
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


def load_returns_for_experiment(
        kind: str,
        reward: float,
        state: StateType,
        trajectory_length: int | None = None,
) -> np.ndarray:
    folder = f"{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{reward}"
    returns = np.zeros((NUM_AGENTS, SEEDS), dtype=float)

    for seed in range(SEEDS):
        filepath = data_dir / state / kind / folder / f"results_seed{seed}.npz"
        with np.load(filepath, allow_pickle=True) as data:
            rewards_by_agent = data["rewards_by_agent"].copy()

        returns[:, seed] = compute_value(rewards_by_agent, trajectory_length=trajectory_length)

    return returns


def plot_mean0_minus_meani_from_agent_means(
        rewards,
        kind: str,                     # "monte-carlo" or "iid"
        baseline_agent: int = 0,
        dpi: int = 300,
        fmt: str = "png",
) -> None:
    state = "agent_on_apple"
    rewards_sorted = sorted(rewards)
    other_agents = [i for i in range(NUM_AGENTS) if i != baseline_agent]

    mean_by_agent = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)
    std_by_agent = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)

    diff_of_means = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)
    diff_err = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)

    for r_idx, r in enumerate(rewards_sorted):
        returns = load_returns_for_experiment(kind, r, state)  # (NUM_AGENTS, num_seeds)

        # mean/std of VALUES for each agent (across seeds) [web:193]
        mean_by_agent[:, r_idx] = returns.mean(axis=1)
        std_by_agent[:, r_idx] = returns.std(axis=1, ddof=0)  # [web:193]

        for i in other_agents:
            # "literally subtract the means"
            diff_of_means[i, r_idx] = mean_by_agent[baseline_agent, r_idx] - mean_by_agent[i, r_idx]

            # error bar from the two stds (simple propagation-style) [web:193]
            diff_err[i, r_idx] = np.sqrt(std_by_agent[baseline_agent, r_idx]**2 + std_by_agent[i, r_idx]**2)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]

    x_base = np.arange(len(rewards_sorted))
    offsets = np.linspace(-0.05, 0.05, len(other_agents))

    for j, i in enumerate(other_agents):
        x = x_base + offsets[j]
        ax.errorbar(
            x,
            diff_of_means[i],
            yerr=diff_err[i],
            fmt="o",
            linestyle="none",
            capsize=3,
            color=colors[i % len(colors)],
            ecolor=colors[i % len(colors)],
            label=f"mean(V{baseline_agent}) − mean(V{i})",
            alpha=0.9,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{r:.2f}" for r in rewards_sorted])
    ax.set_xlabel("Reward (r_picker)")
    ax.set_ylabel("Difference of means")
    ax.set_title(f"{state}: mean(V0) − mean(Vi) ({kind})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    plots_root = data_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    out_path = plots_root / f"diff_of_means_V0_minus_Vi-{kind}-{state}-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}.{fmt}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_mc_prefix_bias_vs_theory(
        trajectory_lengths: Sequence[int],
        *,
        rewards: Sequence[float],
        state: StateType,
        kind: str = "monte-carlo",
        dpi: int = 300,
        fmt: str = "png",
) -> None:
    Ts = np.array(sorted(set(int(t) for t in trajectory_lengths)), dtype=int)

    plots_root = data_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    # Which figures to make (label -> which agents appear in that figure)
    if state == "none_on_apples":
        plot_specs = [("agents0-4", [0, 1, 2, 3])]
    elif state == "agent_on_apple":
        plot_specs = [("agent0", [0]), ("agents1-3", [1, 2, 3])]
    else:
        raise ValueError(f"Unexpected state: {state}")

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]

    for r_pick in rewards:
        r_other = (1.0 - r_pick) / (NUM_AGENTS - 1)

        for fig_label, agent_ids in plot_specs:
            fig, ax = plt.subplots(figsize=(7, 4))

            # --- x-offsets so agent series don't overlap at same T ---
            x_base = Ts.astype(float)
            base_offset = 0.08 * (Ts.max() - Ts.min() + 1) / max(len(Ts), 1)  # heuristic scale
            offsets = np.linspace(-base_offset, base_offset, len(agent_ids)) if len(agent_ids) > 1 else np.array([0.0])  # [web:63]

            for j_agent, agent_id in enumerate(agent_ids):
                v_theory = theoretical_value(
                    r_pick=r_pick,
                    r_other=r_other,
                    p_apple=PROBABILITY_APPLE,
                    num_agents=NUM_AGENTS,
                    gamma=DISCOUNT_FACTOR,
                    state=state,
                    agent_id=agent_id,
                )

                diffs = np.zeros(len(Ts), dtype=float)
                errs = np.zeros(len(Ts), dtype=float)

                for jT, T in enumerate(Ts):
                    vals = np.zeros(SEEDS, dtype=float)

                    for seed in range(SEEDS):
                        rewards_by_agent = load_results(
                            kind=kind,
                            seed=seed,
                            num_agents=NUM_AGENTS,
                            width=W,
                            reward=r_pick,
                            state=state,
                        )

                        T_eff = min(T, rewards_by_agent.shape[1])
                        discounts = DISCOUNT_FACTOR ** np.arange(T_eff)
                        returns_by_agent = (rewards_by_agent[:, :T_eff] * discounts).sum(axis=1)
                        vals[seed] = returns_by_agent[agent_id]

                    mean = vals.mean()
                    std = vals.std(ddof=0)  # NumPy default ddof=0 [web:24]

                    diffs[jT] = v_theory - mean
                    errs[jT] = std

                x_positions = x_base + offsets[j_agent]
                c = colors[agent_id % len(colors)]

                ax.errorbar(
                    x_positions,
                    diffs,
                    yerr=errs,
                    fmt="o",
                    markersize=6,
                    color=c,
                    ecolor=c,
                    elinewidth=1.5,
                    capsize=3,
                    linestyle="none",
                    label=f"Agent {agent_id}",
                    alpha=0.9,
                )  # errorbar supports custom x and yerr [web:32]

            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_xlabel("Trajectory length (prefix T)")
            ax.set_ylabel("Theory − empirical mean return")
            ax.set_title(f"MC hypothesis check ({state}, {fig_label}), r_pick={r_pick}")
            ax.grid(True, axis="y", alpha=0.3)
            ax.grid(False, axis="x")
            ax.legend(title="Agent", frameon=False, ncols=min(len(agent_ids), 5))  # ncols supported [web:44]
            fig.tight_layout()

            out_path = plots_root / f"mc_prefix_bias_vs_theory-{state}-{fig_label}-r{r_pick}.{fmt}"
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)  # close figures in loops [web:51]


def plot_return_std_vs_T(
        trajectory_lengths: Sequence[int],
        *,
        rewards: Sequence[float],
        state: StateType,
        kind: str = "monte-carlo",
        dpi: int = 300,
        fmt: str = "png",
) -> None:
    Ts = np.array(sorted(set(int(t) for t in trajectory_lengths)), dtype=int)

    plots_root = data_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    if state == "none_on_apples":
        plot_specs = [("agents0-4", [0, 1, 2, 3])]   # adjust if you truly have 5 agents
    elif state == "agent_on_apple":
        plot_specs = [("agent0", [0]), ("agents1-3", [1, 2, 3])]
    else:
        raise ValueError(f"Unexpected state: {state}")

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]

    for r_pick in rewards:
        for fig_label, agent_ids in plot_specs:
            fig, ax = plt.subplots(figsize=(7, 4))

            # x-offsets so agent series don't overlap at same T
            x_base = Ts.astype(float)
            base_offset = 0.08 * (Ts.max() - Ts.min() + 1) / max(len(Ts), 1)
            offsets = (
                np.linspace(-base_offset, base_offset, len(agent_ids))
                if len(agent_ids) > 1 else np.array([0.0])
            )

            for j_agent, agent_id in enumerate(agent_ids):
                std_by_T = np.zeros(len(Ts), dtype=float)

                for jT, T in enumerate(Ts):
                    vals = np.zeros(SEEDS, dtype=float)

                    for seed in range(SEEDS):
                        rewards_by_agent = load_results(
                            kind=kind,
                            seed=seed,
                            num_agents=NUM_AGENTS,
                            width=W,
                            reward=r_pick,
                            state=state,
                        )

                        T_eff = min(T, rewards_by_agent.shape[1])
                        discounts = DISCOUNT_FACTOR ** np.arange(T_eff)
                        returns_by_agent = (rewards_by_agent[:, :T_eff] * discounts).sum(axis=1)
                        vals[seed] = returns_by_agent[agent_id]

                    std_by_T[jT] = vals.std(ddof=0)  # ddof default is 0 [web:27]

                x_positions = x_base + offsets[j_agent]
                c = colors[agent_id % len(colors)]

                ax.plot(
                    x_positions,
                    std_by_T,
                    marker="o",
                    linestyle="none",
                    markersize=6,
                    color=c,
                    label=f"Agent {agent_id}",
                    alpha=0.9,
                )

            ax.set_xlabel("Trajectory length (prefix T)")
            ax.set_ylabel("Std. dev. of return across seeds")
            ax.set_title(f"Return std. dev. vs T ({kind}, {state}, {fig_label}), r_pick={r_pick}")
            ax.grid(True, axis="y", alpha=0.3)
            ax.grid(False, axis="x")
            ax.legend(title="Agent", frameon=False, ncols=min(len(agent_ids), 5))
            fig.tight_layout()

            out_path = plots_root / f"return_std_vs_T-{kind}-{state}-{fig_label}-r{r_pick}.{fmt}"
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)


def compare_by_reward(
        rewards: Sequence[float],
        *,
        gen_a: ValueGenerator,
        gen_b: ValueGenerator,
        label_a: str,
        label_b: str,
        states: Sequence[State],
        split_state: State = "agent_on_apple",
) -> None:
    rewards_sorted = sorted(rewards)

    for state in states:
        diff_means = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)
        diff_stds = np.zeros((NUM_AGENTS, len(rewards_sorted)), dtype=float)

        for r_idx, r in enumerate(rewards_sorted):
            a_mean, a_std = gen_a(r, state)
            b_mean, b_std = gen_b(r, state)

            diff_means[:, r_idx] = a_mean - b_mean
            diff_stds[:, r_idx] = np.sqrt(a_std**2 + b_std**2)

        def plot_agents(agent_ids: list[int], suffix: str) -> None:
            fig, ax = plt.subplots(figsize=(7, 4))  # returns (fig, ax) [web:148]
            colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]

            x_base = np.arange(len(rewards_sorted))
            base_offset = 0.05
            offsets = np.linspace(-base_offset, base_offset, len(agent_ids))

            for j, agent_id in enumerate(agent_ids):
                x_positions = x_base + offsets[j]
                ax.errorbar(  # yerr draws vertical errorbars [web:141]
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

            ax.set_xticks(x_base)
            ax.set_xticklabels([f"{r:.2f}" for r in rewards_sorted])
            ax.axhline(0.0, color="red", linestyle=":", linewidth=1.5)

            ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.grid(False, axis="x")

            ax.set_xlabel("Reward")
            ax.set_ylabel(f"{label_a} − {label_b} value")
            ax.set_title(f"Difference in value: {label_a} vs {label_b} ({state}, {suffix})")
            ax.legend(title="Agent", frameon=False, ncol=min(len(agent_ids), 4))

            fig.tight_layout()

            plots_root = data_dir / "plots"
            plots_root.mkdir(parents=True, exist_ok=True)
            exp_name = f"{label_a}_vs_{label_b}-{state}-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{suffix}"
            out_path = plots_root / f"{exp_name}.png"

            fig.savefig(out_path, dpi=300)  # dpi controls output resolution [web:146]
            plt.close(fig)

        if state == split_state:
            plot_agents([0], "agent0")
            plot_agents(list(range(1, NUM_AGENTS)), "agents1-plus")
        else:
            plot_agents(list(range(NUM_AGENTS)), "all_agents")


def empirical_generator(kind: str, *, trajectory_length: int | None) -> ValueGenerator:
    def gen(r: float, state: State) -> tuple[np.ndarray, np.ndarray]:
        returns = load_returns_for_experiment(kind, r, state, trajectory_length=trajectory_length)
        mean = returns.mean(axis=1)
        std = returns.std(axis=1, ddof=0)
        return mean, std
    return gen


def theoretical_generator(*, gamma: float) -> ValueGenerator:
    def gen(r_pick: float, state: State) -> tuple[np.ndarray, np.ndarray]:
        r_other = (1.0 - r_pick) / (NUM_AGENTS - 1)  # adjust if your theory uses a different r_other
        mean = np.array(
            [
                theoretical_value(
                    r_pick=r_pick,
                    r_other=r_other,
                    p_apple=PROBABILITY_APPLE,
                    num_agents=NUM_AGENTS,
                    gamma=gamma,
                    state=state,
                    agent_id=i,
                )
                for i in range(NUM_AGENTS)
            ],
            dtype=float,
        )
        std = np.zeros(NUM_AGENTS, dtype=float)  # deterministic “no sampling noise”
        return mean, std
    return gen


# ------------- driver -----------------

def process(reward, trajectory_length: int | None = None):
    for state in ["none_on_apples", "agent_on_apple"]:
        mc_vals = np.zeros((NUM_AGENTS, SEEDS), dtype=float)
        for seed in range(SEEDS):
            mc_rewards = load_results("monte-carlo", seed, NUM_AGENTS, W, reward, state)
            mc_vals[:, seed] = compute_value(mc_rewards, trajectory_length=trajectory_length)

        iid_vals = np.zeros((NUM_AGENTS, SEEDS), dtype=float)
        for seed in range(SEEDS):
            iid_rewards = load_results("iid", seed, NUM_AGENTS, W, reward, state)
            iid_vals[:, seed] = compute_value(iid_rewards, trajectory_length=trajectory_length)

        r_pick = reward
        r_other = (1 - r_pick) / (NUM_AGENTS - 1)

        plot_distributions(mc_vals, kind="MC", r_pick=r_pick, r_other=r_other, state=state)
        plot_distributions(iid_vals, kind="IID", r_pick=r_pick, r_other=r_other, state=state)


def compare(rewards, trajectory_length: int | None = None):
    STATES = ["none_on_apples", "agent_on_apple"]
    mc_gen = empirical_generator("monte-carlo", trajectory_length=trajectory_length)
    iid_gen = empirical_generator("iid", trajectory_length=trajectory_length)
    th_gen = theoretical_generator(gamma=DISCOUNT_FACTOR)

    compare_by_reward(rewards, gen_a=mc_gen, gen_b=iid_gen, label_a="MC", label_b="IID",
                      states=STATES, split_state="agent_on_apple")

    compare_by_reward(rewards, gen_a=mc_gen, gen_b=th_gen, label_a="MC", label_b="Theory",
                      states=STATES, split_state="agent_on_apple")

    plot_mean0_minus_meani_from_agent_means(rewards, kind="monte-carlo")
    plot_mean0_minus_meani_from_agent_means(rewards, kind="iid")


def get_path_info():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", required=True, help="Reward.", type=int)
    parser.add_argument(
        "--trajectory_length",
        type=int,
        default=None,
        help="Optional prefix length T to truncate each trajectory before discounting (default: full horizon).",
    )  # default=None when omitted [web:68]

    args = parser.parse_args()
    process(args.reward, trajectory_length=args.trajectory_length)
    compare([args.reward], trajectory_length=args.trajectory_length)


if __name__ == "__main__":
    # main()
    plot_mc_prefix_bias_vs_theory(
        trajectory_lengths=[10, 20, 50, 100],
        rewards=[-1],
        state="none_on_apples"
    )

    plot_return_std_vs_T(
        trajectory_lengths=[10, 20, 50, 100],
        rewards=[-1],
        state="none_on_apples",
        kind="monte-carlo",
    )

    # compare([-1])
