import argparse
from typing import Dict, Sequence, Tuple
import numpy as np
from collections.abc import Callable, Sequence
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from scipy.stats import norm

from debug.code.core.enums import (
    DISCOUNT_FACTOR,
    PROBABILITY_APPLE,
    data_dir,
    SEEDS,
)
from debug.code.core.config import load_config
from debug.code.training.monte_carlo import StateType

_DEFAULT_CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
CFG_NUM_AGENTS = int(_DEFAULT_CONFIG.env.num_agents)
CFG_W = int(_DEFAULT_CONFIG.env.width)
CFG_L = int(_DEFAULT_CONFIG.env.length)

ACTOR_ID = 0
State = str  # or your Enum type if you have one
ValueGenerator = Callable[[float, State], tuple[np.ndarray, np.ndarray]]
# returns: (mean_by_agent, std_by_agent), each shape (CFG_NUM_AGENTS,)



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
    elif state == "Z1" and agent_id == ACTOR_ID:
        return 4.48
    elif state == "Z1":
        return 5.14
    elif state == "Y11" and agent_id == ACTOR_ID:
        return 3.93
    elif state == "Y11":
        return 5.59
    elif state == "Y10" and agent_id == ACTOR_ID:
        return 4.93
    elif state == "Y10":
        return 4.93
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
    values: shape (CFG_NUM_AGENTS, SEEDS)
    kind: 'MC' or 'IID' (used in titles / filenames)
    """
    exp_name = f"{kind}-{PROBABILITY_APPLE:.2f}-{CFG_NUM_AGENTS}-{CFG_W}-{r_pick}"
    plots_dir = data_dir / "plots" / state / kind / exp_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    for agent_id in range(CFG_NUM_AGENTS):
        v_theory = theoretical_value(
            r_pick=r_pick,
            r_other=r_other,
            p_apple=PROBABILITY_APPLE,
            num_agents=CFG_NUM_AGENTS,
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
            f"{kind} – Agent {agent_id}\n"
            f"mean={mean:.2f}, std={std:.2f}, theory={v_theory:.2f}"
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
    folder = f"{PROBABILITY_APPLE:.2f}-{CFG_NUM_AGENTS}-{CFG_W}-{reward}"
    returns = np.zeros((CFG_NUM_AGENTS, SEEDS), dtype=float)

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
    other_agents = [i for i in range(CFG_NUM_AGENTS) if i != baseline_agent]

    # Store |mean difference| and asymmetric 95% CI errors (lower/upper)
    abs_mean_diff = np.full((CFG_NUM_AGENTS, len(rewards_sorted)), np.nan, dtype=float)
    abs_err_low   = np.full((CFG_NUM_AGENTS, len(rewards_sorted)), np.nan, dtype=float)
    abs_err_high  = np.full((CFG_NUM_AGENTS, len(rewards_sorted)), np.nan, dtype=float)

    for r_idx, r in enumerate(rewards_sorted):
        returns = load_returns_for_experiment(kind, r, state)  # (CFG_NUM_AGENTS, num_seeds)
        n = returns.shape[1]
        if n < 2:
            raise ValueError(f"Need at least 2 seeds to form a CI; got n={n} for r={r}.")

        tcrit = stats.t.ppf(0.995, df=n - 1)  # 99% two-sided
        base = returns[baseline_agent, :]     # (n,)

        for i in other_agents:
            d = base - returns[i, :]          # paired per-seed diffs, shape (n,)
            md = d.mean()
            sd = d.std(ddof=1)
            se = sd / np.sqrt(n)
            E = tcrit * se                    # half-width for md CI

            lo, hi = md - E, md + E           # CI for md

            y = abs(md)

            # Convert CI for md -> CI for |md|, then turn into asymmetric yerr.
            if lo <= 0.0 <= hi:
                abs_lo = 0.0
                abs_hi = max(abs(lo), abs(hi))
            else:
                abs_lo = min(abs(lo), abs(hi))
                abs_hi = max(abs(lo), abs(hi))

            abs_mean_diff[i, r_idx] = y
            abs_err_low[i, r_idx]   = y - abs_lo
            abs_err_high[i, r_idx]  = abs_hi - y

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]

    x_base = np.arange(len(rewards_sorted))
    offsets = np.linspace(-0.05, 0.05, len(other_agents))

    for j, i in enumerate(other_agents):
        x = x_base + offsets[j]
        yerr_asym = np.vstack([abs_err_low[i], abs_err_high[i]])  # shape (2, N) [web:46]
        ax.errorbar(
            x,
            abs_mean_diff[i],
            yerr=yerr_asym,
            fmt="o",
            linestyle="none",
            capsize=3,
            color=colors[i % len(colors)],
            ecolor=colors[i % len(colors)],
            label=f"Agent {i}",
            alpha=0.9,
        )

    # Reference lines
    ax.axhline(1.66667, linestyle="--", color="red", linewidth=1.5)
    ax.axhline(7,     linestyle="--", color="red", linewidth=1.5)

    ax.axhline(0.0, color="red", linewidth=1, alpha=0.6)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{r:.2f}" for r in rewards_sorted])
    ax.set_xlabel("Reward (r_picker)")
    ax.set_ylabel(f"Absolute mean difference: |V{baseline_agent} − Vi|")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    plots_root = data_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    out_path = plots_root / (
        f"abs_diff_of_means_V{baseline_agent}_minus_Vi-{kind}-{state}-"
        f"{PROBABILITY_APPLE:.2f}-{CFG_NUM_AGENTS}-{CFG_W}.{fmt}"
    )
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
    Ts = np.array(sorted(set(int(t) for t in trajectory_lengths)), dtype=float)

    plots_root = data_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    if state == "none_on_apples":
        plot_specs = [("agents0-4", [0, 1, 2, 3])]
    elif state == "agent_on_apple":
        plot_specs = [("agent0", [0]), ("agents1-3", [1, 2, 3])]
    else:
        raise ValueError(f"Unexpected state: {state}")

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]

    for r_pick in rewards:
        r_other = (1.0 - r_pick) / (CFG_NUM_AGENTS - 1)

        for fig_label, agent_ids in plot_specs:
            fig, ax = plt.subplots(figsize=(7, 4))

            # --- log-safe multiplicative jitter (prevents overlap) ---
            if len(agent_ids) > 1:
                jitters = np.linspace(-0.08, 0.08, len(agent_ids))  # ±4%
            else:
                jitters = np.array([0.0])

            for j_agent, agent_id in enumerate(agent_ids):
                v_theory = theoretical_value(
                    r_pick=r_pick,
                    r_other=r_other,
                    p_apple=PROBABILITY_APPLE,
                    num_agents=CFG_NUM_AGENTS,
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
                            num_agents=CFG_NUM_AGENTS,
                            width=CFG_W,
                            reward=r_pick,
                            state=state,
                        )

                        T_eff = min(int(T), rewards_by_agent.shape[1])
                        discounts = DISCOUNT_FACTOR ** np.arange(T_eff)
                        returns_by_agent = (rewards_by_agent[:, :T_eff] * discounts).sum(axis=1)
                        vals[seed] = returns_by_agent[agent_id]

                    mean = vals.mean()
                    std = vals.std(ddof=1)          # unbiased sample std
                    sem = std / np.sqrt(SEEDS)     # standard error of mean

                    diffs[jT] = v_theory - mean
                    errs[jT] = 2.56 * sem          # 95% CI

                x_positions = Ts * (1.0 + jitters[j_agent])
                c = colors[agent_id % len(colors)]

                ax.errorbar(
                    x_positions,
                    diffs,
                    yerr=errs,
                    fmt="o",
                    linewidth=1.2,
                    markersize=5,
                    color=c,
                    ecolor=c,
                    elinewidth=1.2,
                    capsize=3,
                    label=f"Agent {agent_id}",
                )

            ax.set_xscale("log")
            ax.axhline(0.0, color="red", linewidth=1)

            ax.set_xlabel("Trajectory length")
            ax.set_ylabel("Theory − MC mean (bias)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.grid(False, axis="x")
            ax.legend(title="Agent", frameon=False, ncols=min(len(agent_ids), 5))

            fig.tight_layout()
            out_path = plots_root / f"mc_prefix_bias_vs_theory-{state}-{fig_label}-r{r_pick}.{fmt}"
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)


def plot_distributions_prefix(
        values: np.ndarray,          # shape (CFG_NUM_AGENTS, SEEDS)
        *,
        kind: str,
        r_pick: float,
        r_other: float,
        state: StateType,
        trajectory_length: int,
        agent_ids: Sequence[int] = (0, 3),
        dpi: int = 300,
) -> Path:
    """
    Same plot as plot_distributions, but saved under a per-prefix-length folder.
    Returns the directory containing agent PNGs for this T.
    """
    exp_name = f"{kind}-{PROBABILITY_APPLE:.2f}-{CFG_NUM_AGENTS}-{CFG_W}-{r_pick}-T{trajectory_length}"
    plots_dir = data_dir / "plots" / state / kind / exp_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    for agent_id in agent_ids:
        v_theory = theoretical_value(
            r_pick=r_pick,
            r_other=r_other,
            p_apple=PROBABILITY_APPLE,
            num_agents=CFG_NUM_AGENTS,
            gamma=DISCOUNT_FACTOR,
            state=state,
            agent_id=agent_id,
        )

        vals = values[agent_id]
        mean = vals.mean()
        std = vals.std(ddof=0)

        plt.figure()
        plt.hist(vals, bins=30, alpha=0.7, density=True, color="C0")

        x = np.linspace(vals.min(), vals.max(), 200)
        pdf = norm.pdf(x, loc=mean, scale=std)
        plt.plot(x, pdf, "k-", linewidth=2)

        plt.axvline(v_theory, color="red", linestyle="--", linewidth=2)

        plt.title(
            f"{kind} – Agent {agent_id} returns (T={trajectory_length})\n"
            f"mean={mean:.3f}, std={std:.3f}, theory={v_theory:.3f}"
        )
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.tight_layout()

        fname = plots_dir / f"agent_{agent_id}_returns.png"
        plt.savefig(fname, dpi=dpi)
        plt.close()

    return plots_dir


def stitch_prefix_pngs_one_row(
        plots_dirs_by_T: dict[int, Path],
        *,
        agent_id: int,
        out_path: Path,
        dpi: int = 300,
) -> None:
    Ts = sorted(plots_dirs_by_T.keys())
    fig, axs = plt.subplots(1, len(Ts), figsize=(4.0 * len(Ts), 3.0))

    if len(Ts) == 1:
        axs = [axs]

    for ax, T in zip(axs, Ts):
        img = plt.imread(plots_dirs_by_T[T] / f"agent_{agent_id}_returns.png")  # read PNG [web:19]
        ax.imshow(img)                                                         # show array [web:33]
        ax.set_axis_off()
        ax.set_title(f"T={T}", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def stitch_prefix_pngs_one_row(
        plots_dirs_by_T: dict[int, Path],
        *,
        agent_id: int,
        out_path: Path,
        dpi: int = 300,
) -> None:
    Ts = sorted(plots_dirs_by_T.keys())
    fig, axs = plt.subplots(1, len(Ts), figsize=(4.0 * len(Ts), 3.0))

    if len(Ts) == 1:
        axs = [axs]

    for ax, T in zip(axs, Ts):
        img = plt.imread(plots_dirs_by_T[T] / f"agent_{agent_id}_returns.png")  # read PNG [web:19]
        ax.imshow(img)                                                         # show array [web:33]
        ax.set_axis_off()
        ax.set_title(f"T={T}", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_prefix_distributions_and_stitch(
        trajectory_lengths: Sequence[int],
        *,
        kind: str = "monte-carlo",
        reward: float = -1.0,
        state: StateType = "agent_on_apple",
        agents: Sequence[int] = (0, 3),
        dpi: int = 300,
        fmt: str = "png",
) -> dict[int, Path]:
    Ts = sorted(set(int(t) for t in trajectory_lengths))
    if len(Ts) == 0:
        raise ValueError("trajectory_lengths is empty.")

    r_pick = reward
    r_other = (1.0 - r_pick) / (CFG_NUM_AGENTS - 1)

    # 1) For each T, compute returns across seeds and save per-agent distribution PNGs
    plots_dirs_by_T: dict[int, Path] = {}
    for T in Ts:
        returns = load_returns_for_experiment(kind, r_pick, state, trajectory_length=T)
        plots_dir = plot_distributions_prefix(
            returns,
            kind=("MC" if kind == "monte-carlo" else "IID"),
            r_pick=r_pick,
            r_other=r_other,
            state=state,
            trajectory_length=T,
            agent_ids=agents,
            dpi=dpi,
        )
        plots_dirs_by_T[T] = plots_dir

    # 2) Stitch horizontally for each requested agent
    out_paths: dict[int, Path] = {}
    out_root = data_dir / "plots" / state / kind
    for agent_id in agents:
        out_path = out_root / f"stitched_prefix_distributions-agent{agent_id}-r{r_pick}.{fmt}"
        stitch_prefix_pngs_one_row(
            plots_dirs_by_T,
            agent_id=agent_id,
            out_path=out_path,
            dpi=dpi,
        )
        out_paths[agent_id] = out_path

    return out_paths


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
                            num_agents=CFG_NUM_AGENTS,
                            width=CFG_W,
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
                    linestyle="--",      # dotted straight lines
                    linewidth=1.5,
                    markersize=6,
                    color=c,
                    label=f"Agent {agent_id}",
                    alpha=0.9,
                )

            ax.set_xlabel("Trajectory length (prefix T)")
            ax.set_ylabel("Std. dev. of return across seeds")
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

    N = SEEDS  # number of independent samples used by gen_a / gen_b

    for state in states:
        diff_means = np.zeros((CFG_NUM_AGENTS, len(rewards_sorted)), dtype=float)
        diff_cis = np.zeros((CFG_NUM_AGENTS, len(rewards_sorted)), dtype=float)  # 95% CI half-width

        for r_idx, r in enumerate(rewards_sorted):
            a_mean, a_std = gen_a(r, state)
            b_mean, b_std = gen_b(r, state)

            diff_means[:, r_idx] = a_mean - b_mean

            # SE of mean difference assuming independence
            se_diff = np.sqrt((a_std**2 + b_std**2) / N)

            # 95% confidence interval
            diff_cis[:, r_idx] = 2.56 * se_diff

        def plot_agents(agent_ids: list[int], suffix: str) -> None:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]

            x_base = np.arange(len(rewards_sorted))
            base_offset = 0.05
            offsets = np.linspace(-base_offset, base_offset, len(agent_ids))

            for j, agent_id in enumerate(agent_ids):
                x_positions = x_base + offsets[j]
                c = colors[agent_id % len(colors)]

                ax.errorbar(
                    x_positions,
                    diff_means[agent_id],
                    yerr=diff_cis[agent_id],   # <-- now true 95% CI
                    fmt="o",
                    markersize=6,
                    color=c,
                    ecolor=c,
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
            ax.legend(title="Agent", frameon=False, ncol=min(len(agent_ids), 4))

            fig.tight_layout()

            plots_root = data_dir / "plots"
            plots_root.mkdir(parents=True, exist_ok=True)
            exp_name = f"{label_a}_vs_{label_b}-{state}-{PROBABILITY_APPLE:.2f}-{CFG_NUM_AGENTS}-{CFG_W}-{suffix}"
            out_path = plots_root / f"{exp_name}.png"

            fig.savefig(out_path, dpi=300)
            plt.close(fig)

        if state == split_state:
            plot_agents([0], "agent0")
            plot_agents(list(range(1, CFG_NUM_AGENTS)), "agents1-plus")
        else:
            plot_agents(list(range(CFG_NUM_AGENTS)), "all_agents")


def empirical_generator(kind: str, *, trajectory_length: int | None) -> ValueGenerator:
    def gen(r: float, state: State) -> tuple[np.ndarray, np.ndarray]:
        returns = load_returns_for_experiment(kind, r, state, trajectory_length=trajectory_length)
        mean = returns.mean(axis=1)
        std = returns.std(axis=1, ddof=0)
        return mean, std
    return gen


def theoretical_generator(*, gamma: float) -> ValueGenerator:
    def gen(r_pick: float, state: State) -> tuple[np.ndarray, np.ndarray]:
        r_other = (1.0 - r_pick) / (CFG_NUM_AGENTS - 1)  # adjust if your theory uses a different r_other
        mean = np.array(
            [
                theoretical_value(
                    r_pick=r_pick,
                    r_other=r_other,
                    p_apple=PROBABILITY_APPLE,
                    num_agents=CFG_NUM_AGENTS,
                    gamma=gamma,
                    state=state,
                    agent_id=i,
                )
                for i in range(CFG_NUM_AGENTS)
            ],
            dtype=float,
        )
        std = np.zeros(CFG_NUM_AGENTS, dtype=float)  # deterministic “no sampling noise”
        return mean, std
    return gen


def stitch_agent_pngs_one_row(plots_dir, *, out_path, agents=(0,1,2,3), dpi=300):
    fig, axs = plt.subplots(1, len(agents), figsize=(4.0 * len(agents), 3.0))  # 1xK grid [web:16]
    if len(agents) == 1:
        axs = [axs]

    for ax, agent_id in zip(axs, agents):
        img = plt.imread(plots_dir / f"agent_{agent_id}_returns.png")  # read PNG into array [web:32]
        ax.imshow(img)                                                # display image in axis [web:45]
        ax.set_axis_off()                                             # hide axes/ticks/frame [web:43]

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def stitch_grid_from_existing_pngs(
        kind: str,
        rewards,
        states,
        agents=(0,1,2,3),
        dpi=300,
):
    nrows = len(states) * len(rewards)
    ncols = len(agents)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.0*nrows))  # grid [web:16]

    row = 0
    for state in states:
        for r_pick in rewards:
            exp_name = f"{kind}-{PROBABILITY_APPLE:.2f}-{CFG_NUM_AGENTS}-{CFG_W}-{r_pick}"
            plots_dir = data_dir / "plots" / state / kind / exp_name

            for col, agent_id in enumerate(agents):
                ax = axs[row, col]
                img = plt.imread(plots_dir / f"agent_{agent_id}_returns.png")  # [web:32]
                ax.imshow(img)                                                # [web:45]
                ax.set_axis_off()                                             # [web:43]
            row += 1

    fig.tight_layout()
    out = data_dir / "plots" / f"returns_grid_from_pngs-{kind}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


# ------------- driver -----------------

def process(reward, trajectory_length: int | None = None):
    for state in ["Y11"]:
        mc_vals = np.zeros((CFG_NUM_AGENTS, SEEDS), dtype=float)
        for seed in range(SEEDS):
            mc_rewards = load_results("monte-carlo", seed, CFG_NUM_AGENTS, CFG_W, reward, state)
            mc_vals[:, seed] = compute_value(mc_rewards, trajectory_length=trajectory_length)

        # iid_vals = np.zeros((CFG_NUM_AGENTS, SEEDS), dtype=float)
        # for seed in range(SEEDS):
        #     iid_rewards = load_results("iid", seed, CFG_NUM_AGENTS, CFG_W, reward, state)
        #     iid_vals[:, seed] = compute_value(iid_rewards, trajectory_length=trajectory_length)

        r_pick = reward
        r_other = (1 - r_pick) / (CFG_NUM_AGENTS - 1)

        plot_distributions(mc_vals, kind="MC", r_pick=r_pick, r_other=r_other, state=state)
        # plot_distributions(iid_vals, kind="IID", r_pick=r_pick, r_other=r_other, state=state)


def compare(rewards, trajectory_length: int | None = None):
    STATES = ["none_on_apples", "agent_on_apple"]
    mc_gen = empirical_generator("monte-carlo", trajectory_length=trajectory_length)
    iid_gen = empirical_generator("iid", trajectory_length=trajectory_length)
    th_gen = theoretical_generator(gamma=DISCOUNT_FACTOR)

    compare_by_reward(rewards, gen_a=mc_gen, gen_b=iid_gen, label_a="MC", label_b="IID",
                      states=STATES, split_state="agent_on_apple")

    compare_by_reward(rewards, gen_a=mc_gen, gen_b=th_gen, label_a="MC", label_b="Theory",
                      states=STATES, split_state="agent_on_apple")

    # plot_mean0_minus_meani_from_agent_means(rewards, kind="monte-carlo")
    # plot_mean0_minus_meani_from_agent_means(rewards, kind="iid")


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

    t_length = 100000 if args.trajectory_length is None else args.trajectory_length
    process(args.reward, trajectory_length=t_length)
    # compare([args.reward], trajectory_length=t_length)


if __name__ == "__main__":
    process(-1, 1000)
    # stitch_grid_from_existing_pngs("MC", [-5, -1], ["Y11"])

    # out = plot_prefix_distributions_and_stitch(
    #     trajectory_lengths=[100, 500, 1000, 5000],
    #     kind="monte-carlo",
    #     reward=-1,
    #     state="agent_on_apple",
    #     agents=(0, 3),
    # )
    # print(out)  # {0: Path(...), 3: Path(...)}

    # main()
    # plot_mc_prefix_bias_vs_theory(
    #     trajectory_lengths=[1, 100, 500, 1000, 5000],
    #     rewards=[-5, -1, 5],
    #     state="agent_on_apple"
    # )
    # plot_mc_prefix_bias_vs_theory(
    #     trajectory_lengths=[1, 100, 500, 1000, 5000],
    #     rewards=[-5, -1, 5],
    #     state="none_on_apples"
    # )

    # plot_return_std_vs_T(
    #     trajectory_lengths=[1, 100, 500, 1000, 5000],
    #     rewards=[-5, -1, 5],
    #     state="agent_on_apple",
    #     kind="monte-carlo",
    # )

    # plot_return_std_vs_T(
    #     trajectory_lengths=[1, 100, 500, 1000, 5000],
    #     rewards=[-5, -1, 5],
    #     state="none_on_apples",
    #     kind="monte-carlo",
    # )


    # plot_mean0_minus_meani_from_agent_means(
    #     [-5, -1, 5],
    #     "monte-carlo"
    #     )

    # compare([-5, -1, 5], 5000)
