import argparse
import sys
import multiprocessing as mp
from debug.code.config import EnvironmentConfig, ExperimentConfig, \
    TrainingConfig, data_dir
from debug.code.helpers import set_all_seeds
from debug.code.supervised import Learning
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42069, help="Random seed.")
    parser.add_argument(
        "--timesteps", type=int, default=1000000, help="Number of timesteps."
    )
    parser.add_argument(
        "--picker_r", type=int, default=-1, help="Picker reward."
    )

    parser.add_argument(
        "--num_eval_states", type=int, default=-1, help="Picker reward."
    )

    parser.add_argument(
        "--alpha", type=float, nargs="+", help="Learning rate for critic."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden layer size of critic network.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of layers for critic network."
    )
    parser.add_argument(
        "--supervised",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--reward_learning",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--input_dim", type=int, default=3
    )
    parser.add_argument(
        "--library",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--monte_carlo",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1000
    )
    parser.add_argument(
        "--variance", type=float, default=0
    )
    return parser.parse_args(args)


def set_config(args, i):
    return TrainingConfig(
        alpha=args.alpha[i],
        timesteps=args.timesteps,
        hidden_dimensions=args.hidden_dim,
        num_layers=args.num_layers,
        num_eval_states=args.num_eval_states,
        picker_r=args.picker_r,
        seed=args.seed,
        supervised=args.supervised,
        reward_learning=args.reward_learning,
        input_dim=args.input_dim,
        use_library=args.library,
        monte_carlo=args.monte_carlo,
        num_seeds=args.num_seeds,
        variance=args.variance
    )


def run_one(alpha, base_args, run_idx):
    # base_args: the argparse Namespace is picklable, but to be safe you can pass dict(vars(args))
    seed = int(base_args["seed"]) + int(run_idx)

    train_config = TrainingConfig(
        alpha=float(alpha),
        timesteps=int(base_args["timesteps"]),
        hidden_dimensions=int(base_args["hidden_dim"]),
        num_layers=int(base_args["num_layers"]),
        num_eval_states=int(base_args["num_eval_states"]),
        picker_r=int(base_args["picker_r"]),
        seed=seed,
        supervised=bool(base_args["supervised"]),
        reward_learning=bool(base_args["reward_learning"]),
        input_dim=int(base_args["input_dim"]),
        use_library=bool(base_args["library"]),
        monte_carlo=bool(base_args["monte_carlo"]),
        num_seeds=int(base_args["num_seeds"]),
        variance=float(base_args["variance"]),
    )

    exp_config = ExperimentConfig(env_config=EnvironmentConfig(), train_config=train_config)

    set_all_seeds(seed=seed)
    algo = Learning(exp_config)
    history = algo.train()

    return {"lr": float(alpha), "seed": seed, "history": history}


def main(args):
    args = parse_args(args)
    base_args = vars(args)  # convert Namespace -> dict for safer pickling

    futures = []
    runs = []

    max_workers = min(len(args.alpha), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, a in enumerate(args.alpha):
            futures.append(ex.submit(run_one, a, base_args, i))

        for f in as_completed(futures):
            runs.append(f.result())

    # stable legend order by lr
    runs.sort(key=lambda d: d["lr"])
    plot_multi_run_mae(runs, args)


def plot_multi_run_mae(runs, args):
    """
    runs: list of dicts like:
      {"lr": 1e-3, "history": eval_history}
    where eval_history is a list of {"step": int, "mae_pct_overall": float, ...}
    """
    plt.figure(figsize=(9, 5))

    for run in runs:
        lr = run["lr"]
        hist = run["history"]
        steps = [h["step"] for h in hist]
        maes = [h["mae_pct_overall"] for h in hist]
        plt.plot(steps, maes, marker="o", linewidth=2, label=f"lr={lr:g}")

    plt.xlabel("Training step (evaluation point)")
    plt.ylabel("MAE % of true value")
    plt.title("Overall MAE% over training (each run)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    path = data_dir / "plots" / "supervised_w_variance" / f"res_{args.variance}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(sys.argv[1:])
