import argparse
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from debug.code.config import EnvironmentConfig, ExperimentConfig, \
    TrainingConfig, data_dir
from debug.code.helpers import set_all_seeds
from debug.code.supervised import Learning
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


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
        "--input_dim", type=int, default=0
    )
    parser.add_argument(
        "--forward",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--eligibility",
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
    parser.add_argument(
        "--lmda", type=float, default=0.5
    )
    parser.add_argument(
        "--schedule_lr",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--random_policy",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--top_k_num_apples", type=int, default=1
    )
    parser.add_argument(
        "--centralized",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--concat",
        action=argparse.BooleanOptionalAction,
        default=False
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
        forward=args.forward,
        eligibility=args.eligibility,
        monte_carlo=args.monte_carlo,
        num_seeds=args.num_seeds,
        variance=args.variance,
        schedule_lr=args.schedule_lr,
        lmda=args.lmda,
        random_policy=args.random_policy,
        top_k_num_apples=args.top_k_num_apples,
        centralized=args.centralized,
        concat=args.concat
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
        forward=bool(base_args["forward"]),
        eligibility=bool(base_args["eligibility"]),
        monte_carlo=bool(base_args["monte_carlo"]),
        num_seeds=int(base_args["num_seeds"]),
        variance=float(base_args["variance"]),
        schedule_lr=base_args["schedule_lr"],
        lmda=base_args["lmda"],
        random_policy=base_args["random_policy"],
        top_k_num_apples=base_args["top_k_num_apples"],
        centralized=base_args["centralized"],
        concat=base_args["concat"]
    )

    exp_config = ExperimentConfig(env_config=EnvironmentConfig(), train_config=train_config)

    set_all_seeds(seed=seed)
    algo = Learning(exp_config)
    history = algo.train()

    return {"lr": str(alpha), "seed": seed, "history": history}


def main(args):
    args = parse_args(args)
    base_args = vars(args)  # convert Namespace -> dict for safer pickling
    runs = []
    if args.supervised:
        futures = []
        max_workers = min(len(args.alpha), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for i, a in enumerate(args.alpha):
                futures.append(ex.submit(run_one, a, base_args, i))

            for f in as_completed(futures):
                runs.append(f.result())

        # stable legend order by lr
        runs.sort(key=lambda d: d["lr"])

        # if base_args["schedule_lr"] is False:
        #     base_args["schedule_lr"] = True
        #     res = run_one(args.alpha[0], base_args, len(args.alpha))
        #     res["lr"] = f"schedule_{res["lr"]}"
        #     runs.append(res)
    else:
        runs.append(run_one(args.alpha[0], base_args, len(args.alpha)))

    plot_multi_run_mae(runs, args, data_dir)

# def main(args):
#     args = parse_args(args)
#     all_histories = []
#     for i in range(len(args.alpha)):
#         train_config = set_config(args, i)
#         exp_config = ExperimentConfig(
#             env_config=EnvironmentConfig(), train_config=train_config
#         )
#         algo = Learning(exp_config)
#         set_all_seeds(seed=args.seed)
#         all_histories.append({"lr": exp_config.train_config.alpha, "history": algo.train()})
#
#     plot_multi_run_mae(all_histories, args)


def _next_available_path(path: Path) -> Path:
    """If path exists, return path with _2, _3, ... inserted before suffix."""
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def plot_multi_run_mae(runs, args, data_dir):
    plt.figure(figsize=(9, 5))

    ymax = 0.0
    for run in runs:
        lr = run["lr"]
        hist = run["history"]
        steps = [h["step"] for h in hist]
        maes = [h["mae_pct_overall"] for h in hist]
        if maes:
            ymax = max(ymax, max(maes))
        plt.plot(steps, maes, marker="o", linewidth=2, label=f"lr={lr}")

    ax = plt.gca()
    top = int(np.ceil(ymax / 10.0) * 10)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    plt.xlabel("Training step (evaluation point)")
    plt.ylabel("MAE % of true value")
    plt.title("Overall MAE% over training (each run)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = Path(data_dir) / "plots" / "supervised_w_variance" / f"res_{args.variance}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    path = _next_available_path(path)
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(sys.argv[1:])
