import argparse
import sys

from debug.code.config import EnvironmentConfig, ExperimentConfig, \
    TrainingConfig
from debug.code.helpers import set_all_seeds
from debug.code.supervised import Learning


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
        "--alpha", type=float, default=0.000275, help="Learning rate for critic."
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
    return parser.parse_args(args)


def set_config(args):
    return TrainingConfig(
        alpha=args.alpha,
        timesteps=args.timesteps,
        hidden_dimensions=args.hidden_dim,
        num_layers=args.num_layers,
        num_eval_states=args.num_eval_states,
        picker_r=args.picker_r,
        seed=args.seed,
        supervised=args.supervised,
        reward_learning=args.reward_learning,
        input_dim=args.input_dim
    )


def main(args):
    args = parse_args(args)
    train_config = set_config(args)
    exp_config = ExperimentConfig(
        env_config=EnvironmentConfig(), train_config=train_config
    )
    algo = Learning(exp_config)
    set_all_seeds(seed=args.seed)
    algo.train()


if __name__ == "__main__":
    main(sys.argv[1:])
