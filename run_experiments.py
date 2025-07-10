import argparse
import sys
import numpy as np
import random
import torch
from configs.config import ExperimentConfig, EnvironmentConfig, TrainingConfig
from value_function_learning.train_value_function import (
    CentralizedValueFunction, DecentralizedValueFunction
)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="Centralized", help="Algorithm.")
    parser.add_argument("--width", type=int, default=1, help="Width of the orchard.")
    parser.add_argument("--length", type=int, default=20, help="Length of the orchard.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--seed", type=int, default=42069, help="Random seed.")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of timesteps.")
    parser.add_argument("--apple_life", type=float, default=3, help="Apple mean lifetime.")
    parser.add_argument("--s_target", type=float, default=0.1, help="Expected number of apples spawned per agent per second.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--alpha", type=float, default=0.00125, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers.")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    env_config = EnvironmentConfig(
        s_target=args.s_target,
        apple_mean_lifetime=args.apple_life,
        length=args.length,
        width=args.width,
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        alpha=args.alpha,
        timesteps=args.timesteps,
        num_agents=args.num_agents,
        hidden_dimensions=args.hidden_dim,
        num_layers=args.num_layers,
    )

    exp_config = ExperimentConfig(
        env_config=env_config,
        train_config=train_config,
    )

    if args.algorithm == "Centralized":
        algo = CentralizedValueFunction(exp_config)
    else:
        algo = DecentralizedValueFunction(exp_config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    algo.run()


if __name__ == "__main__":
    main(sys.argv[1:])
