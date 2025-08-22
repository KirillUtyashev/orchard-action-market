import argparse
import sys
import numpy as np
import random
import torch

from actor_critic_beta import ActorCriticBeta
from actor_critic_following_rates import ActorCriticRates
from actor_critic_perfect_info import ActorCriticPerfect, \
    ActorCriticPerfectNoAdvantage
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
    parser.add_argument("--alpha", type=float, default=0.000275, help="Learning rate for critic.")
    parser.add_argument("--actor_alpha", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer size of critic network.")
    parser.add_argument("--hidden_dim_actor", type=int, default=64, help="Hidden layer size of actor network.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers for critic network.")
    parser.add_argument("--num_layers_actor", type=int, default=4, help="Number of layers for actor network.")
    parser.add_argument("--debug", type=bool, default=True, help="Debug.")
    parser.add_argument("--alt_vision", type=int, default=1, help="Alternative Vision.")
    parser.add_argument("--vision", type=int, default=None, help="Vision.")
    parser.add_argument("--skip", type=int, default=1, help="Skip training time.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Random exploration")
    parser.add_argument("--beta_rate", type=float, default=0.99, help="Beta Rate")
    parser.add_argument("--budget", type=float, default=4.0, help="Budget")

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
        actor_alpha=args.actor_alpha,
        timesteps=args.timesteps,
        num_agents=args.num_agents,
        hidden_dimensions=args.hidden_dim,
        hidden_dimensions_actor=args.hidden_dim_actor,
        num_layers=args.num_layers,
        num_layers_actor=args.num_layers_actor,
        alt_input=True if args.alt_vision == 0 else False,
        vision=args.vision,
        skip=True if args.skip == 0 else False,
        epsilon=args.epsilon,
        seed=args.seed,
        beta_rate=args.beta_rate,
        budget=args.budget
    )

    exp_config = ExperimentConfig(
        env_config=env_config,
        train_config=train_config,
        debug=args.debug
    )

    if args.algorithm == "Centralized":
        algo = CentralizedValueFunction(exp_config)
    elif args.algorithm == "Decentralized":
        algo = DecentralizedValueFunction(exp_config)
    elif args.algorithm == "ActorCritic":
        algo = ActorCriticPerfect(exp_config)
    elif args.algorithm == "ActorCriticRates":
        algo = ActorCriticRates(exp_config)
    elif args.algorithm == "ActorCriticNoAdvantage":
        algo = ActorCriticPerfectNoAdvantage(exp_config)
    else:
        algo = ActorCriticBeta(exp_config)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    algo.run()


if __name__ == "__main__":
    main(sys.argv[1:])
