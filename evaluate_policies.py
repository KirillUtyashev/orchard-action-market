from policies.random_policy import random_policy
from policies.nearest import nearest_policy
from configs.config import EnvironmentConfig
from orchard.algorithms import spawn_apple, despawn_apple
from value_function_learning.train_value_function import evaluate_policy, make_baseline_factory


def evaluate_factory(length, width, num_agents):
    env_config = EnvironmentConfig(
        s_target=0.16,
        apple_mean_lifetime=5,
        length=length,
        width=width,
        spawn_algo=spawn_apple,
        despawn_algo=despawn_apple
    )

    return evaluate_policy(
        env_config,
        num_agents=num_agents,
        agent_factory=make_baseline_factory(random_policy),
        timesteps=10000,
        seed=42069
    )


if __name__ == "__main__":
    widths = [1, 3, 6]
    for width in widths:
        print(evaluate_factory(50, width, 10))
