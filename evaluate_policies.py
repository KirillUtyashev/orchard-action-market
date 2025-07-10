from configs.config import EnvironmentConfig
from policies.random_policy import random_policy
from policies.nearest import nearest_policy
from configs.config import EnvironmentConfig
from orchard.algorithms import spawn_apple, despawn_apple
from value_function_learning.train_value_function import evaluate_policy, make_baseline_factory

if __name__ == "__main__":
    env_config = EnvironmentConfig(
        s_target=0.16,
        apple_mean_lifetime=5,
        length=12,
        width=12,
        spawn_algo=spawn_apple,
        despawn_algo=despawn_apple
    )

    baseline_metrics = evaluate_policy(
        env_config,
        num_agents=7,
        agent_factory=make_baseline_factory(random_policy),
        timesteps=10000,
        seed=42069
    )
    print(baseline_metrics)
