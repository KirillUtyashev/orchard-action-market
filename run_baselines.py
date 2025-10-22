import argparse
import json
from configs.config import EnvironmentConfig

# Import the newly fixed evaluate_policy function
from evaluate_policies import evaluate_policy
from policies.random_policy import random_policy
from policies.nearest import nearest_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline policies.")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument(
        "--env_cls", type=str, default="OrchardEuclideanNegativeRewards"
    )
    args = parser.parse_args()

    # Create the environment config, ensuring it matches your main experiments
    env_config = EnvironmentConfig(
        length=args.length,
        width=args.width,
        env_cls=args.env_cls,
        s_target=0.16,
        apple_mean_lifetime=5,
    )

    print("\n===== Running Random Policy Baseline =====")
    random_results = evaluate_policy(
        env_config=env_config,
        num_agents=args.num_agents,
        orchard=args.env_cls,
        policy_func=random_policy,  # Pass the random_policy function
    )
    print(json.dumps(random_results, indent=2))

    print("\n===== Running Nearest Policy Baseline =====")
    nearest_results = evaluate_policy(
        env_config=env_config,
        num_agents=args.num_agents,
        orchard=args.env_cls,
        policy_func=nearest_policy,  # Pass the nearest_policy function
    )
    print(json.dumps(nearest_results, indent=2))

    print("\nBaseline runs complete.")
