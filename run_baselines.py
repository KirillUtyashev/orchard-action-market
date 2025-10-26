import json
import pandas as pd
from configs.config import EnvironmentConfig
from evaluate_policies import evaluate_policy
from policies.random_policy import random_policy
from policies.nearest import nearest_policy
from config import OUT_DIR


def run_all_baselines():
    """
    Runs baseline policies for all specified experimental setups and saves
    the results to a single JSON file.
    """
    # Define the three experimental setups you are testing
    setups = [
        {
            "name": "6x6_2_agents",
            "length": 6,
            "width": 6,
            "num_agents": 2,
            "env_cls": "OrchardEuclideanNegativeRewards",
        },
        {
            "name": "9x9_4_agents",
            "length": 9,
            "width": 9,
            "num_agents": 4,
            "env_cls": "OrchardEuclideanNegativeRewards",
        },
        {
            "name": "12x12_7_agents",
            "length": 12,
            "width": 12,
            "num_agents": 7,
            "env_cls": "OrchardEuclideanNegativeRewards",
        },
    ]

    baseline_results = {}

    for setup in setups:
        print(f"\n===== Running Baselines for Setup: {setup['name']} =====")

        # Consistent environment config for baselines
        env_config = EnvironmentConfig(
            length=setup["length"],
            width=setup["width"],
            env_cls=setup["env_cls"],
            s_target=0.5,  # Match your SLURM script
            apple_mean_lifetime=5,  # Match your SLURM script
        )

        # --- Run Random Policy ---
        print("\n--- Evaluating Random Policy ---")
        random_results = evaluate_policy(
            env_config=env_config,
            num_agents=setup["num_agents"],
            orchard=setup["env_cls"],
            policy_func=random_policy,
            timesteps=10000,  # A shorter run is fine for stable baselines
            seed=42069,
        )
        print(json.dumps(random_results, indent=2))

        # --- Run Nearest Policy ---
        print("\n--- Evaluating Nearest Policy ---")
        nearest_results = evaluate_policy(
            env_config=env_config,
            num_agents=setup["num_agents"],
            orchard=setup["env_cls"],
            policy_func=nearest_policy,
            timesteps=10000,
            seed=42069,
        )
        print(json.dumps(nearest_results, indent=2))

        # Store results
        baseline_results[setup["name"]] = {
            "random": random_results,
            "nearest": nearest_results,
        }

    # Save all baseline results to a file for the plotting script to use
    output_path = OUT_DIR / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\nBaseline results saved to {output_path}")


if __name__ == "__main__":
    run_all_baselines()
