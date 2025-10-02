# generate_params.py
import itertools

# --- Define the Parameter Sets ---
algorithms = ["RewardLearningCNNDecentralized"]
orchard_types = ["OrchardBasic", "OrchardEuclideanNegativeRewards", "OrchardEuclideanRewards"]
mlp_head_hidden_dims = [4, 32, 128]
sizes = [(6, 6, 2), (9, 9, 4), (12, 12, 7)] # width length num_agents
alphas = [0.01, 0.001, 0.0001]
batch_sizes = [1, 128]

# --- Generate All Combinations ---
all_combinations = list(itertools.product(algorithms, orchard_types, mlp_head_hidden_dims, sizes, alphas, batch_sizes))

# Write the final combinations to the map file
with open("parameter_map.txt", "w") as f:
    for combo in all_combinations:
        algo, orchard_type, hidden, size_tuple, alpha, batch_size = combo
        length, width, num_agents = size_tuple
        f.write(f"{algo} {hidden} {length} {width} {num_agents} {orchard_type} {alpha} {batch_size}\n")
    
    # Update the total to use the count from the final list
    f.write(f"# algo hidden length width num_agents orchard_type alpha batch_size, with total:{len(all_combinations)}")

print(f"Successfully generated {len(all_combinations)} parameter combinations in parameter_map.txt")
