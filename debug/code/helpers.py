import random
import torch
import os
import numpy as np

from debug.code.config import NUM_AGENTS, W
from debug.code.environment import Orchard

same_cell_no_reward = 0
count = 0
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Args:
        seed: The seed value to use.
        deterministic: If True, enable extra settings for deterministic
                       behavior in cuDNN at the possible cost of speed.
    """
    # print(f"[seed] Using seed={seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Hash-based things (e.g. dict iteration order)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN settings for (more) deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def teleport(n):
    new_i = np.random.randint(0, n - 1)
    new_j = np.random.randint(0, n - 1)
    return np.array([new_i, new_j])


def random_policy(agent_pos):
    """
    Returns the *new (r, c) position* after taking a uniformly random action
    from {UP, DOWN, LEFT, RIGHT, STAY}. If the sampled move would go off-grid,
    the agent stays in place. [web:198]
    """
    r, c = agent_pos
    a = random.randrange(5)  # 0..4

    nr, nc = r, c
    if a == UP:
        nr = r - 1
    elif a == DOWN:
        nr = r + 1
    elif a == LEFT:
        nc = c - 1
    elif a == RIGHT:
        nc = c + 1
    elif a == STAY:
        pass

    # If illegal, "pick stay" (i.e., revert)
    if not (0 <= nr < W and 0 <= nc < W):
        nr, nc = r, c

    return np.array([nr, nc])


def transition(step, curr_state, env, actor_idx, new_pos):
    if step == -1:
        # init-only: do NOT mutate env
        if curr_state is None:
            curr_state = dict(env.get_state())
        if actor_idx is None:
            actor_idx = random.randint(0, NUM_AGENTS - 1)
        curr_state["actor_id"] = actor_idx
        curr_state["mode"] = 0
        return curr_state, None, None, actor_idx

    env.process_action(
        actor_idx,
        new_pos,
        mode=0,
    )

    semi_state = dict(env.get_state())
    semi_state["actor_id"] = actor_idx
    semi_state["mode"] = 1

    res = env.process_action(actor_idx, None, mode=1)

    if step == NUM_AGENTS - 1:
        env.despawn_apples()
        env.spawn_apples()

    final_state = dict(env.get_state())
    actor_idx = random.randint(0, NUM_AGENTS - 1)
    final_state["actor_id"] = actor_idx
    final_state["mode"] = 0

    # return final_state as the next curr_state
    return final_state, semi_state, res, actor_idx


def make_env(reward_module, p_apple, d_apple, apples, agents, agent_positions):
    return Orchard(W, W, NUM_AGENTS, reward_module, p_apple=p_apple, d_apple=d_apple,
                   start_apples_map=apples, start_agents_map=agents, start_agent_positions=agent_positions)


def eval_performance(
        num_agents,
        agent_controller,
        env,
        name,
        timesteps=5000,
        agents_list=None,
        epsilon=0.1,
        inference=False,
):
    reward = 0
    apples_picked = []
    apples_dropped = []

    # Calculate mean distance between agents and their nearest neighbors at each timestep
    nearest_neighbour_mean_distance = []

    num_of_apples_per_second = []

    # Function to compute distance between two points
    def distance(pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    # Function to find nearest neighbor distance for one agent
    def get_nearest_neighbor_distance(agent_idx, agents):
        current_pos = agents[agent_idx].position
        distances = []
        for i, other_agent in enumerate(agents):
            if i != agent_idx:
                distances.append(distance(current_pos, other_agent.position))
        return min(distances) if distances else float("inf")

    os.makedirs("positions", exist_ok=True)
    for i in range(timesteps):
        num_of_apples_per_second.append(env.apples.sum())
        before = env.total_apples
        after = env.total_apples
        apples_dropped.append(after - before)
        apples_per_second = 0
        for tick in range(num_agents):
            apples_before = env.get_sum_apples()
            env_step(agents_list, env, agent_controller, epsilon, inference)
            change = apples_before - env.get_sum_apples()
            reward += change
            rec.log(agents_list)
            apples_picked.append(apples_per_second)
        timestep_distances = []
        for agent_idx in range(len(agents_list)):
            nearest_dist = get_nearest_neighbor_distance(agent_idx, agents_list)
            timestep_distances.append(nearest_dist)
        nearest_neighbour_mean_distance.append(np.mean(timestep_distances))
        ### IGNORE END #####
        if i % 1000 == 0:
            print(i)
        env.total_despawned += env.despawn_algorithm(env, env.despawn_rate)
        env.total_apples += env.spawn_algorithm(env, env.spawn_rate)
    print("Average number of apples per second: ", np.mean(num_of_apples_per_second))
    print("Average distance:", np.mean(nearest_neighbour_mean_distance))
    print("Number of nearest actions: ", nearest_apple_actions)
    print("Number of idle actions: ", idle_actions)
    print("Results for", name)
    print("Reward: ", reward)
    print("Total Apples: ", env.total_apples)
    print("Apples per agent:", reward / num_agents)
    print("Average Reward: ", reward / env.total_apples)
    print(
        "Picked vs Spawned per agent",
        (reward / num_agents) / (env.total_apples / num_agents),
        )
    if not inference:
        return (
            env.total_apples,  # always here
            reward,
            reward / num_agents,
            (reward / num_agents) / (env.total_apples / num_agents),
            np.mean(nearest_neighbour_mean_distance),
            np.mean(num_of_apples_per_second),
            nearest_apple_actions,
            idle_actions,
        )
    else:
        return personal_q_values, agent_distance_hist
