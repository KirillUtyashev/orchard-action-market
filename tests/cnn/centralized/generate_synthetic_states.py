import random
import numpy as np


def generate_synthetic_state_at_most_1_apples(
    num_agents: int, width: int, height: int, p
) -> tuple[dict, int]:
    """Generates a synthetic state with at most 1 apple per agent.

    Precondition:
        1 <= num_agents <= width * height

    Args:
        num_agents: The number of agents in the environment.
        width: The width of the environment.
        height: The height of the environment.
        p: The probability that an apple is present in a cell (or that reward is 1)
    Returns:
        Tuple:
            0 - dictionary with keys apples and agents, each mapping to a 2D numpy array.
            1 - Reward
    """
    agents = np.zeros((height, width), dtype=int)
    agent_positions = list(np.ndindex((height, width)))
    agent_positions = random.sample(agent_positions, num_agents)
    for pos in agent_positions:
        agents[pos] = 1
    apples = np.zeros((height, width), dtype=int)
    num_apples = np.zeros((height, width), dtype=int)
    reward = 0
    if np.random.rand() < p:
        apple_pos = random.choice(agent_positions)
        apples[apple_pos] = 1
        reward = 1
    return {"agents": agents, "apples": apples}, reward


def generate_synthetic_state_with_agent_picks_apple(
    num_agents, width, height, s_target
) -> tuple[dict, tuple, int]:
    """Generates a synthetic state where an agent is guaranteed to pick up an apple.

    Precondition:
        1 <= num_agents <= width * height

    Args:
        num_agents: The number of agents in the environment.
        width: The width of the environment.
        height: The height of the environment.
        s_target: Target spawn rate to influence the number of additional apples.

    Returns:
        Tuple:
            0 - dictionary with keys 'apples' and 'agents'.
            1 - agent_pos: tuple (row, col) of the acting agent.
            2 - reward: always 1.
    """
    # Initialize empty grids
    agents_grid = np.zeros((height, width), dtype=np.int8)
    apples_grid = np.zeros((height, width), dtype=np.int8)

    total_cells = width * height

    # 1. Place agents and select the acting agent
    agent_indices = np.random.choice(total_cells, size=num_agents, replace=False)
    acting_agent_index = agent_indices[0]
    acting_agent_pos = np.unravel_index(acting_agent_index, (height, width))

    agent_rows, agent_cols = np.unravel_index(agent_indices, (height, width))
    agents_grid[agent_rows, agent_cols] = 1

    # 2. Calculate the per-cell spawn probability, same as the real env
    p_cell = (num_agents / total_cells) * s_target

    # 3. Generate the probabilistic spawn mask for the entire grid
    rand_mat = np.random.rand(height, width)
    spawn_mask = rand_mat < p_cell

    # 4. Force the condition: Guarantee an apple spawns at the agent's location
    #    This ensures the function's promise is met.
    spawn_mask[acting_agent_pos] = True
    apples_grid[spawn_mask] += 1

    state = {"apples": apples_grid, "agents": agents_grid}
    reward = 1

    return state, acting_agent_pos, reward


def generate_synthetic_state_with_agent_doesnt_pick_apple(
    num_agents, width, height, s_target
) -> tuple[dict, tuple, int]:
    """Generates a synthetic state where no agent starts on an apple.

    Precondition:
        1 <= num_agents <= width * height

    Args:
        num_agents: The number of agents in the environment.
        width: The width of the environment.
        height: The height of the environment.
        s_target: Target spawn rate per agent and orchard size.

    Returns:
        Tuple:
            0 - dictionary with keys 'apples' and 'agents'.
            1 - agent_pos: tuple (row, col) of a randomly chosen agent.
            2 - reward: always 0.
    """
    # Initialize empty grids
    agents_grid = np.zeros((height, width), dtype=np.int8)
    apples_grid = np.zeros((height, width), dtype=np.int8)

    # 1. Choose all agent positions at once, ensuring they are unique
    total_cells = width * height
    agent_indices = np.random.choice(total_cells, size=num_agents, replace=False)

    # 2. Place all agents on the grid
    agent_rows, agent_cols = np.unravel_index(agent_indices, (height, width))
    agents_grid[agent_rows, agent_cols] = 1

    # 3. Randomly choose one agent to be the "acting" agent for return consistency
    acting_agent_index = agent_indices[np.random.randint(num_agents)]
    acting_agent_pos = np.unravel_index(acting_agent_index, (height, width))

    # 4. Calculate the per-cell spawn probability (p_cell), same as the real env
    p_cell = (num_agents / total_cells) * s_target

    # 5. Generate the probabilistic spawn mask for the entire grid
    rand_mat = np.random.rand(height, width)
    spawn_mask = rand_mat < p_cell

    # 6. Force the condition: Prevent any apples from spawning on ANY agent.
    #    We use the existing agents_grid to set the spawn_mask to False
    #    wherever an agent is present. This is the crucial step.
    spawn_mask[agents_grid == 1] = False

    # 7. Apply the final, modified mask to the apple grid.
    #    This mimics the `env.apples[spawn_mask] += 1` behavior.
    apples_grid[spawn_mask] += 1

    # --- MODIFICATION END ---

    state = {"apples": apples_grid, "agents": agents_grid}
    reward = 0

    return state, acting_agent_pos, reward
