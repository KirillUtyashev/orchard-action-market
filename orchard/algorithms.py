import numpy as np


def distance(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2))


mean_distances = []


def get_nearest_agent_apple_distance(apple_pos, agents):
    min_distance = float('inf')
    for agent_ in agents:
        dist = distance(agent_.position, apple_pos)
        if dist < min_distance:
            min_distance = dist
    return min_distance

def get_nearest_agent_dirt_distance(dirt_pos, agents):
    min_distance = float('inf')
    for agent_ in agents:
        dist = distance(agent_.position, dirt_pos)
        if dist < min_distance:
            min_distance = dist
    return min_distance


def despawn_apple(env, q_despawn):
    """
    One 'second-boundary' update:
      • despawn apples with geom-hazard q = 1/T
      • spawn new apples in empty cells with p_cell = r · s_target
    Call exactly once after every n micro-ticks.
    """
    H, L = env.apples.shape            # ★ rows = width, cols = length

    # ---- despawn ----
    rand_mat = np.random.rand(H, L)    # ★ same shape as env.apples
    mask_apples = (env.apples != 0)
    removal_mask = mask_apples & (rand_mat < q_despawn)
    total_removed = removal_mask.sum()
    env.apples[removal_mask] -= 1
    return total_removed

def spawn_apple(env, p_cell, total_dirt) :
    H, L = env.apples.shape  # ★ rows = width, cols = length

    rand_mat = np.random.rand(H, L)  # ★ new RNG draw, same shape
    spawn_positions = rand_mat < p_cell
    positions = np.argwhere(spawn_positions).tolist()
    dirt_slowed_rate = 1 - (total_dirt / H * L) ^ 2
    total_spawned = spawn_positions.sum() * dirt_slowed_rate
    Reduced_spawned = spawn_positions.sum() - total_spawned
    while Reduced_spawned > 0:
        for i in positions:
            if np.random.rand() < 0.5 and Reduced_spawned > 0:
                positions.remove(i)
                Reduced_spawned -= 1
    if total_spawned > 0:
        for position in positions:
            mean_distances.append(get_nearest_agent_apple_distance(position, env.agents_list))
    env.apples[spawn_positions] += 1
    return total_spawned

def spawn_dirt(env, p_cell):
    L, W = env.apples.shape  # ★ rows = Length, cols = Width

    rand_mat = np.random.rand(L, W)  # ★ new RNG draw, same shape
    spawn_positions = rand_mat < p_cell
    positions = np.argwhere(spawn_positions)
    total_spawned = spawn_positions.sum()
    if total_spawned > 0:
        for position in positions:
            mean_distances.append(get_nearest_agent_dirt_distance(position, env.agents_list))
    env.dirt[spawn_positions] += 1
    return total_spawned

def spawn_apple_selfless_orchard(env, p_cell):
    H, L = env.apples.shape  # rows = width, cols = length

    rand_mat = np.random.rand(H, L)
    spawn_mask = (rand_mat < p_cell) & (env.apples == 0)  # only empty cells eligible
    positions = np.argwhere(spawn_mask)
    total_spawned = len(positions)

    if total_spawned > 0:
        # assign random agent IDs (0 ... num_agents-1)
        assigned_agents = np.random.randint(1, env.n + 1, size=total_spawned)

        for pos, agent_id in zip(positions, assigned_agents):
            env.apples[tuple(pos)] = agent_id
            mean_distances.append(get_nearest_agent_apple_distance(pos, env.agents_list))

    return total_spawned


def despawn_apple_selfless_orchard(env, q_despawn):
    """
    One 'second-boundary' update:
      • despawn apples with geom-hazard q = 1/T
      • spawn new apples in empty cells with p_cell = r · s_target
    Call exactly once after every n micro-ticks.
    """
    H, L = env.apples.shape            # ★ rows = width, cols = length

    # ---- despawn ----
    rand_mat = np.random.rand(H, L)    # ★ same shape as env.apples
    mask_apples = (env.apples != 0)
    removal_mask = mask_apples & (rand_mat < q_despawn)
    total_removed = removal_mask.sum()
    env.apples[removal_mask] = 0
    return total_removed
