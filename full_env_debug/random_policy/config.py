"""
Shared constants for MC generation and training.
Everything here must be identical between generate_mc.py and train.ipynb.
"""

# Grid
H = 9
W = 9

# Agents
NUM_AGENTS = 4

# Discount
GAMMA = 0.9
GAMMA_STEP = GAMMA ** (1.0 / NUM_AGENTS)

# Encoding
K_NEAREST = 3
INPUT_DIM = 3 + 3 * (NUM_AGENTS - 1) + 4 * K_NEAREST  # = 24

# Spawn/despawn (applied once per round, i.e. every L rounds of N ticks)
L = 1                       # rounds between spawn/despawn events
P_SPAWN = 0.04              # per empty cell per event  (~3.2 apples spawn/round)
P_DESPAWN = 0.12            # per apple per event       (mean lifetime ~8 rounds)
# Steady-state density: P_SPAWN/(P_SPAWN+P_DESPAWN) = 0.25  (~20 apples on 81 cells)

# Action deltas: (row_delta, col_delta)
ACTION_DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay, up, down, left, right