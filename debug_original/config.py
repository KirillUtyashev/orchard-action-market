"""
Shared constants for MC generation and training.
Partner's 2-mode environment parameters, our config style.
"""

# Grid
H = 9
W = 9

# Agents
NUM_AGENTS = 4

# Discount
# Partner uses 2 transitions per agent-step (mode 0 + mode 1),
# so γ_step = γ_round ^ (1 / (2*N)) to get γ_round per full round.
GAMMA = 0.99
GAMMA_STEP = GAMMA ** (1.0 / (2 * NUM_AGENTS))
GAMMA_SEMI = GAMMA ** (1.0 / NUM_AGENTS)

# Encoding (entity-based: 3 scalars + 3 actor + 3*(N-1) others + 4*K apples)
K_NEAREST = 10
INPUT_DIM = 3 + 3 + 3 * (NUM_AGENTS - 1) + 4 * K_NEAREST  # = 55

# Apple dynamics (partner's parameterization: q_agent, apple_life)
Q_AGENT = 1              # expected apples per spawn event
APPLE_LIFE = 8            # mean apple lifetime in rounds
P_SPAWN = Q_AGENT / (W * W)                  # per-cell spawn probability per round
P_DESPAWN = 1.0 / (APPLE_LIFE * NUM_AGENTS)  # per-apple despawn probability per round
L = 1                     # spawn/despawn every L rounds (always 1 in partner's code)
