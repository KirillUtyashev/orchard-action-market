from pathlib import Path
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
data_dir = Path(__file__).parent.parent / "data"

NUM_AGENTS = 4
W, L = 9, 9
PROBABILITY_APPLE = 32.4 / (W * L)
NUM_WORKERS = 8
DISCOUNT_FACTOR = 0.99
SEEDS = 100
