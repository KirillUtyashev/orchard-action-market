from pathlib import Path
from dataclasses import dataclass, field

import torch
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
data_dir = Path(__file__).parent.parent / "data"

NUM_AGENTS = 4
W, L = 9, 9
PROBABILITY_APPLE = 32.4 / (W * L)
NUM_WORKERS = 8
DISCOUNT_FACTOR = 0.99
SEEDS = 1000

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    alpha: float = 0.01
    timesteps: int = 1000000
    hidden_dimensions: int = 16
    num_layers: int = 4
    seed: int = 1234
    num_eval_states: int = 0
    picker_r: int = -5
    supervised: bool = True
    reward_learning: bool = False
    input_dim: int = 0
    forward: bool = False
    eligibility: bool = False
    monte_carlo: bool = False
    num_seeds: int = 1000
    variance: float = 0
    schedule_lr: bool = False
    lmda: float = 0.5
    random_policy: bool = False
    q_agent: float = 0.5
    apple_life: float = 8
    debug: bool = True
    top_k_num_apples: int = 1
    centralized: bool = False
    concat: bool = False


@dataclass
class EnvironmentConfig:
    """Configuration for environment parameters."""

    length: int = L
    width: int = W


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    train_config: TrainingConfig = None
    env_config: EnvironmentConfig = None
    debug: bool = False
