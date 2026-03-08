from pathlib import Path
from dataclasses import dataclass, field

import torch
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
data_dir = Path(__file__).parent.parent / "data"
runs_dir = Path(__file__).parent.parent / "runs"

NUM_AGENTS = 2
W, L = 6, 6
PROBABILITY_APPLE = 0.4
NUM_WORKERS = 8
DISCOUNT_FACTOR = 0.99
SEEDS = 1000

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class NetworkConfig:
    CNN: bool = False
    mlp_dims: tuple = (128, 128)
    conv_channels: list = None
    kernel_size: int = 3
    input_dim: int = 0
    cnn_dim: int = 4


@dataclass
class TrainingConfig:
    """Optimization and learning parameters."""
    alpha: float = 0.01
    timesteps: int = 1_000_000
    seed: int = 1234
    epsilon: float = 0.1
    lmda: float = 0.5
    schedule_lr: bool = False
    load_weights: bool = False


@dataclass
class AlgorithmConfig:
    """TD/RL algorithm choices."""
    forward: bool = False           # forward vs backward view
    eligibility: bool = False       # eligibility traces
    monte_carlo: bool = False
    random_policy: bool = False
    q_agent: float = 0.5
    centralized: bool = False
    concat: bool = False


@dataclass
class RewardConfig:
    """Reward shaping and environment interaction."""
    picker_r: int = -5
    supervised: bool = True
    reward_learning: bool = False
    top_k_num_apples: int = 1


@dataclass
class EvalConfig:
    """Evaluation settings."""
    num_eval_states: int = 0
    num_seeds: int = 1000
    variance: float = 0.0
    debug: bool = True
    action_prob_num_states: int = 100
    action_prob_burnin: int = 500
    action_prob_stride: int = 5


@dataclass
class EnvironmentConfig:
    length: int = 6
    width: int = 6
    apple_life: float = 8.0
    max_apples: int = 9


@dataclass
class LoggingConfig:
    output_dir: str = "runs"
    main_csv_freq: int = 20000
    weight_samples_enabled: bool = True
    weight_samples_per_tensor: int = 16
    weight_samples_freq: int = 0


@dataclass
class ExperimentConfig:
    network: NetworkConfig = None
    train: TrainingConfig = None
    algorithm: AlgorithmConfig = None
    reward: RewardConfig = None
    eval: EvalConfig = None
    env: EnvironmentConfig = None
    logging: LoggingConfig = None
