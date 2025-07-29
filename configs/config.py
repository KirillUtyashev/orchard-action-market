from dataclasses import dataclass, field
from typing import Optional

from orchard.algorithms import despawn_apple, spawn_apple


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 128
    alpha: float = 0.0025
    lr_schedule: dict = field(
        default_factory=lambda: {0.33: 0.00025, 0.625: 0.000075}
    )
    timesteps: int = 1000000
    num_agents: int = 4
    eval_interval: float = 0.2
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    discount: float = 0.99
    alt_input: bool = False
    vision: Optional[int] = None
    hidden_dimensions: Optional[int] = 128
    num_layers: int = 4
    epsilon: float = 0.1
    test: bool = False
    skip: bool = False


@dataclass
class EnvironmentConfig:
    """Configuration for environment parameters."""
    s_target: float = 0.03
    apple_mean_lifetime: float = 3.5
    spawn_algo: callable = spawn_apple
    despawn_algo: callable = despawn_apple
    length: int = 20
    width: int = 1


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""
    train_config: TrainingConfig = None
    env_config: EnvironmentConfig = None
    debug: bool = False

