from dataclasses import dataclass, field
from typing import Optional

from orchard.algorithms import despawn_apple, spawn_apple


@dataclass
class TrainingConfig:
    """Configuration for training parameters.

    Attributes:
        batch_size (int): Number of samples needed before feading it into the neural net.
            Like each agent needs to perform actions batch_size times before we feed to the net.
        alpha (float): Learning rate for the critic network.
        actor_alpha (float): Learning rate for the actor network.
        lr_schedule (dict): Learning rate schedule over training progress.
        timesteps (int): Total number of training timesteps.
        eval_timesteps (int): Total number of evaluation timesteps used in inference.
        num_agents (int): Number of agents in the environment.
        debug (bool): Whether to enable debug mode.
        seed (int): Random seed for reproducibility.
        discount (float): Discount factor for future rewards.
        critic_vision (Optional[int]): Vision range for the critic.
        actor_vision (Optional[int]): Vision range for the actor.
        hidden_dimensions (Optional[int]): Number of hidden units in the critic network.
        hidden_dimensions_actor (Optional[int]): Number of hidden units in the actor network.
        num_layers (int): Number of layers in the critic network.
        num_layers_actor (int): Number of layers in the actor network.
        policy (str): Type of policy used ("value_function" or "policy_network").
        epsilon (float): Exploration rate for epsilon-greedy policies.
        test (bool): Whether to run in test mode.
        skip (bool): Whether to skip training and only run evaluation.
        beta_rate (float): Beta rate for prioritized experience replay.
        budget (float): Budget constraint for the agents.
    """

    batch_size: int = 4
    alpha: float = 0.000275
    actor_alpha: float = 0.0002
    lr_schedule: dict = field(default_factory=lambda: {0.33: 0.00025, 0.625: 0.000075})
    timesteps: int = 1000000
    eval_timesteps: int = 10000
    num_agents: int = 4
    eval_interval: float = 0.2
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    discount: float = 0.99
    critic_vision: Optional[int] = 0
    actor_vision: Optional[int] = 0
    hidden_dimensions: Optional[int] = 16
    hidden_dimensions_actor: Optional[int] = 32
    num_layers: int = 4
    policy: str = "value_function"
    num_layers_actor: int = 4
    epsilon: float = 0.1
    test: bool = False
    skip: bool = False
    seed: int = 1234
    beta_rate: float = 0.0
    budget: float = 0.0


@dataclass
class EnvironmentConfig:
    """Configuration for environment parameters."""

    s_target: float = 0.03
    apple_mean_lifetime: float = 3.5
    spawn_algo: callable = spawn_apple
    despawn_algo: callable = despawn_apple
    length: int = 20
    width: int = 1
    env_cls: str = "OrchardBasic"


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    train_config: TrainingConfig = None
    env_config: EnvironmentConfig = None
    debug: bool = False
