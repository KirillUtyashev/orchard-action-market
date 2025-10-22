import pytest
from configs.config import ExperimentConfig, EnvironmentConfig, TrainingConfig


@pytest.fixture
def default_config() -> ExperimentConfig:
    """
    Provides a default, small-scale ExperimentConfig for testing.
    """
    env_config = EnvironmentConfig(
        length=6, width=6, s_target=0.1, apple_mean_lifetime=5.0
    )
    train_config = TrainingConfig(
        num_agents=2,
        timesteps=1000,
        batch_size=4,  # Small batch size for faster testing
        alpha=0.001,
        epsilon=0.1,
        debug_log_states=False,  # Disable logging during tests
    )
    return ExperimentConfig(
        env_config=env_config, train_config=train_config, debug=False
    )
