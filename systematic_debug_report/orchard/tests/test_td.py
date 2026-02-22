"""Tests for TD update correctness."""

import pytest
import torch

from orchard.enums import EncoderType, EnvType, ModelType, Schedule
from orchard.datatypes import EncoderOutput, EnvConfig, ModelConfig, ScheduleConfig
import orchard.encoding as encoding
from orchard.model import ValueNetwork


def _setup():
    """Initialize encoder and return configs."""
    env_cfg = EnvConfig(
        height=2, width=2, n_agents=2, n_apples=1,
        gamma=0.9, r_picker=-1.0, force_pick=True,
        max_apples=1, env_type=EnvType.DETERMINISTIC,
    )
    model_cfg = ModelConfig(
        input_type=EncoderType.RELATIVE,
        model_type=ModelType.MLP,
        mlp_dims=(16,),
    )
    lr_cfg = ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)
    encoding.init_encoder(EncoderType.RELATIVE, env_cfg)
    return env_cfg, model_cfg, lr_cfg


class TestTrainStep:
    def test_prediction_moves_toward_target(self):
        env_cfg, model_cfg, lr_cfg = _setup()
        net = ValueNetwork(model_cfg, env_cfg, lr_cfg, total_steps=100)

        s_enc = EncoderOutput(scalar=torch.randn(9))
        s_next_enc = EncoderOutput(scalar=torch.randn(9))

        with torch.no_grad():
            pred_before = net(s_enc).item()
            target = 5.0 + 0.9 * net(s_next_enc).item()

        net.train_step(s_enc, reward=5.0, discount=0.9, s_next_enc=s_next_enc)

        with torch.no_grad():
            pred_after = net(s_enc).item()

        # Prediction should move toward target
        assert abs(pred_after - target) < abs(pred_before - target)

    def test_zero_reward_zero_discount(self):
        env_cfg, model_cfg, lr_cfg = _setup()
        net = ValueNetwork(model_cfg, env_cfg, lr_cfg, total_steps=100)

        s_enc = EncoderOutput(scalar=torch.randn(9))
        s_next_enc = EncoderOutput(scalar=torch.randn(9))

        # Target = 0 + 0 * V(s') = 0
        for _ in range(50):
            net.train_step(s_enc, reward=0.0, discount=0.0, s_next_enc=s_next_enc)

        with torch.no_grad():
            pred = net(s_enc).item()
        assert abs(pred) < 0.5  # should be close to 0

    def test_loss_is_squared_error(self):
        env_cfg, model_cfg, lr_cfg = _setup()
        net = ValueNetwork(model_cfg, env_cfg, lr_cfg, total_steps=100)

        s_enc = EncoderOutput(scalar=torch.randn(9))
        s_next_enc = EncoderOutput(scalar=torch.randn(9))

        with torch.no_grad():
            pred = net(s_enc).item()
            target = 3.0 + 0.9 * net(s_next_enc).item()
        expected_loss = (pred - target) ** 2

        actual_loss = net.train_step(s_enc, reward=3.0, discount=0.9, s_next_enc=s_next_enc)
        assert pytest.approx(actual_loss, rel=1e-4) == expected_loss
