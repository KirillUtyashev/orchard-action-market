"""Tests for ValueNetwork and backward-view TD(λ) trace logic."""

import pytest
import torch
import orchard.encoding as encoding

from orchard.enums import Activation, EncoderType, PickMode, WeightInit
from orchard.datatypes import EnvConfig, ModelConfig, StochasticConfig, EncoderOutput
from orchard.model import ValueNetwork, create_networks


def _setup_networks(n_agents=2, weight_init=WeightInit.DEFAULT):
    env_cfg = EnvConfig(
        height=3, width=3, n_agents=n_agents, n_tasks=1,
        gamma=0.99, r_picker=1.0, n_task_types=1,
        task_assignments=tuple((0,) for _ in range(n_agents)),
        pick_mode=PickMode.FORCED, max_tasks_per_type=1,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(16,),
        conv_specs=((4, 3),),
        activation=Activation.RELU,
        weight_init=weight_init
    )
    
    # Must init encoder singleton before creating networks
    encoding.init_encoder(EncoderType.BLIND_TASK_CNN_GRID, env_cfg)
    
    return env_cfg, model_cfg


class TestValueNetworkInit:
    def test_network_creation(self):
        env_cfg, model_cfg = _setup_networks()
        net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.3)
        
        assert hasattr(net, 'conv')
        assert hasattr(net, 'net')
        assert len(net._traces) == len(list(net.parameters()))
        
        # Verify trace tensors are initialized to zero and match parameter shapes
        for name, param in net.named_parameters():
            assert name in net._traces
            assert net._traces[name].shape == param.shape
            assert (net._traces[name] == 0.0).all()

    def test_zero_bias_init(self):
        env_cfg, model_cfg = _setup_networks(weight_init=WeightInit.ZERO_BIAS)
        net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.0)
        
        # All biases should be exactly 0.0
        for name, param in net.named_parameters():
            if 'bias' in name:
                assert (param == 0.0).all()


class TestForwardPass:
    def test_forward_with_grid_and_scalar(self):
        env_cfg, model_cfg = _setup_networks()
        net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.0)
        
        # Mock encoder output: 4 channels, 3x3 grid; 3 scalars
        grid = torch.rand(4, 3, 3)
        scalar = torch.rand(3)
        enc_out = EncoderOutput(grid=grid, scalar=scalar)
        
        out = net(enc_out)
        
        assert out.dim() == 0  # Should be a single scalar float tensor
        assert out.requires_grad

    def test_forward_raw_batching(self):
        env_cfg, model_cfg = _setup_networks()
        net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.0)
        
        # Batch of 5 inputs
        grid = torch.rand(5, 4, 3, 3)
        scalar = torch.rand(5, 3)
        
        out = net.forward_raw(grid, scalar)
        
        assert out.shape == (5,)


class TestTDStep:
    def test_td_step_moves_prediction_toward_target(self):
        env_cfg, model_cfg = _setup_networks()
        net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.0)
        
        s_enc = EncoderOutput(grid=torch.randn(4, 3, 3), scalar=torch.randn(3))
        s_next_enc = EncoderOutput(grid=torch.randn(4, 3, 3), scalar=torch.randn(3))

        with torch.no_grad():
            pred_before = net(s_enc).item()
            target = 5.0 + 0.99 * net(s_next_enc).item()

        # Execute 1 TD step: α=0.1
        net.td_step(s_enc, reward=5.0, discount=0.99, s_next_enc=s_next_enc, alpha=0.1)

        with torch.no_grad():
            pred_after = net(s_enc).item()

        # The new prediction for s should be closer to the target than it was before
        assert abs(pred_after - target) < abs(pred_before - target)

    def test_traces_reset(self):
        env_cfg, model_cfg = _setup_networks()
        net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.5)
        
        s_enc = EncoderOutput(grid=torch.randn(4, 3, 3), scalar=torch.randn(3))
        s_next_enc = EncoderOutput(grid=torch.randn(4, 3, 3), scalar=torch.randn(3))
        
        # Take a step, which will accumulate gradients into the traces
        net.td_step(s_enc, reward=1.0, discount=0.99, s_next_enc=s_next_enc, alpha=0.1)
        
        # Ensure traces are non-zero
        assert any((trace != 0.0).any() for trace in net._traces.values())
        assert net._gamma_prev == 0.99
        
        # Reset
        net.reset_traces()
        
        # Ensure they are all back to zero
        assert all((trace == 0.0).all() for trace in net._traces.values())
        assert net._gamma_prev == 0.0