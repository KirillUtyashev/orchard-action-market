"""Tests for GPU-batched training correctness.

Verifies that the BatchedTrainer produces IDENTICAL parameter updates
and TD errors as the sequential CPU path (ValueNetwork.td_lambda_step),
within float32 precision.
"""

import pytest
import torch

from orchard.enums import (
    Action, Activation, EncoderType, EnvType, LearningType, ModelType,
    PickMode, Schedule, TrainMethod, WeightInit,
)
from orchard.datatypes import (
    EnvConfig, Grid, ModelConfig, ScheduleConfig, State, StochasticConfig,
)
from orchard.encoding.grid import TaskGridEncoder
from orchard.env.deterministic import DeterministicEnv
import orchard.encoding as encoding
from orchard.model import ValueNetwork, create_networks
from orchard.batched_training import BatchedTrainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _make_cfg(n_task_types=2, n_agents=4, pick_mode=PickMode.FORCED):
    return EnvConfig(
        height=5, width=5, n_agents=n_agents, n_tasks=3,
        gamma=0.99, r_picker=1.0,
        n_task_types=n_task_types, r_low=0.0,
        task_assignments=tuple(
            tuple((i + j) % n_task_types for j in range(1))
            for i in range(n_agents)
        ),
        pick_mode=pick_mode,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )


def _make_state():
    return State(
        agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
        task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
        actor=0,
        task_types=(0, 1, 0),
    )


def _make_model_cfg():
    return ModelConfig(
        input_type=EncoderType.TASK_CNN_GRID,
        model_type=ModelType.CNN,
        mlp_dims=(16,),
        conv_specs=((4, 3),),
        activation=Activation.RELU,
        weight_init=WeightInit.ZERO_BIAS,
    )


def _make_lr_cfg():
    return ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)


def _make_networks(env_cfg, n=None):
    """Create N networks with backward-view TD(λ) and initialize encoder."""
    model_cfg = _make_model_cfg()
    lr_cfg = _make_lr_cfg()
    encoding.init_encoder(EncoderType.TASK_CNN_GRID, env_cfg)
    n_networks = n if n is not None else env_cfg.n_agents
    return create_networks(
        model_cfg, env_cfg, lr_cfg, total_steps=1000,
        td_lambda=0.3, train_method=TrainMethod.BACKWARD_VIEW,
        n_networks=n_networks,
    )


def _clone_networks(networks):
    """Deep-copy a list of networks (so we can run both paths from same init)."""
    import copy
    return [copy.deepcopy(net) for net in networks]


# ---------------------------------------------------------------------------
# encode_all_agents correctness
# ---------------------------------------------------------------------------
class TestEncodeAllAgents:
    """encode_all_agents must match N individual encode() calls."""

    def test_basic(self):
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        s = _make_state()

        grids, scalars = enc.encode_all_agents(s)

        assert grids.shape == (4, cfg.n_task_types + 4, 5, 5)
        assert scalars.shape == (4, 2)

        for i in range(4):
            single = enc.encode(s, i)
            assert torch.allclose(grids[i], single.grid, atol=1e-6), \
                f"Grid mismatch for agent {i}"
            assert torch.allclose(scalars[i], single.scalar, atol=1e-6), \
                f"Scalar mismatch for agent {i}"

    def test_different_actors(self):
        """Test with different actor values."""
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)

        for actor in range(4):
            s = State(
                agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
                task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
                actor=actor,
                task_types=(0, 1, 0),
            )
            grids, scalars = enc.encode_all_agents(s)
            for i in range(4):
                single = enc.encode(s, i)
                assert torch.allclose(grids[i], single.grid, atol=1e-6)
                assert torch.allclose(scalars[i], single.scalar, atol=1e-6)

    def test_phase2_pending(self):
        """Test with phase2_pending flag set."""
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 1, 0),
            phase2_pending=True,
        )
        grids, scalars = enc.encode_all_agents(s)
        for i in range(4):
            single = enc.encode(s, i)
            assert torch.allclose(grids[i], single.grid, atol=1e-6)
            assert torch.allclose(scalars[i], single.scalar, atol=1e-6)

    def test_via_module_api(self):
        """Test the encoding module-level API."""
        cfg = _make_cfg()
        encoding.init_encoder(EncoderType.TASK_CNN_GRID, cfg)
        s = _make_state()
        grids, scalars = encoding.encode_all_agents(s)

        for i in range(4):
            single = encoding.encode(s, i)
            assert torch.allclose(grids[i], single.grid, atol=1e-6)
            assert torch.allclose(scalars[i], single.scalar, atol=1e-6)


# ---------------------------------------------------------------------------
# BatchedTrainer correctness: single step
# ---------------------------------------------------------------------------
class TestBatchedTrainerSingleStep:
    """One TD(λ) step: batched must produce same params as sequential."""

    def test_single_step_params_match(self):
        """After one training step, GPU-batched and CPU-sequential
        network parameters must be identical."""
        cfg = _make_cfg()
        networks_cpu = _make_networks(cfg)
        networks_gpu = _clone_networks(networks_cpu)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",  # CPU for deterministic comparison
        )

        s = _make_state()
        env = DeterministicEnv(cfg)
        s_next = env.advance_actor(env.spawn_and_despawn(
            env.apply_action(s, Action.RIGHT)
        ))

        # CPU sequential path
        for i in range(cfg.n_agents):
            s_enc = encoding.encode(s, i)
            s_next_enc = encoding.encode(s_next, i)
            networks_cpu[i].td_lambda_step(s_enc, 0.0, cfg.gamma, s_next_enc, env_step=0)

        # GPU batched path
        grids_t, scalars_t = encoding.encode_all_agents(s)
        grids_next, scalars_next = encoding.encode_all_agents(s_next)
        rewards = torch.zeros(cfg.n_agents)
        trainer.td_lambda_step_batched(
            grids_t, scalars_t, rewards, cfg.gamma,
            grids_next, scalars_next, env_step=0,
        )
        trainer.sync_to_networks()

        # Compare all parameters
        for i in range(cfg.n_agents):
            for (n1, p1), (n2, p2) in zip(
                networks_cpu[i].named_parameters(),
                networks_gpu[i].named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-5), \
                    f"Network {i}, param {n1}: max diff {(p1 - p2).abs().max().item()}"

    def test_single_step_td_error_match(self):
        """TD errors (loss) must match between CPU and GPU paths."""
        cfg = _make_cfg()
        networks_cpu = _make_networks(cfg)
        networks_gpu = _clone_networks(networks_cpu)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        s = _make_state()
        env = DeterministicEnv(cfg)
        s_next = env.advance_actor(env.spawn_and_despawn(
            env.apply_action(s, Action.DOWN)
        ))

        # CPU
        cpu_loss = 0.0
        for i in range(cfg.n_agents):
            cpu_loss += networks_cpu[i].td_lambda_step(
                encoding.encode(s, i), 0.5, cfg.gamma,
                encoding.encode(s_next, i), env_step=0,
            )

        # GPU
        grids_t, scalars_t = encoding.encode_all_agents(s)
        grids_next, scalars_next = encoding.encode_all_agents(s_next)
        rewards = torch.full((cfg.n_agents,), 0.5)
        gpu_loss = trainer.td_lambda_step_batched(
            grids_t, scalars_t, rewards, cfg.gamma,
            grids_next, scalars_next, env_step=0,
        )

        assert abs(cpu_loss - gpu_loss) < 1e-4, \
            f"Loss mismatch: CPU={cpu_loss:.8f}, GPU={gpu_loss:.8f}"


# ---------------------------------------------------------------------------
# BatchedTrainer correctness: multi-step trajectory
# ---------------------------------------------------------------------------
class TestBatchedTrainerMultiStep:
    """Multiple steps: verify params stay in sync across a trajectory."""

    def test_multi_step_params_match(self):
        """Run 20 training steps on both paths from same init.
        Parameters must match at the end."""
        cfg = _make_cfg()
        networks_cpu = _make_networks(cfg)
        networks_gpu = _clone_networks(networks_cpu)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        env = DeterministicEnv(cfg)
        s = _make_state()
        actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.STAY]

        for step in range(20):
            action = actions[step % len(actions)]
            s_next = env.advance_actor(env.spawn_and_despawn(
                env.apply_action(s, action)
            ))
            reward = 1.0 if s_next.is_agent_on_task(s_next.actor) else 0.0

            # CPU
            for i in range(cfg.n_agents):
                networks_cpu[i].td_lambda_step(
                    encoding.encode(s, i), reward, cfg.gamma,
                    encoding.encode(s_next, i), env_step=step,
                )

            # GPU
            grids_t, scalars_t = encoding.encode_all_agents(s)
            grids_next, scalars_next = encoding.encode_all_agents(s_next)
            rewards = torch.full((cfg.n_agents,), reward)
            trainer.td_lambda_step_batched(
                grids_t, scalars_t, rewards, cfg.gamma,
                grids_next, scalars_next, env_step=step,
            )

            s = s_next

        trainer.sync_to_networks()

        for i in range(cfg.n_agents):
            for (n1, p1), (n2, p2) in zip(
                networks_cpu[i].named_parameters(),
                networks_gpu[i].named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-4), \
                    f"Step 20, network {i}, param {n1}: max diff {(p1 - p2).abs().max().item()}"

    def test_multi_step_with_reset(self):
        """Test that reset_traces produces same result on both paths."""
        cfg = _make_cfg()
        networks_cpu = _make_networks(cfg)
        networks_gpu = _clone_networks(networks_cpu)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        env = DeterministicEnv(cfg)
        s = _make_state()

        # 5 steps, reset, 5 more steps
        for phase in range(2):
            for step in range(5):
                global_step = phase * 5 + step
                s_next = env.advance_actor(env.spawn_and_despawn(
                    env.apply_action(s, Action.RIGHT)
                ))

                for i in range(cfg.n_agents):
                    networks_cpu[i].td_lambda_step(
                        encoding.encode(s, i), 0.0, cfg.gamma,
                        encoding.encode(s_next, i), env_step=global_step,
                    )

                grids_t, scalars_t = encoding.encode_all_agents(s)
                grids_next, scalars_next = encoding.encode_all_agents(s_next)
                trainer.td_lambda_step_batched(
                    grids_t, scalars_t, torch.zeros(cfg.n_agents), cfg.gamma,
                    grids_next, scalars_next, env_step=global_step,
                )
                s = s_next

            # Reset traces
            for net in networks_cpu:
                net.reset_traces()
            trainer.reset_traces()
            s = _make_state()

        trainer.sync_to_networks()

        for i in range(cfg.n_agents):
            for (n1, p1), (n2, p2) in zip(
                networks_cpu[i].named_parameters(),
                networks_gpu[i].named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-4), \
                    f"After reset, network {i}, param {n1}: max diff {(p1 - p2).abs().max().item()}"


# ---------------------------------------------------------------------------
# BatchedTrainer correctness: per-agent rewards
# ---------------------------------------------------------------------------
class TestBatchedTrainerPerAgentRewards:
    """Each agent can receive a different reward. Verify correctness."""

    def test_different_rewards_per_agent(self):
        """Agents receive [0.0, 1.0, 0.5, 0.2] — params must match."""
        cfg = _make_cfg()
        networks_cpu = _make_networks(cfg)
        networks_gpu = _clone_networks(networks_cpu)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        s = _make_state()
        env = DeterministicEnv(cfg)
        s_next = env.advance_actor(env.spawn_and_despawn(
            env.apply_action(s, Action.DOWN)
        ))
        per_agent_rewards = [0.0, 1.0, 0.5, 0.2]

        # CPU
        for i in range(cfg.n_agents):
            networks_cpu[i].td_lambda_step(
                encoding.encode(s, i), per_agent_rewards[i], cfg.gamma,
                encoding.encode(s_next, i), env_step=0,
            )

        # GPU
        grids_t, scalars_t = encoding.encode_all_agents(s)
        grids_next, scalars_next = encoding.encode_all_agents(s_next)
        trainer.td_lambda_step_batched(
            grids_t, scalars_t,
            torch.tensor(per_agent_rewards), cfg.gamma,
            grids_next, scalars_next, env_step=0,
        )
        trainer.sync_to_networks()

        for i in range(cfg.n_agents):
            for (n1, p1), (n2, p2) in zip(
                networks_cpu[i].named_parameters(),
                networks_gpu[i].named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-5), \
                    f"Network {i}, param {n1}: max diff {(p1 - p2).abs().max().item()}"


# ---------------------------------------------------------------------------
# BatchedTrainer: sync roundtrip
# ---------------------------------------------------------------------------
class TestBatchedTrainerSync:
    """sync_to_networks and sync_from_networks preserve params."""

    def test_sync_roundtrip(self):
        """sync_to → modify CPU → sync_from → params reflect modification."""
        cfg = _make_cfg()
        networks = _make_networks(cfg)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        # Get initial inference output
        s = _make_state()
        grids, scalars = encoding.encode_all_agents(s)
        out_before = trainer.forward_single_batched(grids, scalars).clone()

        # Modify network 0 on CPU
        with torch.no_grad():
            for p in networks[0].parameters():
                p.add_(10.0)

        # Stale — trainer still has old params
        out_stale = trainer.forward_single_batched(grids, scalars)
        assert torch.allclose(out_stale, out_before, atol=1e-6)

        # sync_from_networks — trainer picks up new params
        trainer.sync_from_networks()
        out_after = trainer.forward_single_batched(grids, scalars)
        assert not torch.allclose(out_after, out_before, atol=1e-2)


# ---------------------------------------------------------------------------
# BatchedTrainer: inference matches sequential
# ---------------------------------------------------------------------------
class TestBatchedTrainerInference:
    """forward_single_batched must match individual network forwards."""

    def test_inference_matches(self):
        cfg = _make_cfg()
        networks = _make_networks(cfg)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        s = _make_state()

        # Sequential
        seq_vals = []
        with torch.no_grad():
            for i, net in enumerate(networks):
                val = net(encoding.encode(s, i)).item()
                seq_vals.append(val)

        # Batched
        grids, scalars = encoding.encode_all_agents(s)
        batched_vals = trainer.forward_single_batched(grids, scalars)

        for i in range(cfg.n_agents):
            assert abs(seq_vals[i] - batched_vals[i].item()) < 1e-5, \
                f"Agent {i}: seq={seq_vals[i]:.8f}, batched={batched_vals[i].item():.8f}"


# ---------------------------------------------------------------------------
# Different N values
# ---------------------------------------------------------------------------
class TestBatchedTrainerScaling:
    """Verify correctness for different numbers of agents."""

    @pytest.mark.parametrize("n_agents", [2, 3, 6, 8])
    def test_different_n(self, n_agents):
        """Same test as single-step but for various N."""
        cfg = EnvConfig(
            height=5, width=5, n_agents=n_agents, n_tasks=3,
            gamma=0.99, r_picker=1.0,
            n_task_types=2, r_low=0.0,
            task_assignments=tuple(
                ((i % 2,),) for i in range(n_agents)
            ),
            pick_mode=PickMode.FORCED,
            max_tasks_per_type=3, max_tasks=12,
            env_type=EnvType.DETERMINISTIC,
        )
        # Make agent positions that fit
        positions = tuple(
            Grid(0, i % cfg.width) for i in range(n_agents)
        )
        s = State(
            agent_positions=positions,
            task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 1, 0),
        )

        encoding.init_encoder(EncoderType.TASK_CNN_GRID, cfg)
        model_cfg = _make_model_cfg()
        lr_cfg = _make_lr_cfg()
        networks_cpu = create_networks(
            model_cfg, cfg, lr_cfg, total_steps=1000,
            td_lambda=0.3, train_method=TrainMethod.BACKWARD_VIEW,
            n_networks=n_agents,
        )
        networks_gpu = _clone_networks(networks_cpu)

        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        env = DeterministicEnv(cfg)
        s_next = env.advance_actor(env.spawn_and_despawn(
            env.apply_action(s, Action.RIGHT)
        ))

        # CPU
        for i in range(n_agents):
            networks_cpu[i].td_lambda_step(
                encoding.encode(s, i), 0.0, cfg.gamma,
                encoding.encode(s_next, i), env_step=0,
            )

        # GPU
        grids_t, scalars_t = encoding.encode_all_agents(s)
        grids_next, scalars_next = encoding.encode_all_agents(s_next)
        trainer.td_lambda_step_batched(
            grids_t, scalars_t, torch.zeros(n_agents), cfg.gamma,
            grids_next, scalars_next, env_step=0,
        )
        trainer.sync_to_networks()

        for i in range(n_agents):
            for (n1, p1), (n2, p2) in zip(
                networks_cpu[i].named_parameters(),
                networks_gpu[i].named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-5), \
                    f"N={n_agents}, net {i}, {n1}: diff {(p1 - p2).abs().max().item()}"


# ---------------------------------------------------------------------------
# Variable discount
# ---------------------------------------------------------------------------
class TestBatchedTrainerVariableDiscount:
    """TD(λ) with variable discount (γ changes per step)."""

    def test_variable_discount_sequence(self):
        """Alternate discount=0.99 and discount=0.0 (terminal).
        Both paths must agree."""
        cfg = _make_cfg()
        networks_cpu = _make_networks(cfg)
        networks_gpu = _clone_networks(networks_cpu)

        lr_cfg = _make_lr_cfg()
        trainer = BatchedTrainer(
            networks_gpu, td_lambda=0.3, lr_schedule=lr_cfg,
            total_steps=1000, device="cpu",
        )

        env = DeterministicEnv(cfg)
        s = _make_state()
        discounts = [0.99, 0.99, 0.0, 0.99, 0.99, 0.0, 0.99, 0.99]

        for step, disc in enumerate(discounts):
            s_next = env.advance_actor(env.spawn_and_despawn(
                env.apply_action(s, Action.RIGHT)
            ))

            for i in range(cfg.n_agents):
                networks_cpu[i].td_lambda_step(
                    encoding.encode(s, i), 0.0, disc,
                    encoding.encode(s_next, i), env_step=step,
                )

            grids_t, scalars_t = encoding.encode_all_agents(s)
            grids_next, scalars_next = encoding.encode_all_agents(s_next)
            trainer.td_lambda_step_batched(
                grids_t, scalars_t, torch.zeros(cfg.n_agents), disc,
                grids_next, scalars_next, env_step=step,
            )

            if disc == 0.0:
                for net in networks_cpu:
                    net.reset_traces()
                trainer.reset_traces()
                s = _make_state()
            else:
                s = s_next

        trainer.sync_to_networks()

        for i in range(cfg.n_agents):
            for (n1, p1), (n2, p2) in zip(
                networks_cpu[i].named_parameters(),
                networks_gpu[i].named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-4), \
                    f"Variable discount, net {i}, {n1}: diff {(p1 - p2).abs().max().item()}"
