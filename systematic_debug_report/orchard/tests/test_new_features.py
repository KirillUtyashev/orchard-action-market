"""Tests for all changes in the March 2026 update:
  1. Choice-pick action masking (get_all_actions state-aware)
  2. TaskSpawnMode (global_unique vs per_type_unique)
  3. use_vec_encode flag (loop fallback matches vectorized output)
  4. RPS-only logging (no greedy_pps in legacy path)
  5. Epsilon-greedy samples from masked action space
"""

import pytest
import torch

from orchard.enums import (
    Action, Activation, DespawnMode, EncoderType, EnvType, LearningType,
    ModelType, PickMode, Schedule, TaskSpawnMode, TrainMethod, WeightInit,
    make_pick_action,
)
from orchard.datatypes import (
    EnvConfig, Grid, ModelConfig, ScheduleConfig, State, StochasticConfig,
)
from orchard.env.deterministic import DeterministicEnv
from orchard.env.stochastic import StochasticEnv
from orchard.policy import (
    get_all_actions, get_phase2_actions,
    epsilon_greedy, epsilon_greedy_batched,
    nearest_correct_task_action, nearest_correct_task_stay_wrong_action,
)
from orchard.eval import rollout_trajectory
from orchard.encoding.grid import TaskGridEncoder, CentralizedTaskGridEncoder
import orchard.encoding as encoding
from orchard.model import create_networks
from orchard.seed import set_all_seeds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _det_cfg(pick_mode=PickMode.FORCED, n_task_types=4, n_agents=4, r_low=0.0):
    return EnvConfig(
        height=5, width=5, n_agents=n_agents, n_tasks=2,
        gamma=0.99, r_picker=1.0,
        n_task_types=n_task_types, r_low=r_low,
        task_assignments=tuple((i % n_task_types,) for i in range(n_agents)),
        pick_mode=pick_mode,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )


def _stoch_cfg(pick_mode=PickMode.FORCED, task_spawn_mode=None, n_task_types=4):
    return EnvConfig(
        height=9, width=9, n_agents=4, n_tasks=2,
        gamma=0.99, r_picker=1.0,
        n_task_types=n_task_types, r_low=0.0,
        task_assignments=tuple((i,) for i in range(n_task_types)),
        pick_mode=pick_mode,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.STOCHASTIC,
        stochastic=StochasticConfig(
            spawn_prob=1.0,  # always spawn when under cap
            despawn_mode=DespawnMode.NONE,
            despawn_prob=0.0,
            task_spawn_mode=task_spawn_mode,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Action masking
# ---------------------------------------------------------------------------

class TestActionMasking:
    """get_all_actions must return state-dependent masked action space."""

    def test_forced_always_5_moves(self):
        """Forced pick: always 5 move actions regardless of position."""
        cfg = _det_cfg(pick_mode=PickMode.FORCED)
        # On a task cell
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        actions = get_all_actions(cfg, s)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_choice_not_on_task_moves_only(self):
        """Choice pick, actor not on any task: only 5 move actions."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(2, 2),),
            actor=0,
            task_types=(0,),
        )
        actions = get_all_actions(cfg, s)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_choice_on_task_stay_plus_pick(self):
        """Choice pick, actor on task cell: phase2 actions are STAY + pick(τ)."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE, n_task_types=4)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(2,),
        )
        actions = get_phase2_actions(s, cfg)
        assert len(actions) == 2
        assert Action.STAY in actions
        assert make_pick_action(2) in actions
        assert not any(a.is_move() and a != Action.STAY for a in actions)

    def test_choice_on_task_no_non_stay_moves(self):
        """Choice pick on task cell: UP/DOWN/LEFT/RIGHT not in phase-2 actions."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(2, 2),),
            actor=0,
            task_types=(0,),
        )
        actions = get_phase2_actions(s, cfg)
        for a in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            assert a not in actions

    def test_choice_multiple_types_on_cell(self):
        """Choice pick on cell with 2 task types: phase-2 is STAY + pick(τ1) + pick(τ2)."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE, n_task_types=4)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 1), Grid(1, 1)),
            actor=0,
            task_types=(1, 3),
        )
        actions = get_phase2_actions(s, cfg)
        assert len(actions) == 3  # STAY + pick(1) + pick(3)
        assert Action.STAY in actions
        assert make_pick_action(1) in actions
        assert make_pick_action(3) in actions

    def test_state_none_returns_move_only(self):
        """state=None returns 5 move actions (used for network sizing)."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        actions = get_all_actions(cfg, state=None)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_forced_state_none_same_as_state(self):
        """Forced mode: state=None and state provided give same result."""
        cfg = _det_cfg(pick_mode=PickMode.FORCED)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        assert get_all_actions(cfg, None) == get_all_actions(cfg, s)


# ---------------------------------------------------------------------------
# 2. Epsilon-greedy samples from masked space
# ---------------------------------------------------------------------------

class TestEpsilonGreedyMasking:
    """Epsilon-greedy random actions must respect action masking."""

    def _make_networks(self, cfg):
        model_cfg = ModelConfig(
            input_type=EncoderType.TASK_CNN_GRID,
            model_type=ModelType.CNN,
            mlp_dims=(8,),
            conv_specs=((4, 3),),
            activation=Activation.LEAKY_RELU,
            weight_init=WeightInit.ZERO_BIAS,
        )
        lr_cfg = ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)
        encoding.init_encoder(EncoderType.TASK_CNN_GRID, cfg)
        return create_networks(
            model_cfg, cfg, lr_cfg, total_steps=100,
            td_lambda=0.3, train_method=TrainMethod.BACKWARD_VIEW,
            n_networks=cfg.n_agents,
        )

    def test_random_action_on_task_cell_is_valid(self):
        """With epsilon=1.0, all sampled actions are in the masked set."""
        set_all_seeds(42)
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        env = DeterministicEnv(cfg)
        networks = self._make_networks(cfg)

        # Force actor onto a task cell
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        valid = get_all_actions(cfg, s)

        for _ in range(50):
            a = epsilon_greedy(s, networks, env, epsilon=1.0)
            assert a in valid, f"Sampled {a} not in valid actions {valid}"

    def test_random_action_not_on_task_is_move(self):
        """With epsilon=1.0, random actions when not on task are moves only."""
        set_all_seeds(99)
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        env = DeterministicEnv(cfg)
        networks = self._make_networks(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(3, 3),),
            actor=0,
            task_types=(1,),
        )
        for _ in range(30):
            a = epsilon_greedy(s, networks, env, epsilon=1.0)
            assert a.is_move(), f"Expected move action, got {a}"


# ---------------------------------------------------------------------------
# 3. TaskSpawnMode
# ---------------------------------------------------------------------------

class TestTaskSpawnMode:
    """Spawn behavior matches declared TaskSpawnMode."""

    def test_global_unique_no_two_tasks_same_cell(self):
        """GLOBAL_UNIQUE: no cell has more than one task of any type."""
        cfg = _stoch_cfg(
            pick_mode=PickMode.FORCED,
            task_spawn_mode=TaskSpawnMode.GLOBAL_UNIQUE,
            n_task_types=4,
        )
        env = StochasticEnv(cfg)
        set_all_seeds(7)
        s = env.init_state()
        # Run many spawn rounds
        for _ in range(200):
            s = env.spawn_and_despawn(s)
        # Every cell should have at most 1 task total
        from collections import Counter
        pos_counts = Counter(s.task_positions)
        for pos, count in pos_counts.items():
            assert count <= 1, f"Cell {pos} has {count} tasks (global_unique violated)"

    def test_per_type_unique_allows_multi_type_coexistence(self):
        """PER_TYPE_UNIQUE: a cell may hold tasks of different types."""
        cfg = _stoch_cfg(
            pick_mode=PickMode.CHOICE,
            task_spawn_mode=TaskSpawnMode.PER_TYPE_UNIQUE,
            n_task_types=4,
        )
        env = StochasticEnv(cfg)
        set_all_seeds(13)
        s = env.init_state()
        # Spawn many rounds — with spawn_prob=1.0 and 4 types, cells will stack
        found_multi = False
        for _ in range(500):
            s = env.spawn_and_despawn(s)
            from collections import Counter
            pos_counts = Counter(s.task_positions)
            if any(c > 1 for c in pos_counts.values()):
                found_multi = True
                break
        assert found_multi, "Expected cells with multiple task types, found none"

    def test_per_type_unique_no_duplicate_type_same_cell(self):
        """PER_TYPE_UNIQUE: a cell never has two tasks of the same type."""
        cfg = _stoch_cfg(
            pick_mode=PickMode.CHOICE,
            task_spawn_mode=TaskSpawnMode.PER_TYPE_UNIQUE,
            n_task_types=4,
        )
        env = StochasticEnv(cfg)
        set_all_seeds(21)
        s = env.init_state()
        for _ in range(300):
            s = env.spawn_and_despawn(s)
            for i, pos_i in enumerate(s.task_positions):
                for j, pos_j in enumerate(s.task_positions):
                    if i >= j:
                        continue
                    if pos_i == pos_j:
                        assert s.task_types[i] != s.task_types[j], (
                            f"Two tasks of type {s.task_types[i]} on same cell {pos_i}"
                        )

    def test_auto_fallback_forced_uses_global_unique(self):
        """task_spawn_mode=None with FORCED pick auto-selects GLOBAL_UNIQUE."""
        cfg = _stoch_cfg(
            pick_mode=PickMode.FORCED,
            task_spawn_mode=None,
            n_task_types=4,
        )
        env = StochasticEnv(cfg)
        set_all_seeds(3)
        s = env.init_state()
        for _ in range(200):
            s = env.spawn_and_despawn(s)
        from collections import Counter
        pos_counts = Counter(s.task_positions)
        for pos, count in pos_counts.items():
            assert count <= 1, f"Auto FORCED mode allowed {count} tasks on {pos}"

    def test_auto_fallback_choice_uses_per_type_unique(self):
        """task_spawn_mode=None with CHOICE pick auto-selects PER_TYPE_UNIQUE."""
        cfg = _stoch_cfg(
            pick_mode=PickMode.CHOICE,
            task_spawn_mode=None,
            n_task_types=4,
        )
        env = StochasticEnv(cfg)
        set_all_seeds(17)
        s = env.init_state()
        found_multi = False
        for _ in range(500):
            s = env.spawn_and_despawn(s)
            from collections import Counter
            if any(c > 1 for c in Counter(s.task_positions).values()):
                found_multi = True
                break
        assert found_multi, "Auto CHOICE mode should allow multi-type coexistence"


# ---------------------------------------------------------------------------
# 4. use_vec_encode — loop fallback matches vectorized output
# ---------------------------------------------------------------------------

class TestUseVecEncode:
    """Loop-based encoding must be numerically identical to vectorized."""

    def _make_state(self):
        return State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(2, 2), Grid(3, 3)),
            task_positions=(Grid(0, 1), Grid(1, 2), Grid(3, 0)),
            actor=0,
            task_types=(0, 2, 3),
        )

    def test_task_encoder_single_matches(self):
        """TaskGridEncoder: loop == vec for encode()."""
        cfg = _det_cfg()
        enc_vec = TaskGridEncoder(cfg, use_vec_encode=True)
        enc_loop = TaskGridEncoder(cfg, use_vec_encode=False)
        s = self._make_state()
        for agent_idx in range(4):
            out_vec = enc_vec.encode(s, agent_idx)
            out_loop = enc_loop.encode(s, agent_idx)
            assert torch.allclose(out_vec.grid, out_loop.grid, atol=1e-6), \
                f"Agent {agent_idx}: grid mismatch"
            assert torch.allclose(out_vec.scalar, out_loop.scalar, atol=1e-6), \
                f"Agent {agent_idx}: scalar mismatch"

    def test_task_encoder_batch_matches(self):
        """TaskGridEncoder: loop == vec for encode_batch_for_actions()."""
        cfg = _det_cfg()
        enc_vec = TaskGridEncoder(cfg, use_vec_encode=True)
        enc_loop = TaskGridEncoder(cfg, use_vec_encode=False)
        env = DeterministicEnv(cfg)
        s = self._make_state()
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        for agent_idx in range(4):
            out_vec = enc_vec.encode_batch_for_actions(s, agent_idx, after_states)
            out_loop = enc_loop.encode_batch_for_actions(s, agent_idx, after_states)
            assert torch.allclose(out_vec.grid, out_loop.grid, atol=1e-6), \
                f"Agent {agent_idx}: batch grid mismatch"
            assert torch.allclose(out_vec.scalar, out_loop.scalar, atol=1e-6), \
                f"Agent {agent_idx}: batch scalar mismatch"

    def test_centralized_encoder_single_matches(self):
        """CentralizedTaskGridEncoder: loop == vec for encode()."""
        cfg = _det_cfg()
        enc_vec = CentralizedTaskGridEncoder(cfg, use_vec_encode=True)
        enc_loop = CentralizedTaskGridEncoder(cfg, use_vec_encode=False)
        s = self._make_state()
        for agent_idx in range(4):
            out_vec = enc_vec.encode(s, agent_idx)
            out_loop = enc_loop.encode(s, agent_idx)
            assert torch.allclose(out_vec.grid, out_loop.grid, atol=1e-6), \
                f"Agent {agent_idx}: centralized grid mismatch"

    def test_centralized_encoder_batch_matches(self):
        """CentralizedTaskGridEncoder: loop == vec for encode_batch_for_actions()."""
        cfg = _det_cfg()
        enc_vec = CentralizedTaskGridEncoder(cfg, use_vec_encode=True)
        enc_loop = CentralizedTaskGridEncoder(cfg, use_vec_encode=False)
        env = DeterministicEnv(cfg)
        s = self._make_state()
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        for agent_idx in range(4):
            out_vec = enc_vec.encode_batch_for_actions(s, agent_idx, after_states)
            out_loop = enc_loop.encode_batch_for_actions(s, agent_idx, after_states)
            assert torch.allclose(out_vec.grid, out_loop.grid, atol=1e-6), \
                f"Agent {agent_idx}: centralized batch mismatch"

    def test_pick_after_state_vec_matches_loop(self):
        """Both encoders agree on pick after-states (task removed)."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        enc_vec = TaskGridEncoder(cfg, use_vec_encode=True)
        enc_loop = TaskGridEncoder(cfg, use_vec_encode=False)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 1), Grid(1, 1), Grid(2, 2), Grid(3, 3)),
            task_positions=(Grid(0, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 2),
        )
        s_picked, _ = env.resolve_pick(s, pick_type=0)
        after_states = [env.apply_action(s, Action(a)) for a in range(5)] + [s_picked]
        for agent_idx in range(4):
            out_vec = enc_vec.encode_batch_for_actions(s, agent_idx, after_states)
            out_loop = enc_loop.encode_batch_for_actions(s, agent_idx, after_states)
            assert torch.allclose(out_vec.grid, out_loop.grid, atol=1e-6), \
                f"Pick after-state: agent {agent_idx} grid mismatch"


# ---------------------------------------------------------------------------
# 5. RPS logging — legacy path no longer emits greedy_pps
# ---------------------------------------------------------------------------

class TestRPSLogging:
    """evaluate_policy_learning always returns greedy_rps, never greedy_pps."""

    def test_legacy_path_returns_rps_not_pps(self):
        """n_task_types=1 path returns greedy_rps, not greedy_pps."""
        from orchard.eval import evaluate_policy_learning
        from orchard.enums import Heuristic
        from orchard.datatypes import EnvConfig
        from orchard.env.deterministic import DeterministicEnv
        import orchard.encoding as enc_mod

        cfg = EnvConfig(
            height=5, width=5, n_agents=2, n_tasks=2,
            gamma=0.9, r_picker=1.0,
            pick_mode=PickMode.FORCED,
            max_tasks=4, env_type=EnvType.DETERMINISTIC,
            task_assignments=((0,), (0,)),
        )
        enc_mod.init_encoder(EncoderType.CNN_GRID, cfg)
        from orchard.model import create_networks
        model_cfg = ModelConfig(
            input_type=EncoderType.CNN_GRID,
            model_type=ModelType.CNN,
            mlp_dims=(),
            conv_specs=((4, 3),),
            activation=Activation.LEAKY_RELU,
            weight_init=WeightInit.ZERO_BIAS,
        )
        lr_cfg = ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)
        networks = create_networks(
            model_cfg, cfg, lr_cfg, total_steps=100,
            td_lambda=0.3, train_method=TrainMethod.BACKWARD_VIEW,
            n_networks=2,
        )
        env = DeterministicEnv(cfg)
        metrics = evaluate_policy_learning(
            networks, env, eval_steps=50,
            batch_actions=False, heuristic=Heuristic.NEAREST_TASK,
        )
        assert "greedy_rps" in metrics, "greedy_rps must always be present"
        assert "greedy_pps" not in metrics, "greedy_pps must never appear"
        assert "nearest_pps" not in metrics, "nearest_pps must never appear"

    def test_task_spec_path_returns_rps(self):
        """n_task_types>1 path also returns greedy_rps."""
        from orchard.eval import evaluate_policy_learning
        from orchard.enums import Heuristic

        cfg = _det_cfg(pick_mode=PickMode.FORCED)
        encoding.init_encoder(EncoderType.TASK_CNN_GRID, cfg)
        model_cfg = ModelConfig(
            input_type=EncoderType.TASK_CNN_GRID,
            model_type=ModelType.CNN,
            mlp_dims=(),
            conv_specs=((4, 3),),
            activation=Activation.LEAKY_RELU,
            weight_init=WeightInit.ZERO_BIAS,
        )
        lr_cfg = ScheduleConfig(start=0.01, end=0.01, schedule=Schedule.NONE)
        networks = create_networks(
            model_cfg, cfg, lr_cfg, total_steps=100,
            td_lambda=0.3, train_method=TrainMethod.BACKWARD_VIEW,
            n_networks=cfg.n_agents,
        )
        env = DeterministicEnv(cfg)
        metrics = evaluate_policy_learning(
            networks, env, eval_steps=50,
            batch_actions=False, heuristic=Heuristic.NEAREST_CORRECT_TASK,
        )
        assert "greedy_rps" in metrics
        assert "greedy_pps" not in metrics


# ---------------------------------------------------------------------------
# 6. Two-phase structure
# ---------------------------------------------------------------------------

class TestTwoPhaseStructure:
    """Verify the two-phase (move then pick) structure is correct."""

    def test_get_all_actions_always_5_moves(self):
        """get_all_actions always returns exactly 5 move actions."""
        for mode in [PickMode.FORCED, PickMode.CHOICE]:
            cfg = _det_cfg(pick_mode=mode)
            assert len(get_all_actions(cfg)) == 5
            assert all(a.is_move() for a in get_all_actions(cfg))

    def test_get_phase2_actions_on_task(self):
        """get_phase2_actions returns STAY + pick(τ) when on a task cell."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(2,),
        )
        actions = get_phase2_actions(s, cfg)
        assert Action.STAY in actions
        assert make_pick_action(2) in actions
        assert len(actions) == 2

    def test_get_phase2_actions_not_on_task(self):
        """get_phase2_actions returns empty list when not on task."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(3, 3),),
            actor=0,
            task_types=(1,),
        )
        assert get_phase2_actions(s, cfg) == []

    def test_forced_pick_auto_picks_correct_type(self):
        """In FORCED mode, rollout auto-picks the task at landing cell."""
        cfg = _det_cfg(pick_mode=PickMode.FORCED, n_task_types=4)
        env = DeterministicEnv(cfg)

        # Agent 0 is assigned type 0; place type-0 task one step to the right
        s = State(
            agent_positions=(Grid(0, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),  # matches agent 0's assignment
        )

        def policy(state, phase2=False):
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, policy, env, n_steps=1))

        # Should be 2 transitions: move then pick
        assert len(transitions) == 2
        move_tr, pick_tr = transitions
        assert move_tr.action == Action.RIGHT
        assert pick_tr.action == make_pick_action(0)
        assert pick_tr.discount == 1.0
        assert pick_tr.rewards[0] == 1.0  # correct type, R_high

    def test_choice_pick_stay_no_reward(self):
        """In CHOICE mode, STAY in phase 2 yields zero reward."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        env = DeterministicEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        def policy(state, phase2=False):
            if phase2:
                return Action.STAY  # decline to pick
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, policy, env, n_steps=1))
        assert len(transitions) == 2
        pick_tr = transitions[1]
        assert pick_tr.action == Action.STAY
        assert pick_tr.discount == 1.0
        assert all(r == 0.0 for r in pick_tr.rewards)
        # Task should still be on grid
        assert len(pick_tr.s_t_after.task_positions) == 1

    def test_choice_pick_wrong_type_rlow(self):
        """In CHOICE mode, picking wrong type yields R_low."""
        cfg = _det_cfg(pick_mode=PickMode.CHOICE, r_low=-1.0, n_task_types=4)
        env = DeterministicEnv(cfg)

        # Agent 0 assigned to type 0, but task is type 3
        s = State(
            agent_positions=(Grid(0, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(3,),
        )

        def policy(state, phase2=False):
            if phase2:
                return make_pick_action(3)  # pick wrong type
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, policy, env, n_steps=1))
        pick_tr = transitions[1]
        assert pick_tr.rewards[0] == -1.0
        assert pick_tr.discount == 1.0

    def test_no_task_single_transition(self):
        """Moving to empty cell yields exactly one transition."""
        cfg = _det_cfg(pick_mode=PickMode.FORCED)
        env = DeterministicEnv(cfg)

        s = State(
            agent_positions=(Grid(0, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(4, 4),),
            actor=0,
            task_types=(0,),
        )

        def policy(state, phase2=False):
            return Action.RIGHT

        transitions = list(rollout_trajectory(s, policy, env, n_steps=1))
        assert len(transitions) == 1
        assert transitions[0].action == Action.RIGHT
        assert transitions[0].discount == 0.99  # gamma

    def test_nearest_correct_task_heuristic_phase2_always_picks(self):
        """NEAREST_CORRECT_TASK heuristic picks in phase 2 regardless of type."""
        from orchard.policy import nearest_correct_task_action
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)
        # Agent 0 on a wrong-type task cell
        s = State(
            agent_positions=(Grid(1, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(3,),  # agent 0 is assigned type 0
        )
        action = nearest_correct_task_action(s, cfg, phase2=True)
        assert action.is_pick()  # always picks in phase 2

    def test_nearest_correct_task_stay_wrong_heuristic(self):
        """NEAREST_CORRECT_TASK_STAY_WRONG STAYs on wrong type, picks correct."""
        from orchard.policy import nearest_correct_task_stay_wrong_action
        cfg = _det_cfg(pick_mode=PickMode.CHOICE)

        # Wrong type at cell
        s_wrong = State(
            agent_positions=(Grid(1, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(3,),  # agent 0 assigned type 0
        )
        assert nearest_correct_task_stay_wrong_action(s_wrong, cfg, phase2=True) == Action.STAY

        # Correct type at cell
        s_right = State(
            agent_positions=(Grid(1, 0), Grid(3, 3), Grid(3, 2), Grid(3, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),  # agent 0's type
        )
        action = nearest_correct_task_stay_wrong_action(s_right, cfg, phase2=True)
        assert action == make_pick_action(0)
