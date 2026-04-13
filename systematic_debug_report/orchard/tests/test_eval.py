"""Tests for evaluation rollouts, 2-phase transitions, and metrics calculation."""

import pytest
from orchard.enums import Action, PickMode, make_pick_action
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.eval import rollout_trajectory, evaluate_policy_metrics


def _make_eval_cfg(pick_mode=PickMode.FORCED) -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=2, n_tasks=2, gamma=0.9, r_picker=1.0,
        n_task_types=2, pick_mode=pick_mode, max_tasks_per_type=2,
        task_assignments=((0,), (1,)), r_low=-1.0,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )


class TestRolloutTrajectory:
    def test_no_pick_yields_one_transition(self):
        cfg = _make_eval_cfg()
        env = StochasticEnv(cfg)
        
        # Agent 0 at (0,0), task at (4,4) - far away
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(4, 4),),
            actor=0, task_types=(0,)
        )
        
        def dummy_policy(state, phase2=False):
            return Action.RIGHT
            
        transitions = list(rollout_trajectory(s, dummy_policy, env, n_steps=1))
        
        # Should be exactly 1 transition (just the move)
        assert len(transitions) == 1
        t = transitions[0]
        assert t.action == Action.RIGHT
        assert t.discount == 0.9 # Regular gamma for movement
        assert t.rewards == (0.0, 0.0)

    def test_forced_pick_yields_two_transitions(self):
        cfg = _make_eval_cfg(pick_mode=PickMode.FORCED)
        env = StochasticEnv(cfg)
        
        # Agent 0 at (0,0), task at (0,1)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )
        
        def dummy_policy(state, phase2=False):
            return Action.RIGHT # Move right onto the task
            
        transitions = list(rollout_trajectory(s, dummy_policy, env, n_steps=1))
        
        # Forced pick: Should be 2 transitions (Move, then Auto-Pick)
        assert len(transitions) == 2
        t_move, t_pick = transitions
        
        assert t_move.action == Action.RIGHT
        assert t_move.discount == 0.9
        assert sum(t_move.rewards) == 0.0
        
        assert t_pick.action == make_pick_action(0) # Environment forces the pick
        assert t_pick.discount == 1.0 # Pick discount is always 1.0
        assert t_pick.rewards[0] == 1.0 # Correct pick reward

    def test_choice_pick_uses_policy_for_phase2(self):
        cfg = _make_eval_cfg(pick_mode=PickMode.CHOICE)
        env = StochasticEnv(cfg)
        
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )
        
        def choice_policy(state, phase2=False):
            if phase2:
                assert state.pick_phase is True
                return Action.STAY # Decline the pick!
            return Action.RIGHT # Phase 1: Move right
            
        transitions = list(rollout_trajectory(s, choice_policy, env, n_steps=1))
        
        assert len(transitions) == 2
        t_move, t_pick = transitions
        
        assert t_pick.action == Action.STAY
        assert t_pick.discount == 1.0
        assert sum(t_pick.rewards) == 0.0 # No pick, no reward


class TestEvaluatePolicyMetrics:
    def test_metrics_calculation(self):
        cfg = _make_eval_cfg(pick_mode=PickMode.CHOICE)
        env = StochasticEnv(cfg)
        
        # Agent 0 (type 0) at (0,0), task 0 at (0,1), task 1 at (0,2)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1), Grid(0, 2)),
            actor=0, task_types=(0, 1)
        )
        
        # A scripted policy that takes exactly 2 steps to pick a correct task and a wrong task
        step_counter = 0
        def scripted_policy(state, phase2=False):
            nonlocal step_counter
            if not phase2:
                step_counter += 1
                return Action.RIGHT # Step 1 lands on type 0, Step 2 lands on type 1
            else:
                # Always pick what we are standing on
                tau = state.task_type_at(state.agent_positions[state.actor])
                return make_pick_action(tau)

        metrics = evaluate_policy_metrics(s, scripted_policy, env, n_steps=2)
        
        # N_steps = 2
        # Step 1: Agent 0 picks Type 0 -> +1.0
        # Step 2: Agent 1 (actor advances!) is at (1,1). It moves RIGHT to (1,2) -> No task. Reward 0.0
        # Wait, the scripted policy is called for whichever agent is active.
        
        # Let's verify the exact sequence:
        # Step 1 (A0): Moves right to (0,1). Picks Type 0 -> Correct! r = +1.0
        # Step 2 (A1): At (1,1). Moves right to (1,2). No task. r = 0.0
        
        # Correct total reward: 1.0 over 2 steps -> rps = 0.5
        assert metrics["rps"] == 0.5
        assert metrics["team_rps"] == 0.5
        # Correct picks: 1 over 2 steps -> correct_pps = 0.5
        assert metrics["correct_pps"] == 0.5
        # Wrong picks: 0 -> wrong_pps = 0.0
        assert metrics["wrong_pps"] == 0.0
