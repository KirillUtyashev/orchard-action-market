import pytest
from orchard.enums import Action, Heuristic, PickMode, make_pick_action
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.policy import (
    get_all_actions,
    get_phase2_actions,
    nearest_task_action,
    nearest_correct_task_action,
    nearest_correct_task_stay_wrong_action,
    heuristic_action,
)

def _make_cfg(pick_mode=PickMode.CHOICE) -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=4, n_tasks=2, gamma=0.99, r_picker=1.0,
        n_task_types=2, pick_mode=pick_mode, max_tasks_per_type=2,
        task_assignments=((0,), (0,), (1,), (1,)),
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )

class TestActionMasking:
    def test_get_all_actions_always_returns_moves(self):
        cfg = _make_cfg()
        actions = get_all_actions(cfg)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_get_phase2_actions_not_on_task(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(0,)
        )
        # Not on a task cell, so phase 2 should return an empty list
        actions = get_phase2_actions(s, cfg)
        assert actions == []

    def test_get_phase2_actions_on_single_task(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(1,)
        )
        actions = get_phase2_actions(s, cfg)
        # Should allow STAY or picking Type 1
        assert len(actions) == 2
        assert Action.STAY in actions
        assert make_pick_action(1) in actions

    def test_get_phase2_actions_on_stacked_tasks(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(2, 2), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(2, 2), Grid(2, 2)),
            actor=0, task_types=(0, 1)
        )
        actions = get_phase2_actions(s, cfg)
        # Should allow STAY, pick(0), or pick(1)
        assert len(actions) == 3
        assert Action.STAY in actions
        assert make_pick_action(0) in actions
        assert make_pick_action(1) in actions


class TestHeuristicPolicies:
    def test_nearest_task(self):
        cfg = _make_cfg()
        # Actor 0 at (0,0), Tasks at (0,2) and (4,0)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(1, 2), Grid(1, 3)),
            task_positions=(Grid(0, 2), Grid(4, 0)),
            actor=0, task_types=(1, 0)
        )
        # Nearest task is at (0,2), distance 2. Action should be RIGHT
        assert nearest_task_action(s, cfg) == Action.RIGHT

    def test_nearest_correct_task_ignores_wrong_types(self):
        cfg = _make_cfg()
        # Actor 0 is assigned Type 0.
        # Type 1 task at (0,1) [distance 1]
        # Type 0 task at (3,0) [distance 3]
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(1, 2), Grid(1, 3)),
            task_positions=(Grid(0, 1), Grid(3, 0)),
            actor=0, task_types=(1, 0)
        )
        # Should ignore the close Type 1 task and move DOWN toward Type 0
        assert nearest_correct_task_action(s, cfg) == Action.DOWN

    def test_nearest_correct_stay_wrong_phase2(self):
        cfg = _make_cfg()
        # Actor 0 (assigned Type 0) is standing on a Type 1 task
        s_wrong = State(
            agent_positions=(Grid(2, 2), Grid(1, 1), Grid(1, 2), Grid(1, 3)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(1,)
        )
        # Phase 2 logic: should STAY because it's the wrong type
        action_wrong = nearest_correct_task_stay_wrong_action(s_wrong.with_pick_phase(), cfg)
        assert action_wrong == Action.STAY

        # Actor 0 standing on a Type 0 task
        s_correct = State(
            agent_positions=(Grid(2, 2), Grid(1, 1), Grid(1, 2), Grid(1, 3)),
            task_positions=(Grid(2, 2),),
            actor=0, task_types=(0,)
        )
        # Phase 2 logic: should pick because it's the correct type
        action_correct = nearest_correct_task_stay_wrong_action(s_correct.with_pick_phase(), cfg)
        assert action_correct == make_pick_action(0)

    def test_heuristic_dispatch(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(1, 2), Grid(1, 3)),
            task_positions=(Grid(0, 1), Grid(3, 0)),
            actor=0, task_types=(1, 0)
        )
        
        # Nearest task goes RIGHT (to 0,1)
        a1 = heuristic_action(s, cfg, Heuristic.NEAREST_TASK)
        assert a1 == Action.RIGHT
        
        # Nearest correct goes DOWN (to 3,0)
        a2 = heuristic_action(s, cfg, Heuristic.NEAREST_CORRECT_TASK)
        assert a2 == Action.DOWN