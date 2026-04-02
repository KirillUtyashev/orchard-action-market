"""Tests for policy functions (chunk 4)."""

import pytest

from orchard.enums import Action, EnvType, Heuristic, PickMode, make_pick_action
from orchard.policy import (
    nearest_task_action, nearest_apple_action,
    nearest_correct_task_action, heuristic_action, get_all_actions,
)
from orchard.datatypes import EnvConfig, Grid, State


def _make_legacy_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=3, width=3, n_agents=2, n_tasks=1,
        gamma=0.9, r_picker=-1.0, pick_mode=PickMode.FORCED,
        max_tasks=1, env_type=EnvType.DETERMINISTIC,
        task_assignments=((0,), (0,)),
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_task_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=5, width=5, n_agents=4, n_tasks=3,
        gamma=0.99, r_picker=1.0,
        n_task_types=4, r_low=0.0,
        task_assignments=((0,), (1,), (2,), (3,)),
        pick_mode=PickMode.FORCED,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


# ---------------------------------------------------------------------------
# nearest_task_action (renamed from nearest_apple_action)
# ---------------------------------------------------------------------------
class TestNearestTask:
    def test_move_toward_task_right(self):
        cfg = _make_legacy_cfg()
        s = State(
            agent_positions=(Grid(1, 0), Grid(2, 2)),
            task_positions=(Grid(1, 2),),
            actor=0,
        )
        assert nearest_task_action(s, cfg) == Action.RIGHT

    def test_move_toward_task_down(self):
        cfg = _make_legacy_cfg()
        s = State(
            agent_positions=(Grid(0, 1), Grid(2, 2)),
            task_positions=(Grid(2, 1),),
            actor=0,
        )
        assert nearest_task_action(s, cfg) == Action.DOWN

    def test_already_on_task(self):
        cfg = _make_legacy_cfg()
        s = State(
            agent_positions=(Grid(1, 1), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        assert nearest_task_action(s, cfg) == Action.STAY

    def test_backward_compat_alias(self):
        cfg = _make_legacy_cfg()
        s = State(
            agent_positions=(Grid(1, 0), Grid(2, 2)),
            task_positions=(Grid(1, 2),),
            actor=0,
        )
        assert nearest_apple_action(s, cfg) == Action.RIGHT

    def test_no_tasks(self):
        cfg = _make_legacy_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(),
            actor=0,
        )
        assert nearest_task_action(s, cfg) == Action.STAY


# ---------------------------------------------------------------------------
# nearest_correct_task_action
# ---------------------------------------------------------------------------
class TestNearestCorrectTask:
    def test_moves_toward_correct_type(self):
        """Agent 0 (type 0) moves toward type-0 task, ignoring type-1."""
        cfg = _make_task_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4), Grid(4, 3), Grid(4, 2)),
            task_positions=(Grid(0, 4), Grid(2, 0)),
            actor=0,
            task_types=(1, 0),  # type 1 at (0,4), type 0 at (2,0)
        )
        # Agent 0 owns type 0 → should move toward (2,0)
        action = nearest_correct_task_action(s, cfg)
        assert action == Action.DOWN

    def test_ignores_wrong_type(self):
        """Agent 0 (type 0) ignores type-1 task even if closer."""
        cfg = _make_task_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4), Grid(4, 3), Grid(4, 2)),
            task_positions=(Grid(0, 1), Grid(4, 0)),
            actor=0,
            task_types=(1, 0),  # type 1 at (0,1) close, type 0 at (4,0) far
        )
        action = nearest_correct_task_action(s, cfg)
        # Should move toward (4,0) not (0,1)
        assert action == Action.DOWN

    def test_no_correct_tasks_stay(self):
        cfg = _make_task_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4), Grid(4, 3), Grid(4, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(1,),  # only type 1, agent 0 owns type 0
        )
        assert nearest_correct_task_action(s, cfg) == Action.STAY

    def test_choice_pick_when_on_correct_type(self):
        """In choice mode phase 2, issue pick(τ) when standing on correct type."""
        cfg = _make_task_cfg(pick_mode=PickMode.CHOICE)
        s = State(
            agent_positions=(Grid(1, 1), Grid(4, 4), Grid(4, 3), Grid(4, 2)),
            task_positions=(Grid(1, 1), Grid(2, 2)),
            actor=0,
            task_types=(0, 1),  # type 0 at (1,1) = agent 0's type
        )
        action = nearest_correct_task_action(s, cfg, phase2=True)
        assert action == make_pick_action(0)
        assert action.is_pick()
        assert action.pick_type() == 0

    def test_choice_move_when_on_wrong_type(self):
        """In choice mode, move toward correct type even if on wrong type."""
        cfg = _make_task_cfg(pick_mode=PickMode.CHOICE)
        s = State(
            agent_positions=(Grid(1, 1), Grid(4, 4), Grid(4, 3), Grid(4, 2)),
            task_positions=(Grid(1, 1), Grid(3, 3)),
            actor=0,
            task_types=(1, 0),  # type 1 at (1,1) = wrong, type 0 at (3,3) = correct
        )
        action = nearest_correct_task_action(s, cfg)
        assert action.is_move()
        assert action == Action.DOWN  # move toward (3,3)


# ---------------------------------------------------------------------------
# heuristic_action dispatch
# ---------------------------------------------------------------------------
class TestHeuristicDispatch:
    def test_nearest_task(self):
        cfg = _make_legacy_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 2),),
            actor=0,
        )
        assert heuristic_action(s, cfg, Heuristic.NEAREST_TASK) == Action.RIGHT

    def test_nearest_correct_task(self):
        cfg = _make_task_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(4, 4), Grid(4, 3), Grid(4, 2)),
            task_positions=(Grid(2, 0),),
            actor=0,
            task_types=(0,),
        )
        assert heuristic_action(s, cfg, Heuristic.NEAREST_CORRECT_TASK) == Action.DOWN


# ---------------------------------------------------------------------------
# get_all_actions
# ---------------------------------------------------------------------------
class TestGetAllActions:
    def test_forced_mode(self):
        cfg = _make_task_cfg(pick_mode=PickMode.FORCED)
        actions = get_all_actions(cfg)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_choice_mode(self):
        cfg = _make_task_cfg(pick_mode=PickMode.CHOICE, n_task_types=4)
        # state=None returns move-only (used for network sizing)
        actions = get_all_actions(cfg)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)

    def test_choice_mode_on_task_cell(self):
        """Choice mode with actor on task cell returns STAY + pick actions."""
        cfg = _make_task_cfg(pick_mode=PickMode.CHOICE, n_task_types=4)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(2,),
        )
        from orchard.policy import get_phase2_actions
        actions = get_phase2_actions(s, cfg)
        assert len(actions) == 2  # STAY + pick(2)
        assert Action.STAY in actions
        assert make_pick_action(2) in actions
        assert not any(a.is_move() and a != Action.STAY for a in actions)

    def test_choice_mode_not_on_task_cell(self):
        """Choice mode with actor not on any task cell returns moves only."""
        cfg = _make_task_cfg(pick_mode=PickMode.CHOICE, n_task_types=4)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(3, 3),),
            actor=0,
            task_types=(1,),
        )
        actions = get_all_actions(cfg, s)
        assert len(actions) == 5
        assert all(a.is_move() for a in actions)
