"""Tests for the custom Action class and related helpers."""

import pytest
from orchard.enums import (
    Action,
    PickMode,
    make_pick_action,
    num_actions,
    NUM_MOVE_ACTIONS,
    ACTION_PRIORITY,
)


class TestActionBasic:
    def test_singletons(self):
        assert Action.UP.value == 0
        assert Action.DOWN.value == 1
        assert Action.LEFT.value == 2
        assert Action.RIGHT.value == 3
        assert Action.STAY.value == 4
        assert Action.PICK.value == 5

    def test_construction_by_value(self):
        a = Action(0)
        assert a == Action.UP
        assert a.value == 0

    def test_delta(self):
        assert Action.UP.delta == (-1, 0)
        assert Action.DOWN.delta == (1, 0)
        assert Action.LEFT.delta == (0, -1)
        assert Action.RIGHT.delta == (0, 1)
        assert Action.STAY.delta == (0, 0)
        assert Action.PICK.delta == (0, 0)

    def test_name(self):
        assert Action.UP.name == 'UP'
        assert Action.STAY.name == 'STAY'
        assert Action.PICK.name == 'PICK'

    def test_repr(self):
        assert repr(Action.UP) == 'Action.UP'

    def test_equality(self):
        assert Action(0) == Action(0)
        assert Action(0) == Action.UP
        assert Action(0) != Action(1)

    def test_hash(self):
        s = {Action.UP, Action(0), Action.DOWN}
        assert len(s) == 2  # UP and Action(0) map to the same hash

    def test_not_equal_to_int(self):
        assert (Action.UP == 0) is NotImplemented or Action.UP != 0


class TestActionMovePick:
    def test_is_move(self):
        assert Action.UP.is_move() is True
        assert Action.STAY.is_move() is True
        assert Action.PICK.is_pick() is True
        assert Action.UP.is_pick() is False

    def test_pick_type_movement(self):
        assert Action.UP.pick_type() is None
        assert Action.STAY.pick_type() is None

    def test_pick_type_generic(self):
        assert Action.PICK.pick_type() == 0

    def test_pick_delta(self):
        """Pick actions have (0,0) delta — no movement."""
        p = make_pick_action(3)
        assert p.delta == (0, 0)


class TestMakePickAction:
    def test_pick_0(self):
        a = make_pick_action(0)
        assert a.value == 5
        assert a.is_pick() is True
        assert a.pick_type() == 0
        assert a.name == 'PICK'  # value 5 maps exactly to 'PICK'

    def test_pick_3(self):
        a = make_pick_action(3)
        assert a.value == 8
        assert a.is_pick() is True
        assert a.pick_type() == 3
        assert a.name == 'PICK_3'

    def test_pick_equality(self):
        assert make_pick_action(2) == make_pick_action(2)
        assert make_pick_action(2) != make_pick_action(3)


class TestNumActions:
    def test_forced(self):
        # Forced mode ignores n_task_types, always 5 move actions
        assert num_actions(PickMode.FORCED, 4) == 5

    def test_choice(self):
        # Choice mode adds n_task_types pick actions
        assert num_actions(PickMode.CHOICE, 4) == 9
        assert num_actions(PickMode.CHOICE, 1) == 6
        assert num_actions(PickMode.CHOICE, 10) == 15


class TestConstants:
    def test_num_move_actions(self):
        assert NUM_MOVE_ACTIONS == 5

    def test_action_priority_length(self):
        assert len(ACTION_PRIORITY) == 5
        # All actions in the priority queue should be move actions
        assert all(a.is_move() for a in ACTION_PRIORITY)