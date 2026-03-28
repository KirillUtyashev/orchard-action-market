"""Tests for State creation and properties (chunk 1)."""

import pytest
from orchard.datatypes import Grid, State, sort_tasks


# ---------------------------------------------------------------------------
# Basic State construction
# ---------------------------------------------------------------------------
class TestStateBasic:
    def test_state_frozen(self):
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
        )
        with pytest.raises(AttributeError):
            s.actor = 1  # type: ignore

    def test_state_hashable(self):
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
        )
        {s}  # should not raise

    def test_n_agents(self):
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(0, 1)),
            task_positions=(),
            actor=0,
        )
        assert s.n_agents == 3


# ---------------------------------------------------------------------------
# is_agent_on_task
# ---------------------------------------------------------------------------
class TestIsAgentOnTask:
    def test_on_task(self):
        s = State(
            agent_positions=(Grid(0, 1), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
        )
        assert s.is_agent_on_task(0) is True
        assert s.is_agent_on_task(1) is False

    def test_not_on_task(self):
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
        )
        assert s.is_agent_on_task(0) is False

    def test_backward_compat_alias(self):
        s = State(
            agent_positions=(Grid(0, 1),),
            task_positions=(Grid(0, 1),),
            actor=0,
        )
        assert s.is_agent_on_apple(0) is True


# ---------------------------------------------------------------------------
# task_types — None in legacy mode
# ---------------------------------------------------------------------------
class TestTaskTypesLegacy:
    def test_task_types_none_by_default(self):
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        assert s.task_types is None

    def test_task_type_at_legacy(self):
        """With task_types=None (legacy), task_type_at returns 0."""
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1), Grid(2, 2)),
            actor=0,
        )
        assert s.task_type_at(Grid(1, 1)) == 0
        assert s.task_type_at(Grid(2, 2)) == 0
        assert s.task_type_at(Grid(3, 3)) is None


# ---------------------------------------------------------------------------
# task_types — with types
# ---------------------------------------------------------------------------
class TestTaskTypesMulti:
    def test_construction(self):
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 1, 2),
        )
        assert s.task_types == (0, 1, 2)

    def test_task_type_at(self):
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 1, 2),
        )
        assert s.task_type_at(Grid(1, 0)) == 0
        assert s.task_type_at(Grid(1, 1)) == 1
        assert s.task_type_at(Grid(2, 0)) == 2
        assert s.task_type_at(Grid(5, 5)) is None

    def test_tasks_at_single(self):
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 0), Grid(1, 1)),
            actor=0,
            task_types=(0, 1),
        )
        result = s.tasks_at(Grid(1, 0))
        assert result == [(0, 0)]  # index 0, type 0

    def test_tasks_at_multiple_types_same_cell(self):
        """Choice pick allows multiple types at same cell."""
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 2, 1),
        )
        result = s.tasks_at(Grid(1, 1))
        assert len(result) == 2
        assert (0, 0) in result  # index 0, type 0
        assert (1, 2) in result  # index 1, type 2

    def test_tasks_at_empty(self):
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(0,),
        )
        assert s.tasks_at(Grid(3, 3)) == []


# ---------------------------------------------------------------------------
# Backward compat aliases
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    def test_apple_positions_alias(self):
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1), Grid(2, 2)),
            actor=0,
        )
        assert s.apple_positions == (Grid(1, 1), Grid(2, 2))
        assert s.apple_positions is s.task_positions


# ---------------------------------------------------------------------------
# sort_tasks
# ---------------------------------------------------------------------------
class TestSortTasks:
    def test_already_sorted(self):
        pos = (Grid(0, 0), Grid(0, 1), Grid(1, 0))
        types = (2, 1, 0)
        sp, st = sort_tasks(pos, types)
        assert sp == pos
        assert st == types

    def test_unsorted(self):
        pos = (Grid(1, 0), Grid(0, 1), Grid(0, 0))
        types = (2, 1, 0)
        sp, st = sort_tasks(pos, types)
        assert sp == (Grid(0, 0), Grid(0, 1), Grid(1, 0))
        assert st == (0, 1, 2)

    def test_empty(self):
        sp, st = sort_tasks((), ())
        assert sp == ()
        assert st == ()

    def test_none_types(self):
        pos = (Grid(1, 0), Grid(0, 0))
        sp, st = sort_tasks(pos, None)
        assert sp == (Grid(0, 0), Grid(1, 0))
        assert st is None

    def test_parallel_consistency(self):
        """Positions and types stay paired after sort."""
        pos = (Grid(2, 2), Grid(0, 0), Grid(1, 1))
        types = (7, 3, 5)
        sp, st = sort_tasks(pos, types)
        # Original pairs: (2,2)->7, (0,0)->3, (1,1)->5
        # Sorted by pos: (0,0)->3, (1,1)->5, (2,2)->7
        assert sp == (Grid(0, 0), Grid(1, 1), Grid(2, 2))
        assert st == (3, 5, 7)
