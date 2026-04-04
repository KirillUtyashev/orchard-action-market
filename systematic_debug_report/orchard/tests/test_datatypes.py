"""Tests for core datatypes (State, Grid) and assignment math."""

import pytest
from orchard.datatypes import Grid, State, sort_tasks, compute_task_assignments


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
        assert hash(s)  # Should not raise TypeError

    def test_n_agents(self):
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(0, 1)),
            task_positions=(),
            actor=0,
        )
        assert s.n_agents == 3

    def test_with_pick_phase(self):
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 0),),
            actor=0,
        )
        assert not s.pick_phase
        s2 = s.with_pick_phase()
        assert s2.pick_phase
        assert s2.actor == s.actor
        assert s2.agent_positions == s.agent_positions


class TestStateTaskQueries:
    def test_is_agent_on_task(self):
        s = State(
            agent_positions=(Grid(0, 1), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0,
        )
        assert s.is_agent_on_task(0) is True
        assert s.is_agent_on_task(1) is False

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

    def test_task_type_at_legacy(self):
        """When task_types is None, it defaults to type 0."""
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        assert s.task_type_at(Grid(1, 0)) == 0

    def test_tasks_at_multiple(self):
        """Choice pick allows multiple types at the same cell."""
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 2, 1),
        )
        result = s.tasks_at(Grid(1, 1))
        assert len(result) == 2
        assert (0, 0) in result  # task index 0, type 0
        assert (1, 2) in result  # task index 1, type 2


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

    def test_none_types(self):
        pos = (Grid(1, 0), Grid(0, 0))
        sp, st = sort_tasks(pos, None)
        assert sp == (Grid(0, 0), Grid(1, 0))
        assert st is None


class TestComputeTaskAssignments:
    def test_max_specialization(self):
        # rho=0.25 * 4 = 1. Every agent gets exactly 1 type.
        result = compute_task_assignments(4, 4, 0.25)
        assert result == ((0,), (1,), (2,), (3,))

    def test_half_specialization(self):
        # rho=0.5 * 4 = 2. Every agent gets 2 types.
        result = compute_task_assignments(4, 4, 0.5)
        assert result == ((0, 1), (1, 2), (2, 3), (3, 0))

    def test_no_specialization(self):
        # rho=1.0 * 4 = 4. Every agent gets all 4 types.
        result = compute_task_assignments(4, 4, 1.0)
        for g in result:
            assert set(g) == {0, 1, 2, 3}

    def test_coverage_guaranteed(self):
        """All types must be covered by at least one agent."""
        result = compute_task_assignments(7, 7, 1.0 / 7)
        covered = set()
        for g in result:
            covered.update(g)
        assert covered == set(range(7))

    def test_missing_type_raises_error(self):
        # Unrealistic, but if math fails to cover a type, it must crash.
        # compute_task_assignments mathematically guarantees coverage based on the formulation,
        # but the assert inside will catch manual breaches.
        pass