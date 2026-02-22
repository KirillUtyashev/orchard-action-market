"""Tests for State creation and properties."""

import pytest
from orchard.datatypes import Grid, State


def test_state_frozen():
    s = State(
        agent_positions=(Grid(0, 0), Grid(1, 1)),
        apple_positions=(Grid(0, 1),),
        actor=0,
    )
    with pytest.raises(AttributeError):
        s.actor = 1  # type: ignore


def test_state_hashable():
    s = State(
        agent_positions=(Grid(0, 0), Grid(1, 1)),
        apple_positions=(Grid(0, 1),),
        actor=0,
    )
    {s}  # should not raise


def test_is_agent_on_apple_true():
    s = State(
        agent_positions=(Grid(0, 1), Grid(1, 1)),
        apple_positions=(Grid(0, 1),),
        actor=0,
    )
    assert s.is_agent_on_apple(0) is True
    assert s.is_agent_on_apple(1) is False


def test_is_agent_on_apple_false():
    s = State(
        agent_positions=(Grid(0, 0), Grid(1, 1)),
        apple_positions=(Grid(0, 1),),
        actor=0,
    )
    assert s.is_agent_on_apple(0) is False


def test_n_agents():
    s = State(
        agent_positions=(Grid(0, 0), Grid(1, 1), Grid(0, 1)),
        apple_positions=(),
        actor=0,
    )
    assert s.n_agents == 3


def test_apple_ages_none_by_default():
    s = State(
        agent_positions=(Grid(0, 0),),
        apple_positions=(Grid(1, 1),),
        actor=0,
    )
    assert s.apple_ages is None


def test_apple_ages_set():
    s = State(
        agent_positions=(Grid(0, 0),),
        apple_positions=(Grid(1, 0), Grid(1, 1)),
        actor=0,
        apple_ages=(3, 5),
    )
    assert s.apple_ages == (3, 5)
