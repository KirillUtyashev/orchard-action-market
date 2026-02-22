"""Tests for policy functions: nearest_apple, greedy, epsilon_greedy."""

import pytest

from orchard.enums import Action, EnvType
from orchard.policy import nearest_apple_action
from orchard.datatypes import EnvConfig, Grid, State


def _make_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=3, width=3, n_agents=2, n_apples=1,
        gamma=0.9, r_picker=-1.0, force_pick=True,
        max_apples=1, env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


class TestNearestApple:
    def test_move_toward_apple_right(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(1, 0), Grid(2, 2)),
            apple_positions=(Grid(1, 2),),
            actor=0,
        )
        assert nearest_apple_action(s, cfg) == Action.RIGHT

    def test_move_toward_apple_down(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(0, 1), Grid(2, 2)),
            apple_positions=(Grid(2, 1),),
            actor=0,
        )
        assert nearest_apple_action(s, cfg) == Action.DOWN

    def test_already_on_apple(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(1, 1), Grid(2, 2)),
            apple_positions=(Grid(1, 1),),
            actor=0,
        )
        # Distance is 0 for STAY, but also for any move that keeps same dist
        # Actually on apple: manhattan=0, STAY gives 0
        # LEFT → (1,0): dist to (1,1) = 1
        # So STAY wins
        action = nearest_apple_action(s, cfg)
        assert action == Action.STAY

    def test_tie_break_priority(self):
        """When multiple actions give same distance, ACTION_PRIORITY breaks tie."""
        cfg = _make_cfg()
        # Agent at (0,0), apple at (1,1): dist=2
        # DOWN → (1,0): dist to (1,1)=1
        # RIGHT → (0,1): dist to (1,1)=1
        # Both give dist=1. Priority: LEFT, DOWN, RIGHT, UP, STAY
        # DOWN comes before RIGHT in priority → DOWN wins
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            apple_positions=(Grid(1, 1),),
            actor=0,
        )
        assert nearest_apple_action(s, cfg) == Action.DOWN

    def test_second_agent_as_actor(self):
        cfg = _make_cfg()
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 0)),
            apple_positions=(Grid(2, 2),),
            actor=1,
        )
        assert nearest_apple_action(s, cfg) == Action.RIGHT
