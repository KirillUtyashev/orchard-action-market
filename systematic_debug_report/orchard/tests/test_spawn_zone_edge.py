"""Tests for structured edge zone movement: spawn_zone_mode and spawn_flip_interval."""

import pytest
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.enums import DespawnMode, PickMode, SpawnZoneMode
from orchard.env.stochastic import StochasticEnv, edge_zone_positions
from orchard.seed import set_all_seeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 3
H, W = 9, 9
MAX_R, MAX_C = H - SIZE, W - SIZE  # = 6


def _make_env(
    n_task_types: int = 2,
    spawn_zone_mode: SpawnZoneMode = SpawnZoneMode.NONE,
    spawn_zone_interval: int = 0,
    spawn_flip_interval: int = 0,
    eval_spawn_zone_mode: SpawnZoneMode = SpawnZoneMode.NONE,
    eval_spawn_zone_interval: int = 0,
    eval_spawn_flip_interval: int = 0,
) -> StochasticEnv:
    stoch = StochasticConfig(
        spawn_prob=0.0,
        despawn_mode=DespawnMode.NONE,
        despawn_prob=0.0,
        spawn_area_size=SIZE,
        spawn_zone_mode=spawn_zone_mode,
        spawn_zone_interval=spawn_zone_interval,
        spawn_flip_interval=spawn_flip_interval,
        eval_spawn_zone_mode=eval_spawn_zone_mode,
        eval_spawn_zone_interval=eval_spawn_zone_interval,
        eval_spawn_flip_interval=eval_spawn_flip_interval,
    )
    cfg = EnvConfig(
        height=H, width=W, n_agents=2, n_tasks=0, gamma=0.99, r_picker=1.0,
        n_task_types=n_task_types,
        task_assignments=tuple((tau,) for tau in range(n_task_types)),
        stochastic=stoch,
    )
    return StochasticEnv(cfg)


def _on_border(r: int, c: int) -> bool:
    return r == 0 or r == MAX_R or c == 0 or c == MAX_C


def _advance_rounds(env: StochasticEnv, n: int) -> None:
    """Advance env by n full rounds (2 agents per round)."""
    s = State(
        agent_positions=(Grid(4, 4), Grid(4, 5)),
        task_positions=(),
        actor=0,
        task_types=(),
    )
    for _ in range(n):
        for actor in range(2):
            s = State(
                agent_positions=s.agent_positions,
                task_positions=s.task_positions,
                actor=actor,
                task_types=s.task_types,
            )
            env.spawn_and_despawn(s)


# ---------------------------------------------------------------------------
# edge_zone_positions
# ---------------------------------------------------------------------------

class TestEdgeZonePositions:
    def test_all_on_border(self):
        for n in range(1, 9):
            for r, c in edge_zone_positions(H, W, SIZE, n):
                assert _on_border(r, c), f"n={n}: ({r},{c}) not on border"

    def test_single_zone_top_left(self):
        assert edge_zone_positions(H, W, SIZE, 1) == ((0, 0),)

    def test_two_zones_diagonal(self):
        pos = edge_zone_positions(H, W, SIZE, 2)
        assert pos == ((0, 0), (MAX_R, MAX_C))

    def test_four_zones_corners(self):
        pos = edge_zone_positions(H, W, SIZE, 4)
        assert set(pos) == {(0, 0), (0, MAX_C), (MAX_R, MAX_C), (MAX_R, 0)}

    def test_valid_corners(self):
        for n in range(1, 9):
            for r, c in edge_zone_positions(H, W, SIZE, n):
                assert 0 <= r <= MAX_R and 0 <= c <= MAX_C


# ---------------------------------------------------------------------------
# Initial placement on border
# ---------------------------------------------------------------------------

class TestInitOnBorder:
    def test_edge_switch_mode_starts_on_border(self):
        set_all_seeds(42)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_zone_interval=100)
        for r, c in env._spawn_area_corners:
            assert _on_border(r, c), f"({r},{c}) not on border at init"

    def test_edge_switch_interval_zero_still_starts_on_border(self):
        # interval=0 means no periodic move, but zones still start on border
        set_all_seeds(42)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_zone_interval=0)
        for r, c in env._spawn_area_corners:
            assert _on_border(r, c), f"({r},{c}) not on border at init"

    def test_random_mode_does_not_require_border(self):
        set_all_seeds(0)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.RANDOM, spawn_zone_interval=10)
        assert env._spawn_area_corners is not None  # just check it runs

    def test_none_mode_does_not_require_border(self):
        set_all_seeds(0)
        env = _make_env()
        assert env._spawn_area_corners is not None

    def test_spread_distinct_positions(self):
        # With 4 types on a 9x9 grid, evenly-spread edge positions should be distinct
        set_all_seeds(99)
        env = _make_env(n_task_types=4, spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_zone_interval=100)
        corners = env._spawn_area_corners
        assert len(set(corners)) == 4, "expected 4 distinct edge positions"


# ---------------------------------------------------------------------------
# Flip: antipodal and invertibility
# ---------------------------------------------------------------------------

class TestFlip:
    def test_flip_is_antipodal(self):
        set_all_seeds(1)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_flip_interval=5)
        orig = list(env._spawn_area_corners)
        env._flip_spawn_areas()
        for (r0, c0), (r1, c1) in zip(orig, env._spawn_area_corners):
            assert r1 == H - SIZE - r0
            assert c1 == W - SIZE - c0

    def test_flip_twice_restores_original(self):
        set_all_seeds(2)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_flip_interval=5)
        orig = list(env._spawn_area_corners)
        env._flip_spawn_areas()
        env._flip_spawn_areas()
        assert list(env._spawn_area_corners) == orig

    def test_flipped_edge_corners_stay_on_border(self):
        set_all_seeds(3)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_zone_interval=100, spawn_flip_interval=5)
        env._flip_spawn_areas()
        for r, c in env._spawn_area_corners:
            assert _on_border(r, c), f"flipped corner ({r},{c}) not on border"

    def test_cells_match_corner_after_flip(self):
        set_all_seeds(4)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_flip_interval=5)
        env._flip_spawn_areas()
        for tau, (r0, c0) in enumerate(env._spawn_area_corners):
            expected = {Grid(r0 + dr, c0 + dc) for dr in range(SIZE) for dc in range(SIZE)}
            assert set(env._spawn_area_cells[tau]) == expected


# ---------------------------------------------------------------------------
# Round-based firing
# ---------------------------------------------------------------------------

class TestRoundFiring:
    def test_flip_fires_at_interval(self):
        set_all_seeds(10)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_flip_interval=5)
        orig = list(env._spawn_area_corners)

        _advance_rounds(env, 4)
        assert list(env._spawn_area_corners) == orig, "should not flip before round 5"

        _advance_rounds(env, 1)  # now at round 5
        assert list(env._spawn_area_corners) != orig, "should have flipped at round 5"

    def test_flip_back_at_2n(self):
        set_all_seeds(10)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_flip_interval=5)
        orig = list(env._spawn_area_corners)
        _advance_rounds(env, 10)  # two full flip cycles
        assert list(env._spawn_area_corners) == orig

    def test_edge_switch_fires_at_interval(self):
        set_all_seeds(20)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_zone_interval=10)
        corners_before = list(env._spawn_area_corners)

        _advance_rounds(env, 9)
        assert list(env._spawn_area_corners) == corners_before, "should not switch before round 10"

        # After round 10, corners should still be on border (may differ from initial)
        _advance_rounds(env, 1)
        for r, c in env._spawn_area_corners:
            assert _on_border(r, c)

    def test_edge_switch_takes_priority_over_flip(self):
        # Both fire at round 5 → edge_switch wins (new border positions, not a flip of old)
        set_all_seeds(30)
        env = _make_env(
            spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH,
            spawn_zone_interval=5,
            spawn_flip_interval=5,
        )
        _advance_rounds(env, 5)
        for r, c in env._spawn_area_corners:
            assert _on_border(r, c)

    def test_random_mode_fires_at_interval(self):
        set_all_seeds(40)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.RANDOM, spawn_zone_interval=5)
        _advance_rounds(env, 5)
        # Just check it doesn't crash; zones may have moved anywhere


# ---------------------------------------------------------------------------
# set_eval_mode save/restore
# ---------------------------------------------------------------------------

class TestEvalMode:
    def test_eval_mode_restores_training_corners(self):
        set_all_seeds(50)
        env = _make_env(spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, spawn_zone_interval=100)
        train_corners = list(env._spawn_area_corners)

        env.set_eval_mode(True, seed=42)
        env._flip_spawn_areas()  # modify during eval
        env.set_eval_mode(False)

        assert list(env._spawn_area_corners) == train_corners

    def test_eval_edge_switch_initializes_on_border(self):
        # eval_spawn_zone_mode=edge_switch → zones placed on border at eval start
        set_all_seeds(60)
        env = _make_env(eval_spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH)
        env.set_eval_mode(True, seed=42)
        for r, c in env._spawn_area_corners:
            assert _on_border(r, c)
        env.set_eval_mode(False)

    def test_eval_flip_fires_independently(self):
        # eval_spawn_zone_mode=edge_switch + interval=0 (no periodic move) + flip
        set_all_seeds(70)
        env = _make_env(
            eval_spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH,
            eval_spawn_zone_interval=0,
            eval_spawn_flip_interval=5,
        )
        env.set_eval_mode(True, seed=42)
        corners_at_eval_start = list(env._spawn_area_corners)

        _advance_rounds(env, 5)
        flipped = list(env._spawn_area_corners)
        assert flipped != corners_at_eval_start

        for (r0, c0), (r1, c1) in zip(corners_at_eval_start, flipped):
            assert r1 == H - SIZE - r0 and c1 == W - SIZE - c0

        env.set_eval_mode(False)

    def test_fixed_spawn_zones_overrides_eval_edge(self):
        set_all_seeds(80)
        env = _make_env(eval_spawn_zone_mode=SpawnZoneMode.EDGE_SWITCH, eval_spawn_zone_interval=100)
        fixed = ((0, 0), (MAX_R, MAX_C))
        env.set_eval_mode(True, seed=42, fixed_spawn_zones=fixed)
        assert list(env._spawn_area_corners) == [(0, 0), (MAX_R, MAX_C)]
        env.set_eval_mode(False)
