"""Tests for stable apple IDs and StableIdEncoder.

Tests apple_ids persistence through pick/spawn/despawn and
encoder slot stability (SMAC-style).
"""

import pytest

from orchard.enums import Action, DespawnMode, EncoderType, EnvType
from orchard.datatypes import EnvConfig, Grid, State, StochasticConfig
from orchard.env.deterministic import DeterministicEnv
from orchard.env.stochastic import StochasticEnv
from orchard.encoding.relative import StableIdEncoder
from orchard.seed import rng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=3, width=3, n_agents=2, n_apples=2,
        gamma=0.9, r_picker=-1.0, force_pick=True,
        max_apples=4, env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _stoch_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=3, width=3, n_agents=2, n_apples=2,
        gamma=0.9, r_picker=-1.0, force_pick=True,
        max_apples=4, env_type=EnvType.STOCHASTIC,
        stochastic=StochasticConfig(
            spawn_prob=1.0,       # always spawn (deterministic for testing)
            despawn_mode=DespawnMode.NONE,
            despawn_prob=0.0,
            apple_lifetime=100,
        ),
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _state_with_ids(agents, apples, actor, ids, ages=None):
    """Convenience: build State with apple_ids."""
    return State(
        agent_positions=tuple(Grid(*a) for a in agents),
        apple_positions=tuple(Grid(*a) for a in apples),
        actor=actor,
        apple_ages=ages,
        apple_ids=tuple(ids),
    )


# ===================================================================
# Part 1: apple_ids present in State
# ===================================================================

class TestStateAppleIds:
    def test_default_none(self):
        """apple_ids defaults to None for backward compat."""
        s = State(
            agent_positions=(Grid(0, 0),),
            apple_positions=(Grid(1, 1),),
            actor=0,
        )
        assert s.apple_ids is None

    def test_explicit_ids(self):
        s = _state_with_ids(
            agents=[(0, 0), (0, 1)],
            apples=[(1, 0), (2, 2)],
            actor=0,
            ids=[0, 1],
        )
        assert s.apple_ids == (0, 1)


# ===================================================================
# Part 2: Deterministic env — init_state sets apple_ids
# ===================================================================

class TestDeterministicAppleIds:
    def test_init_state_has_ids(self):
        cfg = _det_cfg()
        env = DeterministicEnv(cfg)
        s = env.init_state()
        assert s.apple_ids is not None
        assert len(s.apple_ids) == len(s.apple_positions)
        # IDs should be 0, 1, ... for initial apples
        assert s.apple_ids == tuple(range(len(s.apple_positions)))

    def test_apply_action_preserves_ids(self):
        cfg = _det_cfg()
        env = DeterministicEnv(cfg)
        s = env.init_state()
        s2 = env.apply_action(s, Action.DOWN)
        assert s2.apple_ids == s.apple_ids

    def test_advance_actor_preserves_ids(self):
        cfg = _det_cfg()
        env = DeterministicEnv(cfg)
        s = env.init_state()
        s2 = env.advance_actor(s)
        assert s2.apple_ids == s.apple_ids

    def test_resolve_pick_removes_correct_id(self):
        """When apple is picked, its ID is removed; others stay."""
        cfg = _det_cfg()
        env = DeterministicEnv(cfg)
        # Agent 0 on apple at (1,0) which has id=0, other apple at (2,2) with id=1
        s = _state_with_ids(
            agents=[(1, 0), (0, 1)],
            apples=[(1, 0), (2, 2)],
            actor=0,
            ids=[0, 1],
        )
        s_picked, rewards = env.resolve_pick(s)
        assert len(s_picked.apple_positions) == 1
        assert s_picked.apple_positions[0] == Grid(2, 2)
        assert s_picked.apple_ids == (1,)  # id=0 removed, id=1 survives

    def test_spawn_assigns_recycled_id(self):
        """After picking apple id=0, newly spawned apple should get id=0."""
        cfg = _det_cfg(n_apples=1, max_apples=2)
        env = DeterministicEnv(cfg)
        # One apple at (2,2) with id=1 (id=0 was picked earlier)
        s = _state_with_ids(
            agents=[(0, 0), (0, 1)],
            apples=[(2, 2)],
            actor=0,
            ids=[1],
        )
        s_spawned = env.spawn_and_despawn(s)
        if len(s_spawned.apple_positions) > 1:
            # New apple should get id=0 (smallest available)
            assert 0 in s_spawned.apple_ids
            assert 1 in s_spawned.apple_ids
            # IDs parallel positions
            assert len(s_spawned.apple_ids) == len(s_spawned.apple_positions)

    def test_full_step_preserves_id_of_surviving_apple(self):
        """Through a full step where one apple is picked, surviving apple keeps its ID."""
        cfg = _det_cfg(n_apples=2, max_apples=4)
        env = DeterministicEnv(cfg)
        # Agent 0 at (1,0), apple at (1,0) id=0 and (2,2) id=3
        s = _state_with_ids(
            agents=[(1, 0), (0, 1)],
            apples=[(1, 0), (2, 2)],
            actor=0,
            ids=[0, 3],
        )
        # STAY action → agent stays on (1,0) → picks apple id=0
        tr = env.step(s, Action.STAY)
        # Apple (2,2) with id=3 must still be present
        assert Grid(2, 2) in tr.s_t_next.apple_positions
        idx = tr.s_t_next.apple_positions.index(Grid(2, 2))
        assert tr.s_t_next.apple_ids[idx] == 3


# ===================================================================
# Part 3: Stochastic env — apple_ids through spawn/despawn
# ===================================================================

class TestStochasticAppleIds:
    def test_init_state_has_ids(self):
        cfg = _stoch_cfg()
        env = StochasticEnv(cfg)
        rng.seed(42)
        s = env.init_state()
        assert s.apple_ids is not None
        assert len(s.apple_ids) == len(s.apple_positions)
        assert s.apple_ids == tuple(range(len(s.apple_positions)))

    def test_despawn_removes_correct_ids(self):
        """With LIFETIME despawn, expired apple's ID is removed."""
        cfg = _stoch_cfg(
            stochastic=StochasticConfig(
                spawn_prob=0.0,  # no new spawns
                despawn_mode=DespawnMode.LIFETIME,
                despawn_prob=0.0,
                apple_lifetime=2,
            ),
        )
        env = StochasticEnv(cfg)
        # Apple at (1,0) age=0 id=0, apple at (2,2) age=1 id=1
        # After despawn with lifetime=2: age=1 < 2 → keep, age=2 not < 2 → despawn
        # Wait: ages are incremented AFTER filtering. So age=1 stays (1<2), gets aged to 2.
        # apple age=0 id=0 → 0<2 keep → aged to 1
        # apple age=1 id=1 → 1<2 keep → aged to 2
        # Actually both survive. Let me set ages so one dies:
        # age=1 → 1<2 keep. age=2 → 2 not < 2 → despawn. But ages start at 0 so need to track.
        # Let me use lifetime=1 so age=0 survives, age=1 dies
        cfg2 = _stoch_cfg(
            stochastic=StochasticConfig(
                spawn_prob=0.0,
                despawn_mode=DespawnMode.LIFETIME,
                despawn_prob=0.0,
                apple_lifetime=1,
            ),
        )
        env2 = StochasticEnv(cfg2)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0), Grid(2, 2)),
            actor=0,
            apple_ages=(0, 1),  # age=0 survives (0<1), age=1 dies (1 not < 1)
            apple_ids=(5, 2),
        )
        s2 = env2.spawn_and_despawn(s)
        # Only apple at (1,0) with id=5 should survive
        assert len(s2.apple_positions) == 1
        assert s2.apple_positions[0] == Grid(1, 0)
        assert s2.apple_ids == (5,)

    def test_spawn_assigns_smallest_available_id(self):
        """Newly spawned apples get the smallest unused ID."""
        cfg = _stoch_cfg(
            max_apples=4,
            stochastic=StochasticConfig(
                spawn_prob=1.0,  # always spawn
                despawn_mode=DespawnMode.NONE,
                despawn_prob=0.0,
                apple_lifetime=100,
            ),
        )
        env = StochasticEnv(cfg)
        # Currently have apples with ids 1 and 3 (ids 0 and 2 are free)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0), Grid(2, 2)),
            actor=0,
            apple_ids=(1, 3),
        )
        rng.seed(42)
        s2 = env.spawn_and_despawn(s)
        # With spawn_prob=1.0, new apples should get ids from {0, 2} (smallest available)
        used_ids = set(s2.apple_ids)
        surviving = {1, 3}
        new_ids = used_ids - surviving
        # New IDs should be drawn from {0, 2}
        assert new_ids.issubset({0, 2})

    def test_ids_parallel_positions_after_respawn(self):
        """apple_ids stays parallel to apple_positions after spawn+sort."""
        cfg = _stoch_cfg(
            max_apples=4,
            stochastic=StochasticConfig(
                spawn_prob=1.0,
                despawn_mode=DespawnMode.NONE,
                despawn_prob=0.0,
                apple_lifetime=100,
            ),
        )
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(2, 2),),
            actor=0,
            apple_ids=(3,),
        )
        rng.seed(99)
        s2 = env.spawn_and_despawn(s)
        # (2,2) must still map to id=3
        if Grid(2, 2) in s2.apple_positions:
            idx = s2.apple_positions.index(Grid(2, 2))
            assert s2.apple_ids[idx] == 3


# ===================================================================
# Part 4: StableIdEncoder
# ===================================================================

class TestStableIdEncoder:
    def _enc_cfg(self, **kw) -> EnvConfig:
        return _det_cfg(height=3, width=3, n_agents=2, max_apples=4, **kw)

    def test_dim(self):
        cfg = self._enc_cfg()
        enc = StableIdEncoder(cfg)
        # max_apples*3 + (n_agents-1)*3 + 4 = 4*3 + 1*3 + 4 = 19
        assert enc.scalar_dim() == 19

    def test_output_shape(self):
        cfg = self._enc_cfg()
        enc = StableIdEncoder(cfg)
        s = _state_with_ids(
            agents=[(0, 0), (0, 1)],
            apples=[(1, 0), (2, 2)],
            actor=0,
            ids=[0, 1],
        )
        out = enc.encode(s, 0)
        assert out.scalar is not None
        assert out.grid is None
        assert out.scalar.shape == (19,)

    def test_apple_in_correct_slot(self):
        """Apple with id=2 should populate slot 2 (indices 6,7,8), not slot 0."""
        cfg = self._enc_cfg()
        enc = StableIdEncoder(cfg)
        # Single apple at (1,1) with id=2. Agent at (0,0).
        s = _state_with_ids(
            agents=[(0, 0), (2, 2)],
            apples=[(1, 1)],
            actor=0,
            ids=[2],
        )
        out = enc.encode(s, 0)
        feats = out.scalar.tolist()
        dh = cfg.height - 1  # =2
        dw = cfg.width - 1   # =2

        # Slot 0 (id=0): should be zeros (no apple with id=0)
        assert feats[0] == 0.0  # is_present
        assert feats[1] == 0.0  # dr
        assert feats[2] == 0.0  # dc

        # Slot 1 (id=1): should be zeros
        assert feats[3] == 0.0
        assert feats[4] == 0.0
        assert feats[5] == 0.0

        # Slot 2 (id=2): apple at (1,1), agent at (0,0) → dr=1/2, dc=1/2
        assert feats[6] == 1.0  # is_present
        assert pytest.approx(feats[7], abs=1e-6) == 1.0 / dh
        assert pytest.approx(feats[8], abs=1e-6) == 1.0 / dw

        # Slot 3 (id=3): zeros
        assert feats[9] == 0.0

    def test_slot_stability_after_pick(self):
        """After picking one apple, surviving apple stays in same slot."""
        cfg = self._enc_cfg()
        enc = StableIdEncoder(cfg)

        # Before pick: apples at (1,0) id=0 and (2,2) id=3
        s_before = _state_with_ids(
            agents=[(0, 0), (0, 1)],
            apples=[(1, 0), (2, 2)],
            actor=0,
            ids=[0, 3],
        )
        out_before = enc.encode(s_before, 0)

        # After pick of apple id=0: only (2,2) id=3 remains
        s_after = _state_with_ids(
            agents=[(0, 0), (0, 1)],
            apples=[(2, 2)],
            actor=0,
            ids=[3],
        )
        out_after = enc.encode(s_after, 0)

        before = out_before.scalar.tolist()
        after = out_after.scalar.tolist()

        # Slot 3 (indices 9,10,11) should be IDENTICAL before and after
        assert before[9] == after[9] == 1.0  # is_present
        assert pytest.approx(before[10], abs=1e-6) == after[10]  # dr
        assert pytest.approx(before[11], abs=1e-6) == after[11]  # dc

        # Slot 0 (indices 0,1,2): was apple, now zeros
        assert after[0] == 0.0
        assert after[1] == 0.0
        assert after[2] == 0.0

    def test_agent_features_by_index(self):
        """Other agents should be ordered by index, not distance."""
        cfg = _det_cfg(height=3, width=3, n_agents=3, max_apples=4)
        enc = StableIdEncoder(cfg)
        # Agent 0 at (0,0), agent 1 at (2,2) (far), agent 2 at (0,1) (close)
        s = _state_with_ids(
            agents=[(0, 0), (2, 2), (0, 1)],
            apples=[(1, 1)],
            actor=0,
            ids=[0],
        )
        out = enc.encode(s, 0)
        feats = out.scalar.tolist()
        dh = cfg.height - 1  # 2
        dw = cfg.width - 1   # 2
        n = cfg.n_agents      # 3

        # Apple slots: 4*3 = 12 features (indices 0-11)
        # Agent 1 (index 1, far): indices 12, 13, 14
        assert pytest.approx(feats[12], abs=1e-6) == 2.0 / dh  # dr
        assert pytest.approx(feats[13], abs=1e-6) == 2.0 / dw  # dc

        # Agent 2 (index 2, close): indices 15, 16, 17
        assert pytest.approx(feats[15], abs=1e-6) == 0.0 / dh  # dr=0
        assert pytest.approx(feats[16], abs=1e-6) == 1.0 / dw  # dc=1/2

    def test_no_apples_all_zeros(self):
        """With no apples, all apple slots are zero."""
        cfg = self._enc_cfg()
        enc = StableIdEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(),
            actor=0,
            apple_ids=(),
        )
        out = enc.encode(s, 0)
        feats = out.scalar.tolist()
        # First 12 features (4 apple slots × 3) should all be 0
        assert feats[:12] == [0.0] * 12
