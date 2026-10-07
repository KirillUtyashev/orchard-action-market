"""Tests for the learned actor with a swappable team critic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import orchard.encoding as encoding
from orchard.actor_critic import build_phase1_legal_mask, build_phase2_legal_mask
from orchard.config import load_config
from orchard.datatypes import Grid, State
from orchard.env import create_env
from orchard.enums import Action, make_pick_action
from orchard.learned_actor.actor import SharedActor
from orchard.learned_actor.critic import CentralizedTeamCritic, DecentralizedTeamCritic
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer
from orchard.trainer.learned_actor import LearnedActorTrainer

REFERENCE = str(Path(__file__).resolve().parents[1] / "configs" / "reference.yaml")
BASE_OVERRIDES = [
    "env.height=5", "env.width=5", "env.n_agents=3", "env.n_task_types=3",
    "env.n_tasks=2", "env.max_tasks_per_type=3", "env.proficiency_width=2",
    "env.stochastic.sigma_a=1.0", "env.stochastic.sigma_b=8.0",
    "model.conv_specs=[[4,3]]", "model.mlp_dims=[8]",
    "train.discount_method=round_steps",
    "train.algorithm.learned_actor.enabled=true",
]


def _build(learning_type: str = "decentralized", extra: list[str] | None = None, seed: int = 7):
    cfg = load_config(
        REFERENCE,
        BASE_OVERRIDES + [f"train.learning_type={learning_type}", f"train.seed={seed}"] + (extra or []),
    )
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    from orchard.enums import LearningType
    n_nets = 1 if cfg.train.learning_type == LearningType.CENTRALIZED else cfg.env.n_agents
    encoding.init_encoder(cfg.model.encoder, env, n_networks=n_nets)
    trainer = create_trainer(cfg, env)
    assert isinstance(trainer, LearnedActorTrainer)
    return cfg, env, trainer


# 1. Reward consistency ----------------------------------------------------
@pytest.mark.parametrize("reward_generation", ["independent", "circulant_all_to_all"])
def test_dec_targets_sum_to_cen_target(reward_generation):
    extra = [
        f"env.stochastic.reward_generation={reward_generation}",
        "env.n_agents=5", "env.n_task_types=5",
        "env.relatedness_width=1", "env.proficiency_width=1",
    ]
    cfg, env, dec = _build("decentralized", extra)
    _, _, cen = _build("centralized", extra)
    state = env.init_state()
    rng = np.random.default_rng(0)
    nonzero = 0
    for _ in range(300):
        mask = build_phase1_legal_mask(state, cfg.env)
        action = Action(int(rng.choice(np.flatnonzero(mask))))
        s_moved = env.apply_action(state, action)
        if s_moved.is_agent_on_task(s_moved.actor, env.proficiency_positive_types[s_moved.actor]):
            pick = s_moved.with_pick_phase()
            pmask = build_phase2_legal_mask(pick, cfg.env, env.proficiency_positive_types)
            idx = int(rng.choice(np.flatnonzero(pmask)))
            pick_action = Action(idx) if idx < 5 else make_pick_action(idx - 5)
            s_moved, rewards = env.resolve_pick(
                s_moved, pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            dec_targets = dec._critic.targets(rewards)
            cen_targets = cen._critic.targets(rewards)
            assert len(cen_targets) == 1
            assert sum(dec_targets) == pytest.approx(cen_targets[0])
            assert cen_targets[0] == pytest.approx(sum(rewards))
            nonzero += int(abs(sum(rewards)) > 0)
        state = env.advance_actor(env.spawn_and_despawn(s_moved))
    assert nonzero > 0


# 2. Masking -----------------------------------------------------------------
def test_actor_never_picks_without_task_and_masks_illegal_actions():
    cfg, env, trainer = _build()
    actor = trainer.actor_networks[0]
    # No task on the actor's cell: only the 5 moves are legal.
    state = State(
        agent_positions=(Grid(0, 0), Grid(4, 4), Grid(2, 2)),
        task_positions=(Grid(3, 3),), actor=0, task_types=(1,),
    )
    enc = encoding.encode(state, 0)
    mask = build_phase1_legal_mask(state, cfg.env)
    probs = actor.get_action_probabilities(enc, mask)
    assert np.all(probs[~mask] == 0.0)
    assert probs[5:].sum() == 0.0
    set_all_seeds(0)
    picks = [trainer._sample_index(probs) for _ in range(2000)]
    assert all(i < 5 for i in picks)

    # Pick phase with one eligible task type: other pick actions are illegal.
    pick_state = State(
        agent_positions=(Grid(3, 3), Grid(4, 4), Grid(2, 2)),
        task_positions=(Grid(3, 3),), actor=0, task_types=(1,), pick_phase=True,
    )
    pmask = build_phase2_legal_mask(pick_state, cfg.env, env.proficiency_positive_types)
    pprobs = actor.get_action_probabilities(encoding.encode(pick_state, 0), pmask)
    assert set(np.flatnonzero(pmask).tolist()) == {int(Action.STAY.value), 5 + 1}
    assert np.all(pprobs[~pmask] == 0.0)
    assert pprobs[5 + 0] == 0.0 and pprobs[5 + 2] == 0.0
    assert pprobs[5 + 1] > 0.0


# 3. Gradient isolation ------------------------------------------------------
@pytest.mark.parametrize("learning_type", ["decentralized", "centralized"])
def test_actor_update_leaves_critic_gradients_untouched(learning_type):
    _, env, trainer = _build(learning_type)
    state = env.init_state()
    state = trainer.step(state, 0)           # collects experience, round not complete
    assert trainer._batch_states
    critic_params = [p for net in trainer.critic_networks for p in net.parameters()]
    for p in critic_params:
        p.grad = None
    trainer._flush_actor_batch(0)
    assert all(p.grad is None for p in critic_params)
    assert any(p.grad is not None for p in trainer.actor_networks[0].parameters())


def test_actor_rejects_attached_advantages():
    _, _, trainer = _build()
    actor = trainer.actor_networks[0]
    adv = torch.ones(1, requires_grad=True)
    enc = encoding.encode(
        State(agent_positions=(Grid(0, 0), Grid(1, 1), Grid(2, 2)), task_positions=(), actor=0, task_types=()), 0,
    )
    with pytest.raises(ValueError):
        actor.update([enc], np.ones((1, 8), dtype=bool), [4], adv, 0.0, True)


# 4. Critic switch -------------------------------------------------------------
def test_critic_switch_changes_only_the_critic():
    cfg_d, _, dec = _build("decentralized")
    cfg_c, _, cen = _build("centralized")
    assert isinstance(dec._critic, DecentralizedTeamCritic)
    assert isinstance(cen._critic, CentralizedTeamCritic)
    assert len(dec.critic_networks) == cfg_d.env.n_agents
    assert len(cen.critic_networks) == 1
    a_d, a_c = dec.actor_networks[0], cen.actor_networks[0]
    assert type(a_d) is type(a_c) is SharedActor
    sd_d, sd_c = a_d.state_dict(), a_c.state_dict()
    assert sd_d.keys() == sd_c.keys()
    for k in sd_d:
        assert torch.equal(sd_d[k], sd_c[k])
    # Both critics are queried through the same interface.
    env = create_env(cfg_d.env)
    state = env.init_state()
    afters = [env.apply_action(state, Action(i)) for i in range(5)]
    assert dec._critic.team_values(state, afters).shape == (5,)
    assert cen._critic.team_values(state, afters).shape == (5,)


# Advantage ---------------------------------------------------------------------
def test_afterstate_advantage_matches_manual_q_without_double_counting():
    cfg, env, trainer = _build("decentralized", ["train.algorithm.learned_actor.normalize_advantage=false"])
    pick_state = State(
        agent_positions=(Grid(3, 3), Grid(4, 4), Grid(2, 2)),
        task_positions=(Grid(3, 3),), actor=0, task_types=(1,), pick_phase=True,
    )
    mask = build_phase2_legal_mask(pick_state, cfg.env, env.proficiency_positive_types)
    enc, _, probs = trainer._probs(pick_state)
    legal = [int(i) for i in np.flatnonzero(mask)]
    pick_idx = 5 + 1
    action = make_pick_action(1)
    trainer._decide(pick_state, action, enc, mask, probs, 1.0, 0)

    qs = []
    for i in legal:
        a = Action(i) if i < 5 else make_pick_action(i - 5)
        after, rewards = trainer._after_state_for_action(pick_state, a)
        v = trainer._critic.team_values(pick_state, [after])[0].item()
        qs.append(sum(rewards) + 1.0 * v)
    qs = np.array(qs)
    expected = qs[legal.index(pick_idx)] - float(np.dot(probs[legal], qs))
    assert trainer._batch_advantages[-1] == pytest.approx(expected, abs=1e-5)
    # Picking really pays reward here, so r must appear in Q exactly once.
    assert qs[legal.index(pick_idx)] != qs[legal.index(4)]


@pytest.mark.parametrize("advantage", ["afterstate", "td_error"])
@pytest.mark.parametrize("learning_type", ["decentralized", "centralized"])
def test_training_runs_for_all_modes(advantage, learning_type):
    _, env, trainer = _build(learning_type, [f"train.algorithm.learned_actor.advantage={advantage}"])
    state = env.init_state()
    for t in range(30):
        state = trainer.step(state, t)
    metrics = trainer.get_main_metrics()
    assert metrics["policy_entropy_mean"] != ""
    assert np.isfinite(float(metrics["actor_loss_mean"]))


def test_round_discount_applies_gamma_once_per_round():
    cfg, _, trainer = _build()
    s = State(agent_positions=(Grid(0, 0),) * 3, task_positions=(), actor=0, task_types=())
    gammas = [trainer._move_discount(State(s.agent_positions, s.task_positions, a, s.task_types)) for a in range(3)]
    assert gammas == [cfg.env.gamma, 1.0, 1.0]
