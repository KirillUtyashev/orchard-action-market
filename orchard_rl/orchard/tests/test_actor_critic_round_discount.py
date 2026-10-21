"""discount_method is honoured by the per-agent actor-critic trainer."""

import pytest

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.env import create_env
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer

CFG = "orchard/configs/reference.yaml"


def _trainer(method):
    cfg = load_config(CFG, [f"train.discount_method={method}"])
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, env, n_networks=cfg.env.n_agents)
    return cfg, env, create_trainer(cfg, env)


def test_round_steps_discounts_once_per_round():
    cfg, env, tr = _trainer("round_steps")
    s = env.init_state()
    got = []
    for t in range(2 * cfg.env.n_agents):
        got.append((s.actor, tr._move_discount(s)))
        s = tr.step(s, t)
    for actor, d in got:
        assert d == (cfg.env.gamma if actor == 0 else 1.0)


def test_team_steps_discounts_every_move():
    cfg, env, tr = _trainer("team_steps")
    s = env.init_state()
    for actor in range(cfg.env.n_agents):
        assert tr._move_discount(s) == cfg.env.gamma
        s = env.advance_actor(s)
