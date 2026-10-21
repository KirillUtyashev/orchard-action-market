"""All-action (expected) policy gradient for the per-agent actor-critic path."""

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.actor_critic import PolicyNetwork
from orchard.config import load_config
from orchard.env import create_env
from orchard.seed import set_all_seeds

CFG = "orchard/configs/reference.yaml"


def _net():
    cfg = load_config(CFG)
    set_all_seeds(0)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, env, n_networks=cfg.env.n_agents)
    return cfg, env, PolicyNetwork(cfg.model, cfg.env, lr=0.1)


def test_default_flag_is_off():
    assert load_config(CFG).train.algorithm.all_action_gradient is False
    cfg = load_config(CFG, ["train.algorithm.all_action_gradient=true"])
    assert cfg.train.algorithm.all_action_gradient is True


def test_all_action_gradient_matches_expected_single_sample_gradient():
    # E_{a~pi}[A(a) grad log pi(a)] computed exactly must equal the all-action gradient.
    cfg, env, net = _net()
    s = env.init_state()
    enc = encoding.encode(s, s.actor)
    mask = np.zeros(net.output_dim, dtype=bool); mask[:5] = True
    adv = np.array([1.0, -0.5, 2.0, 0.0, -1.5] + [0.0] * (net.output_dim - 5))
    adv = adv - float((net.get_action_probabilities(enc, mask) * adv).sum())  # baseline-subtracted

    probs = torch.as_tensor(net.get_action_probabilities(enc, mask))
    expected = None
    for a in range(5):
        net.zero_grad()
        lp = torch.log_softmax(net._masked_logits(net.forward(enc), mask), dim=1)[0, a]
        (-(probs[a] * adv[a]) * lp).backward()
        g = torch.cat([p.grad.flatten() for p in net.parameters()])
        expected = g if expected is None else expected + g

    net.zero_grad()
    before = [p.detach().clone() for p in net.parameters()]
    net.add_all_action_experience(enc, mask, np.where(mask, adv, 0.0))
    net.train_batch()
    # SGD step: param_new = param - lr * grad  -> recover grad
    got = torch.cat([((b - p.detach()) / 0.1).flatten() for b, p in zip(before, net.parameters())])
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_illegal_actions_get_no_gradient_signal():
    cfg, env, net = _net()
    s = env.init_state(); enc = encoding.encode(s, s.actor)
    mask = np.zeros(net.output_dim, dtype=bool); mask[:5] = True
    net.add_all_action_experience(enc, mask, np.where(mask, np.arange(net.output_dim, dtype=float), 0.0))
    m = net.train_batch()
    assert np.isfinite(m["loss"])
    assert float(net.get_action_probabilities(enc, mask)[~mask].sum()) == 0.0
