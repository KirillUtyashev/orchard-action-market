"""Train per-agent actors (CNN[x,x] k3 + MLP[x]) by imitating argmax-Q labels, then play them.

Usage (from orchard_rl/):
  python distill_train.py --width 8 --seed 1 --epochs 15 --out output/distill_results

Each agent i gets its OWN PolicyNetwork trained only on the decisions where agent i acts
(decentralized actors). Validation = the files in --val-parts (held out from training).
After training, the greedy (argmax) per-agent policy is played for --eval-steps steps from
three fresh eval seeds; team_rps is the same metric as metrics.csv. Writes <out>/w<x>_s<seed>.json
and the actor weights <out>/w<x>_s<seed>.pt (list of state_dicts, one per agent).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import orchard.encoding as encoding
from orchard.actor_critic import PolicyNetwork, policy_index_to_action
from orchard.actor_critic.policy_network import _stack_encoder_outputs
from orchard.config import load_config
from orchard.datatypes import EncoderOutput
from orchard.env import create_env
from orchard.eval import evaluate_policy_metrics
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer

CONFIG = "orchard/configs/baseline_match_n11_actor.yaml"


def load_parts(paths):
    ds = [np.load(p) for p in paths]
    return {k: np.concatenate([d[k] for d in ds]) for k in ("grid", "scalar", "mask", "label", "actor")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", default="output/distill_data")
    ap.add_argument("--val-parts", nargs="*", default=["part17", "part18"])
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)

    cfg = load_config(CONFIG, ["train.discount_method=round_steps"])
    x = a.width
    am = cfg.actor_model
    am = type(am)(**{**am.__dict__, "conv_specs": ((x, 3), (x, 3)), "mlp_dims": (x,)})
    set_all_seeds(a.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, env, n_networks=cfg.env.n_agents)
    tr = create_trainer(cfg, env)  # only used for state encoding and legal masks

    val_files = [f"{a.data}/{p}.npz" for p in a.val_parts]
    train_files = [f for f in sorted(glob.glob(f"{a.data}/part*.npz")) if f not in val_files]
    tr_d, va_d = load_parts(train_files), load_parts(val_files)
    N = cfg.env.n_agents
    nets = [PolicyNetwork(am, cfg.env, lr=a.lr) for _ in range(N)]
    n_params = sum(p.numel() for p in nets[0].parameters())

    def sub(d, i):
        m = d["actor"] == i
        return (torch.as_tensor(d["grid"][m], dtype=torch.float32), torch.as_tensor(d["scalar"][m]),
                torch.as_tensor(d["mask"][m]), torch.as_tensor(d["label"][m]))

    history = []
    for i, net in enumerate(nets):
        G, S, M, Y = sub(tr_d, i)
        Gv, Sv, Mv, Yv = sub(va_d, i)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        for ep in range(a.epochs):
            perm = torch.randperm(len(Y))
            for j in range(0, len(Y), 128):
                idx = perm[j:j + 128]
                lg = net._masked_logits(net.forward(EncoderOutput(scalar=S[idx], grid=G[idx])), M[idx].numpy())
                loss = F.cross_entropy(lg, Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tr_acc = (net._masked_logits(net.forward(EncoderOutput(scalar=S[:5000], grid=G[:5000])), M[:5000].numpy()).argmax(1) == Y[:5000]).float().mean().item()
            va_acc = (net._masked_logits(net.forward(EncoderOutput(scalar=Sv, grid=Gv)), Mv.numpy()).argmax(1) == Yv).float().mean().item()
        history.append({"agent": i, "n_train": len(Y), "n_val": len(Yv), "train_acc": tr_acc, "val_acc": va_acc})
        print(f"w{x} agent {i}: n_train {len(Y)} train_acc {tr_acc:.3f} val_acc {va_acc:.3f}", flush=True)

    def policy(s):
        enc = tr._encode_actor_state(s, s.actor)
        probs = nets[s.actor].get_action_probabilities(enc, tr._legal_mask(s))
        return policy_index_to_action(int(np.argmax(probs)))

    rps = []
    for es in (101, 102, 103):
        env.set_eval_mode(True, seed=es)
        rps.append(evaluate_policy_metrics(env.init_state(), policy, env, a.eval_steps)["team_rps"])
    res = {"width": x, "seed": a.seed, "params_per_actor": n_params, "val_acc_mean": float(np.mean([h["val_acc"] for h in history])),
           "train_acc_mean": float(np.mean([h["train_acc"] for h in history])), "team_rps_evals": rps,
           "team_rps_mean": float(np.mean(rps)), "epochs": a.epochs, "per_agent": history}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"w{x}_s{a.seed}.json").write_text(json.dumps(res, indent=1))
    torch.save([n.state_dict() for n in nets], out / f"w{x}_s{a.seed}.pt")
    print(f"w{x} seed {a.seed}: val_acc {res['val_acc_mean']:.3f} team_rps {res['team_rps_mean']:.3f} {rps}", flush=True)


if __name__ == "__main__":
    main()
