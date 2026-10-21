"""Collect (state, legal mask, argmax-Q label) tuples from a trained value-based critic.

Teacher = argmax over legal actions of sum_j [r_j + discount * V_j(afterstate)] using the
critics in --critic (e.g. Kirill/Tadd's 8M-step dec CNN[8,8] checkpoint). Rollout actions
are the teacher's with probability 1-eps, uniformly random legal with probability eps,
so the states cover some off-teacher situations. Labels are always the teacher's argmax.

Usage (from orchard_rl/):
  python distill_collect.py --config orchard/configs/baseline_match_n11_actor.yaml \
      --critic output/vb_critic_8M_seed1235.pt --seed 11 --steps 100000 --out output/distill_data/part11.npz
"""
from __future__ import annotations

import argparse

import numpy as np

import orchard.encoding as encoding
from orchard.actor_critic import policy_index_to_action
from orchard.config import load_config
from orchard.env import create_env
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config, ["train.discount_method=round_steps"])
    set_all_seeds(a.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, env, n_networks=cfg.env.n_agents)
    tr = create_trainer(cfg, env)
    tr.load_critic_checkpoint(a.critic)
    rng = np.random.default_rng(a.seed)

    grids, scalars, masks, labels, actors = [], [], [], [], []
    env.set_eval_mode(True, seed=a.seed)
    s = env.init_state()
    n_dec = 0
    for _ in range(a.steps):
        # one environment turn = a move decision plus (optionally) a pick decision
        for _phase in range(2):
            mask = tr._legal_mask(s)
            disc = 1.0 if s.pick_phase else tr._move_discount(s)
            q, *_ = tr._enumerate_action_objectives(s, mask, disc)
            label = int(np.where(mask, q, -np.inf).argmax())
            enc = tr._encode_actor_state(s, s.actor)
            grids.append(enc.grid.numpy().astype(np.uint8)); scalars.append(enc.scalar.numpy().astype(np.float32))
            masks.append(mask); labels.append(label); actors.append(s.actor); n_dec += 1
            act_idx = label if rng.random() > a.eps else int(rng.choice(np.flatnonzero(mask)))
            act = policy_index_to_action(act_idx)
            if s.pick_phase:
                s2, _ = env.resolve_pick(s, pick_type=act.pick_type() if act.is_pick() else None)
                s = env.advance_actor(env.spawn_and_despawn(s2))
                break
            sm = env.apply_action(s, act)
            if sm.is_agent_on_task(sm.actor, env.proficiency_positive_types[sm.actor]):
                s = sm.with_pick_phase()
                continue
            s = env.advance_actor(env.spawn_and_despawn(sm))
            break
    np.savez_compressed(a.out, grid=np.stack(grids), scalar=np.stack(scalars), mask=np.stack(masks),
                        label=np.array(labels), actor=np.array(actors))
    print(f"saved {n_dec} decisions to {a.out}")


if __name__ == "__main__":
    main()
