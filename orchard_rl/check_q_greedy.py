"""Check the learned-actor Q computation against a trained critic.

Evaluates the policy argmax_a [ r_team(a) + discount * V_team(afterstate_a) ] with a
critic checkpoint (for example one from value-based training). If this does not
reach roughly the value-based greedy result, the problem is in the critic, the
discounting or the masking, not in the actor update.

Usage:
  python check_q_greedy.py --config orchard/configs/reference.yaml \
      --critic path/to/step_N.pt [--override k=v ...] [--steps 5000]
Use the same env/model/learning_type overrides that the critic was trained with.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.actor_critic import policy_index_to_action
from orchard.config import load_config
from orchard.enums import LearningType
from orchard.env import create_env
from orchard.eval import evaluate_policy_metrics
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--critic", required=True, help="checkpoint with critic weights")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        ["train.discount_method=round_steps", "train.algorithm.learned_actor.enabled=true"] + args.override,
    )
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    n_nets = 1 if cfg.train.learning_type == LearningType.CENTRALIZED else cfg.env.n_agents
    encoding.init_encoder(cfg.model.encoder, env, n_networks=n_nets)
    trainer = create_trainer(cfg, env)
    trainer.load_critic_checkpoint(args.critic)

    def q_greedy(state):
        mask = trainer._legal_mask(state)
        legal = [int(i) for i in np.flatnonzero(mask)]
        outs = [trainer._after_state_for_action(state, policy_index_to_action(i)) for i in legal]
        discount = trainer._move_discount(state) if not state.pick_phase else 1.0
        q = torch.tensor([float(sum(o[1])) for o in outs]) + discount * trainer._critic.team_values(
            state, [o[0] for o in outs],
        )
        return policy_index_to_action(legal[int(q.argmax())])

    env.set_eval_mode(True, seed=None)
    print("Q-greedy:", evaluate_policy_metrics(env.init_state(), q_greedy, env, args.steps))


if __name__ == "__main__":
    main()
