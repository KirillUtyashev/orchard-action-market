"""Reference lines evaluated with the SAME protocol as distill_train.py (eval seeds 101-103, 1500 steps):
teacher Q-greedy (value-based 8M dec critic) and the nearest heuristic."""
import json, numpy as np
import orchard.encoding as encoding
from orchard.actor_critic import policy_index_to_action
from orchard.config import load_config
from orchard.env import create_env
from orchard.eval import evaluate_policy_metrics
from orchard.policy import heuristic_action
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer
cfg = load_config("orchard/configs/baseline_match_n11_actor.yaml", ["train.discount_method=round_steps"])
set_all_seeds(1); env = create_env(cfg.env)
encoding.init_encoder(cfg.model.encoder, env, n_networks=cfg.env.n_agents)
tr = create_trainer(cfg, env); tr.load_critic_checkpoint("output/vb_critic_8M_seed1235.pt")
def qg(s):
    m = tr._legal_mask(s); d = 1.0 if s.pick_phase else tr._move_discount(s)
    q, *_ = tr._enumerate_action_objectives(s, m, d)
    return policy_index_to_action(int(np.where(m, q, -np.inf).argmax()))
res = {"teacher_qgreedy": [], "nearest": []}
for es in (101, 102, 103):
    env.set_eval_mode(True, seed=es); st = env.init_state()
    res["teacher_qgreedy"].append(evaluate_policy_metrics(st, qg, env, 1500)["team_rps"])
    res["nearest"].append(evaluate_policy_metrics(st, lambda s: heuristic_action(s, env, cfg.train.heuristic), env, 1500)["team_rps"])
res = {k: {"evals": v, "mean": float(np.mean(v))} for k, v in res.items()}
json.dump(res, open("output/distill_results/refs.json", "w"), indent=1); print(res)
