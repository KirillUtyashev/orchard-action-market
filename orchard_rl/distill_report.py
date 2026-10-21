"""Aggregate output/distill_results/w*_s*.json -> table + figures in notes/figures."""
import glob, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = collections.defaultdict(list)
for f in glob.glob("output/distill_results/w*_s*.json"):
    r = json.load(open(f)); R[r["width"]].append(r)
refs = json.load(open("output/distill_results/refs.json"))
W = sorted(R)
mean = lambda k: np.array([np.mean([r[k] for r in R[w]]) for w in W])
std = lambda k: np.array([np.std([r[k] for r in R[w]], ddof=1) if len(R[w]) > 1 else 0.0 for w in W])
out = Path("/Users/luozihan/Desktop/MARL/notes/figures"); out.mkdir(parents=True, exist_ok=True)
rows = ["width,params_per_actor,n_seeds,team_rps_mean,team_rps_std,val_acc_mean,train_acc_mean"]
for i, w in enumerate(W):
    rows.append(f"{w},{R[w][0]['params_per_actor']},{len(R[w])},{mean('team_rps_mean')[i]:.3f},{std('team_rps_mean')[i]:.3f},{mean('val_acc_mean')[i]:.3f},{mean('train_acc_mean')[i]:.3f}")
(out / "distill_table.csv").write_text("\n".join(rows)); print("\n".join(rows))
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(W, mean("team_rps_mean"), yerr=std("team_rps_mean"), marker="o", capsize=3, label="per-agent actor (imitation)")
ax.axhline(refs["teacher_qgreedy"]["mean"], ls="--", c="g", label=f"teacher Q-greedy ({refs['teacher_qgreedy']['mean']:.2f})")
ax.axhline(refs["nearest"]["mean"], ls=":", c="r", label=f"nearest heuristic ({refs['nearest']['mean']:.2f})")
ax.set_xscale("log", base=2); ax.set_xticks(W); ax.set_xticklabels(W)
ax.set_xlabel("actor width x  (CNN[x,x] k3, MLP[x])"); ax.set_ylabel("team reward per step"); ax.legend(); fig.tight_layout()
fig.savefig(out / "fig_width_vs_reward.png", dpi=160)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(W, mean("val_acc_mean"), "o-", label="validation"); ax.plot(W, mean("train_acc_mean"), "s--", label="train")
ax.set_xscale("log", base=2); ax.set_xticks(W); ax.set_xticklabels(W); ax.set_xlabel("actor width x"); ax.set_ylabel("agreement with teacher argmax"); ax.legend(); fig.tight_layout()
fig.savefig(out / "fig_width_vs_accuracy.png", dpi=160)
