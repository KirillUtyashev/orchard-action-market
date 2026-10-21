"""Actor-width sweep launcher. Runs every (width, actor_lr, seed) as its own process.

Usage (from orchard_rl/):
  python run_actor_width_sweep.py --widths 1 8 64 --lrs 1e-4 3e-4 1e-3 --seeds 1 \
      --steps 100000 --out output/actor_width_phase1 [--max-parallel 9]

Each run gets a resolved YAML (written with PyYAML so nested lists survive) derived
from --base, changing ONLY: actor_model (CNN[x,x] k3 + MLP[x]), train.actor_lr,
train.total_steps, train.seed, logging.output_dir. Critic block and critic lr are
untouched. Blocks until every run has finished.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="orchard/configs/baseline_match_n11_actor.yaml")
    ap.add_argument("--widths", type=int, nargs="+", required=True)
    ap.add_argument("--lrs", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-parallel", type=int, default=9)
    a = ap.parse_args()

    base = yaml.safe_load(Path(a.base).read_text())
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    jobs = []
    for x in a.widths:
        for lr in a.lrs:
            for seed in a.seeds:
                out = Path(a.out) / f"w{x}_alr{lr:g}_s{seed}"
                out.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(base)
                cfg["actor_model"]["conv_specs"] = [[x, 3], [x, 3]]
                cfg["actor_model"]["mlp_dims"] = [x]
                cfg["train"]["actor_lr"] = {"start": lr, "end": lr, "schedule": "none"}
                cfg["train"]["total_steps"] = a.steps
                cfg["train"]["seed"] = seed
                cfg["logging"]["output_dir"] = str(out) + "/"
                (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                jobs.append(out)

    running: list[tuple[Path, subprocess.Popen]] = []
    pending = list(jobs)
    while pending or running:
        running = [(o, p) for o, p in running if p.poll() is None]
        while pending and len(running) < a.max_parallel:
            out = pending.pop(0)
            log = open(out / "stdout.log", "w")
            p = subprocess.Popen(
                [sys.executable, "-u", "-m", "orchard.train", "--config", str(out / "config.yaml")],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            running.append((out, p))
            print(f"started {out.name}", flush=True)
        time.sleep(20)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
