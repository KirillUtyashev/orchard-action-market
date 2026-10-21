"""Wait for phase 1 to finish, pick the actor lr, then launch phase 2.

Rule (mechanical, decided before seeing data): for each actor lr, average over widths
{1,8,64} of the mean greedy_team_rps of the last 5 evaluations; pick the lr with the
highest average. Phase 2: widths 1,2,4,8,16,32,64 x seeds 1,2 x 300k steps at that lr.
"""
import csv, glob, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

P1 = Path("output/actor_width_phase1")
while not any("ALL DONE" in l for l in open("output/phase1_driver.log")):
    time.sleep(60)

scores = defaultdict(list)
for f in glob.glob(str(P1 / "w*_alr*_s1" / "*" / "metrics.csv")):
    tag = Path(f).parents[1].name            # w8_alr0.0003_s1
    lr = tag.split("_alr")[1].split("_s")[0]
    rows = list(csv.DictReader(open(f)))[-5:]
    scores[lr].append(sum(float(r["greedy_team_rps"]) for r in rows) / len(rows))
avg = {lr: sum(v) / len(v) for lr, v in scores.items()}
best = max(avg, key=avg.get)
Path("output/phase2_lr_choice.txt").write_text(f"avg by lr: {avg}\nchosen: {best}\n")
subprocess.run([sys.executable, "-u", "run_actor_width_sweep.py",
                "--widths", "1", "2", "4", "8", "16", "32", "64",
                "--lrs", best, "--seeds", "1", "2", "--steps", "300000",
                "--out", "output/actor_width_phase2", "--max-parallel", "9"],
               stdout=open("output/phase2_driver.log", "w"), stderr=subprocess.STDOUT)
