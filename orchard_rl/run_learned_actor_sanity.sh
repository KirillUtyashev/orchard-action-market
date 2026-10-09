#!/usr/bin/env bash
# Run the learned-actor sanity point for 5 seeds with the decentralized and/or
# centralized critic. Outputs stay under output/ (git-ignored).
#
# Usage:
#   ./run_learned_actor_sanity.sh [dec|cen|both] [total_steps]
# Example (smoke):  ./run_learned_actor_sanity.sh both 10000
#
# Runs are launched in the background, one process per (critic, seed).
# Summarize with:   python summarize_learned_actor.py output/learned_actor_sanity
set -euo pipefail
cd "$(dirname "$0")"
# One thread per process so that many seeds can run side by side.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

MODE="${1:-both}"
STEPS="${2:-3000000}"
PY="${PYTHON:-python}"
CONFIG="orchard/configs/learned_actor_sanity.yaml"
ROOT="output/learned_actor_sanity"

case "$MODE" in
  dec)  TYPES=(decentralized) ;;
  cen)  TYPES=(centralized) ;;
  both) TYPES=(decentralized centralized) ;;
  *) echo "mode must be dec, cen or both" >&2; exit 1 ;;
esac

for lt in "${TYPES[@]}"; do
  tag=${lt:0:3}   # dec / cen
  for seed in 1 2 3 4 5; do
    out="$ROOT/$tag/seed$seed/"
    mkdir -p "$out"
    nohup "$PY" -m orchard.train --config "$CONFIG" --override \
      train.learning_type="$lt" train.seed="$seed" \
      train.total_steps="$STEPS" logging.output_dir="$out" \
      > "$out/stdout.log" 2>&1 &
    echo "started $tag seed=$seed -> $out"
  done
done
echo "All runs started. Check progress with: tail -n 3 $ROOT/*/seed*/stdout.log"
