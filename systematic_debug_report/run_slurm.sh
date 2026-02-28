srun --partition=gpunodes --gpus=1 --mem=2G --time=2:00:00 \
  python3 -m orchard.train --config /u/taddmao/code/orchard-action-market/systematic_debug_report/orchard/configs/3x3/policy_learning.yaml