#!/bin/sh
# exp param
algo="Centralized"

num_agents=2

for dimension in 4 16 64 128
do
  echo "8 by 8, ${num_agents}"

  python ../run_experiments.py \
  --algorithm ${algo} --width 8 --length 8 --num_agents 2 --seed 42069 --timesteps 1000000 --apple_life 5 \
  --s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim ${dimension} --num_layers 4
done
