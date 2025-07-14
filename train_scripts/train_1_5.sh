#!/bin/sh

algo="Centralized"
num_agents=1

for width in 1 2
do
  echo "20 by ${width}, ${num_agents}"
  python -u ../run_experiments.py \
  --algorithm ${algo} --width ${width} --length 5 --num_agents ${num_agents} --seed 42069 --timesteps 1000000 --apple_life 5 \
  --s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim 16 --num_layers 4 > output.log 2>&1
done
