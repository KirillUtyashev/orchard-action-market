#!/bin/sh

algo="Centralized"
num_agents=1

echo "5 by 1, 1"
python -u ../run_experiments.py \
--algorithm ${algo} --width 1 --length 5 --num_agents ${num_agents} --seed 42069 --timesteps 1000000 --apple_life 5 \
--s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim 16 --num_layers 4 > output.log 2>&1
