#!/bin/sh

algo="Decentralized"
num_agents=10
length=50

for width in 3
do
  echo "${length} by ${width}, ${num_agents}"
  python ../run_experiments.py \
  --algorithm ${algo} --width ${width} --length ${length} --num_agents ${num_agents} --seed 42069 --timesteps 600000 --apple_life 5 \
  --s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim 64 --num_layers 4
done

