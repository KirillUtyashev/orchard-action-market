#!/bin/sh
# exp param
algo="Decentralized"
num_agents=4

#for width in 8
#do
#  echo "20 by ${width}, ${num_agents}"
#  python ../run_experiments.py \
#  --algorithm ${algo} --width ${width} --length 20 --num_agents ${num_agents} --seed 42069 --timesteps 1000000 --apple_life 5 \
#  --s_target 0.16 --batch_size 256 --alpha 0.000625 --hidden_dim 256 --num_layers 4
#done


algo="Centralized"
num_agents=4

for width in 5
do
  echo "20 by ${width}, ${num_agents}"
  python -u ../run_experiments.py \
  --algorithm ${algo} --width ${width} --length 20 --num_agents ${num_agents} --seed 42069 --timesteps 1000000 --apple_life 5 \
  --s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim 128 --num_layers 4
done
