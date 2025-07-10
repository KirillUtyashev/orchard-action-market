#!/bin/sh
# exp param
#algo="Decentralized"
#
#num_agents=4
#
#for dimension in 128 256 512
#do
#  echo "8 by 8, ${num_agents}"
#  python ../run_experiments.py \
#  --algorithm ${algo} --width 12 --length 12 --num_agents 4 --seed 42069 --timesteps 1000000 --apple_life 5 \
#  --s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim ${dimension} --num_layers 4
#done
#
algo="Decentralized"

num_agents=4

for dimension in 256
do
  echo "8 by 8, ${num_agents}"
  python ../run_experiments.py \
  --algorithm ${algo} --width 12 --length 12 --num_agents 4 --seed 42069 --timesteps 1000000 --apple_life 5 \
  --s_target 0.16 --batch_size 256 --alpha 0.00125 --hidden_dim ${dimension} --num_layers 6
done

