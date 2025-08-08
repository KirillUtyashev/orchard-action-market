#!/bin/sh
algo="Decentralized"
num_agents=4
length=9
width=9

set -a
source /Users/utya.kirill/Desktop/orchard-action-market/.env
set +a

for hidden in 4
do
  echo "${length} by ${width}, ${num_agents}"
  python ../run_experiments_vm.py 2 \
  --algorithm ${algo} --width ${width} --length ${length} --num_agents ${num_agents} --seed 42 --timesteps 1000000 --apple_life 5 \
  --s_target 0.16 --batch_size 4 --alt_vision 1 --vision 0 --alpha 0.000275 --hidden_dim ${hidden} --num_layers 4 --skip 1 --epsilon 0.1
done
