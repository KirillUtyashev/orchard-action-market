#!/bin/sh

algo="Decentralized"
num_agents=4
length=9
width=9

for hidden_dim in 64
do
  echo "${length} by ${width}, ${num_agents}"
  cd .. && python run_experiments.py \
  --algorithm ${algo} --width ${width} --length ${length} --num_agents ${num_agents} --seed 42069 --timesteps 1000000 --apple_life 5 \
  --s_target 0.16 --batch_size 4 --alt_vision 1 --vision 0 --alpha 0.000275 --hidden_dim ${hidden_dim} --num_layers 4 --skip 1 --epsilon 0.1 \
  --env_cls "OrchardEuclideanRewards"
done

