#!/bin/sh
algo="Decentralized"
num_agents=2
length=6
width=6

for hidden in 16 256
do
  echo "${length} by ${width}, ${num_agents}"
  python ../run_experiments.py \
  --algorithm ${algo} --width ${width} --length ${length} --num_agents ${num_agents} --seed 42069 \
  --timesteps 1000000 --apple_life 5 --s_target 0.16 --batch_size ${num_agents} --critic_vision 0 \
  --new_input 1 --alpha 0.000275 --hidden_dim ${hidden} --num_layers 4 --skip 1 --epsilon 0.1 --env_cls "OrchardEuclideanRewards" --new_dynamic 0
done
