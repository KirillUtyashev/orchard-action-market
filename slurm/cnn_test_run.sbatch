#!/bin/bash

#-----------------

Job Info
#-----------------
#SBATCH --job-name=reward_learning
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

#-----------------

Resources
#-----------------
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00

NOTE: The partition and GPU request below are examples.
You may need to change them based on your cluster's configuration.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#-----------------

Environment Setup
#-----------------

Load necessary modules (adjust as needed for your environment)
module load python/3.9
module load cuda/11.7
Activate your virtual environment if you are using one
source ../bin/activate
#-----------------

Job Execution
#-----------------

This is the command that will run your Python script.
Replace your_script.py with the actual name of your script.
python run_experiments.py 

--algorithm "RewardLearningCNNDecentralized" 

--width "6" 

--length "6" 

--num_agents "2" 

--seed "42069" 

--timesteps "5000000" 

--eval_timesteps "5000" 

--apple_life "5" 

--s_target "0.16" 

--batch_size "128" 

--critic_vision "0" 

--new_input "1" 

--alpha "0.003" 

--hidden_dim "64" 

--num_layers "4" 

--skip "0" 

--epsilon "0.1" 

--env_cls "OrchardEuclideanNegativeRewards" 

--new_dynamic "1"