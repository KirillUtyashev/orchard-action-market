# Action Market (Orchard Implementation) (Alex's README)

This repository implements aspects of the Content Market and the Action Market. 

The Action Market is implemented with respect to the Orchard MARL problem. To maintain relative compatibility, the Content Market is also implemented with an (unused) Orchard environment. 

On the MARL side, there are multiple "approaches" which we benchmark and compare against each other. The current implementations are the Centralized and Decentralized Value Functions, the Actor-Critic (with Content Market), and the Localized Decentralized and Actor-Critic approaches.

The Content Market contains specific experiments with the an emergent influencer and with relevant parameter sweeps. The Action Market implementations contain an Actor-Critic approach with an elected "influencer", but mostly exploration and testing has been performed with the set influencer.

## Packages and Set-up 

Pytorch with CUDA is highly recommended. I run most experiments on an RTX-4060 (laptop version), though I also use Google Colab. The Pytorch (especially CUDA version) is not necessarily required to perfectly match. See here: https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/ and https://developer.nvidia.com/cuda-downloads

## Folders

`agents` contain the logic for most of the Agent objects (which act both in the Orchard Action Market and the Content Market).\
`alloc` contains the logic for solving optimization problems as in the rate allocation of agents.\
`models` contains the code for the (PyTorch) neural network models which are used for both Value Functions and policy approximation / learning.\
`orchard` contains the implementation of the Orchard (apple-picking) environment, including the global state, apple spawning (and despawning), apple picking, and time evolution.\
`policies` contains benchmark policies for use within the Orchard environment.\
`testing` contains two files which can be used to load Actor-Critic or Value Function model weights ("checkpoints") to test their performance within an Orchard. 

## Top-Level Files

`learning.py`: The file from which Centralized and Decentralized Value Functions are trained. These implementations can be found in `train_central.py` and `train_decentral.py` respectively.

`orchard_set_influencer_set_critic_cleaner.py`: An Actor-Critic approach to the Orchard MARL but with the Content Market influencer implementation (with rate allocation, etc.) but with a set influencer. For ease of testing, we only use a single "policy iteration" step (i.e. training one value function, one actor-critic).

`orchard_ac_variable_influencer.py`: An Actor-Critic approach to the Orchard MARL but with a variable influencer. Only tested tentatively for a small environment (4 agents, 10 space).

`ac_content_EI_observer.py`: An Actor-Critic approach to the Content Market (with an emergent influencer). All content market dynamics apply.

## Project Structure (Kirill's README)

This repository implements multi-agent reinforcement learning in the **Orchard** environment. Below is an overview of the main folders and files:

### **agents/**
- `agent.py` – Parent class for all agent implementations.
- `simple_agent.py` – Agent used for *Centralized Value Function Learning* (no communication).
- `communicating_agent.py` – Agent used for *Decentralized Value Function Learning* (perfect communication).
- `actor_critic_agent.py` – Agent for *Actor-Critic Learning*, including variations with following rates from content market theory.

### **helpers/**
- `helpers.py` – General utility functions.
- `rate_updater.py` – Class for solving the optimization problem behind following rates.
- `controller.py` – Classes that manage agent communication for different algorithms.

### **models/**
- `main_net.py` – Defines a simple MLP architecture for critic and actor networks.
- `value_function.py` – Object for training the critic’s value network.
- `actor_network.py` – Object for training the actor’s policy network.

### **orchard/**
- `environment.py` – Orchard environment definition.
- `algorithms.py` – Logic for spawning and respawning apples.

### **policies/**
- Baseline policy implementations (e.g., random actions, nearest-apple movement).

### **policyitchk/**
- Stores neural network weights during training.

### **setup_vm/**, **train_vm/**, **train_scripts/**
- Shell scripts for launching and managing simulations on virtual machines.

### **value_function_learning/**
- `train_value_function.py` – Training logic for centralized and decentralized value function learning.

### **actor_critic/**
- `actor_critic.py` – Parent class for actor-critic algorithms.
- `actor_critic_perfect_info.py` – Actor-critic in perfect-information setup.
- `actor_critic_following_rates.py` – Actor-critic with following rates optimization.

### **Core Files**
- `algorithm.py` – Parent class for all training algorithms; contains main training loop logic.
- `main.py` – Evaluation function for trained algorithms.
- `run_experiments.py` – Entry point for launching experiments locally.
- `run_experiments_vm.py` – Entry point for launching experiments on virtual machines.
- `tests.py` – Basic unit tests for core functionality.  
