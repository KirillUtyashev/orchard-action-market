# Action Market (Orchard Implementation)

This repository implements aspects of the Content Market and the Action Market. 

The Action Market is implemented with respect to the Orchard MARL problem. To maintain relative compatibility, the Content Market is also implemented with an (unused) Orchard environment. 

On the MARL side, there are multiple "approaches" which we benchmark and compare against each other. The current implementations are the Centralized and Decentralized Value Functions, the Actor-Critic (with Content Market), and the Localized Decentralized and Actor-Critic approaches.

The Content Market contains specific experiments with the an emergent influencer and with relevant parameter sweeps. The Action Market implementations contain an Actor-Critic approach with an elected "influencer", but mostly exploration and testing has been performed with the set influencer.

## Packages and Set-up 

Pytorch with CUDA is highly recommended. I run most experiments on an RTX-4060 (laptop version), though I also use Google Colab. The Pytorch (especially CUDA version) is not necessarily required to perfectly match. See here: https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/ and https://developer.nvidia.com/cuda-downloads

## Folders

`agents` contain the logic for most of the Agent objects (which act both in the Orchard Action Market and the Content Market).
`alloc` contains the logic for solving optimization problems as in the rate allocation of agents.
`models` contains the code for the (PyTorch) neural network models which are used for both Value Functions and policy approximation / learning.
`orchard` contains the implementation of the Orchard (apple-picking) environment, including the global state, apple spawning (and despawning), apple picking, and time evolution.
`policies` contains benchmark policies for use within the Orchard environment.
`testing` contains two files which can be used to load Actor-Critic or Value Function model weights ("checkpoints") to test their performance within an Orchard. 

## Top-Level Files

`learning.py`: The file from which Centralized and Decentralized Value Functions are trained. These implementations can be found in `train_central.py` and `train_decentral.py` respectively.

`orchard_set_influencer_set_critic_cleaner.py`: An Actor-Critic approach to the Orchard MARL but with the Content Market influencer implementation (with rate allocation, etc.) but with a set influencer. For ease of testing, we only use a single "policy iteration" step (i.e. training one value function, one actor-critic).

`orchard_ac_variable_influencer.py`: An Actor-Critic approach to the Orchard MARL but with a variable influencer. Only tested tentatively for a small environment (4 agents, 10 space).

`ac_content_EI_observer.py`: An Actor-Critic approach to the Content Market (with an emergent influencer). All content market dynamics apply.

