from debug.code.agents.controllers import AgentControllerCentralized, AgentControllerDecentralized
from debug.code.nn.encoders import (
    CenConcatEncoder,
    CenEntityEncoder,
    CenGridEncoder,
    DecEntityEncoder,
    DecGridEncoder,
)
from debug.code.core.enums import L, NUM_AGENTS, W
from debug.code.env.environment import Orchard
from debug.code.training.helpers import random_policy, teleport
from debug.code.agents.simple_agent import SimpleAgent
from debug.code.nn.value_function import VNetwork


class LearningSetupMixin:
    def _build_encoder(self):
        cfg = self.exp_config
        k = cfg.reward.top_k_num_apples

        if cfg.algorithm.centralized:
            if cfg.network.CNN:
                self.encoder = CenGridEncoder(W, W, NUM_AGENTS)
            elif cfg.algorithm.concat:
                dec = DecEntityEncoder(W, W, NUM_AGENTS, k)
                self.encoder = CenConcatEncoder(dec)
            else:
                self.encoder = CenEntityEncoder(W, W, NUM_AGENTS, k)
        else:
            if cfg.network.CNN:
                self.encoder = DecGridEncoder(W, W, NUM_AGENTS)
            else:
                self.encoder = DecEntityEncoder(W, W, NUM_AGENTS, k)

    def _init_critic_networks(self):
        cfg = self.exp_config

        if cfg.algorithm.centralized:
            self.critic_networks.append(
                VNetwork(
                    self.encoder,
                    1,
                    cfg.train.alpha,
                    self.discount_factor,
                    reward_learning=cfg.reward.reward_learning,
                    mlp_dims=tuple(cfg.network.mlp_dims),
                    num_training_steps=self.trajectory_length,
                    lam=self.exp_config.train.lmda,
                    schedule=cfg.train.schedule_lr,
                    conv_channels=cfg.network.conv_channels,
                    kernel_size=cfg.network.kernel_size,
                )
            )
        else:
            for i in range(NUM_AGENTS):
                nn = VNetwork(
                    self.encoder,
                    1,
                    cfg.train.alpha,
                    self.discount_factor,
                    reward_learning=cfg.reward.reward_learning,
                    mlp_dims=tuple(cfg.network.mlp_dims),
                    lam=self.exp_config.train.lmda,
                    num_training_steps=self.trajectory_length,
                    schedule=cfg.train.schedule_lr,
                    conv_channels=cfg.network.conv_channels,
                    kernel_size=cfg.network.kernel_size,
                )
                if cfg.train.load_weights:
                    nn.import_net_state(self.crit_blobs[i]["blob"])
                self.critic_networks.append(nn)

    def _init_agents_for_training(self):
        policy_fn = teleport(W) if not self.exp_config.algorithm.random_policy else random_policy
        for i in range(NUM_AGENTS):
            net = self.critic_networks[0] if self.exp_config.algorithm.centralized else self.critic_networks[i]
            self.agents.append(SimpleAgent(policy_fn, i, net))

    def build_experiment(self):
        self._build_encoder()
        self._init_critic_networks()
        self._init_agents_for_training()

        p_apple = self.exp_config.algorithm.q_agent / (W**2)
        d_apple = 1 / self.exp_config.env.apple_life

        if self.exp_config.algorithm.random_policy:
            self._generate_evaluation_states(p_apple, d_apple)
        elif self.exp_config.reward.reward_learning:
            self._generate_evaluation_states_reward_learning()

        if self.exp_config.algorithm.centralized:
            self.agent_controller = AgentControllerCentralized(
                self.agents,
                self.encoder,
                self.discount_factor,
                self.exp_config.train.epsilon,
            )
        else:
            self.agent_controller = AgentControllerDecentralized(
                self.agents,
                self.encoder,
                self.discount_factor,
                self.exp_config.train.epsilon,
            )

        self.env = Orchard(
            W,
            L,
            NUM_AGENTS,
            self.reward_module,
            p_apple,
            d_apple,
            max_apples=self.exp_config.env.max_apples,
        )
        self.env.set_positions()
        self._networks_for_eval = self.critic_networks
        self._init_weight_sample_indices()
