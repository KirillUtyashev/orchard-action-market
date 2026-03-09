from debug.code.agents.controllers import AgentControllerCentralized, AgentControllerDecentralized
from debug.code.nn.encoders import (
    CenConcatEncoder,
    CenEntityEncoder,
    CenGridEncoder,
    DecEntityEncoder,
    DecGridEncoder,
)
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
                self.encoder = CenGridEncoder(self.width, self.length, self.num_agents)
            elif cfg.algorithm.concat:
                dec = DecEntityEncoder(self.width, self.length, self.num_agents, k)
                self.encoder = CenConcatEncoder(dec)
            else:
                self.encoder = CenEntityEncoder(self.width, self.length, self.num_agents, k)
        else:
            if cfg.network.CNN:
                self.encoder = DecGridEncoder(self.width, self.length, self.num_agents)
            else:
                self.encoder = DecEntityEncoder(self.width, self.length, self.num_agents, k)

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
            for i in range(self.num_agents):
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
        if self.exp_config.algorithm.random_policy:
            policy_fn = lambda agent_pos: random_policy(agent_pos, width=self.width, length=self.length)
        else:
            policy_fn = lambda _agent_pos: teleport(self.width, self.length)
        for i in range(self.num_agents):
            net = self.critic_networks[0] if self.exp_config.algorithm.centralized else self.critic_networks[i]
            self.agents.append(SimpleAgent(policy_fn, i, net))

    def build_experiment(self):
        self._build_encoder()
        self._init_critic_networks()
        self._init_agents_for_training()

        p_apple = self.exp_config.algorithm.q_agent / float(self.width**2)
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
            self.length,
            self.width,
            self.num_agents,
            self.reward_module,
            p_apple,
            d_apple,
            max_apples=self.exp_config.env.max_apples,
        )
        self.env.set_positions()
        self._networks_for_eval = self.critic_networks
        self._init_weight_sample_indices()
