from pathlib import Path

import numpy as np
import torch

from debug.code.agents.actor_critic_agent import ACAgent
from debug.code.agents.controllers import AgentControllerActorCritic, AgentControllerCentralized, AgentControllerDecentralized
from debug.code.nn.encoders import (
    CenConcatEncoder,
    CenEntityEncoder,
    CenGridEncoder,
    DecCenteredGridEncoder,
    DecEntityEncoder,
    DecGridEncoder,
)
from debug.code.env.environment import Orchard
from debug.code.training.helpers import random_policy, teleport
from debug.code.agents.simple_agent import SimpleAgent
from debug.code.nn.policy_network import PolicyNetwork
from debug.code.nn.value_function import VNetwork


class LearningSetupMixin:
    def _critic_network_cfg(self):
        return getattr(self.exp_config, "critic_network", None) or self.exp_config.network

    def _actor_network_cfg(self):
        return getattr(self.exp_config, "actor_network", None) or self.exp_config.network

    def _shared_network_cfg(self):
        critic_cfg = self._critic_network_cfg()
        actor_cfg = self._actor_network_cfg()
        if bool(getattr(actor_cfg, "CNN", False)) != bool(getattr(critic_cfg, "CNN", False)):
            raise ValueError("actor_network.CNN and critic_network.CNN must match in the current implementation.")
        if bool(getattr(actor_cfg, "self_centered_grid", False)) != bool(
            getattr(critic_cfg, "self_centered_grid", False)
        ):
            raise ValueError(
                "actor_network.self_centered_grid and critic_network.self_centered_grid must match in the current implementation."
            )
        return critic_cfg

    def _critic_schedule_lr(self) -> bool:
        specific = getattr(self.exp_config.train, "critic_schedule_lr", None)
        if specific is None:
            return bool(getattr(self.exp_config.train, "schedule_lr", False))
        return bool(specific)

    def _actor_schedule_lr(self) -> bool:
        specific = getattr(self.exp_config.train, "actor_schedule_lr", None)
        if specific is None:
            return bool(getattr(self.exp_config.train, "schedule_lr", False))
        return bool(specific)

    def _build_encoder(self):
        cfg = self.exp_config
        net_cfg = self._shared_network_cfg()
        k = cfg.reward.top_k_num_apples

        if cfg.algorithm.centralized:
            if net_cfg.CNN:
                self.encoder = CenGridEncoder(self.width, self.length, self.num_agents)
            elif cfg.algorithm.concat:
                dec = DecEntityEncoder(self.width, self.length, self.num_agents, k)
                self.encoder = CenConcatEncoder(dec)
            else:
                self.encoder = CenEntityEncoder(self.width, self.length, self.num_agents, k)
        else:
            if net_cfg.CNN:
                if bool(getattr(net_cfg, "self_centered_grid", False)):
                    self.encoder = DecCenteredGridEncoder(self.width, self.length, self.num_agents)
                else:
                    self.encoder = DecGridEncoder(self.width, self.length, self.num_agents)
            else:
                self.encoder = DecEntityEncoder(self.width, self.length, self.num_agents, k)

    def _load_supervised_teacher_blobs(self) -> list[dict]:
        sup_cfg = self.exp_config.supervised
        weights_path = Path(str(sup_cfg.weights_path)).expanduser()
        if not weights_path.is_absolute():
            weights_path = (Path.cwd() / weights_path).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Supervised weights file not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location="cpu")

        if isinstance(ckpt, dict) and "critics" in ckpt:
            critics = ckpt.get("critics", [])
            blobs = []
            for item in critics:
                if isinstance(item, dict) and "blob" in item:
                    blobs.append(item["blob"])
                else:
                    blobs.append(item)
            return blobs

        if isinstance(ckpt, dict) and "weights" in ckpt:
            return [ckpt]

        raise ValueError(
            "Unsupported supervised checkpoint format. Expected {'critics':[{'blob':...}]} or a single net blob."
        )

    def _init_supervised_networks(self) -> None:
        self.supervised_networks = []
        if not self.supervised_enabled:
            return

        if bool(self.exp_config.supervised.CNN) != bool(self._critic_network_cfg().CNN):
            raise ValueError(
                "supervised.CNN must match critic_network.CNN so encoder/model input types are compatible."
            )

        teacher_blobs = self._load_supervised_teacher_blobs()
        expected = len(self.critic_networks)
        if len(teacher_blobs) != expected:
            raise ValueError(
                f"Supervised teacher/student mismatch: expected {expected} teacher networks, "
                f"got {len(teacher_blobs)}."
            )

        sup_cfg = self.exp_config.supervised
        for blob in teacher_blobs:
            teacher = VNetwork(
                self.encoder,
                1,
                self.exp_config.train.alpha,
                self.discount_factor,
                reward_learning=False,
                supervised=False,
                mlp_dims=tuple(sup_cfg.mlp_dims),
                lam=self.exp_config.train.lmda,
                num_training_steps=self.trajectory_length,
                schedule=False,
                conv_channels=sup_cfg.conv_channels,
                kernel_size=sup_cfg.kernel_size,
            )
            teacher.import_net_state(blob)
            teacher.set_eval_mode()
            for param in teacher.model.parameters():
                param.requires_grad = False
            self.supervised_networks.append(teacher)

    def _init_critic_networks(self):
        cfg = self.exp_config
        net_cfg = self._critic_network_cfg()
        critic_schedule_lr = self._critic_schedule_lr()

        if cfg.algorithm.centralized:
            net = VNetwork(
                    self.encoder,
                    1,
                    cfg.train.alpha,
                    self.discount_factor,
                    reward_learning=cfg.reward.reward_learning,
                    supervised=self.supervised_enabled,
                    mlp_dims=tuple(net_cfg.mlp_dims),
                    use_mlp=bool(getattr(net_cfg, "MLP", True)),
                    num_training_steps=self.trajectory_length,
                    lam=self.exp_config.train.lmda,
                    schedule=critic_schedule_lr,
                    conv_channels=net_cfg.conv_channels,
                    kernel_size=net_cfg.kernel_size,
                )
            if self.crit_blobs:
                net.import_net_state(self.crit_blobs[0]["blob"])
            if bool(getattr(cfg.train, "freeze_critics", False)):
                net.set_eval_mode()
                for param in net.model.parameters():
                    param.requires_grad = False
            self.critic_networks.append(net)
        else:
            for i in range(self.num_agents):
                nn = VNetwork(
                    self.encoder,
                    1,
                    cfg.train.alpha,
                    self.discount_factor,
                    reward_learning=cfg.reward.reward_learning,
                    supervised=self.supervised_enabled,
                    mlp_dims=tuple(net_cfg.mlp_dims),
                    use_mlp=bool(getattr(net_cfg, "MLP", True)),
                    lam=self.exp_config.train.lmda,
                    num_training_steps=self.trajectory_length,
                    schedule=critic_schedule_lr,
                    conv_channels=net_cfg.conv_channels,
                    kernel_size=net_cfg.kernel_size,
                )
                if self.crit_blobs:
                    nn.import_net_state(self.crit_blobs[i]["blob"])
                if bool(getattr(cfg.train, "freeze_critics", False)):
                    nn.set_eval_mode()
                    for param in nn.model.parameters():
                        param.requires_grad = False
                self.critic_networks.append(nn)

    def _actor_alpha(self) -> float:
        actor_alpha = getattr(self.exp_config.train, "actor_alpha", None)
        if actor_alpha is None:
            actor_alpha = self.exp_config.train.alpha
        return float(actor_alpha)

    def _init_policy_networks(self):
        self.policy_networks = []
        if not bool(getattr(self.exp_config.algorithm, "actor_critic", False)):
            return

        cfg = self.exp_config
        net_cfg = self._actor_network_cfg()
        actor_alpha = self._actor_alpha()
        expected = self.num_agents
        if cfg.train.load_weights and self.actor_blobs and len(self.actor_blobs) != expected:
            raise ValueError(
                f"Expected {expected} actor blobs when loading actor-critic weights, got {len(self.actor_blobs)}."
            )

        for i in range(self.num_agents):
            net = PolicyNetwork(
                self.encoder,
                output_dim=5,
                alpha=actor_alpha,
                discount=self.discount_factor,
                mlp_dims=tuple(net_cfg.mlp_dims),
                use_mlp=bool(getattr(net_cfg, "MLP", True)),
                num_training_steps=self.trajectory_length,
                schedule=self._actor_schedule_lr(),
                conv_channels=net_cfg.conv_channels,
                kernel_size=net_cfg.kernel_size,
            )
            if cfg.train.load_weights:
                if not self.actor_blobs:
                    raise ValueError("Missing actor weights in checkpoint for actor-critic load.")
                blob = self.actor_blobs[i]
                net.import_net_state(blob["blob"] if isinstance(blob, dict) and "blob" in blob else blob)
            self.policy_networks.append(net)

    def _init_agents_for_training(self):
        if self.exp_config.algorithm.random_policy:
            policy_fn = lambda agent_pos: random_policy(agent_pos, width=self.width, length=self.length)
        else:
            policy_fn = lambda _agent_pos: teleport(self.width, self.length)
        for i in range(self.num_agents):
            net = self.critic_networks[0] if self.exp_config.algorithm.centralized else self.critic_networks[i]
            if bool(getattr(self.exp_config.algorithm, "actor_critic", False)):
                self.agents.append(
                    ACAgent(
                        policy_fn,
                        i,
                        net,
                        self.policy_networks[i],
                        np.zeros(self.num_agents, dtype=float),
                    )
                )
            else:
                self.agents.append(SimpleAgent(policy_fn, i, net))

    def build_experiment(self):
        self._build_encoder()
        self._init_critic_networks()
        self._init_policy_networks()
        self._init_supervised_networks()
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
        elif bool(getattr(self.exp_config.algorithm, "actor_critic", False)):
            self.agent_controller = AgentControllerActorCritic(
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
        if self.supervised_enabled:
            self._generate_evaluation_states_supervised()
        self._networks_for_eval = self.policy_networks if self.policy_networks else self.critic_networks
        self._init_weight_sample_indices()
