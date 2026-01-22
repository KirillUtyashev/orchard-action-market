import random
from debug.code.simple_agent import SimpleAgent
from config import (
    NUM_AGENTS,
    W,
    L,
    PROBABILITY_APPLE,
)

from debug.code.environment import Orchard
from debug.code.helpers import teleport
from debug.code.reward import Reward
from models.value_function import VNetwork


class SupervisedLearning:
    def __init__(self, picker_r, trajectory_length):
        self.env = None
        self.agent_controller = None
        self.input_type = None
        self.agents = []
        self.critic_networks = []
        self.reward_module = Reward(picker_r, NUM_AGENTS)
        self.trajectory_length = trajectory_length

        self._networks_for_eval = []

        self.build_experiment()

    def _init_critic_networks(self):
        for i in range(NUM_AGENTS):
            self.critic_networks[i] = VNetwork

    def _init_agents_for_training(self):
        for i in range(NUM_AGENTS):
            self.agents[i] = SimpleAgent(teleport(W), i, self.critic_networks[i])

    def build_experiment(self):
        # 1. Initialize our CNN critic network.
        self._init_critic_networks()

        # 2.
        self._init_agents_for_training()

        # 3. Initialize OUR agent controller. ignore test flag.
        self.agent_controller = AgentControllerCentralizedCNN(self.agents)

        # 4. Create the environment.
        self.env = Orchard(
            W,
            L,
            NUM_AGENTS,
            self.reward_module,
            PROBABILITY_APPLE,
        )

        # 5. Set up the network for evaluation, consistent with the parent class.
        self._networks_for_eval = self.critic_networks

    def step_and_collect_observation(self, step: int) -> None:
        """
        Takes the result of a single, clean environment step and adds the
        corresponding experience to the training buffer.
        """
        for step in range(self.trajectory_length):
            actor_idx = random.randint(0, NUM_AGENTS - 1)
            res = self.env.process_action(actor_idx, self.agents[actor_idx].policy)

            for i in range(NUM_AGENTS):
                processed_old_state, processed_intermediate_state, processed_final_state = None, None, None
                self.critic_networks[i].add_experience(processed_old_state, processed_intermediate_state, 0)
                self.critic_networks[i].add_experience(processed_intermediate_state, processed_final_state, res.reward_vector[i])



        # assert isinstance(self._agents_list[0].policy_value, ValueCNNCentralized)
        # valueCNN: ValueCNNCentralized = self._agents_list[0].policy_value
        # for tick in range(self.train_config.num_agents):
        #     env_step_result = self.single_agent_env_step(tick)
        #     # remove apple if picked
        #     if self.train_config.new_dynamic and env_step_result.picked:
        #         pos = self._agents_list[env_step_result.acting_agent_id].position
        #         self.env.remove_apple(pos)
        #     reward = sum(env_step_result.reward_vector)
        #     processed_state = valueCNN.raw_state_to_nn_input(env_step_result.old_state)
        #     processed_new_state = valueCNN.raw_state_to_nn_input(
        #         env_step_result.new_state
        #     )
        #     valueCNN.add_experience(processed_state, processed_new_state, reward)
