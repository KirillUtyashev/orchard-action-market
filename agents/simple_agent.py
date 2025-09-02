from agents.value_agent import ValueAgent


class SimpleAgent(ValueAgent):
    def add_experience(self, old_state, new_state, reward):
        self.policy_value.add_experience(old_state, new_state, reward)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state[:self.policy_value.get_input_dim()])
