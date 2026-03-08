class SimpleAgent:
    def __init__(self, policy, id_, value_network):
        self.policy = policy
        self.id = id_
        self.policy_value = value_network

    def add_experience(self, old_state, new_state, reward):
        self.policy_value.add_experience(old_state, new_state, reward)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state)
