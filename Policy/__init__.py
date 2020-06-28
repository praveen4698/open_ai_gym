

class BasePolicy:

    action_value = None
    num_actions = None

    def __init__(self, action_value, num_actions):
        print(" in BasePolicy ")
        self.action_value = action_value
        self.num_actions = num_actions

    def get_action_probabilities(self, state):
        pass

    def get_action(self, state):
        pass
