from Policy import BasePolicy
import numpy

class EpsilonGreedy(BasePolicy):

    epsilon = None
    def __init__(self, epsilon = 0.1, *args, **kwargs):
        self.epsilon = epsilon
        super(EpsilonGreedy, self).__init__(*args, **kwargs)

    def get_action_probabilities(self, state):
        action_probabilities = numpy.ones(self.num_actions, dtype=float) * self.epsilon / self.num_actions

        best_action = numpy.argmax(self.action_value[state])
        action_probabilities[best_action] += (1.0 - self.epsilon)

        return action_probabilities

    def get_action(self, state):
        action_probabilities = self.get_action_probabilities(state)

        return numpy.random.choice(numpy.arange(len(action_probabilities)), p=action_probabilities)