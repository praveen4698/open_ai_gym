import itertools

import numpy
from collections import defaultdict

from Policy import BasePolicy
from Policy.epsilon_greedy import EpsilonGreedy


class QLearning:

    env = None
    discount_factor = None
    alpha = None
    policy = None
    epsilon = None
    _total_reward = 0
    _number_of_episode = 0
    _episode_reward_list = list()

    def __init__(self, env, discount_factor = 0.1, alpha=0.6, epsilon=0.1, policy='random'):
        self.env = env
        self.Q = defaultdict(lambda: numpy.zeros(env.action_space.n))
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.__initialize_policy(policy)

    def __initialize_policy(self, policy):
        if policy == 'random':
            self.policy = BasePolicy(action_value=self.Q, num_actions=self.env.action_space.n)
        elif policy == 'epsilon_greedy':
            self.policy = EpsilonGreedy(action_value=self.Q, epsilon=self.epsilon, num_actions=self.env.action_space.n)
        else:
            raise Exception('unidentified policy')


    def run_episode(self, render=False):

        state = self.env.reset()
        episode_reward = 0

        for t in itertools.count():

            action = self.policy.get_action(state)

            next_state, reward, done, info = self.env.step(action)
            if render:
                self.env.render()

            best_next_action = numpy.argmax(self.Q[next_state])
            td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
            td_delta = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_delta

            # For statistics
            episode_reward += reward
            if done:
                break

            state = next_state
        self._update_total_reward(episode_reward=episode_reward)
        return

    def _update_total_reward(self, episode_reward):
        self._total_reward += episode_reward
        self._episode_reward_list.append(episode_reward)
        self._number_of_episode += 1

    def get_average_reward(self, number_of_episode=None):
        if number_of_episode and number_of_episode < self._number_of_episode:
            total_reward = sum(self._episode_reward_list[-number_of_episode:])
            return round(total_reward/number_of_episode, 2)
        return round(self._total_reward/self._number_of_episode, 2)



