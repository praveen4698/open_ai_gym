import gym
import numpy as np

from learning_algorithm.q_learning import QLearning
from discretized_observation_wrapper.discretized_observation_wrapper import DiscretizedObservationWrapper


def CartPoleWithQLearnig():
    env = gym.make('CartPole-v0')
    env = DiscretizedObservationWrapper(
        env,
        n_bins=200,
        low=np.array([-2.4, -200.0, -0.42, -350]),
        high=np.array([2.4, 200.0, 0.42, 350])
    )


    agent = QLearning(env, alpha=0.6, discount_factor=1, policy='epsilon_greedy', epsilon=0.1)
    for i in range(1, 1000):
        agent.run_episode(render=True)
        print(i, agent.get_average_reward(), agent.get_average_reward(number_of_episode=10))
