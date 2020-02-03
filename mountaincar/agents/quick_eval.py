"""
FYI:
pos = observation0
velocity = observation1
"""

import itertools
import numpy as np
import gym


class PlagihAgent_A:

    def decide(self, observation):
        observation0, observation1 = observation
        if -observation1 + min(observation1, observation0 + 1.025) > observation1:
            return 0
        else:
            return 2


class SimpleAgent:

    def decide(self, observation):
        observation0, observation1 = observation
        if observation1 < 0:
            return 0
        else:
            return 2


class FixAgent:

    def decide(self, observation):
        pos, velocity = observation
        lb = min(-0.09 * (pos + 0.25) ** 2 + 0.03,
                 0.3 * (pos + 0.9) ** 4 - 0.008)
        ub = -0.07 * (pos + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action


class TestAgent:

    def decide(self, observation):
        pos, velocity = observation
        if (velocity <= 0.7 - 0.07 * (pos + 0.38) ** 2) \
                & \
                (min(
                    0.03 - 0.09 * (pos + 0.25)**2,
                    0.3*(pos + 0.9)**4 - 0.008) <= velocity):
            return 2
        else:
            return 0


class TestFixNoLowerbound:
    """
    I randomly found out, that the upper bound is not good for anything
    """
    def decide(self, observation):
        pos, velocity = observation
        lb = min(-0.09 * (pos + 0.25)**2 + 0.03,
                 0.3*(pos + 0.9)**4 - 0.008)

        if lb < velocity:
            return 2
        else:
            return 0


class TestCombined:
    """
    I found this candidate within 1 minute of gp
    """
    def decide(self, observation):
        pos, velocity = observation
        if (velocity <= 0.63) and (min(-0.09*(pos + 0.25)**2.0 + 0.03, 0.3*(pos + 0.9)**4.0 - 0.01) <= velocity):
            return 2
        else:
            return 0


class TestTmp:
    """
    random test
    """

    def decide(self, observation):
        pos, velocity = observation
        if (velocity <= 0.63) and (min(-0.09 * (pos + 0.25) ** 2.0 + 0.03, 0.3 * (pos + 0.9) ** 4.0 - 0.01) <= velocity):
            return 2
        else:
            return 0


def play_once(env, agent, render=False, verbose=False):
    observation = env.reset()
    episode_reward = 0.
    for step in itertools.count():
        if render:
            env.render()
        action = agent.decide(observation)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    # if verbose:
    #     print('get {} rewards in {} steps'.format(
    #         episode_reward, step + 1))
    return episode_reward


def compare_simple():
    for agent in agents:

        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)

        episode_rewards = [play_once(env, agent) for _ in range(100)]
        print('average episode rewards = {}'.format(np.mean(episode_rewards)))
        env.close()


def render_simple():
    for agent in agents:
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)

        play_once(env, agent, render=True)
        env.close()


all_agents = [SimpleAgent(), PlagihAgent_A(), FixAgent(), TestAgent(), TestFixNoLowerbound(), TestTmp()]
agents = [SimpleAgent(), FixAgent(), TestTmp()]

compare_simple()
