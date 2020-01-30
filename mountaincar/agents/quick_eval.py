import itertools
import numpy as np
import gym


class FixAgent:

    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action


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


class TestAgent:

    def decide(selfself, observation):
        observation0, observation1 = observation
        if (observation1 <= 0.7 - 0.07 * (observation0 + 0.38) ** 2) & \
                (min(0.03 - 0.09 * (observation0 + 0.25)**2, 0.3*(observation0 + 0.9)**4 - 0.008) <= observation1):
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
    if verbose:
        print('get {} rewards in {} steps'.format(
            episode_reward, step + 1))
    return episode_reward


agents = [SimpleAgent(), FixAgent(), PlagihAgent_A(), TestAgent()]
for agent in agents:

    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    episode_rewards = [play_once(env, agent) for _ in range(100)]
    print('average episode rewards = {}'.format(np.mean(episode_rewards)))
    env.close()
