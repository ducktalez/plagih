import itertools

import numpy as np
import gym
import matplotlib.pyplot as plt


class SimpleAgent:

    def decide(self, observation):
        if observation[1] < 0: #PLAGI
            return 0
        else:
            return 2


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


def move_towards_direction(env, agent, episodes, reward_interval):

    reward_list, reward_list_avg = [], []

    for i in range(episodes):
        observation = env.reset()
        done = False
        reward_tot = 0
        reward_list_preavg = []

        while(done is False):
            action = agent.decide(observation)
            observation, reward, done, info = env.step(action)
            reward_tot += reward
        reward_list_preavg.append(reward_tot)

        if i % reward_interval == 0:

            reward_list.append(reward_tot)
            reward_list_avg.append(np.mean(reward_list[-reward_interval:]))
            print('EZ-Episode {} Average Reward: {}'.format(i + 1, reward_tot))

    return reward_list, reward_list_avg


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


agent = FixAgent()

np.random.seed(0)

env = gym.make('MountainCar-v0')
env.seed(0)
episode_rewards = [play_once(env, agent) for _ in range(100)]
print('average episode rewards = {}'.format(np.mean(episode_rewards)))
env.close()
