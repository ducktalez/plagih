
import time
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

import numpy as np
import gym


def compare_simple(agents):
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('CartPole-v0')
        env.seed(0)
        episode_rewards = [play_once(env, agent) for _ in range(100)]
        failcount = sum([1 for x in episode_rewards if x < 195])
        print('{} \thad average episode rewards = {}. Failed {} times. \tTime needed: idk'.format(name, np.mean(episode_rewards), failcount))
        env.close()


def render_ntimes(agents, n, verbose=False, sleep=0):
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('CartPole-v0')
        env.seed(0)
        for _ in range(n):
            episode_rewards = play_once(env, agent, render=True, verbose=verbose, sleep=sleep)
            print('Reward sum', episode_rewards)
        env.close()


def play_once(env, agent, render=False, verbose=False, sleep=0):
    observation = env.reset()
    episode_reward = 0.
    if verbose:
        print('New agent')
    # for step in itertools.count():
    for step in range(2000):
        if render:
            env.render()
        action = agent.decide(observation, verbose=verbose)
        time.sleep(sleep)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        # if verbose:
        #     print('cart_pos {:1.2f} cart_vel {:1.2f} \tangle {:1.2f} pole_vel {:1.2f} {}'.format(observation[0], observation[1], observation[2], observation[3], '--->' if action == 1 else '<---'))
        if done:
            pass
            # break

    # if verbose:
    #     print('get {} rewards in {} steps'.format(
    #         episode_reward, step + 1))
    return episode_reward


class angle_only:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]
        if pole_angle < 0:
            return 0
        else:
            return 1


class poleVel_only:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]
        if pole_vel < 0:
            return 0
        else:
            return 1


class SimonsFirst200:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]

        if abs(pole_vel) > 0.1:
            if pole_vel < 0:
                return 0
            else:
                return 1

        # move_dir =
        if cart_vel < 0:
            return 0
        else:
            return 1


class SimonsBest:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]

        if abs(pole_vel) > 0.1:
            if pole_vel < 0:
                return 0
            else:
                return 1

        if abs(cart_vel) > 0.1:
            if cart_vel < 0:
                return 0
            else:
                return 1

        if cart_pos > 0:
            return 0
        else:
            return 1


cartpole_agents = {1: ('angle_only', angle_only()),
                   2: ('poleVel_only', poleVel_only()),
                   3: ('SimonsFirst200', SimonsFirst200()),
                   4: ('SimonsBest', SimonsBest())}

oneAgent = {cartpole_agents[3]}
twoAgents = {cartpole_agents[4], cartpole_agents[3]}

# compare_simple(oneAgent)
render_ntimes(oneAgent, 1, verbose=True)
