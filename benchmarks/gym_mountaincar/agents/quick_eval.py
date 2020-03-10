"""
FYI:
pos = observation0
velocity = observation1
"""
import time

from benchmarks.gym_mountaincar.agents.agent_groups import *
import matplotlib.pyplot as plt
import itertools
import pickle
from pathlib import Path

import numpy as np
import gym


def plot_decisions(env, agent, name):
    # Creates the plot which displays current position/speed and the corresponding action

    poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
    vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
    positions, velocities = np.meshgrid(poses, vels)

    @np.vectorize
    def decide(position, velocity):
        return agent.decide((position, velocity))

    action_values = decide(positions, velocities)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(positions, velocities, action_values)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.colorbar(c, ax=ax, boundaries=[-.5, .5, 1.5, 2.5], ticks=[0, 1, 2])
    plt.title(name)
    fig.show()
    plt.savefig(Path('img/{}.jpg'.format(name)))
    plt.close()
    return


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


def compare_simple(agents):
    for name, agent in agents.items():
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)
        time_start = time.perf_counter()
        episode_rewards = [play_once(env, agent) for _ in range(100)]
        print('{} \thad average episode rewards = {}. \tTime needed: {:1.3f}s'.format(name, np.mean(episode_rewards), time.perf_counter()-time_start))
        env.close()


def render_simple(agents):
    for name, agent in agents.items():
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)

        play_once(env, agent, render=True)
        env.close()


def plot_simple(agents):
    for name, agent in agents.items():
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)

        plot_decisions(env, agent, 'agent_{}'.format(name))
        env.close()


with Path.open(Path(sarsa_file_75), 'rb') as file:
    sarsa_agent_75 = pickle.load(file)
    print('Loaded sarsa 75')

with Path.open(Path(sarsa_file_200), 'rb') as file:
    sarsa_agent_200 = pickle.load(file)
    print('Loaded sarsa 200')

with Path.open(Path(sarsa_file_1000), 'rb') as file:
    sarsa_agent_1000 = pickle.load(file)
    print('Loaded sarsa 1000')

with Path.open(Path(sarsa_file_10000), 'rb') as file:
    sarsa_agent_10000 = pickle.load(file)
    print('Loaded sarsa 10000')

all_agents = {'v1_simple__': SimpleAgent(),
              'v1_improved': PlagihAgent_A(),
              'xiao_base__': FixAgent(),
              'xiao_short_': TestFixNoLowerbound(),
              'sarsa_75___': sarsa_agent_75,
              'sarsa_200__': sarsa_agent_200,
              'sarsa_1000_': sarsa_agent_1000,
              'sarsa_10000': sarsa_agent_10000,
              'test_tmp': TestTmp()}

compare_simple(all_agents)
# plot_simple(all_agents)
