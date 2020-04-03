"""
FYI:
pos = observation0
velocity = observation1
pos, vel = observation
"""
import time

from benchmarks.gym_mountaincar.agents.agent_groups import *
import matplotlib.pyplot as plt
import itertools
import pickle
from pathlib import Path
from benchmarks.gym_mountaincar.agents.generated_agents import *

import numpy as np
import gym


def plot_decisions(env, agent, name):
    # Creates the plot which displays current position/speed and the corresponding action

    poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
    vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
    positions, velocities = np.meshgrid(poses, vels)

    @np.vectorize
    def decide(position, velocity):
        try:
            action = agent.decide((position, velocity))  # todo
        except:
            action = 1
        return action

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


def play_once(env, agent, render=False, verbose=False, sleep=0.0):
    observation = env.reset()
    episode_reward = 0.
    for step in itertools.count():
        if render:
            env.render()
        try:
            action = agent.decide(observation)
        except:
            action = 1
        time.sleep(sleep)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    # if verbose:
    #     print('get {} rewards in {} steps'.format(
    #         episode_reward, step + 1))
    return episode_reward


def compare_simple(agents):
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)
        time_start = time.perf_counter()
        episode_rewards = [play_once(env, agent) for _ in range(100)]
        failcount = sum([1 for x in episode_rewards if x == -200])
        print('{} \thad average episode rewards = {}. Failed {} times. \tTime needed: {:1.3f}s'.format(name, np.mean(episode_rewards),failcount, time.perf_counter() - time_start))
        env.close()


def render_ntimes(agents, n, verbose=False, sleep=0.0):
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)
        for _ in range(n):
            episode_rewards = play_once(env, agent, render=True, verbose=verbose, sleep=sleep)
            print('episode_rewards', episode_rewards)
        env.close()


def plot_simple(agents):
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)

        plot_decisions(env, agent, 'agent_{}'.format(name))
        env.close()


def load_sarsas():

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
    return sarsa_agent_75, sarsa_agent_200, sarsa_agent_1000, sarsa_agent_10000


# sarsa_agent_75, sarsa_agent_200, sarsa_agent_1000, sarsa_agent_10000 = None, None, None, None
sarsa_agent_75, sarsa_agent_200, sarsa_agent_1000, sarsa_agent_10000 = load_sarsas()
# sarsa_agent_75, _, _, _ = load_sarsas()


mountain_agents = {1: ('v1_simple', SimpleAgent()),
                   2: ('v1_improved', PlagihAgent_A()),
                   3: ('xiao_base', FixAgent()),
                   4: ('xiao_short', TestFixNoLowerbound()),
                   5: ('sarsa_75', sarsa_agent_75),
                   6: ('sarsa_200', sarsa_agent_200),
                   7: ('sarsa_1000', sarsa_agent_1000),
                   8: ('sarsa_10000', sarsa_agent_10000),
                   9: ('test_tmp', TestTmp()),
                   10: ('AgentV1p40', AgentV1p40()),
                   11: ('SimonsBest', SimonsGpFriendly()),
                   12: ('test', SimonsCheckpoints()),
                   13: ('TestCombined', TestCombined()),
                   14: ('SimonTesting', SimonsTesting())}

gen_agents = all_agents
agent_tuples = agent_tuples

oneAgent = {mountain_agents[12]}
twoAgents = {mountain_agents[14], mountain_agents[11]}

plot_simple(agent_tuples)
# compare_simple(agent_tuples)
# render_ntimes(mountain_agents.values(), 3, verbose=True, sleep=0.0)

# todo idee: gp vs. nn entscheidungen clustern.