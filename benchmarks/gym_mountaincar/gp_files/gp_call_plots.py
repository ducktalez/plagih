import numpy as np
import gym
import matplotlib.pyplot as plt
from pathlib import Path


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


def plot_simple(agents):
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('MountainCar-v0')
        env.seed(0)

        plot_decisions(env, agent, 'agent_{}'.format(name))
        env.close()