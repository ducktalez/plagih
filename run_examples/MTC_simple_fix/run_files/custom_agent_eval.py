"""
pos = observation0
velocity = observation1
pos, vel = observation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gym


def mtc_plot_decisions(agent, name, folder='img/'):
    # Creates the plot which displays current position/speed and the corresponding action
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)
    observation = env.reset()

    poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
    vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
    positions, velocities = np.meshgrid(poses, vels)

    @np.vectorize
    def decide(position, velocity):
        try:
            action = agent.decide((position, velocity))
        except:
            action = 1
        return action

    action_values = decide(positions, velocities)
    env.close()

    fig, ax = plt.subplots()
    c = ax.pcolormesh(positions, velocities, action_values)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.colorbar(c, ax=ax, boundaries=[-.5, .5, 1.5, 2.5], ticks=[0, 1, 2])
    plt.title(name)
    plt.savefig(Path(folder) / '{}.jpg'.format(name))
    plt.close()
    return


def mtc_play(agent, render=False, mp4=False, n=1):
    # if mp4:
    #     env = wrappers.Monitor(env, Path('videos/{}/'.format(time.time() % 100000)))
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)
    reward_sum = 0
    fail_sum = 0
    for _ in range(n):
        episode_reward = 0
        observation = env.reset()
        while True:
            if render:
                env.render()
            try:
                action = agent.decide(observation)
            except:
                action = 1
            observation, reward, done, _ = env.step(action)

            episode_reward += reward
            if done:
                reward_sum += episode_reward
                break

        if episode_reward == 199:
            fail_sum += 1
    reward_average = reward_sum / n
    env.close()

    return reward_average, fail_sum


def eval_agents(agent_list, folder=Path('img/')):
    if not Path.is_dir(folder):
        Path.mkdir(folder)

    agent_performance = []
    for name, agent in agent_list:
        mtc_plot_decisions(agent, name, folder=folder)
        avg_reward, fails = mtc_play(agent, n=100)
        agent_performance.append([name, avg_reward, fails])

    y = [x[1] for x in agent_performance]
    x = list(range(len(agent_performance)))
    plt.bar(x, y)
    # names = [x[0] for x in agent_performance]; plt.xticks(x, names)
    plt.savefig(folder / 'agent_perf.jpg')

    with Path.open(folder / 'summary.txt', 'w') as file:
        file.write('\n'.join(['Tree {} has real average reward {} and failed {} times.'.format(x[0], x[1], x[2]) for x in agent_performance]))


if __name__ == '__main__':
    print('executing!')
    eval_agents(agent_tuples, folder=folder)
