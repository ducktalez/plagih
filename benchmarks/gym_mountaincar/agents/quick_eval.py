"""
pos = observation0
velocity = observation1
pos, vel = observation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gym


def mtc_plot_decisions_space(agent, name='space_test', folder='img/'):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
    vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
    positions, velocities = np.meshgrid(poses, vels)

    @np.vectorize
    def decide(position, velocity):
        action = agent.decide((position, velocity))
        return action

    action_values = decide(positions, velocities)
    env.close()

    # generating plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(positions, velocities, action_values)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.colorbar(c, ax=ax, boundaries=[-.5, .5, 1.5, 2.5], ticks=[0, 1, 2])
    plt.title(name)
    folder = Path(folder)
    if not Path.is_dir(folder):
        Path.mkdir(folder)
    plt.savefig(Path(folder) / '{}.jpg'.format(name))
    plt.close()
    return


def mtc_heatmap_data(env, agent, num_splits, n=100):
    # Making the data for the plot
    behaviour = []
    for _ in range(n):
        observation = env.reset()
        while True:
            action = agent.decide(observation)
            behaviour.append([observation[0], observation[1], action])
            observation, reward, done, _ = env.step(action)

            if done:
                break
    env.close()
    behaviour = np.array(behaviour)  # [(1,2,3), (1,2,3), (1,2,3), (1,2,3), (1,2,3), ...]
    positions, velocities, action_values = behaviour.T  # [[1,1,1, ...], [2,2,2, ...], [3,3, ...]]

    x_min = env.unwrapped.min_position
    x_max = env.unwrapped.max_position
    y_min = -env.unwrapped.max_speed
    y_max = env.unwrapped.max_speed

    # Making the data for the plot
    splits_x = np.linspace(x_min, x_max, num_splits)
    splits_y = np.linspace(y_min, y_max, num_splits)

    def linspace_pos(x, rangetuple, num_splits):
        num_splits -= 1
        range_spread = rangetuple[1] - rangetuple[0]
        xadd = -rangetuple[0]
        x = (x + xadd) * (num_splits / range_spread)  # normalized
        x = round(x * (num_splits)) / (num_splits)
        x = int(round(x))
        return x

    results_0 = np.zeros((num_splits, num_splits))
    results_1 = np.zeros((num_splits, num_splits))
    results_2 = np.zeros((num_splits, num_splits))
    results = np.zeros((num_splits, num_splits))
    heatmap_list = [results, results_0, results_1, results_2]

    for tup in behaviour:
        res_x = linspace_pos(tup[0], (x_min, x_max), num_splits)
        res_y = linspace_pos(tup[1], (y_min, y_max), num_splits)
        results[res_x, res_y] += 1

    return splits_x, splits_y, results


def mtc_plot_heatmap(agent, n=100, name='heatmap_test', folder='img/', num_splits=128, decisions=None):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    splits_x, splits_y, results = mtc_heatmap_data(env, agent, num_splits)

    # generating plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(splits_x, splits_y, results, cmap='Greys')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.colorbar(c, ax=ax)
    plt.title(name)
    folder = Path(folder)
    if not Path.is_dir(folder):
        Path.mkdir(folder)
    plt.savefig(Path(folder) / '{}.jpg'.format(name))
    plt.close()
    return


def mtc_plot_differences(agent_a, agent_b, agent_a_dummy=None, n=100, num_splits=256, name='differences', folder='img/', abs_diff=True, cmap='viridis'):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, num_splits)
    vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, num_splits)
    positions, velocities = np.meshgrid(poses, vels)

    @np.vectorize
    def decide_diff(position, velocity):
        action_a = agent_a.decide((position, velocity))
        action_b = agent_b.decide((position, velocity))
        return action_a - action_b

    result = decide_diff(positions, velocities)

    if abs_diff:
        result = abs(result)
        boundaries=np.linspace(-0.5,2.5,4)
        ticks=np.linspace(0,2,3)
    else:
        boundaries=np.linspace(-2.5,2.5,6)
        ticks=np.linspace(-2,2,5)
        cmap='PiYG'

    if agent_a_dummy:
        hm_x, hm_y, hm_res = mtc_heatmap_data(env, agent_a, num_splits)
        dummy_res = abs(np.sign(hm_res))
        result = result * dummy_res


    env.close()

    # generating plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(positions, velocities, result, cmap=cmap)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.colorbar(c, ax=ax, boundaries=boundaries, ticks=ticks)
    plt.title(name)
    folder = Path(folder)
    if not Path.is_dir(folder):
        Path.mkdir(folder)
    plt.savefig(Path(folder) / '{}.jpg'.format(name))
    plt.close()
    return


def mtc_play(agent, render=False, mp4=False, n=1):
    # if mp4:
    #     env = wrappers.Monitor(env, Path('videos/{}/'.format(time.time() % 100000)))
    np.random.seed(0);env = gym.make('MountainCar-v0');env.seed(0)
    reward_sum = 0
    fail_sum = 0
    for asd in range(n):
        episode_reward = 0
        observation = env.reset()
        while True:
            if render:
                env.render()
            action = agent.decide(observation)
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


def eval_agent_list(agent_list, folder=Path('img/')):
    if not Path.is_dir(folder):
        Path.mkdir(folder)

    agent_performance = []
    for name, agent in agent_list:
        mtc_plot_decisions_space(agent, name, folder=folder)
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
    eval_agent_list(agent_tuples)  # , folder=folder
