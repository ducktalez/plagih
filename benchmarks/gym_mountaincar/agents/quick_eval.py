"""
pos = observation0
velocity = observation1
pos, vel = observation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gym


def mtc_plot_decisions_space(agent, name='space_test', folder='img/', cmap='coolwarm'):
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
    c = ax.pcolormesh(positions, velocities, action_values, cmap=cmap)
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


def mtc_heatmap_data(env, agent, num_splits, n):
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
    behaviour = np.array(behaviour)  # [(pos, vel, act), (-0.5,0.01,2), ...]
    positions, velocities, action_values = behaviour.T  # [[pos, pos, ..], [vel, ..], [act, ..]]

    x_min = env.unwrapped.min_position
    x_max = env.unwrapped.max_position
    y_min = -env.unwrapped.max_speed
    y_max = env.unwrapped.max_speed

    # Making the data for the plot
    splits_x = np.linspace(x_min, x_max, num_splits)
    splits_y = np.linspace(y_min, y_max, num_splits)

    def linspace_pos(x, rangetuple, num_splits):  # returns the position of the bucket the calue is in
        num_splits -= 1
        range_spread = rangetuple[1] - rangetuple[0]
        xadd = -rangetuple[0]
        x = (x + xadd) * (num_splits / range_spread)  # normalized
        x = round(x * (num_splits)) / (num_splits)
        x = int(round(x))
        return x

    results = np.zeros((num_splits, num_splits))

    for tup in behaviour:
        res_x = linspace_pos(tup[0], (x_min, x_max), num_splits)
        res_y = linspace_pos(tup[1], (y_min, y_max), num_splits)
        results[res_y, res_x] += 1  # the mesh is not a coordinate system...

    return splits_x, splits_y, results


def mtc_plot_heatmap(agent, n=100, name='heatmap_test', folder='img/', splits=128, dummymap=False, cmap='Greys'):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    splits_x, splits_y, results = mtc_heatmap_data(env, agent, splits, n)

    if dummymap:
        results = abs(np.sign(results))
        boundaries = np.linspace(-0.5, 1.5, 3)
        ticks = np.linspace(0, 1, 2)

    # generating plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(splits_x, splits_y, results, cmap=cmap)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')

    fig.colorbar(c, ax=ax)  # needed, plot is stretched otherwise
    plt.title(name)
    folder = Path(folder)
    if not Path.is_dir(folder):
        Path.mkdir(folder)
    plt.savefig(Path(folder) / '{}.jpg'.format(name))
    plt.close()
    return


def mtc_plot_differences(agent_a, agent_b, agent_a_dummy=None, n=100, num_splits=256, name='differences', folder='img/', abs_diff=True, cmap='coolwarm'):
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
        boundaries = np.linspace(-0.5, 2.5, 4)
        ticks = np.linspace(0, 2, 3)
    else:
        boundaries = np.linspace(-2.5, 2.5, 6)
        ticks = np.linspace(-2, 2, 5)

    if agent_a_dummy:
        _, _, heatmap_result = mtc_heatmap_data(env, agent_a, num_splits, n)
        dummy_res = abs(np.sign(heatmap_result))
        result = result * dummy_res

        boundaries = np.linspace(-0.5, 2.5, 4)
        ticks = np.linspace(0, 2, 3)

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


def mtc_play(agent, render=False, n=1):
    # if mp4:
    #     env = wrappers.Monitor(env, Path('videos/{}/'.format(time.time() % 100000)))
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)
    reward_sum = 0
    list_episode_rewards = []
    fail_sum = 0
    for _ in range(n):
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
                list_episode_rewards.append(episode_reward)
                break

        if episode_reward == 199:
            fail_sum += 1
    reward_average = reward_sum / n
    env.close()

    return reward_average, fail_sum, list_episode_rewards


def mtc_plot_episode_performance(agent, name='episode perfoemance', folder=Path('img/'), n=100, color='b'):

    if not Path.is_dir(folder):
        Path.mkdir(folder)

    mtc_plot_decisions_space(agent, name, folder=folder)
    _, _, reward_list = mtc_play(agent, n=n)

    x = np.arange(n)
    plt.plot(x, reward_list, color=color)

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title(name)

    plt.ylim(-200, -80)

    plt.savefig(folder / '{}.jpg'.format(name))


def eval_agent_list(agent_list, folder=Path('img/')):
    if not Path.is_dir(folder):
        Path.mkdir(folder)

    agent_performance = []
    for name, agent in agent_list:
        mtc_plot_decisions_space(agent, name, folder=folder)
        avg_reward, fails, _ = mtc_play(agent, n=100)
        agent_performance.append([name, avg_reward, fails])

    y = [x[1] for x in agent_performance]
    x = list(range(len(agent_performance)))
    plt.bar(x, y)
    # names = [x[0] for x in agent_performance]; plt.xticks(x, names)
    plt.savefig(folder / 'agent_perf.jpg')

    with Path(folder / 'summary.txt', 'w').open as file:
        file.write('\n'.join(['Tree {} has real average reward {} and failed {} times.'.format(x[0], x[1], x[2]) for x in agent_performance]))
