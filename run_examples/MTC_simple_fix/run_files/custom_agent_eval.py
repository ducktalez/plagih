"""
pos = observation0
velocity = observation1
pos, vel = observation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gym
import pickle


def mtc_plot_decisions_space(agent, name='space_test', folder='img/', cmap='bwr', dummy=False, n=100, boundaries=None, ticks=None, nan_style=None, no_colorbar=False):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    x_linspace = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
    y_linspace = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
    positions, velocities = np.meshgrid(x_linspace, y_linspace)

    @np.vectorize
    def decide(position, velocity):
        action = agent.decide((position, velocity))
        return action

    results = decide(positions, velocities)
    if dummy:
        _, _, result_dummy = mtc_heatmap_helper(env, agent, 256, n, dummy=dummy)
        results = results * result_dummy

    env.close()

    boundaries = [-.5, .5, 1.5, 2.5]
    ticks = [0, 1, 2]

    mtc_plot(x_linspace, y_linspace, results, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)

    return


def mtc_heatmap_helper(env, agent, num_splits, n, dummy=False):
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
    x_linspace = np.linspace(x_min, x_max, num_splits)
    y_linspace = np.linspace(y_min, y_max, num_splits)

    def linspace_pos(x, rangetuple, num_splits):  # returns the position of the bucket the calue is in
        num_splits -= 1
        range_spread = rangetuple[1] - rangetuple[0]
        xadd = -rangetuple[0]
        x = (x + xadd) * (num_splits / range_spread)  # normalized
        x = round(x * (num_splits)) / (num_splits)
        x = int(round(x))
        return x

    result = np.zeros((num_splits, num_splits))
    # result[:] = np.nan  # np.nan instead of zero for better visualisation

    for tup in behaviour:
        res_x = linspace_pos(tup[0], (x_min, x_max), num_splits)
        res_y = linspace_pos(tup[1], (y_min, y_max), num_splits)
        result[res_y, res_x] += 1  # the mesh is not a coordinate system...

    if dummy:
        result = np.vectorize(lambda x: np.nan if x == 0 else 1)(result)
    else:
        result = np.vectorize(lambda x: np.nan if x == 0 else x)(result)  # sfeh: normalize values?

    return x_linspace, y_linspace, result


def mtc_plot_heatmap(agent, n=100, name='heatmap_test', folder='img/', splits=128, dummy=False, cmap='Greys', boundaries=None, ticks=None, nan_style=None, no_colorbar=False):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    x_linspace, y_linspace, result = mtc_heatmap_helper(env, agent, splits, n, dummy=dummy)

    mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)

    return


def mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=False, boundaries=None, ticks=None, nan_style=None, no_colorbar=False):
    # generating plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x_linspace, y_linspace, result, cmap=cmap)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')

    if no_colorbar:
        boundaries = np.array([0, 1])  # don't know why, but makes the bar disappear somehow

    if dummy:

        mask_nan = np.ma.masked_where(result == np.nan, result)
        plt.pcolor(x_linspace, y_linspace, mask_nan, hatch=None, cmap=cmap, alpha=1)

        fig.colorbar(c, ax=ax, boundaries=boundaries, ticks=ticks)  # needed, plot is stretched otherwise  # todo: whitespace, find better solution

        # sfeh normalize?
        # plt.cm.get_cmap().set_bad(color='white')
        # plt.imshow(result)
    else:
        fig.colorbar(c, ax=ax)  # needed, plot is stretched otherwise

    # get data you will need to create a "nan_style patch" to your plot
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xy = (xmin, ymin)
    width = xmax - xmin
    height = ymax - ymin

    # create the patch and place it in the back of countourf (zorder!)

    p = patches.Rectangle(xy, width, height, fill=None, zorder=0.5)  # "zorder=-10" -> nan_style

    if nan_style:
        ax.set_facecolor(nan_style[0])
        p.set_color(nan_style[0])
        p.fill = True
        p.set_hatch(nan_style[1])
        p.set_edgecolor(nan_style[2])
        plt.rcParams['hatch.linewidth'] = nan_style[3]
    else:
        # p.fill = False
        # p.set_color()  # default 'white' is okay
        ax.set_facecolor('xkcd:light grey')
        p.set_color('xkcd:light grey')
        p.fill = True
        p.set_hatch('//')
        p.set_edgecolor('xkcd:dark grey')
        plt.rcParams['hatch.linewidth'] = 0.2
    ax.add_patch(p)

    # saving as jpg
    folder = Path(folder)
    if not Path.is_dir(folder):
        Path.mkdir(folder)
    plt.savefig(Path(folder) / '{}.png'.format(name))
    plt.close()


def mtc_plot_differences(agent_a, agent_b, agent_a_dummy=None, n=100, num_splits=256, name='diff', folder='img/',
                         abs_diff=True, cmap='bwr', nan_style=None, no_colorbar=False):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    x_linspace = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, num_splits)
    y_linspace = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, num_splits)
    positions, velocities = np.meshgrid(x_linspace, y_linspace)

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
        _, _, result_dummy = mtc_heatmap_helper(env, agent_a, num_splits, n, dummy=True)
        result = result * result_dummy  # just transfer all 'nan', done here as x*nan=nan, x*1=1

        # boundaries = np.linspace(-0.5, 2.5, 4)
        # ticks = np.linspace(0, 2, 3)

    env.close()

    mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=agent_a_dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)

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
    #
    # mtc_plot_decisions_space(agent, name, folder=folder)
    _, _, reward_list = mtc_play(agent, n=n)

    x = np.arange(n)
    plt.plot(x, reward_list, color=color)

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title(name)

    plt.ylim(-200, -80)

    plt.savefig(folder / '{}.png'.format(name))


def eval_agent_list(agent_list, folder=Path('img/')):

    if not Path.is_dir(folder):
        Path.mkdir(folder)

    # with Path.open(Path('sarsa_agent_75.p'), 'rb') as file:
    #     sarsa_agent_75 = pickle.load(file)

    agent_performance = []
    for name, agent in agent_list:
        mtc_plot_decisions_space(agent, name=name, folder=folder, dummy=True)
        # mtc_plot_differences(agent, sarsa_agent_75, name='diff-{}'.format(name), folder=folder, abs_diff=False, agent_a_dummy=True)
        avg_reward, fails, _ = mtc_play(agent, n=100)
        agent_performance.append([name, avg_reward, fails])

    y = [x[1] for x in agent_performance]
    x = list(range(len(agent_performance)))
    plt.bar(x, y)
    # names = [x[0] for x in agent_performance]; plt.xticks(x, names)
    plt.savefig(folder / 'agent_perf.png')

    summary_text = '\n'.join(['Tree {} has real average reward {} and failed {} times.'.format(x[0], x[1], x[2]) for x in agent_performance])
    with (folder / 'summary.txt').open('w') as file:
        file.write(summary_text)



if __name__ == '__main__':
    print('executing!')
    eval_agent_list(agent_tuples, folder=folder)
