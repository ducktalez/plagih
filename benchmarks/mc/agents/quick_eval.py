"""
pos = observation0
velocity = observation1
pos, vel = observation
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
import gym
import pickle
# import multiprocessing as mp


def mtc_plot_decisions_space(agent, folder='img/', name='space_test', cmap='bwr', dummy=False, n=100, nan_style=None, no_colorbar=False):
    """
    plotting the decision space
    """
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    x_linspace = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
    y_linspace = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
    positions, velocities = np.meshgrid(x_linspace, y_linspace)

    env.close()

    @np.vectorize
    def decide(position, velocity):
        action = agent.decide((position, velocity))
        return action

    results = decide(positions, velocities)
    if dummy:
        _, _, result_dummy = mtc_heatmap_helper(agent, 256, n, dummy=dummy)
        results = results * result_dummy

    ticks = np.linspace(0, 2, 3)
    boundaries = np.linspace(-0.5, 2.5, 4)

    mtc_plot(x_linspace, y_linspace, results, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)

    return


def mtc_heatmap_helper(agent, num_splits, n, dummy=None):
    # Making the data for the plot
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

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

    if dummy == 1:
        result = np.vectorize(lambda x: np.nan if x == 0 else 1)(result)
    else:  # either dummy=2 OR no dummy
        result = np.vectorize(lambda x: np.nan if x == 0 else x)(result)  # sfeh: envstate_normalize values?

    return x_linspace, y_linspace, result


def mtc_plot_heatmap(agent, n=100, name='heatmap_test', folder='img/', splits=128, dummy=False, cmap='Greys', boundaries=None, ticks=None, nan_style=None, no_colorbar=False):

    x_linspace, y_linspace, result = mtc_heatmap_helper(agent, splits, n, dummy=dummy)
    mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)

    return


def mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=False, boundaries=None, ticks=None, vmin=None, vmax=None, nan_style=None, no_colorbar=False):
    # generating plot
    if vmin and vmax:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None
    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x_linspace, y_linspace, result, cmap=cmap, norm=norm)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')

        if no_colorbar:
            boundaries = np.array([0, 1])  # don't know why, but makes the bar disappear somehow

        if dummy:
            mask_nan = np.ma.masked_where(result == np.nan, result)
            ax.pcolor(x_linspace, y_linspace, mask_nan, hatch=None, cmap=cmap, alpha=1)
            fig.colorbar(c, ax=ax, boundaries=boundaries, ticks=ticks)  # needed, plot is stretched otherwise

        else:
            fig.colorbar(c, ax=ax, boundaries=boundaries, ticks=ticks)  # insert empty colorbar, plot is stretched otherwise

        # get data you will need to create a "nan_style patch" to your plot
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xy = (xmin, ymin)
        width = xmax - xmin
        height = ymax - ymin

        p = patches.Rectangle(xy, width, height, fill=None, zorder=0.5)  # "zorder=-10" -> nan_style

        if nan_style:
            ax.set_facecolor(nan_style[0])
            p.set_color(nan_style[0])
            p.fill = True
            p.set_hatch(nan_style[1])
            p.set_edgecolor(nan_style[2])
            fig.rcParams['hatch.linewidth'] = nan_style[3]
        else:
            p.set_color('xkcd:dark grey')
            p.fill = True
            p.set_hatch('//')
            # ax.set_facecolor('xkcd:light grey')
            # p.set_edgecolor('xkcd:dark grey')
            # plt.rcParams['hatch.linewidth'] = 0.2
        ax.add_patch(p)

        # saving as png
        folder = Path(folder)
        if not Path.is_dir(folder):
            Path.mkdir(folder)
        fig.savefig(Path(folder) / f'{name}.pdf')
        plt.close('all')


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


def mtc_plot_episode_performance(agent, name='episode performance', folder=Path('img/'), n=100, color='b'):
    """
    plot the performance of all tested episodes (e.g. for showing outliers)
    delete this?
    """
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

    plt.savefig(folder / f'{name}.pdf')


def mtc_plot_differences(agent, diff_agent, dummy_result=None, boarders=1, num_splits=256, name='diff', folder='img/',
                         abs_diff=True, cmap='bwr', nan_style=None, no_colorbar=False):
    """
    Creates the difference-plot
    """
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    x_linspace = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, num_splits)
    y_linspace = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, num_splits)
    env.close()

    positions, velocities = np.meshgrid(x_linspace, y_linspace)

    @np.vectorize
    def decide_diff(position, velocity):
        action_a = agent.decide((position, velocity))
        action_b = diff_agent.decide((position, velocity))
        return action_a - action_b

    result = decide_diff(positions, velocities)

    if abs_diff:
        # result = abs(result)
        ticks = np.linspace(0, 2*boarders, min(3*boarders, 10))
        boundaries = np.linspace(-(0.5+0), 0.5+2*boarders, 1+min(3*boarders, 10))
        vmin = 0
        vmax = 2 * boarders
    else:
        vmin = -2 * boarders
        vmax = +2 * boarders
        ticks = np.linspace(-(2*boarders), 2*boarders, 1+min(4*boarders, 10))
        boundaries = np.linspace(-(0.5+2*boarders), 0.5+2*boarders, 2+min(4*boarders, 10))

    # if boarders > 1:
    #     boundaries = boundaries * boarders
    #     ticks = ticks * boarders
    #     ticks = None  #

    if dummy_result is None:
        mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=False, boundaries=boundaries, ticks=ticks, vmin=vmin, vmax=vmax, nan_style=nan_style, no_colorbar=no_colorbar)
    else:
        result = result * dummy_result  # just transfer all 'nan', done here as x*nan=nan, x*1=1
        mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=True, boundaries=boundaries, ticks=ticks, vmin=vmin, vmax=vmax, nan_style=nan_style, no_colorbar=no_colorbar)

    return


def eval_agent_list(agent_list, goal_agent, n=100, dir_save=Path('img/')):
    """
    Automatically evalueate gp-agents (difference plot, decision plot, performance)

    """
    if not Path.is_dir(dir_save):
        Path.mkdir(dir_save)

    agent_performance = []
    _, _, sarsa_dummy = mtc_heatmap_helper(goal_agent, 256, n, dummy=1)

    for name, agent in agent_list:
        print('Evaluating agent: {}'.format(name))
        mtc_plot_decisions_space(agent, name=name, folder=dir_save, dummy=True)
        mtc_plot_differences(agent, goal_agent, dummy_result=sarsa_dummy, boarders=1, name=f'diff-{name}', folder=dir_save, abs_diff=False)
        avg_reward, fails, _ = mtc_play(agent, n=n)
        agent_performance.append([name, avg_reward, fails])

    y = [x[1] for x in agent_performance]
    x = list(range(len(agent_performance)))

    plt.bar(x, y)
    # names = [x[0] for x in agent_performance]; plt.xticks(x, names)
    plt.savefig(dir_save / 'agent_perf.pdf')

    summary_text = '\n'.join(['Tree {} has real average reward {} and failed {} times.'.format(x[0], x[1], x[2]) for x in agent_performance])
    with (dir_save / 'summary.txt').open('w') as file:
        file.write(summary_text)


def auto_evaluate_run_end(root_dir, prepared_run, sarsa_agent, n=100):
    """
    asd
    """

    class DummyMcAgent:

        def __init__(self, pycode):
            self.mcAction = pycode

        def decide(self, input):
            cartPos, cartVel = input
            try:
                mc_actn = eval(self.mcAction)
            except:
                raise  # sfeh
            return int(round(max(0, min(2, mc_actn))))

    _, _, sarsa_dummy = mtc_heatmap_helper(sarsa_agent, 256, n, dummy=1)

    """
    Just create the subfolder
    """
    dir_save = root_dir / 'sfehs_eval'
    if not Path.is_dir(dir_save):
        Path.mkdir(dir_save)

    """
    load pareto front from old run
    """
    with Path.open(root_dir / 'backup/backup.p', 'rb') as file:
        gp_backup_data = pickle.load(file)
    gen_id, pareto, pop_base, monitor_pd, a_helping_dict = gp_backup_data

    agent_performance = []
    for (parsim, fitness, cooltree) in pareto:
        agent_name = f'{prepared_run}_{parsim:.0f}'
        print(f'Evaluating MC Agent: {agent_name}')
        pycode = cooltree.get_pycode()
        mcAgent = DummyMcAgent(pycode)
        try:
            avg_reward, fails, _ = mtc_play(mcAgent, n=n)
            agent_performance.append([agent_name, parsim, avg_reward, fails])
            mtc_plot_decisions_space(mcAgent, folder=dir_save, name=agent_name, dummy=True)
            mtc_plot_differences(mcAgent, sarsa_agent, folder=dir_save, name=f'diff-{agent_name}', dummy_result=sarsa_dummy, boarders=1, abs_diff=False)  # diff at start for diashow
        except Exception as ex:
            print(f'MTC eval failed because of: {ex}')
            # agent_performance.append([agent_name, parsim, 0, 0])

    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        fig.tight_layout()
        x = [x[1] for x in agent_performance]
        y = [x[2] for x in agent_performance]
        ax.bar(x, y)
        fig.savefig(dir_save / f'evaled_overview.pdf')

    summary_text = '\n'.join(['Tree {} has real average reward {} and failed {} times.'.format(x[0], x[1], x[2]) for x in agent_performance])
    with (dir_save / 'summary.txt').open('w') as file:
        file.write(summary_text)
