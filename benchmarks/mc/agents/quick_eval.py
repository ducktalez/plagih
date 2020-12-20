"""
sfeh-specific file. not required otherwise.

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
import math
import pickle
# import multiprocessing as mp
from plagih.file_interaction import *


def mtc_plot_decisions_space(agent, folder, name, cmap='bwr', dummy=False, n=100, nan_style=None, no_colorbar=False, backup_results1=None):
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

    if backup_results1 is None:

        @np.vectorize
        def decide(position, velocity):
            action = agent.decide((position, velocity))
            return action

        results = decide(positions, velocities)
        result_dummy = None
        if dummy:
            _, _, result_dummy = mtc_heatmap_helper(agent, 256, n, dummy=dummy)
            results = results * result_dummy
        backup_results1 = (results, result_dummy)
    else:
        results, result_dummy = backup_results1

    ticks = np.linspace(0, 2, 3)
    boundaries = np.linspace(-0.5, 2.5, 4)
    mtc_plot(x_linspace, y_linspace, results, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)
    return backup_results1


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


def mtc_plot_heatmap(agent, n=100, name='heatmap_test', folder=Path.cwd() / 'img/', splits=128, dummy=False, cmap='Greys', boundaries=None, ticks=None, nan_style=None, no_colorbar=False):
    """

    """
    x_linspace, y_linspace, result = mtc_heatmap_helper(agent, splits, n, dummy=dummy)
    mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, nan_style=nan_style, no_colorbar=no_colorbar)
    return


def mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=False, boundaries=None, ticks=None, vmin=None, vmax=None, nan_style=None, no_colorbar=False):
    """
    Mountaincar-specific space-plot
    plots the whole state space
    """
    if vmin and vmax:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        plt.yticks(MTC_XTICKS[0], MTC_XTICKS[1])
        plt.xticks(MTC_YTICKS[0], MTC_YTICKS[0])
        c = ax.pcolormesh(x_linspace, y_linspace, result, cmap=cmap, norm=norm, rasterized=True)
        ax.set(xlabel='position', ylabel='velocity')

        if no_colorbar:  # sfeh weird solution... placeholder for the space
            boundaries = np.array([0, 1])  # don't know why, but makes the bar disappear somehow

        if dummy:
            mask_nan = np.ma.masked_where(result == np.nan, result)
            ax.pcolor(x_linspace, y_linspace, mask_nan, hatch=None, cmap=cmap, alpha=1, rasterized=True)

        fig.colorbar(c, ax=ax, boundaries=boundaries, ticks=ticks)  # needed, plot is stretched otherwise

        # get data you will need to create a "nan_style patch" to your plot
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xy = (xmin, ymin)
        width = xmax - xmin
        height = ymax - ymin

        p = patches.Rectangle(xy, width, height, fill=True, zorder=0.5, rasterized=True)  # "zorder=-10" -> nan_style

        if nan_style:
            ax.set(facecolor=nan_style[0], color=nan_style[0])
            p.set(hatch=nan_style[1], edgecolor=nan_style[2], rasterized=True)
            fig.rcParams['hatch.linewidth'] = nan_style[3]
        else:
            p.set(hatch='//', color='xkcd:dark grey', rasterized=True)

        ax.add_patch(p)

        path_mcmeshplot = path_make_dir(folder / f'{name}.pdf')
        fig.savefig(path_mcmeshplot)
        print(f'saved {path_mcmeshplot}')
        plt.close('all')

    return


def mtc_play(agent, render=False, n=1):
    """
    going through n amounts of MC-runs
    """
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
    print(f'asd {np.average(list_episode_rewards)}')

    return reward_average, fail_sum, list_episode_rewards


# def mtc_plot_episode_performance(agent, name='episode performance', folder=Path('img/'), n=100, color='b'):
#     """
#     plot the performance of all tested episodes (e.g. for showing outliers)
#     delete this?
#     """
#     if not Path.is_dir(folder):
#         Path.mkdir(folder)
#     #
#     # mtc_plot_decisions_space(agent, name, folder=folder)
#     _, _, reward_list = mtc_play(agent, n=n)
#
#     x = np.arange(n)
#     plt.plot(x, reward_list, color=color)
#
#     plt.xlabel('episode')
#     plt.ylabel('reward')
#     plt.title(name)
#
#     plt.ylim(-200, -80)
#
#     plt.savefig(folder / f'{name}.pdf')


def mtc_plot_differences(agent, diff_agent, dummy_result=None, boarders=1, num_splits=256, name='diff', folder=Path.cwd() / 'img/',
                         abs_diff=True, cmap='bwr', nan_style=None, no_colorbar=False, backup_results2=None):
    """
    Creates the difference-plot
    """
    if backup_results2:
        pass
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # Making the data for the plot
    x_linspace = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, num_splits)
    y_linspace = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, num_splits)
    env.close()

    positions, velocities = np.meshgrid(x_linspace, y_linspace)

    if backup_results2 is None:
        @np.vectorize
        def decide_diff(position, velocity):
            action_a = agent.decide((position, velocity))
            action_b = diff_agent.decide((position, velocity))
            return action_a - action_b

        result = decide_diff(positions, velocities)

        if dummy_result is not None:
            result = result * dummy_result  # just transfer all 'nan', done here as x*nan=nan, x*1=1
            dummy = True
        else:
            dummy = False
        backup_results2 = (result, dummy)
    else:
        result, dummy = backup_results2 

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
    mtc_plot(x_linspace, y_linspace, result, cmap, folder, name, dummy=dummy, boundaries=boundaries, ticks=ticks, vmin=vmin, vmax=vmax, nan_style=nan_style, no_colorbar=no_colorbar)

    return backup_results2


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

        mtc_plot_decisions_space(agent, folder=dir_save, name=name, dummy=True)
        mtc_plot_differences(agent, goal_agent, dummy_result=sarsa_dummy, boarders=1, folder=dir_save, name=f'diff-{name}', abs_diff=False)
        avg_reward, fails, _ = mtc_play(agent, n=n)
        agent_performance.append([name, avg_reward, fails, dir_save / name])

    y = [x[1] for x in agent_performance]
    x = list(range(len(agent_performance)))

    plt.bar(x, y)
    # names = [x[0] for x in agent_performance]; plt.xticks(x, names)
    plt.savefig(dir_save / 'agent_perf.pdf')

    summary_text = '\n'.join([f'Tree {x[0]} has real average reward {x[1]} and failed {x[2]} times.' for x in agent_performance])
    with (dir_save / 'summary.txt').open('w') as file:
        file.write(summary_text)


def auto_evaluate_run_end(root_dir, sarsa_agent, n=100):
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

    try:
        sarsa_dummy, bur_lut = pickle_load(root_dir / 'backup/mcevalbackup.p')  # sarsa_dummy  (results, result_dummy)   (result, dummy)
    except Exception:
        bur_lut = {}
        _, _, sarsa_dummy = mtc_heatmap_helper(sarsa_agent, 256, n, dummy=1)

    dir_save = path_make_dir(root_dir / 'sfehs_eval')

    gp_backup_data = pickle_load(root_dir / 'backup/backup.p')
    gen_id, pareto, pop_base, monitor_pd, a_helping_dict = gp_backup_data

    agent_performance = {}

    for (parsim, fitness, cooltree) in pareto:
        agent_name = f'{parsim:.0f}'
        print(f'Evaluating MC Agent: {parsim:.0f}')
        pycode = cooltree.get_pycode()
        mc_gent = DummyMcAgent(pycode)
        try:
            # bur1, bur2, avg_reward, fails = bur_lut.get(parsim, (None, None, None, None))
            # if avg_reward is None:  # or fails is None:
            bur1 = bur2 = None
            avg_reward, fails, _ = mtc_play(mc_gent, n=n)

            # sfehsfeh save  time comment this todotodo todo
            bur1 = mtc_plot_decisions_space(mc_gent, folder=dir_save, name=agent_name, dummy=True, backup_results1=bur1)
            bur2 = mtc_plot_differences(mc_gent, sarsa_agent, folder=dir_save, name=f'diff-{agent_name}', dummy_result=sarsa_dummy, boarders=1, abs_diff=False, backup_results2=bur2)  # diff at start for diashow
            mtc_plot_decisions_space(mc_gent, folder=dir_save, name=f'space-{agent_name}', dummy=False)
            bur_lut[parsim] = (bur1, bur2, avg_reward, fails)
            agent_performance[parsim] = [parsim, fitness, avg_reward, fails, None, None]
        except Exception as ex:
            print(f'MTC eval failed because of: {ex}')
            agent_performance[parsim] = [parsim, fitness, np.nan, np.nan, None, None]

    # pickle_dump(root_dir / 'backup/mcevalbackup.p', (sarsa_dummy, bur_lut))

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        agentperflist = list(zip(*agent_performance.values()))
        x = agentperflist[0]
        y = agentperflist[2]
        tuples = [[parsim, fitness] for (parsim, fitness, cooltree) in pareto]
        xx, yy = np.array(tuples).T
        ax.step(xx, yy, linestyle='dotted', marker='.', where='post')
        ax.set(xlabel='complexity', ylabel='regression error', ylim=(0, 1))
        # ax.legend('upper right')

        ax2 = ax.twinx()
        ax2.set(ylim=(-95, -200), xlim=(0, 35))
        ax2.step(x, y, linestyle='None', marker='x', label='real reward')
        ax2.legend(loc='upper right')

        fig.savefig(dir_save / f'evaled_overview.pdf')

    # summary_text = '\n'.join([f'Tree {x[0]} (regr. error {x[1]}) has real average reward {x[2]} and failed {x[3]} times.' for x in agent_performance.values()])
    # file_dump(dir_save / 'summary.txt', summary_text)
    yaml_dump(dir_save / 'summary.yaml', agent_performance)

    return agent_performance
