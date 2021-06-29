"""
making several stylish plots for the thesis (same style, all at once)
not relevant for the framework (at all)
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
from plagih.util import *
from benchmarks.ib.ib_eval_agents import eval_combined_agents
from benchmarks.mc.agents.try_agents import *


def autolabel_xx(ax, rects, width):
    """
    sfeh maybee
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + width / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    return ax, rects, width


# def thesis_ib_agent_results():
#     ibx = eval_agent(ib_agent)
#     ibx_safe = eval_agent(ib_agent, complete=False)
#     ibx_r50 = eval_agent(ib_agent, randomize=50, repeat_avg=10)
#     ibx_safe_r50 = eval_agent(ib_agent, complete=False, randomize=50, repeat_avg=10)


def thesis_plot_mc_comparisson():
    """
    Creates one single plot for the master thesis with the mountaincar results
    """
    names = ['Sarsa 75', 'Sarsa 200', 'Sarsa 1000', 'Sarsa 10000', 'Momentum', 'Momentum (ge)', 'Parabolic']  # sfeh , 'if-Expert'
    mc_steps = [103.41, 98.68, 106.78, 109.31, 119.41, 129.62, 102.61]  # sfeh thesis , 100.3
    mc_reward = [-103.41, -98.68, -106.78, -109.31, -119.41, -129.62, -102.61]

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        fig.figsize = plplot_size_up
        x = np.arange(len(names))
        width = 0.8
        rects = ax.bar(x, mc_steps, color='b', width=width)
        ax.set(ylabel='average steps', ylim=(90, 140))

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + width/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, fontsize=11)
        savepath = path_make_dir(Path.cwd() / f'benchmarks/MC/agents/img/mc_reward_compared.pdf')
        fig.savefig(savepath)
        plt.close('all')


def thesis_plot_ib_comparisson():
    """
    Plots that compare the main industrial benchmark solutions
    """
    tuples = {'Random': [-6809, -6632],
              'Nothing': [-6077, -6068],
              'Hein-21': [-5278, -5573],
              'Hein-27': [-5268, -5548],
              'Hein-29': [-5232, -5542],
              '50-50-50': [-8273, -8321]}

    def autolabel(rects, add_texty=0, textcolor='b'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + width / 2, height),
                        xytext=(0, -12+add_texty),
                        textcoords="offset points",
                        color=textcolor,
                        ha='center', va='bottom')

    names = tuples.keys()
    reward = [x[0] for x in tuples.values()]
    reward_safe = [x[1] for x in tuples.values()]

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        fig.figsize = plplot_size_up
        x = np.arange(len(names))
        width = 0.8/2
        rects1 = ax.bar(x-width/2, reward, color='b', width=width, label='regular')
        rects2 = ax.bar(x+width/2, reward_safe, color='g', width=width, label='safe')

        ax.legend(loc='lower left')
        plt.yticks(IB_YICKS[0], IB_YICKS[1])

        # rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
        # rects2 = ax.bar(x + width / 2, women_means, width, label='Women')
        ax.set(ylabel='reward [x1000]', ylim=(-15000, -4000))

        autolabel(rects1, textcolor='b')
        # autolabel(rects2, add_texty=-9)  # only main results

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, fontsize=11)
        savepath = path_make_dir(Path.cwd() / f'MA_lyx/img/IB/ib_reward_compared.pdf')
        fig.savefig(savepath)
        plt.close('all')


def thesis_mc_comparisson():
    xx = np.linspace(-5, 5, 100)
    yy = np.abs(xx)
    xxd = np.linspace(-5, 5, 10+1)
    yyd = np.round(xx)

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        fig.figsize = plplot_size_up
        ax.legend(loc='lower left')
        ax.plot(xx, yy)


def thesisplot_tempdiff():
    """

    """
    def get_probs(cnt):
        fairness_bonus = np.log(cnt) + 1
        p = np.geomspace(1 + fairness_bonus, cnt + fairness_bonus, num=cnt)[::-1]
        p = p / np.sum(p)  # the sum must be equal to 1
        return np.arange(cnt), p

    # = [get_probs(x) for x in [5, 10, 20, 30, 50]]
    xx1, pp1 = get_probs(5)
    xx2, pp2 = get_probs(10)
    xx3, pp3 = get_probs(20)
    xx4, pp4 = get_probs(30)
    xx5, pp5 = get_probs(50)
    # sfeh xxx make this more beautiful
    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        ax.plot(xx1, pp1, linestyle='dotted', marker='.', label='5 values')
        ax.plot(xx2, pp2, linestyle='dotted', marker='.', label='10 values')
        ax.plot(xx3, pp3, linestyle='dotted', marker='.', label='20 values')
        ax.legend(loc='upper right')
        # ax.plot(xx4, pp4, linestyle='dotted', marker='.')
        # ax.plot(xx5, pp5, linestyle='dotted', marker='.')
        ax.set(xlabel='age', ylabel='probability')
        savepath = path_make_dir(Path.cwd() / 'MA_lyx/img/pyplots_custom/distribution_obs_time.pdf')
        fig.savefig(savepath)
        plt.close('all')


def thesis_decision_plots_fullspace(folder=Path.cwd() / 'benchmarks/mc/agents/img'):
    # sarsa agents

    mtc_plot_decisions_space(sarsa_agent_75, folder=folder, name='decisions-sarsa_agent_75')
    mtc_plot_decisions_space(sarsa_agent_200, folder=folder, name='decisions-sarsa_agent_200')
    mtc_plot_decisions_space(sarsa_agent_1000, folder=folder, name='decisions-sarsa_agent_1000')
    mtc_plot_decisions_space(sarsa_agent_10000, folder=folder, name='decisions-sarsa_agent_10000')

    mtc_plot_decisions_space(SimpleAgent(), folder=folder, name='decisions-SimpleAgent')
    mtc_plot_decisions_space(PlagihAgent_A(), folder=folder, name='decisions-PlagihAgent_A')
    mtc_plot_decisions_space(XiaoPresetAgent(), folder=folder, name='decisions-XiaoPresetAgent')
    mtc_plot_decisions_space(XiaoPresetNoLowerbound(), folder=folder, name='decisions-XiaoPresetNoLowerbound')
    mtc_plot_decisions_space(AgentV1p40(), folder=folder, name='decisions-AgentV1p40')
    mtc_plot_decisions_space(Good_Expert(), folder=folder, name='decisions-Good_Expert')

    mtc_plot_decisions_space(TestCombined(), folder=folder, name='decisions-Combined_AgentV1p40')

    mtc_plot_decisions_space(sarsa_agent_75, folder=folder, name='dummy-sarsa_agent_75', dummy=True)
    mtc_plot_decisions_space(sarsa_agent_200, folder=folder, name='dummy-sarsa_agent_200', dummy=True)


# def main2():
#
#     compile_files = list(Path('benchmarks/').glob('slurm_runs*/**/analysis_overview_plus.tex'))
#     for file in compile_files:
#
#         print(f'===NEWFILE===\n{file.as_posix()}\n\n')
#
#         try:
#             os.system(f'pdflatex {file.as_posix()}')
#         except Exception as ex:
#             print(f'Failed conversion for {file.name}. ignoring. Reason: {ex}')
#
#     return


def compare_savehuman_vs_sarsa():
    """
    plot for the thesis
    sarsa-75 is better, but has an outlier
    """
    sarsa75, _, _, _ = load_sarsas(set_epsilon=0)
    simple_agent = SimpleAgent()

    _, _, rewards_sarsa = mtc_play(sarsa75, n=100)
    _, _, rewards_mtc = mtc_play(simple_agent, n=100)

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(rewards_sarsa)), rewards_sarsa, label='Sarsa-75')
        ax.plot(np.arange(len(rewards_mtc)), rewards_mtc, label='Maximize velocity')
        ax.set(xlabel='episode', ylabel='reward', xlim=(0, 100))  # 1.05  # top * 1.05 for better style
        ax.legend(loc='lower left')
        plt.show()
        savepath = path_make_dir(Path.cwd() / 'MA_lyx/img/pyplots_custom/sarsa_vs_human.pdf')
        fig.savefig(savepath)
        plt.close('all')


if __name__ == "__main__":
    print(f'Making all the plots for the thesis (so they are python-style)')
    # compare_savehuman_vs_sarsa()
    # thesis_decision_plots_fullspace()
    thesisplot_tempdiff()
    # thesis_plot_ib_comparisson()
    # # thesis_plot_mc_comparisson()
    # # thesis_decision_plots_dummied()
    # res = mtc_play(Good_Expert(), n=100)
    # res = mtc_play(sarsa_agent_200, n=100)
    # print(res)
    # mtc_plot_decisions_space(Good_Expert(), folder=Path.cwd(), name='decisions-Good_Expert')
    # mtc_plot_decisions_space(FixAgentRe(), folder=Path.cwd(), name='decisions-FixAgentRe_DEL')
