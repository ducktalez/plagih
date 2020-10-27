"""
making several stylish plots for the thesis (same style, all at once)
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
from benchmarks.mc.agents.try_agents import *


def makeonelatexplothere():
    """
    Creates one single plot for the master thesis with the mountaincar results
    """
    names = ['Sarsa 75', 'Sarsa 200', 'Sarsa 1000', 'Sarsa 10000', 'Momentum', 'Momentum (ge)', 'Preset Policy']
    mc_steps = [103.41, 98.68, 106.78, 109.31, 119.41, 129.62, 102.61]
    mc_reward = [-103.41, -98.68, -106.78, -109.31, -119.41, -129.62, -102.61]

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        fig.figsize = plplot_size_up
        x = np.arange(len(names))
        width = 0.8
        rects = ax.bar(x, mc_steps, color='b', width=width)
        ax.set(ylabel='average steps', ylim=(90, 140))
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + width/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, fontsize=11)
        savepath = path_make_dir(Path.cwd() / f'benchmarks/MC/agents/img/mc_reward_compared.pdf')
        fig.savefig(savepath)
        plt.close('all')


def thesisplot_tempdiff():
    """

    """
    def get_probs(cnt):
        fairness_bonus = np.log(cnt) + 1
        p = np.geomspace(1 + fairness_bonus, cnt + fairness_bonus, num=cnt)[::-1]
        p = p / np.sum(p)  # the sum must be equal to 1
        return np.arange(cnt), p

    xx, pp = get_probs(30)
    xx2, pp2 = get_probs(10)
    # sfeh xxx make this more beautiful
    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        ax.plot(xx, pp)
        ax.plot(xx2, pp2)
        fig.savefig(Path.cwd() / 'MA_lyx/img/pyplots_custom/distribution_obs_time.pdf')
        plt.show()
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


if __name__ == "__main__":
    thesisplot_tempdiff()
    print('SFEH COMMENTED! asd')
    # makeonelatexplothere()
    # thesis_decision_plots_dummied()
