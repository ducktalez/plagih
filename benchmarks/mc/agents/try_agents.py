from benchmarks.mc.agents.quick_eval import *
from benchmarks.mc.agents.mtc_agent_sarsa import *


def compare_plot_styles():
    pass
    # mtc_plot_decisions_space(sarsa_agent_75, name='sarsa75 decisions', cmap='coolwarm')
    # mtc_plot_decisions_space(sarsa_agent_75, name='sarsa75 decisions dummy', cmap='coolwarm', dummy=True)
    #
    # mtc_plot_heatmap(sarsa_agent_75, name='heatmap sarsa75 1', splits=128, cmap='Greys')
    # mtc_plot_heatmap(SimpleAgent(), name='heatmap simple 1', splits=128, dummy=False, cmap='Greys')
    #
    # mtc_plot_heatmap(sarsa_agent_75, name='heatdummy sarsa75 2', splits=128, dummy=True, cmap='Greys')

    # mtc_plot_heatmap(sarsa_agent_75, name='sarsa75 states 1', splits=128, cmap='gray', dummy=True, nan_style=('white', '///', 'grey', 0.2), no_colorbar=True)
    # mtc_plot_heatmap(sarsa_agent_75, name='sarsa75 states 2', splits=128, cmap='binary', dummy=True, nan_style=('black', '', 'grey', 0.2), no_colorbar=True)
    # mtc_plot_heatmap(sarsa_agent_75, name='sarsa75 states 3', splits=128, cmap='binary', dummy=True, nan_style=('black', '///', 'grey', 0.2), no_colorbar=True)


sarsa_agent_75, sarsa_agent_200, sarsa_agent_1000, sarsa_agent_10000 = load_sarsas()

mountain_agents = [('simple', SimpleAgent()),
                   ('v1_improved', PlagihAgent_A()),
                   ('xiao_base', XiaoPresetAgent()),
                   ('xiao_short', XiaoPresetNoLowerbound()),
                   ('sarsa_75', sarsa_agent_75),
                   ('sarsa_200', sarsa_agent_200),
                   ('sarsa_1000', sarsa_agent_1000),
                   ('sarsa_10000', sarsa_agent_10000),
                   ('AgentV1p40', AgentV1p40()),
                   ('Good Expert', Good_Expert()),
                   ('TestCombined', TestCombined()),
                   ('test_tmp', TestTmp()),
                   ('test', SimonsCheckpoints()),
                   ('SimonTesting', SimonsTesting())]


class Test_Agent:
    """
    Ifte(Nand((cartVel == 0), Nand((-0.6 <= cartPos), (cartPos <= -0.4))), Ifte(True, 2, 0), Ifte(False, Ifte((cartVel < 0), 0, 2), Ifte((cartVel < 0), 0, 2)))
    """
    def decide(self, input):
        cartPos, cartVel = input

        # startposition
        if cartVel == 0 and -0.6 <= cartPos <= -0.4:  # the starting state
            if True:
                return 2
            else:
                return 0
        else:
            if False:
                if cartVel < 0:
                    return 0
                else:
                    return 2
            else:
                if cartVel < 0:
                    return 0
                else:
                    return 2


# eval_agent_list([('lelel_MTC_simple13', MTC_simple13())])

# mtc_plot_differences(MTC_simple13(), sarsa_agent_75, name='lelel_MTC_simple13 vs sarsa', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
# mtc_plot_differences(MTC_simple13(), SimpleAgent(), name='lelel_MTC_simple13 vs simple', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
# mtc_plot_differences(SimpleAgent(), MTC_simple13(), name='lelel simple vs 13', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
#
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
# mtc_plot_differences(sarsa_agent_75, SimpleAgent(), name='sarsa vs simple', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
#
# mtc_plot_differences(sarsa_agent_200, sarsa_agent_75, name='sarsa200 vs sarsa75', abs_diff=False, agent_a_dummy=True)
# mtc_plot_decisions_space(MTC_simple13(), name='decisions_13', dummy=True)
# mtc_plot_decisions_space(SimpleAgent(), name='decisions_simple', dummy=True)
# mtc_plot_decisions_space(MTC_simple13(), name='decisionsr-13')
# mtc_plot_decisions_space(SimpleAgent(), name='decisionsr-simple')

# eval_agent_list([('simyo', MTC_simple13())], goal_agent=sarsa_agent_75)

def thesis_decision_plots_fullspace():
    # sarsa agents
    mtc_plot_decisions_space(sarsa_agent_75, name='decisions-sarsa_agent_75')
    mtc_plot_decisions_space(sarsa_agent_200, name='decisions-sarsa_agent_200')
    mtc_plot_decisions_space(sarsa_agent_1000, name='decisions-sarsa_agent_1000')
    mtc_plot_decisions_space(sarsa_agent_10000, name='decisions-sarsa_agent_10000')

    mtc_plot_decisions_space(SimpleAgent(), name='decisions-SimpleAgent')
    mtc_plot_decisions_space(PlagihAgent_A(), name='decisions-PlagihAgent_A')
    mtc_plot_decisions_space(XiaoPresetAgent(), name='decisions-XiaoPresetAgent')
    mtc_plot_decisions_space(XiaoPresetNoLowerbound(), name='decisions-XiaoPresetNoLowerbound')
    mtc_plot_decisions_space(AgentV1p40(), name='decisions-AgentV1p40')
    mtc_plot_decisions_space(Good_Expert(), name='decisions-Good_Expert')

    mtc_plot_decisions_space(TestCombined(), name='decisions-Combined_AgentV1p40')


def thesis_decision_plots_dummied():
    mtc_plot_decisions_space(sarsa_agent_75, name='dummy-sarsa_agent_75', dummy=True)
    mtc_plot_decisions_space(sarsa_agent_200, name='dummy-sarsa_agent_200', dummy=True)


thesis_decision_plots_dummied()
thesis_decision_plots_fullspace()

# print('results:', mtc_play(Good_Expert(), n=100))
# print('results:', mtc_play(sarsa_agent_75, n=100))
