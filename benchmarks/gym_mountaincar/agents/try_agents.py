from benchmarks.gym_mountaincar.agents.quick_eval import *
from benchmarks.gym_mountaincar.agents.mtc_agent_sarsa import *

# from pathlib import Path

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


    # # background style
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 1', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('black', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 2', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('black', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 3', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('xkcd:dark grey', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 4', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:dark grey', None, 'grey', 0.2))
    #
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 5', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 6', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('xkcd:off white', None, 'grey', 0.2))
    #
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 7', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 8', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('grey', None, 'grey', 0.2))

    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 color 9', abs_diff=False, agent_a_dummy=True, cmap='cool', nan_style=('xkcd:dark grey', None, 'xkcd:dark grey', 0.2))

    # hatch patterns
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 2 hatch 1', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '///', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 2 hatch 2', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', 'XXX', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 2 hatch 3', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '...', 'black', 0.2))

    # hatch pattern density
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 3 hatchdense 1', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '/', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 3 hatchdense 2', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '//', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 3 hatchdense 3', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '///', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 3 hatchdense 4', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '////', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 3 hatchdense 5', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '/////', 'black', 0.2))

    # hatch line thickness
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick 1', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'black', 0.1))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick 2', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick 3', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'black', 0.3))
    # hatch line thickness compared with a lighter color
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick gr 1', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'grey', 0.1))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick gr 2', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick gr 3', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'grey', 0.3))
    # # hatch line thickness light grey
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick ow 1', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:dark grey', '///', 'xkcd:off white', 0.1))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick ow 2', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:dark grey', '///', 'xkcd:off white', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 4 hatchthick ow 3', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:dark grey', '///', 'xkcd:off white', 0.3))
    #
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 5 cmap 1', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 5 cmap 1', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('white', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 5 cmap 1', abs_diff=False, agent_a_dummy=True, cmap='PiYG', nan_style=('white', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 5 cmap 1', abs_diff=False, agent_a_dummy=True, cmap='gray', nan_style=('white', '///', 'grey', 0.2))

    # Final options?
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt bwr 2', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '...', 'grey', 0.3))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt bwr 3', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt bwr 4', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('grey', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt bwr 5', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt bwr 6', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('grey', '///', 'xkcd:dark grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt bwr 7', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:dark grey', '///', 'black', 0.2))

    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt cw 1', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('white', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt cw 3', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('xkcd:off white', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt cw 4', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('xkcd:off white', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt cw 5', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('xkcd:off white', '///', 'xkcd:dark grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff opt cw 6', abs_diff=False, agent_a_dummy=True, cmap='coolwarm', nan_style=('xkcd:dark grey', '///', 'black', 0.2))

    # Just for fun, experiments
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-diff 1 fun 1', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:magenta', 'XXX', 'black', 1))
    #

    # black/white data scale
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-abs opt 2', abs_diff=True, agent_a_dummy=True, cmap='Greys', nan_style=('xkcd:off white', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-abs opt 2', abs_diff=True, agent_a_dummy=True, cmap='Greys', nan_style=('xkcd:off white', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-abs opt 3', abs_diff=True, agent_a_dummy=True, cmap='Greys', nan_style=('xkcd:light grey', None, 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-abs opt 3', abs_diff=True, agent_a_dummy=True, cmap='Greys', nan_style=('xkcd:light grey', '///', 'grey', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-abs opt 3', abs_diff=True, agent_a_dummy=True, cmap='Greys', nan_style=('xkcd:light grey', None, 'black', 0.2))
    # mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='vs-abs opt 3', abs_diff=True, agent_a_dummy=True, cmap='Greys', nan_style=('xkcd:light grey', '///', 'black', 0.2))
    #
    # mtc_plot_episode_performance(sarsa_agent_75, name='sarsa75 individual performance', color='y')
    # mtc_plot_episode_performance(SimpleAgent(), name='simple individual performance', color='b')


mountain_agents = [('simple', SimpleAgent()),
                   ('v1_improved', PlagihAgent_A()),
                   ('xiao_base', FixAgent()),
                   ('xiao_short', TestFixNoLowerbound()),
                   ('sarsa_75', sarsa_agent_75),
                   ('sarsa_200', sarsa_agent_200),
                   ('sarsa_1000', sarsa_agent_1000),
                   ('sarsa_10000', sarsa_agent_10000),
                   ('test_tmp', TestTmp()),
                   ('AgentV1p40', AgentV1p40()),
                   ('Good Expert', Good_Expert()),
                   ('test', SimonsCheckpoints()),
                   ('TestCombined', TestCombined()),
                   ('SimonTesting', SimonsTesting())]


class MTC_simple13:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-2.0) * cartPos) + 0.555) + (lambda x, y: x / y if y != 0 else 0)(math.sin(cartPos), (cartPos * cartVel)))
        return max(0, min(2, int(round(action))))


# eval_agent_list([('lelel_MTC_simple13', MTC_simple13())])

# mtc_plot_differences(MTC_simple13(), sarsa_agent_75, name='lelel_MTC_simple13 vs sarsa', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
# mtc_plot_differences(MTC_simple13(), SimpleAgent(), name='lelel_MTC_simple13 vs simple', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
# mtc_plot_differences(SimpleAgent(), MTC_simple13(), name='lelel simple vs 13', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
#
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
# mtc_plot_differences(sarsa_agent_75, SimpleAgent(), name='sarsa vs simple', abs_diff=False, agent_a_dummy=True, cmap='bwr', nan_style=('xkcd:light grey', '///', 'xkcd:dark grey', 0.2))
#
# mtc_plot_differences(sarsa_agent_200, sarsa_agent_75, name='sarsa200 vs sarsa75', abs_diff=False, agent_a_dummy=True)
# mtc_plot_decisions_space(MTC_simple13(), name='decisions-13', dummy=True)
# mtc_plot_decisions_space(SimpleAgent(), name='decisions-simple', dummy=True)
# mtc_plot_decisions_space(MTC_simple13(), name='decisionsr-13')
# mtc_plot_decisions_space(SimpleAgent(), name='decisionsr-simple')


eval_agent_list([('simyo', MTC_simple13())])
