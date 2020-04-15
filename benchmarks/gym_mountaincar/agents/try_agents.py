from benchmarks.gym_mountaincar.agents.quick_eval import *
from benchmarks.gym_mountaincar.agents.mtc_agent_sarsa import *
# from pathlib import Path


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

# mtc_plot_decisions_space(sarsa_agent_75, name='sarsa75 decisions', cmap='coolwarm')
#
# mtc_plot_heatmap(sarsa_agent_75, name='sarsa75 states heatmap', splits=128, cmap='Greys')
mtc_plot_heatmap(sarsa_agent_75, name='sarsa75 states', splits=128, dummymap=True, cmap='Greys')
mtc_plot_heatmap(SimpleAgent(), name='simple states', splits=128, dummymap=False, cmap='Greys')
#
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa75 differences', abs_diff=False, cmap='PiYG')
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa75 differences in states PiYG', abs_diff=False, agent_a_dummy=True, cmap='PiYG')
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa75 differences in states Greys', abs_diff=False, agent_a_dummy=True, cmap='Greys')
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa75 differences in states coolwarm', abs_diff=False, agent_a_dummy=True, cmap='coolwarm')
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs sarsa75 absolute differences', abs_diff=True, cmap='Greys')
#
# mtc_plot_episode_performance(sarsa_agent_75, name='sarsa75 individual performance', color='y')
# mtc_plot_episode_performance(SimpleAgent(), name='simple individual performance', color='b')
