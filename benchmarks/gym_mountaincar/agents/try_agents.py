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

# mtc_plot_heatmap(sarsa_agent_75, name='sarsa75-heatmap')
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs. Sarsa - diff', abs_diff=False)
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs. Sarsa - diff + dummy', abs_diff=False, agent_a_dummy=True)
mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs. Sarsa - diff + dummy', abs_diff=False, agent_a_dummy=True)
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs. Sarsa')
# mtc_plot_differences(SimpleAgent(), sarsa_agent_75, name='simple vs. Sarsa +dummies', agent_a_dummy=True)
