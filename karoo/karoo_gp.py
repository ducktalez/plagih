import sys; sys.path.append('modules/') # add directory 'modules' to the current path
import karoo.modules.plagih_gp_base_class_xai; gp = karoo.modules.plagih_gp_base_class_xai.Base_GP()
from pathlib import Path

# • Generational (g): pauses after each generation is complete
# • Interactive (i): pauses with the completion of each section (e.g. tournament, gene pool, genetic operators)
# • DeBug (db): displays the internal workings of the genetic operators
# • Minimal (m): displays only the multivariate expression of each tree
# • Silent (s): displays only the summary of each generations

kernel = 'r'    		# [r,c,m]			fitness function: (r)egression, (c)lassification, or (m)atching
tree_type = 'g'    		# [f,g,r]			Tree type: (f)ull, (g)row, or (r)amped half/half
tree_depth_base = 3		# [3...10]			maximum Tree depth for initial population
tree_depth_max = 10		# [3...10]			maximum Tree depth for entire run
tree_depth_min = 3		# [3 to 2^(bas +1) - 1]	minimum number of nodes
tree_pop_max = 60    	# [10...1000]		number of trees in each generational population
gen_max = 10			# [1...100]			number of generations
tourn_size = 10			# [7 per 100]		number of trees selected for tournament

evolve_repro = 0.1   	# [0.0...1.0]  		decimal percent of pop generated through Reproduction
evolve_point = 0.25  	# [0.0...1.0]  		decimal percent of pop generated through Point Mutation
evolve_branch = 0.4     # [0.0...1.0]  		decimal percent of pop generated through Branch Mutation
evolve_cross = 0.25  	# [0.0...1.0]  		decimal percent of pop generated through Crossover

evolve_repro = int(evolve_repro*tree_pop_max)
evolve_point = int(evolve_point*tree_pop_max)
evolve_branch = int(evolve_branch*tree_pop_max)
evolve_cross = int(evolve_cross*tree_pop_max)
evolve_missing = 0

evolve_total = evolve_repro + evolve_point + evolve_branch + evolve_cross

if evolve_total > tree_pop_max:
    print('Error: A new generation has more than', evolve_total, 'candidates. It has', tree_pop_max)
    exit()
elif evolve_total < tree_pop_max:
    print('Error: A new generation has less than', evolve_total, 'candidates. It has', tree_pop_max)
    print('Missing', tree_pop_max-evolve_total, 'will be random o')
    evolve_missing = evolve_missing + tree_pop_max-evolve_total
evolve_distribution = [evolve_repro, evolve_point, evolve_branch, evolve_cross, evolve_missing]

kernel = 'r'
display = 's'  # display mode is set to (s)ilent
precision = 6  # number of floating points for the round function in 'fx_fitness_eval'
swim = 'p'  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
mode = 'db'  # pause at the (d)esktop when complete, awaiting further user interaction; or terminate in (s)erver mode



tourn_size = 4
gen_max = 20
tree_pop_max = 50
samples_file = Path('../mountaincar/karoo_files/behaviour_samples.csv')
origin_tree_file = Path('../mountaincar/karoo_files/test_tree.csv')
operators_file = Path('../mountaincar/karoo_files/operators.csv')
# display = 'gewsivto'  #
gene_pool_threshold = 0.5  # this amount of percent a tree needs to fulfill to be in the gene pool
parsimony_min_max = [15, 50]
display = {'generation_steps': 'y',
           'errors': 'y',
           'warnings': 'y',
           'verbosity_lvl': '3',
           'function': 'n',
           'custom': 'y',
           }
monitor = {'verbosity': 'g',  # every generation
           'gen_fitness_avg': 'y',
           'sympify_errors': 'y',}

# origin_tree_file = Path('../mountaincar/karoo_files/test_tree2.csv')
gp.plagih_gp(kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, gen_max,
                   tourn_size, operators_file, samples_file, origin_tree_file, evolve_distribution, display,
                   precision, swim, mode, gene_pool_threshold, parsimony_min_max, monitor)


