import os
import sys; sys.path.append('modules/') # add directory 'modules' to the current path
import argparse
import karoo.modules.karoo_gp_base_class_xai; gp = karoo.modules.karoo_gp_base_class_xai.Base_GP()

kernel= 'r'    			# [r,c,m]			fitness function: (r)egression, (c)lassification, or (m)atching
tree_type= 'g'    		# [f,g,r]			Tree type: (f)ull, (g)row, or (r)amped half/half
tree_depth_base= 3		# [3...10]			maximum Tree depth for initial population
tree_depth_max= 10		# [3...10]			maximum Tree depth for entire run
tree_depth_min= 3		# [3 to 2^(bas +1) - 1]	minimum number of nodes
tree_pop_max= 100    	# [10...1000]		number of trees in each generational population
gen_max= 10				# [1...100]			number of generations
tourn_size= 10			# [7 per 100]		number of trees selected for tournament
evolve_repro= 0.25   	# [0.0...1.0]  		decimal percent of pop generated through Reproduction
evolve_point= 0.25   	# [0.0...1.0]  		decimal percent of pop generated through Point Mutation
evolve_branch= 0.25   	# [0.0...1.0]  		decimal percent of pop generated through Branch Mutation
evolve_cross= 0.25   	# [0.0...1.0]  		decimal percent of pop generated through Crossover

#++++++++++++++++++++++++++++++++++++++++++
#   Conduct the GP run                    |
#++++++++++++++++++++++++++++++++++++++++++
kernel='p'
gp.plagih_karoo_gp(kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, gen_max, tourn_size, filename, evolve_repro, evolve_point, evolve_branch, evolve_cross, display, precision, swim, mode)


