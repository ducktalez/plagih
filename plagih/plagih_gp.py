import sys; sys.path.append('modules/') # add directory 'modules' to the current path
import plagih.modules.plagih_gp_base_class_xai as plagih
from pathlib import Path
from plagih.modules.plagih_sympy_extras import plagih_sympify


def start_plagih(run='plagih_gp_run', manual_expr=''):

    evolve_repro = 0.1   	# [0.0...1.0]  		decimal percent of pop generated through Reproduction
    evolve_point = 0.25  	# [0.0...1.0]  		decimal percent of pop generated through Point Mutation
    evolve_branch = 0.4     # [0.0...1.0]  		decimal percent of pop generated through Branch Mutation
    evolve_cross = 0.25  	# [0.0...1.0]  		decimal percent of pop generated through Crossover
    # [3 to 2^(bas +1) - 1]	minimum number of nodes
    tree_pop_max = 120
    # gen_with_max_parsimony = int(gen_max/2)

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

    gp_ops_distribution_dict = {'reproduce': evolve_repro,
                                'mutate_point': evolve_point,
                                'mutate_branch': evolve_branch,
                                'crossover': evolve_cross,
                                'missing': evolve_missing}

    config_dict = {
        'name': '_MTC_tree_012',
        'kernel': 'r',    		    # [r,c,m]			fitness function: (r)egression, (c)lassification, or (m)atching
        'tree_depth_base': 3,		# [3...10]			maximum Tree depth for initial population
        'tree_depth_max': 10,		# [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 3,	    # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': 60,   	        # [10...1000]		number of trees in each generational population
        'gen_max': 10,			    # [1...100]			number of generations
        'gp_tourn_size': 3,			# [7 per 100]		number of trees selected for tournament
        'precision': 6,             # number of floating points for the round function in 'fx_fitness_eval'
        'swim': 'p',                # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'display': 'gewsivtott',      # To display absolutely all: ewgggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'parsimony_min_max': [15, 100],
    }

    file_dict = {
        'samples_file': Path('../mountaincar/karoo_files/behaviour_samples.csv'),
        'origin_tree_file': Path('../mountaincar/karoo_files/test_tree_012.csv'),
        'operators_file': Path('../mountaincar/karoo_files/operators.csv')
    }

    monitor = {'verbosity': 'end',      # every [generation] or at the [end]
               'gen_fitness_average': 'y',
               'sympify_errors': 'y',
               'genepool_size': 'y'
               }

    gp = plagih.ExplainableGP(config_dict, file_dict, gp_ops_distribution_dict, monitor)

    if run == 'plagih_gp_run':
        gp.plagih_gp_run()
    elif run == 'expr_manual_fitness_test':
        gp.manual_expr_fitness(manual_expr)
    else:
        print('This run Mode is not specified:', run)
        raise

    return


# start_plagih(run='expr_manual_fitness_test', manual_expr=str(expr))
# start_plagih(run='expr_manual_fitness_test', manual_expr=str(expr2))

start_plagih()
