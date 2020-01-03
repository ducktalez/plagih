import sys;

sys.path.append('modules/')  # add directory 'modules' to the current path
import plagih.modules.plagih_gp_base_class_xai as plagih
from pathlib import Path


def start_plagih(run='plagih_gp_run', manual_expr=''):
    evolve_repro = 0.1  # [0.0...1.0]  		decimal percent of pop generated through Reproduction
    evolve_point = 0.25  # [0.0...1.0]  		decimal percent of pop generated through Point Mutation
    evolve_branch = 0.4  # [0.0...1.0]  		decimal percent of pop generated through Branch Mutation
    evolve_cross = 0.25  # [0.0...1.0]  		decimal percent of pop generated through Crossover
    # [3 to 2^(bas +1) - 1]	minimum number of nodes
    tree_pop_max = 120

    evolve_repro = int(evolve_repro * tree_pop_max)
    evolve_point = int(evolve_point * tree_pop_max)
    evolve_branch = int(evolve_branch * tree_pop_max)
    evolve_cross = int(evolve_cross * tree_pop_max)
    evolve_missing = 0
    evolve_total = evolve_repro + evolve_point + evolve_branch + evolve_cross

    if evolve_total > tree_pop_max:
        print('Error: A new generation has more than', evolve_total, 'candidates. It has', tree_pop_max)
        exit()
    elif evolve_total < tree_pop_max:
        print('Error: A new generation has less than', evolve_total, 'candidates. It has', tree_pop_max)
        print('Missing', tree_pop_max - evolve_total, 'will be random o')
        evolve_missing = evolve_missing + tree_pop_max - evolve_total

    config_dict = {
        'name': '_MTC_tree_012',
        'kernel': 'regression',  # [regression, classification, match]
        'precision': 6,  # number of floating points for the round function in 'fx_fitness_eval'
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'display': 'ggewsivott',  # To display absolutely all: ewggggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_uniform',
        'tree_depth_base': 3,  # [3...10]			maximum Tree depth for initial population
        'tree_depth_max': 10,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 3,
        'tree_parsimony_min_max': [15, 100],  # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': 60,  # [10...1000]		number of trees in each generational population
        'gen_max': 30,  # [1...100]			number of generations
        'gp_tourn_size': 3,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'evolve_ratio': {'reproduce': evolve_repro,
                         'mutate_point': evolve_point,
                         'mutate_branch': evolve_branch,
                         'crossover': evolve_cross,
                         'missing': evolve_missing}
    }

    return plagih.ExplainableGP(config_dict)


gp = start_plagih()

file_dict = {
    'samples_file': Path('../mountaincar/karoo_files/behaviour_samples.csv'),
    # 'origin_tree_file': Path('../mountaincar/karoo_files/test_tree_012.csv'),
    'operators_file': Path('../mountaincar/karoo_files/operators.csv')
}
gp.data_load_data(file_dict['samples_file'])
gp.data_load_operators(file_dict['operators_file'])
label_list = ['Ifte', '<', '0', 'Ifte', 'observation1', '0', 'True', '2', '0']
permanent_list = [0, 1, 0, 0, 1, 1, 1, 0, 0]
gp.load_origin_tree(label_list=label_list, permanent_list=permanent_list)
gp.plagih_gp_run()
