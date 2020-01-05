import sys;

sys.path.append('modules/')  # add directory 'modules' to the current path
import plagih.modules.plagih_gp_base_class_xai as plagih
from pathlib import Path


def get_evolve_rates_dict(rates, tree_pop_max):

    amounts = [int(x * tree_pop_max) for x in rates]
    evolve_total = sum(amounts)
    missing = 0
    if evolve_total > tree_pop_max:
        print('Error: A new generation has more than', evolve_total, 'candidates. It has', tree_pop_max)
        exit()
    elif evolve_total < tree_pop_max:
        print('Error: A new generation has less than', evolve_total, 'candidates. It has', tree_pop_max)
        print('Missing', tree_pop_max - evolve_total, 'will be TODO')
        missing = tree_pop_max - evolve_total

    evolve_rates = {'Reproduce': amounts[0],
                    'Point Mutation': amounts[1],
                    'Point Filter': amounts[2],
                    'Branch Mutation': amounts[3],
                    'Crossover': amounts[4],
                    'missing': amounts[5] + missing}
    return evolve_rates


def start_plagih(tree_pop_max, gen_max, evolve_rates):
    config_dict = {
        'name': '_MTC_tree_012',
        'kernel': 'regression',  # [regression, classification, match]
        'precision': 6,  # number of floating points for the round function in 'fx_fitness_eval'
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'display': 'gggewsivott',  # To display absolutely all: ewggggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_uniform',
        'tree_depth_base': 5,  # [3...10]			maximum Tree depth for initial population
        'tree_depth_max': 15,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 3,
        'tree_parsimony_min_max': [15, 100],  # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': tree_pop_max,  # [10...1000]		number of trees in each generational population
        'gen_max': gen_max,  # [1...100]			number of generations
        'gp_tourn_size': 3,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'evolve_rates': evolve_rates
    }

    return plagih.ExplainableGP(config_dict)


"""
The must crucial parameters for testing are here
"""
tree_pop_max = 100
gen_max = 20

file_dict = {
    'samples_file': Path('../mountaincar/karoo_files/behaviour_samples.csv'),
    # 'origin_tree_file': Path('../mountaincar/karoo_files/test_tree_012.csv'),
    'operators_file': Path('../mountaincar/karoo_files/operators.csv')
}
label_list = ['Ifte', '<', '0', 'Ifte', 'observation1', '0', 'True', '2', '0']
permanent_list = [0, 1, 0, 0, 1, 1, 1, 0, 0]
#         repro, point, filter, branch, cross
evolve_rates_list = [0.1, 0.1, 0.2, 0.3, 0.3, 0]
evolve_rates = get_evolve_rates_dict(evolve_rates_list, tree_pop_max)

gp = start_plagih(tree_pop_max, gen_max, evolve_rates)
gp.data_load_samples_csv(file_dict['samples_file'])
gp.data_load_operators(file_dict['operators_file'])
gp.load_origin_tree(label_list=label_list, permanent_list=permanent_list)
gp.plagih_gp_run()
