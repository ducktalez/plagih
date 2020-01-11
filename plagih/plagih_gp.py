import sys;

sys.path.append('modules/')  # add directory 'modules' to the current path
import plagih.modules.plagih_gp_base_class_xai as plagih
from pathlib import Path


def get_evolve_rates_dict(rates, pop_max):

    amounts = [int(x * pop_max) for x in rates]
    evolve_total = sum(amounts)
    missing = 0
    if evolve_total > pop_max:
        print('Error: A new generation has more than', evolve_total, 'candidates. It has', pop_max)
        exit()
    elif evolve_total < pop_max:
        print('Error: A new generation has less than', evolve_total, 'candidates. It has', pop_max)
        print('Missing', pop_max - evolve_total, 'will be TODO')
        missing = pop_max - evolve_total

    evolve_rates = {'Reproduce': amounts[0],
                    'Point Mutation': amounts[1],
                    'Point Filter': amounts[2],
                    'Branch Mutation': amounts[3],
                    'Crossover': amounts[4],
                    'Create Random': amounts[5] + missing}
    return evolve_rates


def create_config_dict(pop_max, gen_max, evolve_rates):
    config_dict = {
        'name': '_MTC_tree_012',
        'kernel': 'regression',  # [regression, classification, match]
        'precision': 6,  # number of floating points for the round function in 'fx_fitness_eval'
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'display': 'ggewsivott',  # To display absolutely all: ewggggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_uniform',
        'tree_depth_base': 5,  # [3...10]			maximum Tree depth for initial population
        'tree_depth_max': 15,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 3,
        'tree_parsimony_min_max': [15, 100],  # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': pop_max,  # [10...1000]		number of trees in each generational population
        'gen_max': gen_max,  # [1...100]			number of generations
        'gp_tourn_size': 3,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'evolve_rates': evolve_rates,
        'time_max': 60*1  # 60 = 1 min
    }

    return config_dict


"""
The must crucial parameters for testing are here
"""
pop_max = 200
gen_max = 20

file_dict = {
    'samples_file': Path('../mountaincar/karoo_files/data_samples/behaviour_samples.csv'),
    'samples_pickle': Path('../mountaincar/karoo_files/data_samples/plagih_data_prepared.p'),
    'operators_file': Path('../mountaincar/karoo_files/operators/operators.csv')
}
label_list = ['Ifte', '<', '0', 'Ifte', 'observation1', '0', 'True', '2', '0']
permanent_list = [0, 1, 0, 0, 1, 1, 1, 0, 0]
evolve_rates_list = [0.1,  # repro, point, filter, branch, cross
                     0.1,  # point
                     0.1,  # filter
                     0.1,  # branch
                     0.3,  # cross
                     0.3]  # Create Random
evolve_rates = get_evolve_rates_dict(evolve_rates_list, pop_max)

config_dict = create_config_dict(pop_max, gen_max, evolve_rates)
gp = plagih.ExplainableGP(config_dict)
gp.data_from_csv(file_dict['samples_file'], save_pickle_path=file_dict['samples_pickle'])
# gp.data_from_pickle(file_dict['samples_pickle'])
gp.data_load_operators(file_dict['operators_file'])
gp.load_origin_tree(label_list=label_list, permanent_list=permanent_list)
gp.plagih_gp_run()
