import sys;

sys.path.append('modules/')  # add directory 'modules' to the current path
import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import data_from_csv, data_load_pickle, save_data_pickle, load_operators_from_csv, load_pop_from_csv
from pathlib import Path


def get_evolve_rates_dict(evolve_rates, pop_max):
    for (type, rate) in evolve_rates.items():
        evolve_rates[type] = int(pop_max * rate)

    return evolve_rates


def create_config_dict():
    config_dict = {
        'name': 'MTC_tree_012',
        'kernel': 'regression',  # [regression, classification, match]
        'precision': 6,  # number of floating points for the round function in 'fx_fitness_eval'
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'display': 'ggewsiivoa',  # To display absolutely all: ewggggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_uniform',
        'tree_depth_base': 8,
        'tree_depth_max': 50,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 3,
        'tree_parsimony_min_max': [15, 200],  # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': 1000,
        'gen_max': 5000,
        'gp_tourn_size': 3,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'period': {'time_monitor': 60 * 0.5,  # in sec
                   'time_save': 60 * 0.5,  # in sec
                   'gen_monitor': None,  # in gen counts
                   'gen_save': None},  # in gen counts
        'evolve_rates': {'Reproduce': 0, 'Reproduce gen': 0.1, 'Reproduce Olymp': 0,
                         'Point': 0, 'Point Mutation': 0.2, 'Point Filter': 0.1,
                         'Branch': 0, 'Branch mutate one': 0.0, 'Branch nodebased': 0.3, 'Branch 2': 0, 'Branch 3': 0,
                         'Crossover': 0, 'Crossover one Branch': 0.3, 'Crossover 2': 0, 'Crossover 3': 0,
                         'Create Random': 0.0},
        'time_max': int(60 * 60 * 12),  # 60 = 1 min
        'float_accuracy': 200
    }

    return config_dict


file_dict = {
    'samples_file': Path('../mountaincar/karoo_files/data_samples/behaviour_samples.csv'),
    'samples_pickle': Path('../mountaincar/karoo_files/data_samples/plagih_data_prepared.p'),
    'operators_file': Path('../mountaincar/karoo_files/operators/operators.csv'),
    'backup_pop': Path('runs/Best_of/old_v1/population_new.csv')}

# gp.data_from_pickle(file_dict['samples_pickle'])
tree_v1 = ['Ifte', '<', '0.0', '2.0', 'observation1', '0.0']
tree_v1_modify = [0, 1, 0, 0, 1, 1]
tree_v2 = ['Ifte', '<', '0.0', 'Ifte', 'observation1', '0.0', 'True', '2.0', '1.0']
tree_v2_modify = [0, 1, 0, 0, 1, 1, 1, 0, 0]
tree_v3 = ['Ifte', '&', '2', '0', '<=', '<=', 'Mini', 'observation1', 'observation1', '+', '+', '-', '*', '0.7', '*', '0.03', '*', '0.008', '-0.07', '**',
           '-0.09', '**', '0.3', '**', '+', '2', '+', '2', '+', '4', 'observation0', '0.38', 'observation0', '0.25', 'pos', '0.9']
tree_v3_modify = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

config_dict = create_config_dict()

# data_save_pickle(prepared_data, file_dict['samples_pickle'])
# prepared_data = data_from_csv(file_dict['samples_file'])
prepared_data = data_load_pickle(file_dict['samples_pickle'])
op_array = load_operators_from_csv(file_dict['operators_file'])
pop = load_pop_from_csv(file_dict['backup_pop'])

gp = plagih.ExplainableGP(config_dict)
gp.activate_data(prepared_data)
gp.activate_operators(op_array)
gp.activate_pop(pop)

gp.load_origin_tree(label_list=tree_v2, permanent_list=tree_v2_modify)

gp.plagih_gp_run()
