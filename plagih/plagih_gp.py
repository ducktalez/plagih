import sys
sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current path

import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import *
from pathlib import Path
from plagih.modules.Examples import *


def prepare_data_pickle(prepared_data, file_dict):
    save_data_pickle(prepared_data, file_dict['samples_pickle'])


def get_evolve_rates_dict(evolve_rates, pop_max):
    for (type, rate) in evolve_rates.items():
        evolve_rates[type] = int(pop_max * rate)

    return evolve_rates


def create_config_dict():
    config_dict = {
        'path': Path.cwd(),
        'name': 'MTC_tree_',
        'kernel': 'regression',  # [regression, classification, match]
        'precision': 6,  # rounding the fitness
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'display': 'ggewsiivoa',  # To display absolutely all: ewggggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_random',
        'tree_depth_base': 7,
        'tree_depth_max': 50,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 5,
        'tree_parsimony_min_max': [15, 200],  # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': 600,
        'gen_max': 1000,
        'gp_tourn_size': 5,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'period': {'time_monitor': 60 * 0.5,  # in sec
                   'time_save': 60 * 0.5,  # in sec
                   'gen_monitor': None,  # in gen counts
                   'gen_save': None},  # in gen counts
        'evolve_rates': {'Reproduce gen': 0.05, 'Reproduce Olymp': 0.0,
                         'Point Mutation': 0.5, 'Point Filter': 0.05,
                         'Branch nodebased': 0.1,
                         'Crossover one Branch': 0.35,
                         'Create Random': 0.35},
        # 'evolve_rates': {'Reproduce': 0, 'Reproduce gen': 0.0, 'Reproduce Olymp': 0.0,
        #                  'Point': 0, 'Point Mutation': 0.0, 'Point Filter': 0.0,
        #                  'Branch': 0, 'Branch mutate one': 0.0, 'Branch nodebased': 0.0, 'Branch 2': 0, 'Branch 3': 0,
        #                  'Crossover': 0, 'Crossover one Branch': 0.0, 'Crossover 2': 0, 'Crossover 3': 0,
        #                  'Create Random': 0.0},
        'time_max': int(60 * 60 * 12),  # 60 = 1 min
        'float_accuracy': 200
    }

    return config_dict


def fast_run(config):
    config['pop_max'] = 200
    config['gen_max'] = 15
    config['gp_tourn_size'] = 3
    return config


def mountaincar_prepare(config_dict, path):

    # prepared_data = data_from_csv(file_dict['samples_file'])

    gp = plagih.ExplainableGP(config_dict)
    prepared_data = data_load_pickle(path / MountainCarExamples.files['samples_pickle'])
    gp.activate_data(prepared_data)
    op_array = load_operators_from_csv(path / MountainCarExamples.files['operators_file'])
    gp.activate_operators(op_array)
    return gp


def mountaincar_v1(config_dict, path):
    config_dict['name'] = 'MTC_v1_'
    config_dict['path'] = path
    gp = mountaincar_prepare(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v1_list, modify_list=MountainCarExamples.tree_v1_modify)
    return gp


def mountaincar_v2(config_dict, path):
    config_dict['name'] = 'MTC_v2_'
    config_dict['path'] = path
    gp = mountaincar_prepare(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def mountaincar_v3(config_dict, path):
    config_dict['name'] = 'MTC_v3_'
    config_dict['path'] = path
    gp = mountaincar_prepare(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    return gp


def mountaincar_tmp(config_dict, path):
    config_dict['name'] = 'MTC_tmp_'
    config_dict['path'] = path
    config_dict = fast_run(config_dict)
    gp = mountaincar_prepare(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def run(root_dir):
    config_dict = create_config_dict()
    gp = mountaincar_tmp(config_dict, root_dir)
    gp.plagih_gp_run()

