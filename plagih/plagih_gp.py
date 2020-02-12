import sys
sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current path

import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import *
from pathlib import Path
from plagih.modules.Examples import *
from plagih.modules.plagih_data import *


def create_config_dict():
    config_dict = {
        'path': Path.cwd(),
        'name': 'MTC_tree_',
        'kernel': 'regression',  # [regression, classification, match]
        'precision': 6,  # rounding the fitness
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'print_type': 'ggewsiivoa',  # To print_type absolutely all: ewggggsiiiivvvtopppttt
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_random',
        'tree_depth_base': 7,
        'tree_depth_max': 50,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 5,
        'tree_parsimony_min_max': [15, 200],  # [3 to 2^(bas +1) - 1]	minimum number of nodes
        'pop_max': 600,
        'gen_max': 1000,
        'gp_tourn_size': 7,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'period': {'time_monitor': None,    # in sec
                   'time_save': None,       # in sec
                   'gen_monitor': 1,        # in gen counts
                   'gen_save': 1},          # in gen counts
        'evolve_rates': {'Reproduce gen': 0.05, 'Reproduce Olymp': 0.05, 'Reproduce reduce': 0.05,
                         'Point Mutation': 0.05, 'Point Filter': 0.05,
                         'Branch nodebased': 0.1,
                         'Crossover one Branch': 0.35,
                         'Create Random': 0.30},
        'float_accuracy': 200,
        'time_max': 60 * 60 * 7  # 60min * 60s * hours
    }

    return config_dict


def mountaincar_load_corefiles(config_dict, path):

    # prepared_data = data_from_csv(file_dict['samples_file'])

    gp = plagih.ExplainableGP(config_dict)
    prepared_data = data_load_pickle(path / MountainCarExamples.files['samples_pickle'])
    gp.activate_data(prepared_data)
    op_array = load_operators_from_csv(path / MountainCarExamples.files['operators_file'])
    gp.activate_operators(op_array)
    return gp


def mountaincar_v1(config_dict, path):
    config_dict['name'] = 'MTC_v1'
    config_dict['path'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v1_list, modify_list=MountainCarExamples.tree_v1_modify)
    return gp


def mountaincar_v2(config_dict, path):
    config_dict['name'] = 'MTC_v2'
    config_dict['path'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def mountaincar_v3(config_dict, path):
    config_dict['name'] = 'MTC_v3'
    config_dict['path'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    return gp


def mountaincar_v4_scratch(config_dict, path):
    """
    mountain car example from scratch. origin is empty
    """
    config_dict['name'] = 'MTC_v4_scratch'
    config_dict['path'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list='')
    return gp


def mountaincar_test(config_dict, path):
    config_dict['name'] = 'MTC_test'
    config_dict['path'] = path
    config_dict['pop_max'] = 200
    config_dict['gen_max'] = 18
    config_dict['gp_tourn_size'] = 3
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def run(root_dir):
    config_dict = create_config_dict()
    # gp = mountaincar_v4_scratch(config_dict, root_dir)
    gp = mountaincar_test(config_dict, root_dir)
    # gp = mountaincar_v1(config_dict, root_dir)
    gp.plagih_gp_run()
