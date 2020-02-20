import sys
sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.Examples import *
from plagih.modules.plagih_data import *


def create_config_dict():
    config_dict = {
        'root_dir': Path.cwd(),
        'name': 'Plagih_tree',

        'kernel_name': 'regression',  # [regression, regression bounded, classification, match]
        'precision': 3,  # rounding the fitness
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'crossover_type_safety_mode': 'replace_same_types',
        'print_type': 'ggewwsiivoa',  # To print_type absolutely all: ewggggsiiiivvvtopppttt
        'overwrite periodic files': True,  # If True, the file gets overwritten. If False, in every generation a new file is created.
        'gene_pool_threshold': 0.5,  # this amount of percent a tree needs to fulfill to be in the gene pool
        'tree_growth': 'depth_base_random',
        'tree_depth_base': 7,
        'tree_depth_max': 50,  # [3...10]			maximum Tree depth for entire run
        'tree_depth_min': 5,
        'parsimony_tmp': 15,
        'gen_id_max_parsimony': 50,
        'parsimony_max': 80,  # right value is the maximum parsimony. left value not used, but was meant to set parsimony for the first generations. [3 to 2^(bas +1) - 1]
        'pop_max': 1000,  # Maximum amount of trees in a population. Only used evolve rates, condition is never tested.
        'gen_max': 1000,  # Maximum amount of generations
        'complexity_measure': 'ted',
        'force_new_run': False,  # especially for testing. Instead of deleting the old folder each time, you can set this to False to init a new run again # todo delete old files?
        'tourn_size': 7,  # [7 per 100]		number of trees selected for tournament
        'monitor': {'verbosity': 'end',  # every [generation] or at the [end]
                    'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'genepool_size': 'y'
                    },
        'period': {'time_monitor': None,    # in sec
                   'time_save': None,       # in sec
                   'gen_monitor': 1,        # in gen counts
                   'gen_save': 1},          # in gen counts
        'evolve_rates': {'repro one': 0.05, 'repro pareto': 0.05, 'repro reduced one': 0.05,
                         'filter floats': 0.05,
                         'point mutate function': 0.05,
                         'branch mutate insert': 0.1,
                         'crossover branches': 0.40,
                         'random from origin': 0.25, 'random from scratch': 0,
                         },
        'time_max': int(60 * 60 * 12),  # 60 = 1 min
        'float_accuracy': 200
    }

    return config_dict


def fast_run(config):
    config['pop_max'] = 200
    config['gen_max'] = 18
    config['tourn_size'] = 3
    return config


def create_samples_pickle(path):
    """
    Saving your data file
    """
    prepared_data = data_from_csv(path / MountainCarExamples.files['samples_file'])
    data_save_pickle(prepared_data, path / MountainCarExamples.files['samples_pickle'])


def mountaincar_load_corefiles(config_dict, path):
    gp = plagih.ExplainableGP(config_dict)
    prepared_data = data_load_pickle(path / MountainCarExamples.files['samples_pickle'])
    gp.activate_data(prepared_data)
    op_array = load_funcarray_from_csv(path / MountainCarExamples.files['operators_file'])
    gp.activate_operators(op_array)
    return gp


def run_mountaincar_origin2_fix(config_dict, path):
    description = 'Mountaincar from 2-decision Origin (simple)'
    config_dict['name'] = 'MTC_v1'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v1_list, modify_list=MountainCarExamples.tree_v1_modify)
    return gp


def run_mountaincar_origin3_fix(config_dict, path):
    config_dict['name'] = 'MTC_v2'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def run_mountaincar_originbest_fix(config_dict, path):
    config_dict['name'] = 'MTC_v3'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    return gp


def run_mountaincar_scratch_fix(config_dict, path):
    config_dict['name'] = 'MTC_v4_scratch'
    config_dict['root_dir'] = path
    config_dict['gen_max'] = 15
    config_dict['kernel_name'] = 'regression bounded'
    config_dict['complexity_measure'] = 'total_count_nodes'
    config_dict['evolve_rates']['random from scratch'] += config_dict['evolve_rates']['random from origin']
    config_dict['evolve_rates']['random from origin'] = 0
    gp = mountaincar_load_corefiles(config_dict, path)
    return gp


def run_mountaincar_test(config_dict, path):
    config_dict['name'] = 'MTC_test'
    config_dict['root_dir'] = path
    config_dict['pop_max'] = 200
    config_dict['gen_max'] = 12
    config_dict['tourn_size'] = 3
    config_dict['kernel_name'] = 'regression'
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def run(root_dir):
    # create_samples_pickle(root_dir)
    # mtc_runs = [mountaincar_v1, mountaincar_v2, mountaincar_v3, mountaincar_v4_scratch]
    config_dict = create_config_dict()
    gp_test = run_mountaincar_test(config_dict, root_dir)
    gp_test.plagih_gp_run()
    # gp_1 = mountaincar_v4_scratch(config_dict, root_dir)
    # gp_2 = mountaincar_test(config_dict, root_dir)
    # gp_3 = mountaincar_v1(config_dict, root_dir)
    # gp_1.plagih_gp_run()
