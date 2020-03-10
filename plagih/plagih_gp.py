import sys
sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.Examples import *
from plagih.modules.plagih_data import *

# todo clean up this class... make extra class or folder (!) with all cases to be tested

# import warnings
# warnings.filterwarnings('error')

def create_config_dict():
    config_dict = {
        'root_dir': Path.cwd(),
        'name': 'Plagih_name_dummy',  # please set a name
        'mode': 'run',  # ['run', 'analyze']

        # (!) Relevant for result
        'pop_max': 1000,  # Maximum amount of trees in a population. Only used evolve rates, condition is never tested.
        'parsimony_max': 100,  # right value is the maximum parsimony. left value not used, but was meant to set parsimony for the first generations. [3 to 2^(bas +1) - 1]
        'kernel_name': 'regression bounded',  # [regression, regression bounded, classification, match]
        'complexity_measure': 'ted',

        # rather irrelevant
        'parsimony_tmp': 15,
        'precision': 3,                 # rounding the fitness
        'float_accuracy': 200,
        'swim': 'p',                    # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        'print_type': 'gggewwsiivoaa',    # To print_type absolutely all: ewggggsiiiivvvtopppttt
        'overwrite periodic files': True,  # If True, the file gets overwritten. If False, in every generation a new file is created.
        'force_new_run': False,         # especially for testing. Instead of deleting the old folder each time, you can set this to False to init a new run again #
        'delete_old_file': False,       # sfeh, delete old files. be very careful
        'monitor': {'gen_fitness_average': 'y',
                    'sympify_errors': 'y',
                    'population_tmp_done-size': 'y'
                    },
        'period': {'time_monitor': None,    # in sec
                   'time_save': None,       # in sec
                   'gen_monitor': 1,        # in gen counts
                   'gen_save': 1},          # in gen counts

        # GP-evolve specific parameters
        'crossover_type_safety_mode': 'replace_same_types',
        'gen_num_max_parsimony': 50,    # Increase tmp_parsim to this generation
        'tree_growth': 'v2',
        'tree_depth_base': 7,
        'tree_depth_max': 50,           # maximum Tree depth for entire run
        'tree_depth_min': 5,
        'tree_branch_nodes_base': 32,
        'tourn_size': 4,                # [7 per 100]		number of trees selected for tournament
        'evolve_rates': {'repro one': 0.03, 'repro pareto': 0.04, 'repro reduced one': 0.03,
                         'filter floats': 0.05,
                         'point mutate function': 0.1,
                         'branch mutate insert': 0.10,
                         'crossover branches': 0.40,
                         'random from origin_meta': 0.25, 'random from scratch': 0,
                         },

        # When to stop the run
        'time_max': int(60 * 60 * 12),  # 60 = 1 min
        'gen_max': 800,  # Maximum amount of generations
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
    gp.activate_dataset(prepared_data)
    op_array = load_funcarray_from_csv(path / MountainCarExamples.files['operators_file'])
    gp.activate_operators(op_array)
    return gp


def run_mountaincar_v1(config_dict, path):
    description = 'Mountaincar from 2-decision Origin (simple)'
    config_dict['name'] = 'MTC_v1'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v1_list, modify_list=MountainCarExamples.tree_v1_modify)
    return gp


def run_mountaincar_v2(config_dict, path):
    config_dict['name'] = 'MTC_v2'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def run_mountaincar_v3(config_dict, path):
    config_dict['name'] = 'MTC_v3'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    return gp


def run_mountaincar_v4(config_dict, path):
    config_dict['name'] = 'MTC_v4_scratch'
    config_dict['root_dir'] = path
    config_dict['kernel_name'] = 'regression bounded'
    config_dict['evolve_rates']['random from scratch'] += config_dict['evolve_rates']['random from origin_meta']
    config_dict['evolve_rates']['random from origin_meta'] = 0
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v1_list)
    return gp


def mountaincar_update_analysis_files(config_dict, path):
    all_runs = ['MTC_v1', 'MTC_v2', 'MTC_v3', 'MTC_v4']

    config_dict['root_dir'] = path
    config_dict['mode'] = 'analyse'

    config_dict['name'] = 'MTC_v4_scratch'
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.plagih_update_analysis()
    return


def run_mountaincar_test(config_dict, path):
    config_dict['name'] = 'MTC_test'
    test_run = Path.cwd() / folder_runs / config_dict['name']
    print('Test dir is:', test_run)
    config_dict['root_dir'] = path
    config_dict['pop_max'] = 100
    config_dict['gen_max'] = 500
    config_dict['tourn_size'] = 3
    config_dict['kernel_name'] = 'regression bounded'
    config_dict['parsimony_max'] = 50
    config_dict['print_type'] = 'ewaaaggggsiiiivvvtopppttt'    # To print_type absolutely all: ewaaaiiiiggggvvvpppttt
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.load_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def analyse_old_run(root_dir):
    config_dict = create_config_dict()
    mountaincar_update_analysis_files(config_dict, root_dir)


def run(root_dir):
    # create_samples_pickle(root_dir)  # todo outsource
    # analyse_old_run(root_dir)
    config_dict = create_config_dict()
    gp = run_mountaincar_v4(config_dict, root_dir)
    gp.plagih_gp_run()
