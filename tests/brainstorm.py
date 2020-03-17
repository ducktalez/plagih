dct = {1: {'a': 10, 'b': 4},
       3: {'a': 30, 'b': 3},
       2: {'a': 40, 'b': 2},
       4: {'a': 20, 'b': 1}}

import json
from pathlib import Path

config = {
            'root_dir': None,  # TODO
            'mode': 'run',  # ['run', 'analyze']

            # (!) Relevant for result
            'pop_max': 1000,  # Maximum amount of trees in a population. Only used evolve rates, condition is never tested.
            'parsimony_max': 100,  # right value is the maximum parsimony. left value not used, but was meant to set parsimony for the first generations. [3 to 2^(bas +1) - 1]
            'kernel_name': 'regression bounded',  # [regression, regression bounded, classification, match]
            'complexity_measure': 'ted',

            # rather irrelevant
            'parsimony_tmp': 15,
            'precision': 3,  # rounding the fitness
            'float_accuracy': 200,
            'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
            'print_type': 'gggewwsiivoaa',  # To print_type absolutely all: ewggggsiiiivvvtopppttt
            'overwrite periodic gp_files': True,  # If True, the file gets overwritten. If False, in every generation a new file is created.
            'force_new_run': False,  # especially for testing. Instead of deleting the old folder each time, you can set this to False to init a new run again #
            'delete_old_file': False,  # sfeh, delete old gp_files. be very careful
            'monitor': {'gen_fitness_average': 'y',
                        'sympify_errors': 'y',
                        'population_tmp_done-size': 'y'
                        },
            'period': {'time_monitor': None,  # in sec
                       'time_save': None,  # in sec
                       'gen_monitor': 1,  # in gen counts
                       'gen_save': 1},  # in gen counts

            # GP-evolve specific parameters
            'crossover_type_safety_mode': 'replace_same_types',
            'gen_num_max_parsimony': 50,  # Increase tmp_parsim to this generation
            'tree_growth': 'node-based',  # node-based, depth-based
            'tree_depth_base': 7,  # [3..10]
            'tree_depth_max': 25,  # maximum Tree depth for entire run
            'tree_depth_min': 5,
            'tree from scratch: min_nodes': 3,
            'tree from scratch: max_nodes': 50,
            'tree branch: base nodes': 20,
            'tourn_size': 4,  # [7 per 100] number of trees selected for tournament
            'evolve_rates': {'repro one': 0.03,
                             'repro pareto': 0.04,
                             'repro reduced one': 0.03,
                             'filter floats': 0.05,
                             'point mutate function': 0.1,
                             'branch mutate insert': 0.10,
                             'crossover branches': 0.40,
                             'random from origin_tree': 0.15,
                             'random from scratch': 0.15,
                             },

            # When to stop the run
            'time_max': None,  # int(60 * 60 * 12),  # 60 = 1 min
            'gen_max': 800,  # Maximum amount of generations
        }

with open('config_all.json', 'w') as f:
    json.dump(config, f, indent = 4)
