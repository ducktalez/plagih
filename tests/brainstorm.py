import yaml
from pathlib import Path

tree_components = {

    'csv samples': {

        'cartPos': {

            'use as': 'constant',
            'type': 'float',
            'label as': 'cartPos',
            'z (for future)': {
                'insert min': None,
                'insert max': 5,
                'value box/shape': [-2.4, 2.4],
                'time delta': None,
                'time delta 0 name': None,
                'description': 'The current position of the cart'}
        },
        'cartVel': {
            'type': 'float',
        },
        'poleAngle': {
            'type': 'float',
        },
        'poleVel': {
            'type': 'float',
        },
        'someOtherValue': {
            'use': False
        },
        'action0': {
            'use': 'result',
            'type': 'float',
        },

    },
    'number of observations': None,
    'number of actions': 1,
}

config = {
    # 'root_dir': root_dir,  # TODO
    'run': {
        'mode': 'gp-run',  # ['run', 'analyze']
        'kernel_name': 'regression bounded',  # [regression, regression bounded, classification, match]
        'time_max': None,  # int(60 * 60 * 12),  # 60 = 1 min
        'gen_max': 800,  # Maximum amount of generations
        'print_type': 'gggwwsivoaa',  # To print_type absolutely all: wggggsiiiivvvtopppttt
    },

    'evolve': {
        'pop_max': 1000,  # Maximum amount of trees in a population. Only used evolve rates, condition is never tested.
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
        'tree_growth': 'node-based',  # node-based, depth-based
        'tree_depth_base': 7,
        'tree_depth_min': 5,
        'random from scratch': {
            'rate': 0.15,
            'random from scratch min_nodes': 8,
            'random from scratch max nodes': 50,
        },
        'tree branch base nodes': 20,
        'crossover_type_safety_mode': 'replace_same_types',
        'choose': {'operators': ['random', 'from distribution'],
                   'terminals': ['random'],
                   'constants': ['gauss', 'observation samples'],
                   'observations': ['random'],
                   }
    },

    'tree': {
        'parsimony_max': 100,  # right value is the maximum parsimony. left value not used, but was meant to set parsimony for the first generations. [3 to 2^(bas +1) - 1]
        'depth_max': 20,  # maximum Tree depth for entire run
        'complexity_measure': 'ted',
        'choose': {
            'choose rules': ['first', 'choose', 'random'],
            'operators': ['random', 'from distribution'],
            'constants': ['gauss', 'random', 'observation samples'],
            'default': 'random'
        },
    },

    'rather irrelevant': {
        'description': 'No description set',
        'parsimony_tmp': 15,
        'precision': 3,  # rounding the fitness
        'float_accuracy': 200,
        'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
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

        'gen_num_max_parsimony': 50,  # Increase tmp_parsim to this generation
    }
}

print(yaml.dump(tree_components))

with Path.open(Path.cwd() / 'pltest.yaml', 'w') as file:
    yaml.dump(tree_components, file)
