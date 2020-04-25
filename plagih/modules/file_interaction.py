import pickle
from plagih.modules.plagih_tree import *
from plagih.modules.printing import *
import csv
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
import sklearn.model_selection as skcv
import numpy as np
import yaml

example_runs = 'run_examples/'

run_files = 'run_files/'
folder_plots = 'plots/'
folder_steps = 'steps/'
folder_pop_analysis = 'pop_dist/'

file_pareto = 'info/pareto.txt'
info_config_yaml = 'info/config.yaml'
file_info_config_json = 'info/config.json'
file_backup_pickle = 'backup/backup.p'  # backup-version is set here
file_conclusion = 'conclusion.txt'

file_config_yaml = 'run_files/config.yaml'
file_info_evolve_dict_yaml = 'info/evolve_list.yaml'
file_config_json = 'run_files/config.json'
samples_ready_p = 'run_files/samples_ready.p'
file_evolve_functions = 'run_files/evolve_functions.yaml'
env_variables_yaml = 'run_files/env_variables.yaml'
samples_csv = 'run_files/samples.csv'
operators = 'run_files/operators.csv'
distributions_file = 'run_files/distributions_file.yaml'
tree_expr_txt = 'run_files/tree_expr.txt'
tree_labels_csv = 'run_files/tree_labels.csv'
tree_numpy_csv = 'run_files/tree_numpy.csv'

callable_user_python_script = 'run_files/custom_agent_eval.py'  # sfeh make pretty solution

folder_solutions = 'agents/'
trees_tex = 'agents_trees.tex'
file_pycode = 'agents.py'
file_pycode_eval = 'eval_agents.py'

T_num_lines = 15  # sfeh this var is not found otherwise


def make_dir(path):
    """
    Checks if the folders for the specified path exist and creates them otherwise.
    Apparently, this procedure is used often.
    """
    if not Path.is_dir(path):
        Path.mkdir(path)
    return path


def file_make_dir(file_path):
    """
    Creates the folder only knowing the file.
    paff/tuuu/fyle.txe  ->  *mkdir* paff/tuuu/
    """
    p = Path(file_path)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)
    return p


def write_file_pareto_text(pareto, root_path):
    """
    Save all the pareto efficient candidates to file
    """

    pth = file_make_dir(root_path / file_pareto)

    with Path.open(pth, 'w') as file:
        for parsim, meta in sorted(list(pareto.items())):
            fitness = meta['fitness_train']
            algo_sym = meta['expr_sym']  # save raw version, not the sympified one
            file.write('\nParsimony: \t{0} Fitness: \t{1} Expr: \t{2}'.format(parsim, fitness, algo_sym))
    printez('f', '{}'.format(file_pareto))


def open_force_write_text(p, text):
    p = Path(p)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)
    p.write_text(text)


def experiment_data(experiment_yaml):
    if Path.is_file(experiment_yaml):  # Load config.yaml
        with Path.open(experiment_yaml, 'r') as file:
            experiment_infos = yaml.load(file, Loader=yaml.FullLoader)

    wer = {
        'env': {
            'observations': {
                'cartPos': {
                    'use as': 'constant',
                    'type': 'float',
                    'custom label': 'cartPos',
                    'insert min': None,
                    'insert max': None,
                    'value box/shape': None,
                    'time delta': None,
                    'time delta 0 name': None,
                    'description': None
                },
                'cartVel': {
                    'type': 'float',
                },
                'poleAngle': {
                    'type': 'float',
                },
                'poleVel': {'type':
                                'float',
                            },
            },
            'action0': {
                'use': 'result',
                'type': 'float',
            },

            'number of observations': None,
            'number of actions': 1,
        }
    }


def save_data_pickle(data_prepared, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        pickle.dump(data_prepared, file, protocol=pickle.HIGHEST_PROTOCOL)  # not sure if the protocol matters

    printez('f', '{}'.format(data_pickle_path))

    return


def write_file_population_karoo(population, pop_name, path, gen_id):
    """
    Save population_* to disk.

    """
    file_name = 'population_{}.csv'.format(str(pop_name))
    file_path = path / 'info/' / file_name
    file_path = file_make_dir(file_path)
    # sfeh? function to tree_ and append each tree
    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for ii, tree in enumerate(population):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[ii][row]])

    printez('f', '{}'.format(file_name))

    return


def pickle_load(path):
    """
    loads a data_csv_path file that was already split with the csv reader
    """

    with Path.open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data


def pickle_dump(path, data):
    """
    saves prepared plagih data to pickle file
    """

    path = file_make_dir(path)
    with Path.open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return


def yaml_load(yaml_path):

    with Path.open(yaml_path, 'r') as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


def yaml_dump(path, data):
    """
    saves prepared plagih data to pickle file
    """

    path = file_make_dir(path)
    with Path.open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    return
