import pickle
from plagih.modules.plagih_tree import *
from plagih.modules.printing import *
import csv
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
import sklearn.model_selection as skcv
import numpy as np
import yaml

example_runs = 'examples/'

folder_save = 'save/'
folder_plots = 'plots/'
folder_info = 'info/'
folder_steps = 'steps/'
folder_trees = 'trees/'
folder_pop_analysis = 'pop_dist/'

file_pareto = 'pareto.txt'
complete_file_pareto = 'pareto.txt'
file_config = 'config.csv'
file_backup_pickle = folder_info + 'backup.p'  # backup-version is set here
file_conclusion = 'conclusion.txt'
file_pycode = 'agents.py'

run_files = 'run_files'
config_yaml = 'config.yaml'
samples_ready = 'samples_ready.p'
samples_csv = 'samples.csv'
operators = 'operators.csv'
tree_expr_txt = 'tree_expr.txt'
tree_labels_csv = 'tree_labels.csv'
tree_numpy_csv = 'tree_numpy.csv'

T_num_lines = 15  # todo this var is not found otherwise


def make_dir(path):
    """
    Checks if the folders for the specified path exist and creates them otherwise.
    Apparently, this procedure is used often.
    """
    if not Path.is_dir(path):
        Path.mkdir(path)
    return path


def get_path(gen_id='tmp'):
    """
    ! Only used for population plots right now
    Returns the path where a file is located
    get_path('config') -> *root_dir*/info/config.csv
    """
    path = 'plots/pop_dist/fitness_{}.jpg'


def data_load_pickle(data_prepared_pickle_path):
    """
    Loads a data_csv_path file that was already split with the csv reader
    """
    with Path.open(data_prepared_pickle_path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data  # input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


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
    return


def data_load_data_split(data_x, data_y, test_size):
    x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=test_size)  # 80/20 TRAIN/TEST
    data_train = np.c_[x_train, y_train]  # recombine each row of data_csv_path with its class label (right column)
    data_control = np.c_[x_test, y_test]  # recombine each row of data_csv_path with its class label (right column)

    data_train_rows = len(data_train[:, 0])

    return data_train_rows, data_train, data_control


def file_population_karoo(population, pop_name, path, gen_id):
    """
    Save population_* to disk.

    """

    pop_path = make_dir(path / folder_info)

    file_path = pop_path / 'population_{}.csv'.format(str(pop_name))
    # todo function to tree_ and append each tree
    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for ii, tree in enumerate(population):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[ii][row]])

    return
