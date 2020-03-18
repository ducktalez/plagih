import pickle
from plagih.modules.plagih_tree import *
from plagih.modules.printing import *
import csv
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
import sklearn.model_selection as skcv
import numpy as np


example_runs = 'run_examples/'

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

run_files = 'run_files'
config_json = 'config.json'
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


def data_load_pickle(prepared_data_pickle_path):
    """
    Loads a data_csv_path file that was already split with the csv reader
    """
    with Path.open(prepared_data_pickle_path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data  # input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


def save_data_pickle(prepared_data, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        pickle.dump(prepared_data, file, protocol=pickle.HIGHEST_PROTOCOL)  # not sure if the protocol matters
    return


def load_operators_from_csv(op_csv_path):
    """
    Load all operators ready-to-use from a file
    """

    functions = np.loadtxt(op_csv_path, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
    # Part 3.5: Split the functions in 5 types

    # rows are the function types (f2f)
    # columns are the arity
    op_array = [[[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []]]
    for fun in functions:
        label = fun[0]
        arity = op[label]['arity']  # arity = int(fun[1])
        xtype = op[label]['xtype']

        if xtype == 'f2f':
            op_array[f2f][arity].append(label)
        elif xtype == 'f2b':
            op_array[f2b][arity].append(label)
        elif xtype == 'b2b':
            op_array[b2b][arity].append(label)
        elif xtype == 'b2f':
            op_array[b2f][arity].append(label)
        elif xtype == 'b2f2f':
            op_array[b2f2f][arity].append(label)

    return op_array


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


def data_from_csv(samples_file):
    """
    loads the goal-data_csv_path from .csv file. first observations then actions.
    Both can have any shape specified in the gym.env "spaces" (dimensions: 1-n, type: int-floatstring?)

    Mountaincar .csv first lines (11.12.2019):
    --------------------------------------------------------
    observation0:float,     observation1:float, action0:float
    -0.5031261704876531,    0.0,                2
    --------------------------------------------------------
    """

    num_observations, num_actions = 0, 0
    var_types = []
    input_dict = {'all': {},
                  'float': {},
                  'bool': {}}
    variables_dict = {'all': [],
                      'types': [],
                      'float': [],
                      'bool': []}

    action_dict = {}

    # 1. Read file
    with Path.open(samples_file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        for i, row in enumerate(reader):
            if i == 0:  # variable identifiers
                # all_variables = [x.rsplit(':', 1)[0] for x in row]  # ['observation0:float'] -> ['observation0']
                for var_name in row:
                    var_types.append(var_name.split(':', 1)[1])
                    if var_name.startswith('o'):  # found an observation
                        num_observations += 1
                        term = var_name.rsplit(':', 1)[0]
                        term_type = var_name.split(':', 1)[1]
                        input_dict[term] = term_type
                        variables_dict['all'].append(term)
                        variables_dict['types'].append(term_type)
                        if term_type == 'float':
                            variables_dict['float'].append(term)
                            variables_dict['float'].append(term)
                        elif term_type == 'bool':
                            variables_dict['bool'].append(term)
                        else:
                            raise
                    elif var_name.startswith('a'):  # found an action
                        num_actions += 1
                        action = var_name.split(':', 1)[0]
                        action_type = var_name.split(':', 1)[1]
                        action_dict[action] = action_type  # Do not use this:# '2b' if 'bool' in action_type else '2f'
                    else:
                        print_e('Behaviour samples first line: Variables have to start with "o" or "a" to be recognized. Is actually: {}'.format(var_name))
                        raise

                data_x, data_y = [], []

            else:  # convert every 'string' element to its data_csv_path type
                row_as_data = [locate(var_types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123
                data_x.append(row_as_data[:num_observations])
                data_y.append(row_as_data[num_observations:])
        csvFile.close()
    unique_outputs_num = len(np.unique(data_y))  # load the user defined true labels for classification or solutions for regression

    data_train_rows, data_train, data_control = data_load_data_split(data_x, data_y, test_size=0.2)

    # self.printplg('g', 'Loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
    return input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


