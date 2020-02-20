from pathlib import Path
import pickle
from plagih.modules.dicts import *
from plagih.modules.printing import *
import csv
import matplotlib.pyplot as plt
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
import sklearn.model_selection as skcv
import numpy as np


folder_runs = 'runs/'

folder_save = 'save/'
folder_plots = 'plots/'
folder_info = 'info/'
folder_steps = 'steps/'

file_pareto = 'pareto.txt'
file_config = 'config.txt'
file_backup_pickle = folder_info + 'backup.p'  # backup-version is set here
file_conclusion = 'conclusion.txt'


def data_load_pickle(prepared_data_pickle_path):
    """
    loads a data_csv_path file that was already split with the csv reader
    """
    with Path.open(prepared_data_pickle_path, 'rb') as file:
        pickle_data = pickle.load(file)

    # self.printplg('g', 'Pickle-loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
    return pickle_data  # input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


def save_data_pickle(prepared_data, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        pickle.dump(prepared_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return


def file_population_write_plagih(population, pop_name, path, gen_id):
    file_path = path / 'population_plagih_{}.csv'.format(str(pop_name))

    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a (-> file too large)
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for tree in range(1, len(population)):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[tree][row]])

    return


def file_population_write_karoo(population, pop_name, path, gen_id):
    """
    Save population_* to disk.

    """
    file_path = path / 'population_{}.csv'.format(str(pop_name))

    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. (-> file too large)
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for tree in range(1, len(population)):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[tree][row]])

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


def load_pop_from_csv(pop_csv):
    """
    This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_r
    replaces population_a in the karoo_gp/runs/[date-time]/ directory.
    """

    with Path.open(pop_csv, 'r') as csv_file:
        target = csv.reader(csv_file, delimiter=',')
        n = 0  # track row count

        for row in target:

            n = n + 1
            if n == 1:
                pass  # skip first empty row

            elif n == 2:
                population_a = [row]  # write header to population_a

            else:
                if not row:
                    tree = np.array([[]])  # initialise Tree array

                else:
                    if tree.shape[1] == 0:
                        tree = np.append(tree, [row], axis=1)  # append first row to Tree

                    else:
                        tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree

                if tree.shape[0] == T_num_lines:
                    population_a.append(tree)  # append complete Tree to population list

    return population_a


def plot_end(data_2d, path, plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='', yscale='linear', only_dots=None):

    x, y = [], []
    for a, b in data_2d:
        x.append(a)
        y.append(b)

    if only_dots:
        plt.plot(x, y, linestyle='', marker='o', label=plt_curve_label)
    else:
        plt.plot(x, y, label=plt_curve_label)

    if plt_x_label and plt_y_label:
        plt.xlabel(plt_x_label)
        plt.ylabel(plt_y_label)

    if plt_title:
        plt.title(plt_title)

    # plt.legend()
    plt.yscale(yscale)
    plt.ylim(0)
    plt.xlim(0)
    path_plot = path / 'plots'
    if not Path.is_dir(path_plot):
        Path.mkdir(path_plot)
    plt.savefig(path_plot / '{}-plot.jpg'.format(plt_title))
    plt.close()
    return
