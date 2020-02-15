import pickle
from pathlib import Path
import sklearn.model_selection as skcv
import numpy as np
import csv
from plagih.modules.printing import *
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
from plagih.modules.dicts import *


def data_load_data_split(data_x, data_y, test_size):
    x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=test_size)  # 80/20 TRAIN/TEST split
    data_train = np.c_[x_train, y_train]  # recombine each row of data_csv_path with its associated class label (right column)
    data_control = np.c_[x_test, y_test]  # recombine each row of data_csv_path with its associated class label (right column)

    data_train_rows = len(data_train[:, 0])

    return data_train_rows, data_train, data_control


def data_from_csv(samples_file, test_size=0.2):
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

    if not Path.is_file(samples_file):
        print_e('samples_file does not exist here.')

    # 1. Read file
    with Path.open(samples_file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        for i, row in enumerate(reader):
            if i == 0:  # variable identifiers
                # all_variables = [x.rsplit(':', 1)[0] for x in row]  # ['observation0:float'] -> ['observation0']
                for var_name in row:
                    var_types.append(var_name.split(':', 1)[1])
                    term = var_name.rsplit(':', 1)[0]
                    term_type = var_name.split(':', 1)[1]

                    if term_type != 'float' and term_type != 'bool':
                        raise Exception(str(term_type))

                    if var_name.startswith('o'):  # found an observation
                        num_observations += 1
                        input_dict[term] = term_type
                        variables_dict['all'].append(term)
                        variables_dict['types'].append(term_type)
                        variables_dict[term_type].append(term)

                    elif var_name.startswith('a'):  # found an action
                        num_actions += 1
                        action_dict[term] = term_type  # Do not use this:# '2b' if 'bool' in action_type else '2f'

                    else:
                        print_e('Behaviour samples first line: Variables have to start with "o" or "a" to be recognized. Is actually: {}'.format(var_name))
                        raise

                data_obs, data_act = [], []

            else:  # convert every 'string' element to its data_csv_path type
                row_as_data = [locate(var_types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123
                data_obs.append(row_as_data[:num_observations])
                data_act.append(row_as_data[num_observations:])
        csvFile.close()

    if num_actions > 1:
        print_e('More than one result is not yet supported!')
        raise

    unique_outputs_num = len(np.unique(data_act))  # load the user defined true labels for classification or solutions for regression
    action_min_max = [None, None]
    if action_dict[first_action] == 'float':
        data_act_one = [x[0] for x in data_act]
        action_min_max[0] = min(data_act_one)
        action_min_max[1] = max(data_act_one)

    data_train_rows, data_train, data_control = data_load_data_split(data_obs, data_act, test_size=test_size)

    return input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control, action_min_max


def data_load_pickle(prepared_data_pickle_path):
    """
    loads a data_csv_path file that was already split with the csv reader
    """
    with Path.open(prepared_data_pickle_path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data  # input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


def data_save_pickle(prepared_data, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        pickle.dump(prepared_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return
