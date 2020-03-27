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


def samples_header_line(row):
    observations_bundle = {'info': {}, 'bool': [], 'float': [], 'all': {}}  # to identify all observation types # sfeh remove 'all'
    param_at = {}
    actions = {}  # action: type, xtype,

    for ii, header in enumerate(row):
        header_split = header.split('|')  # split 1: cartVel|type=float|role=input -> {cartVel, type=float, role=input]
        name = header_split[0]
        observations_bundle[name] = {'type': 'float', 'role': None, 'pos': ii}  # sfeh we assume it is float
        param_at[ii] = {'name': name, 'type': 'float', 'role': None, 'pos': ii}  # sfeh we assume it is float
        try:
            for col_param in header_split[1:]:
                param, value = col_param.split('=')
                observations_bundle[name][param] = value
        except:
            print('Could not load samples csv correctly...')

        # try to guess if it is observation or action
        role = observations_bundle[name]['role']
        type = observations_bundle[name]['type']
        xtype = '2b' if 'bool' in type else '2f'

        param_at[ii] = {'name': name, 'type': type, 'xtype': xtype, 'label': name}

        if any(x in role for x in ['input', 'observation', 'obs']):
            observations_bundle['info'][name] = {'type': type, 'xtype': xtype, 'custom label': None}
            observations_bundle[type].append(name)
        elif any(x in role for x in ['result', 'output', 'out', 'action']):
            actions[name] = {'type': type, 'xtype': xtype, 'label': name, 'pos': ii}
        else:
            if ii < len(row) - 1:
                observations_bundle[name]['role'] = 'input'
            else:
                observations_bundle[name]['role'] = 'action'

        # sfeh solve this later
        observations_bundle['all'] = param_at

    return observations_bundle, actions, param_at


def data_from_csv(samples_file, samples_info=None, test_size=0.2):
    """
    loads the goal-data_csv_path from .csv file. first observations_bundle then actions.
    Both can have any shape specified in the gym.env "spaces" (dimensions: 1-n, type: int-floatstring?)

    Mountaincar .csv first lines (11.12.2019):
    --------------------------------------------------------
    observation0:float,     observation1:float, action0:float
    -0.5031261704876531,    0.0,                2
    --------------------------------------------------------
    """

    data_obs, data_results = [], []
    # 1. Read file
    with Path.open(samples_file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        for i, row in enumerate(reader):
            if i == 0:  # variable identifiers
                # all_variables = [x.rsplit(':', 1)[0] for x in row]  # ['observation0:float'] -> ['observation0']
                observations_bundle, actions, param_at = samples_header_line(row)

                if len(actions) > 1:
                    print_e('More than one result is not yet supported!')
                    raise

            else:  # convert every 'string' element to its data_csv_path type

                types = [param_at[x]['type'] for x in range(len(row))]
                row_as_data = [locate(types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123

                num_observations = min(len(observations_bundle['all']), len(row) - 1)

                data_obs.append(row_as_data[:num_observations])
                data_results.append(row_as_data[num_observations:])

    print('DFGFDSFSDFDS')

    unique_outputs_num = len(np.unique(data_results))  # load the user defined true labels for classification or solutions for regression

    data_train_rows, data_train, data_control = data_load_data_split(data_obs, data_results, test_size=test_size)
    # self.printplg('g', 'Loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

    action_min_max = (min(data_results), max(data_results))

    return observations_bundle, actions, param_at, unique_outputs_num, data_train_rows, data_train, data_control, action_min_max


def data_load_pickle(data_prepared_pickle_path):
    """
    loads a data_csv_path file that was already split with the csv reader
    """
    with Path.open(data_prepared_pickle_path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data


def data_save_pickle(data_prepared, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        pickle.dump(data_prepared, file, protocol=pickle.HIGHEST_PROTOCOL)
    return


def data_save_yaml(data_prepared, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        yaml.dump(data_prepared, file, protocol=pickle.HIGHEST_PROTOCOL)
    return
