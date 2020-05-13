import pickle
import yaml
from pathlib import Path
import sklearn.model_selection as skcv
import numpy as np
import csv
from plagih.modules.printing import *
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
from plagih.modules.operators import *


def samples_header_line(row):
    """
    samples.csv headerline:
    cartVel|type=float|role=input|minmax=(-0.07, 0.07)  # todo better solution please
    goal_action: (todo) only one action can be the result. vectors should be implemented someday

    env_variables = {'obs_name': {}, '2b': [], '2f': [], 'action_at': {}, cartPos: {}, cartVel: {}}
    env_variables['obs_name'} = {'type': 'float', 'role': None, 'pos': ii}
    env_variables['action_at'} = {'name': name, 'type': type, 'xtype': xtype, 'label': name, 'pos': ii}
    """
    env_variables = {'obs_name': {}, '2b': [], '2f': [], 'action_at': {}}  # to identify all observation types
    param_at = {}

    for ii, header in enumerate(row):
        header_split = header.split('|')  # split 1: cartVel|type=float|role=input -> {cartVel, type=float, role=input]
        name = header_split[0]
        env_variables[name] = {'type': 'float', 'role': None, 'pos': ii}  # if no type is specified -> float
        param_at[ii] = {'name': name, 'type': 'float', 'role': None, 'pos': ii}
        try:
            for col_param in header_split[1:]:
                param, value = col_param.split('=')
                env_variables[name][param] = value
        except:
            print('Could not load samples csv correctly')

        # todo fill obs_name or diretly in names??

        col_type = env_variables[name].get('type')
        if col_type is None:
            col_type = 'float'  # it probably is float anyways
        else:
            col_type = col_type.replace('int', 'float')  # sfeh: int is currently not really used
            col_type = col_type.replace('num', 'float')  # sfeh: num is currently not really used

        xtype = '2b' if 'bool' in col_type else '2f'

        minmax = env_variables[name].get('minmax')
        if minmax is None:
            printez('w', 'mimax value tuple not provided. Trying to use the min-max values from the samples later.')
        else:
            minmax = [float(x) for x in minmax.split(',')]
            env_variables['obs_name']['minmax'] = minmax

        param_at[ii] = {'name': name, 'type': col_type, 'xtype': xtype, 'label': name}

        role = env_variables[name].get('role')
        if role is None:
            if ii < len(row) - 1:
                env_variables[name]['role'] = 'input'
            else:
                env_variables[name]['role'] = 'action'
        else:
            if any(x in role for x in ['input', 'observation', 'obs']):
                env_variables['obs_name'][name] = {'type': col_type, 'xtype': xtype, 'label': name, 'pos': ii}
                env_variables[xtype].append(name)
            elif any(x in role for x in ['result', 'output', 'out', 'action']):
                env_variables['action_at'][0] = {'name': name, 'type': col_type, 'xtype': xtype, 'label': name, 'pos': ii}
            else:
                print_warning('w', 'role ignored: {}'.format(role))

    if len(env_variables['action_at']) > 1:
        print_e('More than one result is not yet supported! Is done in next milestone though.')

    return env_variables, param_at


def data_from_csv(samples_file, test_size=0.2):
    """
    loads the goal-data_csv_path from .csv file. first env_var_dummy then actions.
    Both can have any shape specified in the gym.env "spaces" (dimensions: 1-n, type: int-floatstring?)

    Mountaincar .csv first lines (11.12.2019):
    --------------------------------------------------------
    observation0:float,     observation1:float, action0:float
    -0.5031261704876531,    0.0,                2
    --------------------------------------------------------
    """

    data_obs, data_results = [], []
    with Path.open(samples_file) as file:
        reader = csv.reader(file, delimiter=',')

        for i, row in enumerate(reader):
            if i == 0:
                # env_variables
                env_variables, param_at = samples_header_line(row)
                num_observations = min(len(env_variables['obs_name']), len(row) - 1)
            else:  # convert every 'string' element to its data_csv_path type

                types = [param_at[x]['type'] for x in range(len(row))]  # sfeh allow int again
                row_as_data = [locate(types[i])(x) for i, x in enumerate(row)]  # ['varA:float'] + ['0.123'] --> float(['0.123']) --> 0.123

                data_obs.append(row_as_data[:num_observations])
                data_results.append(row_as_data[num_observations:])

    unique_outputs_num = len(np.unique(data_results))  # load the user defined true labels for classification or solutions for regression
    action_min_max = min(data_results)[0], max(data_results)[0]  # sfeh: better to get this from the environment!
    env_variables['action_at'][0]['unique_outputs_num'] = unique_outputs_num
    env_variables['action_at'][0]['minmax'] = action_min_max

    x_train, x_test, y_train, y_test = skcv.train_test_split(data_obs, data_results, test_size=test_size)  # 80/20 TRAIN/TEST split
    data_train = np.c_[x_train, y_train]  # recombine each row of data_csv_path with its associated class label (right column)
    data_control = np.c_[x_test, y_test]  # recombine each row of data_csv_path with its associated class label (right column)

    data_prepared = env_variables, data_train, data_control

    return data_prepared
