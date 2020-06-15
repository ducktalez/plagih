import pickle
import yaml
from pathlib import Path
import sklearn.model_selection as skcv
import numpy as np
import csv
from plagih.modules.printing import *
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
from plagih.modules.operators import *
import pandas as pd
import re


def samples_header_line(row):
    """
    samples.csv headerline:
    cartVel|type=float|role=input|minmax=(-0.07, 0.07)
    goal_action: (todo) only one action can be the result. vectors should be implemented someday

    env_variables = {'obs_name': {}, '2b': [], '2f': [], 'action_at': {}, cartPos: {}, cartVel: {}}  # sfeh name vs label??
    env_variables['obs_name'] = {'type': 'float', 'role': None, 'pos': ii}
    env_variables['action_at'}[0] = {'name': name, 'type': type, 'xtype': xtype, 'label': name, 'pos': ii, 'unique_outputs_num': None}

    param_at[ii] = {'name': name, 'type': col_type, 'xtype': xtype, 'role': role}
    """
    env_variables = {'obs_name': {}, '2b': [], '2f': [], 'action_at': {}, 'param_at': {}}  # to identify all observation types
    env_observation = {}
    env_xtype_list = {'2b': [], '2f': []}  # for choosing random variables
    env_param_at = {}  #
    param_at = {}  #
    env_action_at = {}  #
    env_observation_family = {}
    env_param_lookup = {}
    column_data = {}

    for ii, header in enumerate(row):
        header_split = header.split('|')  # split 1: cartVel|type=float|role=input --> {cartVel, type=float, role=input]
        col_label = header_split[0]
        column_meta_values = {}
        column_meta_values[col_label] = {'type': 'float', 'role': None, 'pos': ii}  # if no type is specified -> float  # todo rename pos
        # param_at[ii] = {'name': name, 'type': 'float', 'role': None, 'pos': ii}
        try:
            for col_param in header_split[1:]:
                param, value = col_param.split('=')
                column_meta_values[param] = value
        except Exception as ex:
            print('Could not load samples csv correctly: {}'.format(ex))

        col_type = header_entry_get_type(column_meta_values)
        xtype = '2b' if 'bool' in col_type else '2f'
        minmax = header_entry_get_minmax(column_meta_values)
        role = header_entry_get_roleguess(env_observation, col_label, ii, row)

        all_meta_dict = {'name': col_label, 'type': col_type, 'xtype': xtype, 'label': col_label, 'pos': ii, 'role': role, 'minmax': minmax}

        if any(x in role for x in ['input', 'observation', 'obs']):
            temp_diff = header_entry_get_tempdiff(col_label, column_meta_values)
            core_label = envvariable_get_corelabel(col_label)
            try:
                env_observation_family[core_label].extend([col_label])  # todo insert in sorted list... also t = [1,2,5,10]?
            except (IndexError, KeyError):
                env_observation_family[core_label] = [col_label]

            all_meta_dict['temp_diff'] = temp_diff
            all_meta_dict['core_label'] = core_label
            # todo test
            env_observation[col_label] = all_meta_dict
            env_xtype_list[xtype].append(col_label)
        elif any(x in role for x in ['result', 'output', 'out', 'action']):
            action_count = len(env_action_at)
            all_meta_dict['unique_outputs_num'] = None  # sfeh add this so a later pycharm warning goes away?
            env_action_at[action_count] = all_meta_dict
        else:
            raise

        param_at[ii] = {'name': col_label, 'type': col_type, 'xtype': xtype, 'role': role}

    # last-chance use of env_variables. sfeh: use many dicts instead! hmmm
    env_variables['action_at'].update(env_action_at)
    env_variables['obs_name'].update(env_observation)
    env_variables.update(env_xtype_list)  # sfeh that is ugly code
    env_variables['env_observation_family'] = env_observation_family

    return env_variables, param_at


def header_entry_get_type(column_meta_values):
    """

    """
    col_type = column_meta_values.get('type')
    if col_type is None:
        col_type = 'float'  # it probably is float anyways
    else:
        col_type = col_type.replace('int', 'float')  # sfeh: int is currently not really used
        col_type = col_type.replace('num', 'float')  # sfeh: num is currently not really used
    return col_type


def header_entry_get_minmax(column_meta_values):
    """

    """
    minmax = column_meta_values.get('minmax')
    if minmax is None:
        printez('w', 'mimax value tuple not provided. Trying to use the min-max values from the samples later.')
    else:
        minmax = [float(x) for x in minmax.split(',')]
    return minmax


def header_entry_get_roleguess(env_observation, name, ii, row):
    """

    """
    role = env_observation.get('role')
    if role is None:
        if 'a_' in name[:2]:
            role = 'action'
        elif ii < len(row) - 1:
            role = 'input'
        else:
            role = 'action'
        print_warning('w', 'role not given. Role is interpreted as: {}'.format(role))
    return role


def envvariable_get_tempdiff(name, re_pattern='_\d+$'):
    """
    'cartVel_12' -> core_label = 'cartVel', temp_diff = 12
    can be used to enrich old runs 'manually', otherwise only used in header_entry_get_tempdiff
    """
    re_search = re.search(re_pattern, name)  # todo what is the best solution? '\_\d+$' is the correct regex using search.
    if re_search:
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
    else:
        temp_diff = 0  # there is only the latest version

    return int(temp_diff)


def envvariable_get_corelabel(name, re_pattern='_\d+$'):
    """

    """
    core_label = re.split(re_pattern, name)  #
    if core_label:
        core_label = core_label[0]
    else:
        core_label = name
    return core_label


def header_entry_get_tempdiff(name, column_meta_values, re_pattern='_\d+$'):
    """
    checking if an observation/variable is a past/historic value 'cartVel_10'
    """

    temp_diff = column_meta_values.get('temp_diff')

    if temp_diff is not None:
        temp_diff = int(temp_diff)  # convert time-difference to int-number
    else:
        temp_diff = envvariable_get_tempdiff(name, re_pattern=re_pattern)

    return temp_diff


def data_from_csv(samples_file, test_size=0.2, delimiter=','):
    """
    loads the data for the regression from a .csv-file.


    Mountaincar .csv first lines (11.12.2019):
    --------------------------------------------------------
    cartPos:float,     cartVel:float, action:float
    -0.5031261704876531,    0.0,                2
    --------------------------------------------------------

    sfeh: pandas seems prettier to read the data
    for now, i just used what was given, which is read line and the header is specifically analysed
    """

    dtype_dict = {'float': np.float32,  # fix float, dtype float32 etc
                  'int': np.int32,
                  'bool': np.bool}

    data_obs, data_results = [], []
    with Path.open(samples_file) as samples_csv_file:
        # reader = csv.reader(file, delimiter=delimiter)
        # header_row = next(reader)
        header_row = list(pd.read_csv(samples_csv_file, delimiter=delimiter, nrows=1))

        #######
        env_variables, param_at = samples_header_line(header_row)
        # env_observation = env_variables['obs_name']
        env_action_at = env_variables['action_at']
        #######

        dtypes_from_header = {}
        colnames_from_header = []

        # param_at[ii] = {'name': name, 'type': col_type, 'xtype': xtype, 'role': role}
        for ii, param_values in param_at.items():
            param_name = param_values['name']
            ptype = param_values['type']
            dtype = dtype_dict[ptype]
            colnames_from_header.append(param_name)
            dtypes_from_header[param_name] = dtype

        # for k, v in env_observation.items():
        #     dtype = dtype_dict[v['type']]
        #     dtypes_from_header[k] = dtype
        #
        # for action_name in env_action_at.values():
        #     name = action_name['name']
        #     dtype = dtype_dict[action_name['type']]
        #     dtypes_from_header[name] = dtype

        df = pd.read_csv(samples_csv_file, skiprows=1, names=colnames_from_header, dtype=dtypes_from_header)

        # # num_observations = min(len(env_variables['obs_name']), len(header_row) - 1)  # obsolete, is now observation_count
        # observation_columns = [x['pos'] for x in env_observation.values()]
        # action_columns = [x['pos'] for x in env_action_at.values()]

    #     for i, row in enumerate(data_nparray):  # all lines containing actual data (leaving out the header line)
    #         # if i == 0:
    #         #     # env_variables
    #         #     env_variables, param_at = samples_header_line(row)
    #         #     num_observations = min(len(env_variables['obs_name']), len(row) - 1)
    #         # else:  # convert every 'string' element to its data_csv_path type
    #
    #         types = [param_at[x]['type'] for x in range(len(row))]  # sfeh allow int again
    #         row_as_data = [locate(types[i])(x) for i, x in enumerate(row)]  # ['varA:float'] + ['0.123'] --> float(['0.123']) --> 0.123
    #         data_obs.append(row_as_data[observation_columns])
    #         data_results.append(row_as_data[action_columns])
    #
    # for x in range(len(data_results[0])):
    #
    #     unique_outputs_num = len(np.unique(data_results))  # load the user defined true labels for classification or solutions for regression
    #     action_min_max = min(data_results)[0], max(data_results)[0]
    #     env_variables['action_at'][x]['unique_outputs_num'] = unique_outputs_num
    #     env_variables['action_at'][x]['minmax'] = action_min_max

    for act_ii, action_info in env_action_at.items():
        # action_name = action_info['name']
        df_col = df[action_info['name']]
        unique_outputs_num = len(df_col.unique())  # load the user defined true labels for classification or solutions for regression
        env_variables['action_at'][act_ii]['unique_outputs_num'] = unique_outputs_num

        if env_variables['action_at'][act_ii].get('minmax') is None:  # find out own min-max (if not provided)
            action_min = df_col.min()
            action_max = df_col.max()
            env_variables['action_at'][act_ii]['minmax'] = (action_min, action_max)

    data_train_panda, data_test_panda = skcv.train_test_split(df, test_size=test_size)  # splitting the data in 80% train and 20% testdata for validation

    # Sfeh: no numpy for version 1.0
    data_train_numpy = data_train_panda.to_numpy()
    data_test_numpy = data_test_panda.to_numpy()

    return env_variables, data_train_panda, data_test_panda, data_train_numpy, data_test_numpy
