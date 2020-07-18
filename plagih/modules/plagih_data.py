from pathlib import Path
import sklearn.model_selection as skcv
from plagih.modules.printing import *
import numpy as np
import pandas as pd
import re


class EnvObservations:

    def __init__(self, name):
        self.obs_info = {}
        self.obs_familys = {}
        self.xtype_obs = {'2f': [], '2b': []}


class EvalAction:

    def __init__(self, minmax=None):
        self.obs_info = {}


class EnvVars:

    def __init__(self):
        self.action_info = {}


def data_from_csv(samples_file, action_name, test_size=0.2, delimiter=','):
    """
    Loads .csv data files.
    Information that we need to extract for each column:
    1. observation or action? (-> choosing leaf nodes)
        - observation
            - is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
            (- observation min max (deprecated, for histograms))
        - action
            - which is the action for the regression? -> more than one action might be required (IB has three action dimensions)
            - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values
                 todo solve this inside the kernel (?)

    """

    def is_action(name, role):
        if 'role' is None:
            if any(x in role for x in ['out', 'action', 'act', 'result']):
                is_act = True
            elif any(x in role for x in ['input', 'observation']):
                is_act = False
            else:
                raise Exception(f'Role not known! {role}')
        else:
            if 'a_' == name[:2]:
                is_act = True
            elif ii == len(df.columns) - 1:
                is_act = True
            elif any(x in name for x in ['out', 'action', 'act', 'result']):
                is_act = True
            else:
                is_act = False
        return is_act


    env_vars = EnvVars()

    """
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use
    """
    with Path.open(samples_file) as file:
        df = pd.read_csv(file, delimiter=delimiter, nrows=1)

    rename_columns = {}
    for ii, header in enumerate(df):

        header_split = header.split('|')  # split 1: cartVel|type=float|role=input --> {cartVel, type=float, role=input]
        column_name = header_split[0]  # The first entry is always the column name
        rename_columns[header_split] = column_name

        col_infos = {}  # column_name: {'type': 'float', 'role': None, 'colpos': ii}}
        try:
            for col_param in header_split[1:]:
                param, value = col_param.split('=')
                col_infos[param] = value
        except Exception as ex:
            print(f'Could not load samples csv correctly: {ex}')

        if is_action(name, col_infos.get('role')):
            if column_name == action_name:

                cmin = col_infos.get('min')
                cmax = col_infos.get('max')
                if cmin is None or cmax is None:
                    cmin = df[header].min()
                    cmax = df[header].max()

                ctype = float
                cmin = float(cmin)
                cmax = ctype(cmax)
                env_vars.eval_Action = EvalAction(minmax=(cmin, cmax))
            else:
                printez('i', f'Ignoring action {column_name} for this run')
    df.rename(columns=rename_columns, inplace=True)

    """
    Splitting df into train and testdata
    """
    data_train, data_test = skcv.train_test_split(df, test_size=test_size, random_state=0)  # 80% train 20% test-validation
    # if 'type' in column_meta_values:
    #         #     ctype = column_meta_values['type']
    #         #     ctype = ctype.replace('num', 'float')  # sfeh: num is currently not really used
    #         # else:
    #         #     dtype = df[header].dtype
    #         #     if column_meta_values.get('type') == 'float' \
    #         #             or dtype == np.dtype('float') \
    #         #             or dtype == np.dtype('int'):  # sfeh yes, int aswell
    #         #         xtype = '2f'
    #         #         ctype = float
    #         #         tf_dtype = tf.float32
    #         #     elif dtype == np.dtype('bool'):
    #         #         xtype = '2b'
    #         #         ctype = float
    #         #         tf_dtype = tf.float32
    ctype = float
    tf_type = tf.float32
    xtype = '2f'




    else:

        obs_family, obs_index = observation_get_family_and_time(column_name)
        try:
            env_vars.obs_familys[obs_family].extend([column_name])  # todo insert in sorted list... also t = [1,2,5,10]?
        except:
            env_vars.obs_familys[obs_family] = [column_name]

    if any(x in role for x in ['result', 'output', 'out', 'action']):
        action_count = len(env_action_at)
        env_action_at[action_count] = all_meta_dict
    else:
        raise

    param_at[ii] = {'name': column_name, 'type': col_type, 'xtype': xtype, 'role': role}

    # last-chance use of env_vars. sfeh: use many dicts instead! hmmm
    env_vars['action_at'].update(env_action_at)
    env_vars['obs_name'].update(obs_name)
    env_vars.update(env_xtype_list)  # sfeh that is ugly code
    env_vars['env_observation_family'] = env_observation_family

    env_vars, param_at = samples_header_line(header_row)
    #######
    dtypes_from_header = {}
    # colnames_from_header = []

    # param_at[ii] = {'name': name, 'type': col_type, 'xtype': xtype, 'role': role}
    # for ii, param_values in param_at.items():
        # colnames_from_header.append(param_values['name'])
        # dtypes_from_header[param_values['name']] = tf.float32 if param_values['type'] == 'float' else tf.bool

    with Path.open(samples_file) as file:
        df = pd.read_csv(file)  # skiprows=1, names=colnames_from_header, dtype=dtypes_from_header)

    env_action_at = env_vars['action_at']
    for act_ii, action_info in env_action_at.items():
        df_col = df[action_info['name']]
        env_vars['action_at'][act_ii]['unique_outputs_num'] = len(df_col.unique())
        if env_vars['action_at'][act_ii].get('minmax') is None:  # find out own min/max (if not provided)
            env_vars['action_at'][act_ii]['minmax'] = (df_col.min(), df_col.max())

    obs_infoz = env_vars['obs_name']
    for ii, (obs_name, obs_info) in enumerate(obs_infoz.items()):
        if obs_info.get('minmax') is None:
            minmax = (df[obs_name].min(), df[obs_name].max())
            print_warning('w', f'Tried to get the observation min/max from data, which is {minmax}.')
            env_vars['obs_name'][obs_name]['minmax'] = minmax



    return env_vars, data_train, data_test


def observation_get_family_and_time(name, re_pattern='_\d+$', none_return=0):
    """
    """

    core_label = re.split(re_pattern, name)[0]

    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception:
        temp_diff = none_return  # there is only the latest version
    return core_label, temp_diff


def obs_get_timedelta(name, re_pattern='_\d+$', none_return=0):
    """
    'cartVel_12' -> core_label = 'cartVel', temp_diff = 12
    can be used to enrich old runs 'manually', otherwise only used in header_entry_get_tempdiff
    """
    re_search = re.search(re_pattern, name)  # re_search => ['_12']
    # todo what is the best solution? '\_\d+$' is the correct regex using search.
    if re_search:
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        return int(temp_diff)
    else:
        return none_return  # there is only the latest version


def obs_get_family(name, re_pattern='_\d+$'):
    """
    variable like 'temperature_12' is variable 12 steps from past (13th)
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
        temp_diff = obs_get_timedelta(name, re_pattern=re_pattern)

    return temp_diff


if __name__ == '__main__':
    import tensorflow as tf
    env_vars, data_train_panda, data_test_panda = data_from_csv(Path('../../benchmarks/run_sources/MTC/samples75.csv'))
    obsnames = env_vars['obs_name'].keys()
    for (name, data) in data_train_panda.iteritems():
        tf_dtype = tf.float32 if data.dtype == np.float32 else tf.bool
        x = tf.constant(data.to_numpy(), dtype=tf.float32)
    print(x)
