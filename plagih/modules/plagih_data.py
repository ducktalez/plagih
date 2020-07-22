from pathlib import Path
import sklearn.model_selection as skcv
from plagih.modules.printing import *
from plagih.modules.operators import op
import numpy as np
import pandas as pd
import re
from pydoc import locate
import random

import tensorflow as tf


class Obs:

    def filter_new_index(self):
        if self.index_minmax is None:
            return
        else:
            new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
            self.obs_index = new_index
            self.name = f'{self.family}_{new_index}'

    def __init__(self, name, ctype):
        self.name = name
        self.ctype = ctype
        self.tf_type = tf.float32
        self.xtype = '2f'

        obs_family, obs_index = observation_get_family_and_time(name, none_return=None)
        self.family = obs_family
        self.obs_index = obs_index  # is None when no index but 0 when

        self.index_minmax = None
        self.fun_filter_index = lambda: None  # as default, return own index


def choice_weights(obs_list, max_hist=10):
    """
    chooses variables but weighting how old they are.
    obs_list = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4']
    -> [0.28, 0.23, 0.19, 0.16, 0.13]
    sfeh: what about larger steps?
    0, 1, 2, 3 is good, but
    0, 5, 10, 15 is worse
    what if variables are not all of same diff?
    todo create list at start of run
    """
    obs_list = np.delete(obs_list, np.s_[max_hist:])
    x = len(obs_list)
    fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
    p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
    p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
    return p


class EvalAction:
    """
    - minmax for histograms
    - minmax for regression-bounded
    """

    def __init__(self, name, ctype, cmin, cmax, uniques):
        self.name = name

        self.name = name
        self.ctype = ctype  # ! may be int!
        self.tf_type = tf.float32  # sfeh especiall when the type is integer
        self.xtype = '2f'

        self.uniques = uniques

        self.minmax = (ctype(cmin), ctype(cmax))


class EnvVars:
    """
    observation
    - take ~100 samples as constants
    - get xtype for buildin tree from expr
    - eval - get tensors with correct tf-type
    - filter index
    eval_action
    """

    def __init__(self):
        self.obs_krazy = {}  # lookup table with all observations - if an observation is not in here, it is a float
        self.obs_infos = {}
        self.eval_action: EvalAction = None
        self.choose_obs = {'f2': None,
                           '2b': None}


def data_from_csv(samples_file, action_name, test_size=0.2, delimiter=','):
    """
    Loads .csv data files.
    Information that we need to extract for each column:
    - choose_xtype choosing a random observation for leaf nodes
    - filtering observation index
        is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
    - evalaction data_train, data_test, is the action for the regression? -> more than one action might be required (IB has three action dimensions)
            - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values
                 todo solve this inside the kernel (?)

    """

    def is_action(name, role):
        if 'role' is None:
            is_act = any(x in role for x in ['out', 'action', 'act', 'result'])
        else:
            is_act = any(['a_' == name[:2],
                          ii == len(df.columns) - 1,
                          any(x in name for x in ['out', 'action', 'act', 'result'])])
        return is_act

    """
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use
    """
    with Path.open(samples_file) as file:
        df = pd.read_csv(file, delimiter=delimiter)

    eval_action = None
    drop_actions = []
    obs_list = []
    rename_columns = {}

    for ii, header in enumerate(df):

        header_split = header.split('|')  # split 1: cartVel|type=float|role=input --> {cartVel, type=float, role=input]
        column_name = header_split[0]  # The first entry is always the column name
        if column_name in op:
            raise Exception(f'Your samples hold a column that matches the potential tree operator {column_name}. That might end up in confusion, please rename the column.')
        rename_columns[header] = column_name

        col_infos = {}  # column_name: {'type': 'float', 'role': None, 'colpos': ii}}
        try:
            for col_param in header_split[1:]:
                param, value = col_param.split('=')
                col_infos[param] = value
        except Exception as ex:
            print(f'Could not load samples csv correctly: {ex}')

        ctype = locate(col_infos.get('type', 'float'))

        if is_action(column_name, col_infos.get('role')):
            if column_name == action_name or action_name is None:
                cmin = col_infos.get('min', df[header].min())
                cmax = col_infos.get('max', df[header].max())
                uniques = df[header].unique()
                eval_action = EvalAction(column_name, ctype, cmin, cmax, uniques)
            else:
                drop_actions.append(column_name)  # drop from data
                printez('i', f'Ignoring action {column_name} for this run')
        else:
            obs_list.append(Obs(column_name, ctype))

    df.rename(columns=rename_columns, inplace=True)
    df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P Following design pattern #YOLO
    df = df.drop(drop_actions, axis=1)  # no need to keep other actions

    """
    choosing random observations made easy
    """
    env_vars = EnvVars()
    obs_fams = [fam for fam in list(set(obs.family for obs in obs_list))]
    choose_obs_2f = []
    choose_obs_p = []
    obs_info = {}
    for fam in obs_fams:
        family_meeting = sorted([x for x in obs_list if x.family == fam], key=lambda lulz: lulz.obs_index)
        if len(family_meeting) > 1:
            choose_obs_2f.extend([x for x in family_meeting])
            choose_obs_p.extend(list(choice_weights(family_meeting)))

            index_minmax = (family_meeting[0].obs_index, family_meeting[-1].obs_index)
            for obs_tmp in family_meeting:
                obs_tmp.index_minmax = index_minmax
                env_vars.obs_infos[obs_tmp.name] = obs_tmp
                # obs_tmp.fun_filter_index = lambda: obs_tmp.filter_new_index()  # int(max(min(round(random.gauss(obs_tmp.obs_index, 1)), index_minmax[1]), 0))
                obs_info[obs_tmp.name] = obs_tmp
        else:
            # LOL UMAD? only one family member (probably even more common)
            obs_tmp = family_meeting[0]
            # obs_tmp.fun_filter_index = lambda: None
            obs_info[obs_tmp.name] = obs_tmp
            choose_obs_2f.append(obs_tmp)
            choose_obs_p.append(1)

    obs_2f = lambda: random.choices(choose_obs_2f, weights=choose_obs_p)[0]
    choose_obs = {'2f': obs_2f,
                  '2b': None}  # todo consider max age?

    env_vars.choose_obs = choose_obs
    env_vars.obs_infos = obs_info

    if eval_action:
        env_vars.eval_action = eval_action
    else:
        raise

    data_train, data_test = skcv.train_test_split(df, test_size=test_size, random_state=0)

    return env_vars, data_train, data_test


def observation_get_family_and_time(name, re_pattern='_\d+$', none_return=None):
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


# if __name__ == '__main__':
#     import tensorflow as tf
#
#     env_vars, data_train_panda, data_test_panda = data_from_csv(Path('../../benchmarks/run_sources/MTC/samples75.csv'), action_name=None)
#     obsnames = env_vars['obs_name'].keys()
#     for (name, data) in data_train_panda.iteritems():
#         tf_dtype = tf.float32 if data.dtype == np.float32 else tf.bool
#         x = tf.constant(data.to_numpy(), dtype=tf.float32)
#     print(x)
