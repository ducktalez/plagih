"""
This file takes care of the strong typing in plagih.
'Types' of nodes:
- Terminals: 2f, 2b
- Functions: f2f, f2b, b2b, b2f, b2f2f (last one being if-then-else)
This notation offers the opportunity to actually search for parts, e. g.
> if '2b' in function:  # This returns True if the label evaluates to boolean
Be careful with if-then-else though. This needs boolean and two float inputs to produce one float
"""
import random

from plagih.modules.operators import *
from plagih.modules.printing import *
import numpy as np


def label_get_arity(node_label):
    """
    return terminal or function according to the label
    """

    if node_label in op:
        arity_sfehdebug = op[node_label]['arity']
        return arity_sfehdebug
    else:
        return 0


def xtype_equi_outcome(a_xtype, b_xtype):
    """
    Dummy. Returns, whether two xtypes are equal
    """
    if a_xtype[-2:] == b_xtype[-2:]:
        equal = True
    else:
        equal = False
    return equal


def xtype_get_converters(xtype):
    """
    convert b-to-a dummy
    """
    # 'lf left is boolean, give a converter to my type'

    if '2b' in xtype:
        return 'Ftob', 'Btof'
    if '2f' in xtype:
        return 'Btof', 'Ftob'
    else:
        print('e', 'Wrong data_csv_path type? Should be 2b or 2f, but is {}'.format(xtype))
        raise


def random_choose_tempobs(obs_list, max_hist=10):
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
    # p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
    new_obs = random.choices(obs_list, weights=p)[0]  # random.choices returns a list
    return new_obs


def choose_term(xtype, random_obs, choose_distribution, float_decimals):
    """
    Returns a terminal of xtype.

    function: f2b -> 2b needed
    terminal:  2f -> 2f needed
    --> check if it is function, aka _2f
    --> check if it is terminal, aka f2

    Modes:
    var_and_const: return randomly (50:50) a variable or a constant
    terminal_only: return                  a variable

    input options: f2f, f2b, b2f, b2b, f2b2b, 2f, 2b
    """

    # insert a ?
    if random.choice(['obs', 'distrib']) == 'obs' and random_obs[xtype]:
        # obs_list = random.choice(list(env_vars['env_observation_family'].values()))
        term = random_obs[xtype]()
    else:
        dist_fun = random.choice(choose_distribution[xtype])
        term = dist_fun()
        if '2f' in xtype:  # sfeh int aswell?
            term = round(term * float_decimals) / float_decimals

    return str(term)  # sfeh str necessary?


def choose_operator(xtype, choose_oparray=None, choose_oparray2=None, arity=None):
    """
    Randomly choosing an operator for a given xtype.
    choose_oparray must be given, as they are different between runs.
    arity can also be set optionally, e.g. for point mutation
    """
    if delete_this_version1 and choose_oparray2 is not None:
        func_list, probability_list = choose_oparray2[xtype][arity]
    else:
        func_list, probability_list = xtype_get_func_list(choose_oparray, xtype=xtype, arity=arity)
    if not func_list:
        print_e(f'No function found with xtype={xtype}, arity={arity}.\nfunc_arr_dummy:\n{choose_oparray}')
    func = random.choices(func_list, weights=probability_list)[0]  # returns a list, so we choose the first element
    arity = label_get_arity(func)
    xtype = op[func]['xtype']
    return func, arity, xtype


def xtypes_from_labels(label_list, obs_krazy=None):
    xtype_list = [xtype_get_from_label(label, obs_krazy) for label in label_list]
    return xtype_list


def xtype_get_from_label(label, obs_krazy=None):
    """
    returns xtype for a label
    if you are not 100% sure that it is a function.
    """

    if label in ['True', 'False']:
        xtype = '2b'
    elif label in op:
        xtype = op[label]['xtype']
    else:
        try:
            label = label[1:] if label[0] == '-' else label
            xtype = obs_krazy[label]
        except:
            xtype = '2f'

    return xtype
