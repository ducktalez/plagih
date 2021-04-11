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

from plagih.operators import *
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
    Dummy. Returns, whether two xtypes have the same outcome
    """
    return a_xtype[-2:] == b_xtype[-2:]


def random_choose_tempobs(obs_list, max_hist=10):
    """
    chooses variables but weighting how old they are.
    obs_list = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4']
    -> [0.28, 0.23, 0.19, 0.16, 0.13]
    sfeh: what about larger steps?
    0, 1, 2, 3 is good, but
    0, 5, 10, 15 is worse
    what if variables are not all of same diff?
    """
    obs_list = np.delete(obs_list, np.s_[max_hist:])
    x = len(obs_list)
    fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
    p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
    # p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
    new_obs = random.choices(obs_list, weights=p)[0]  # random.choices returns a list
    return new_obs


def choose_term(xtype, choose_obs, choose_distribution, float_decimals):
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
    if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[xtype]:
        # obs_list = random.choice(list(env_vars['env_observation_family'].values()))
        term = choose_obs[xtype]()
        term = term.name
    else:
        dist_fun = random.choice(choose_distribution[xtype])
        term = dist_fun()
        if '2f' in xtype:  # sfeh int aswell?
            term = float(round(term, float_decimals))

    return str(term)  # sfeh str necessary?


def choose_operator(xtype, choose_oparray2, arity=None):
    """
    Randomly choosing an operator for a given xtype.
    choose_oparray2 must be given, as they are different between runs.
    arity can also be set optionally, e.g. for point mutation
    """
    func_list, probability_list = choose_oparray2[xtype][arity]
    # except:
    #     raise Exception('sfeh DELETE IF NOT REQUIRED, 22.07.')

    ops = random.choices(func_list, weights=probability_list)[0]  # [0] as this function can only return lists...
    return ops


def xtypes_from_labels(label_list, obs_infos=None):
    xtype_list = [xtype_get_from_label(label, obs_infos) for label in label_list]
    return xtype_list


def xtype_get_from_label(label, obs_infos=None):
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
            xtype = obs_infos[label]['xtype']
        except:
            xtype = '2f'

    return xtype
