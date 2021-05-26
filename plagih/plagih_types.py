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
    return arity of a label
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


def choose_term(coolxtype_out, choose_obs, choose_distributions, float_decimals):
    """

    """

    # sfeh 50% chance observation/value
    if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[coolxtype_out]:
        obs = choose_obs[coolxtype_out]()
        # print('SAME???', obs.name, obs.label)  # sfeh
        return obs
    else:
        dist_fun = random.choice(choose_distributions[coolxtype_out])
        value = dist_fun()
        if coolxtype_out == float:  # sfeh int aswell?
            value = float(round(value, float_decimals))
            const = FloatConstant(value)
        elif coolxtype_out == bool:
            const = BoolConstant(value)
        else:
            raise Exception('ASDASD NOOO WHYY')
        return const


def choose_operator(coolxtype_key, choose_oparray3):
    """
    Randomly choosing an operator-label for a given xtype.
    choose_oparray3 must be given, as they are different between runs.
    arity can also be set optionally, e.g. for point mutation
    todoo DOUBLE-check if this coolxtype is chosen correcrtly... better: replace it
    """
    func_list, probability_list = choose_oparray3[coolxtype_key]
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
