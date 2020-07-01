"""
This file takes care of the strong typing in plagih.
'Types' of nodes:
- Terminals: 2f, 2b
- Functions: f2f, f2b, b2b, b2f, b2f2f (last one being if-then-else)
This notation offers the opportunity to actually search for parts, e. g.
> if '2b' in function:  # This returns True if the label evaluates to boolean
Be careful with if-then-else though. This needs boolean and two float inputs to produce one float
"""
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


def choose_term(xtype, env_vars, choose_distribution, float_accuracy):
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
    if np.random.choice(['obs', 'distrib']) == 'obs' and env_vars[xtype]:
        # todo take temp_diff into consideration
        term = np.random.choice(env_vars[xtype])
    else:
        term = choose_constant(xtype, choose_distribution, float_accuracy)

    term = str(term)  # sfeh

    return term


def random_choose_tempobs(var_list):
    """
    # todo filter
    """
    x = len(var_list)
    fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
    p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
    p = p / np.sum(p)  # the sum must be equal to 1
    new_obs = np.random.choice(var_list, p=p)
    return new_obs


def choose_constant(xtype, choose_distributions, accuracy):
    """

    Returns a constant that fits into the position
    -- xtype = 'float'
    """
    const = np.random.choice(choose_distributions[xtype])()
    if '2f' in xtype:
        const = round(const*accuracy)/accuracy

    return const

#
# def xtype_choose_func_v2(func_arr_dummy, xtype=None, arity=None):
#     """
#     This fills in a function that fits the type of the function/terminal before.
#     terminal  '2f' -> '_2f', arity
#     function 'f2f' -> '_2f', arity
#     function 'b2f2f' -> '_2f', arity
#     > ->
#     """
#     if xtype:
#         if '2f' in xtype:
#             choose_operator = sum(func_arr_dummy[f2f] + func_arr_dummy[b2f] + func_arr_dummy[b2f2f], [])
#         elif '2b' in xtype:
#             choose_operator = sum(func_arr_dummy[f2b] + func_arr_dummy[b2b], [])
#         else:
#             print_e('What kind of type is that? {}'.format(xtype))
#             raise
#     else:
#         choose_operator = sum(func_arr_dummy[f2f] +
#                           func_arr_dummy[b2f] +
#                           func_arr_dummy[b2f2f] +
#                           func_arr_dummy[f2b] +
#                           func_arr_dummy[b2b], [])
#
#     # Attention! do not choose out of an dictionary.
#     # every function is inside there only once, so no higher chance for functions that are more often in the list
#     # label = np.random.choice(self.xype_func_dict['2f'])
#
#     label = np.random.choice(choose_operator)
#
#     return label, op[str(label)]['arity']


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
        print_e('No function found with xtype={}, arity={}.\nfunc_arr_dummy:\n{}'.format(xtype, arity, choose_oparray))
    func = np.random.choice(func_list, p=probability_list)
    arity = label_get_arity(func)
    xtype = op[func]['xtype']
    return func, arity, xtype


def xtypes_from_labels(label_list, env_vars=None):
    xtype_list = [xtype_get_from_label(label, env_vars) for label in label_list]
    return xtype_list


def xtype_get_from_label(label, env_vars=None):
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
            xtype = env_vars['obs_name'][label]['xtype']
        except:
            xtype = '2f'

    return xtype
