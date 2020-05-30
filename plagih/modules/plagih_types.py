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
        return op[node_label]['arity']
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


def choose_term(xtype, env_variables, choose_distribution, float_accuracy):
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
    if np.random.choice(['obs', 'distrib']) == 'obs' and env_variables[xtype]:
        term = np.random.choice(env_variables[xtype])
    else:
        term = choose_constant(xtype, choose_distribution, float_accuracy)

    term = str(term)  # sfeh

    return term


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


def choose_operator(xtype, choose_oparray, arity=None):
    """
    Randomly choosing an operator for a given xtype.
    choose_oparray must be given, as they are different between runs.
    arity can also be set optionally, e.g. for point mutation
    """
    func_list = xtype_get_func_list(choose_oparray, xtype=xtype, arity=arity)
    if not func_list:
        print_e('No function found with xtype={}, arity={}.\nfunc_arr_dummy:\n{}'.format(xtype, arity, choose_oparray))
    func = np.random.choice(func_list)
    arity = label_get_arity(func)
    xtype = op[func]['xtype']
    return func, arity, xtype


def choose_func_old(choose_oparray, xtype, arity=None):
    """
    chooses a function that fits at the spot randomly. func array is created from user functions
    - get a list with potantial functions: ['+', '-', '*']
    - chooses one: '+'

    notes
    - xtype was optional. (xtype=None)
    """
    func_list = xtype_get_func_list(choose_oparray, xtype=xtype, arity=arity)
    if not func_list:
        print_e('No function found with xtype={}, arity={}.\nfunc_arr_dummy:\n{}'.format(xtype, arity, choose_oparray))
    func = np.random.choice(func_list)
    arity = label_get_arity(func)
    xtype = op[func]['xtype']
    return func, arity, xtype


def xtype_get_func_list(choose_oparray, xtype=None, arity=None):
    """
    returns a function list out of the given 2d-op array randomly
    This fills in a function that fits the type of the function/terminal before.
    terminal  '2f' -> '_2f', arity
    function 'f2f' -> '_2f', arity
    function 'b2f2f' -> '_2f', arity

    Note: arity-0 functions (e.g. dummies, that calculate a problem specific value) are terminals!
    """
    func_list = []
    funcs_float = [f2f, b2f, b2f2f]
    funcs_bool = [f2b, b2b]

    # arity and xtype
    if arity is not None and xtype:
        xtype_row = ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f'].index(xtype)
        func_list.extend(choose_oparray[xtype_row][arity])

    # arity
    if arity is not None and xtype is None:
        func_list = sum([xtype_row[arity] for xtype_row in choose_oparray], [])

    # xtype
    if arity is None and xtype is not None:
        if '2f' in xtype:
            func_list = sum([sum(choose_oparray[funcs], []) for funcs in funcs_float], [])
        elif '2b' in xtype:
            func_list = sum([sum(choose_oparray[funcs], []) for funcs in funcs_bool], [])
        else:
            print_e('xtype {} is not accepted. Must be \'2f\' or \'2b\'.'.format(xtype))
            raise

    # return all functions
    if arity is None and xtype is None:
        func_list = sum(sum(choose_oparray, []), [])

    return func_list


def xtype_get_func_list_OLD(oparray, xtype=None, arity=None):
    """
    returns a function list out of the given 2d-op array randomly
    This fills in a function that fits the type of the function/terminal before.
    terminal  '2f' -> '_2f', arity
    function 'f2f' -> '_2f', arity
    function 'b2f2f' -> '_2f', arity
    """
    func_list = []
    funcs_float = [f2f, b2f, b2f2f]
    funcs_bool = [f2b, b2b]

    # arity and xtype
    if arity is not None and xtype:
        xtype_row = ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f'].index(xtype)
        func_list.extend(oparray[xtype_row][arity])

    # arity
    if arity is not None and xtype is None:
        func_list = sum([xtype_row[arity] for xtype_row in oparray], [])

    # xtype
    if arity is None and xtype is not None:
        if '2f' in xtype:
            func_list = sum([sum(oparray[funcs], []) for funcs in funcs_float], [])
        elif '2b' in xtype:
            func_list = sum([sum(oparray[funcs], []) for funcs in funcs_bool], [])
        else:
            print_e('xtype {} is not accepted. Must be \'2f\' or \'2b\'.'.format(xtype))
            raise

    # return all functions
    if arity is None and xtype is None:
        func_list = sum(sum(oparray, []), [])

    return func_list


def xtypes_from_labels(label_list, env_variables):
    xtype_list = [xtype_get_from_label(label, env_variables) for label in label_list]
    return xtype_list


def xtype_get_from_label(label, env_variables):
    """
    returns xtype for a label
    if you are not 100% sure that it is a function.
    """

    if env_variables == 'ö':
        # todo deleteable?
        print_warning('www', 'Sfeh, we knowingly create a xtype-dummy')
        return 'ö'

    if label in env_variables['obs_name']:
        xtype = env_variables['obs_name'][label]['xtype']
    elif label in ['True', 'False']:
        xtype = '2b'
    elif label in op:
        xtype = op[label]['xtype']
    else:
        # if delete_this and isinstance(label, bool):
        #     xtype = '2b'
        #     print_e('This should never happen!!! labels  must be string')
        # try:
        #     float(label)
        #     xtype = '2f'
        # except:
        #     raise Exception('This label is not known at all: {}'.format(label))
        xtype = '2f'

    return xtype
