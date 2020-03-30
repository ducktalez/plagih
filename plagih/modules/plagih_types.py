"""
This file takes care of the strong typing in plagih.
'Types' of nodes:
- Terminals: 2f, 2b
- Functions: f2f, f2b, b2b, b2f, b2f2f (last one being if-then-else)
This notation offers the opportunity to actually search for parts, e. g.
> if '2b' in function:  # This returns True if the label evaluates to boolean
Be careful with if-then-else though. This needs boolean and two float inputs to produce one float
"""
from plagih.modules.dicts import *
from plagih.modules.printing import *
import numpy as np
from plagih.modules.dicts import name_observation


def op_label_get_basictype(node_label):
    """
    return terminal or function according to the label
    """

    if node_label in op:
        return 'func'
    else:
        return 'term'


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


def xtype_label_get_child_xtypes(label, node_arity, variables_dict):
    """
    reverse stuff
    """
    xtype = xtype_get_from_label(label, variables_dict)
    if xtype == 'b2f2f':
        xtypes = ['2b', '2f', '2f']
    else:
        xtypes = [xtype[:2][::-1]] * node_arity
    return xtypes


def xtype_choose_term_v2(node_xtype, variables_dict):
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

    # node_xtype == '2f' or 'f2' in node_xtype:
    if '2f' in node_xtype:
        terminals_type = variables_dict['float']
        the_type = 'float'
    elif '2b' in node_xtype:
        terminals_type = variables_dict['bool']
        the_type = 'bool'
    else:
        print_e('Probably, you have to check if your "function" is actually a terminal. xtype: {}'.format(node_xtype))
        raise

    if np.random.choice(['var', 'const']) == 'var' and terminals_type:  # Is there an entry in the list? # todo sfeh choice
        term = np.random.choice(terminals_type)  # ...so we return one
    else:
        term = choose_constant(term_type=the_type)  # otherwise: constant (There are always constants :P)

    return term


def choose_constant(term_type='', mode='float-1to1', uniform_range=None):
    """

    Returns a constant that fits into the position
    -- term_type = 'float'
    """
    if uniform_range:
        return np.random.uniform(uniform_range[0], uniform_range[1])

    if term_type == 'bool':
        const = np.random.choice([True, False])
    elif term_type == 'float':
        if mode == 'float-1to1':
            const = np.random.uniform(-1, 1)
        elif mode == 'intTotal_10':
            const = np.random.random_integers(-10, 10)
        elif mode == 'random_optimised':
            const = np.random.choice([-10, -5, -2, -1, -1, -0.8, -0.6, -0.5, -0.4, -0.2, 0, 10,
                                      5, 2, 1, 1, 0.8, 0.6, 0.5, 0.4, 0.2, 0])
        elif mode == 'idk a nema yet':
            const = float('Math.pi')
            # todo constants can easily be loaded as arity 0 function
            pass
        else:
            # sfeh: gibt viele Verteilungen: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
            print_e('You did not take care of the kind of numbers you want to have')
            raise
    elif term_type == 'int':
        # TODO give more opportunities, similar to random floats
        const = np.random.random_integers(-10, 10)
    else:
        print_warning('w', 'Please specify your desired datatype if possible. Trying to return c1 similar to terminals.')
        print_e('This term type should not occur, I guess {}'.format(term_type))
        const = choose_constant(term_type=term_type)
    return str(const)

#
# def xtype_choose_func_v2(func_array, xtype=None, arity=None):
#     """
#     This fills in a function that fits the type of the function/terminal before.
#     terminal  '2f' -> '_2f', arity
#     function 'f2f' -> '_2f', arity
#     function 'b2f2f' -> '_2f', arity
#     > ->
#     """
#     if xtype:
#         if '2f' in xtype:
#             choose_func = sum(func_array[f2f] + func_array[b2f] + func_array[b2f2f], [])
#         elif '2b' in xtype:
#             choose_func = sum(func_array[f2b] + func_array[b2b], [])
#         else:
#             print_e('What kind of type is that? {}'.format(xtype))
#             raise
#     else:
#         choose_func = sum(func_array[f2f] +
#                           func_array[b2f] +
#                           func_array[b2f2f] +
#                           func_array[f2b] +
#                           func_array[b2b], [])
#
#     # Attention! do not choose out of an dictionary.
#     # every function is inside there only once, so no higher chance for functions that are more often in the list
#     # label = np.random.choice(self.xype_func_dict['2f'])
#
#     label = np.random.choice(choose_func)
#
#     return label, op[str(label)]['arity']


def xtype_choose_func(func_array, xtype, arity=None):
    """
    chooses a function that fits at the spot randomly. func array is created from user functions
    - get a list with potantial functions: ['+', '-', '*']
    - chooses one: '+'

    notes
    - xtype was optional. (xtype=None)
    """
    func_list = xtype_get_func_list(func_array, xtype=xtype, arity=arity)
    if not func_list:
        print_e('No function found with xtype={}, arity={}.\nfunc_array:\n{}'.format(xtype, arity, func_array))
    func = np.random.choice(func_list)
    arity = label_get_arity(func)
    xtype = op[func]['xtype']
    return func, arity, xtype


def xtype_get_func_list(func_array, xtype=None, arity=None):
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
        func_list.extend(func_array[xtype_row][arity])

    # arity
    if arity is not None and xtype is None:
        func_list = sum([xtype_row[arity] for xtype_row in func_array], [])

    # xtype
    if arity is None and xtype is not None:
        if '2f' in xtype:
            func_list = sum([sum(func_array[funcs], []) for funcs in funcs_float], [])
        elif '2b' in xtype:
            func_list = sum([sum(func_array[funcs], []) for funcs in funcs_bool], [])
        else:
            print_e('xtype {} is not accepted. Must be \'2f\' or \'2b\'.'.format(xtype))
            raise

    # return all functions
    if arity is None and xtype is None:
        func_list = sum(sum(func_array, []), [])

    # TODO what about arity-0 functions? those are effectively terminals and currently do not exist

    return func_list


def xtypes_from_labels(label_list, observations_bundle):
    xtype_list = [xtype_get_from_label(label, observations_bundle) for label in label_list]
    return xtype_list


def xtype_get_from_label(label, variables_dict):
    """
    returns xtype for a label
    if you are not 100% sure that it is a function.
    """
    if variables_dict == 'ö':
        print_warning('www', 'Sfeh, we knowingly create a xtype-dummy')
        return 'ö'

    if label in variables_dict['info']:
        xtype = variables_dict['info'][label]['xtype']
    elif label in ['True', 'False']:
        xtype = '2b'
    elif label in op:
        xtype = op[label]['xtype']
    else:
        try:
            float(label)
            xtype = '2f'
        except:
            raise Exception('This label is not known at all: {}'.format(label))

    return xtype
