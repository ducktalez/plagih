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
        print_e('Probably, you have to check if your "function" is actually a terminal. xtype {}'.format(node_xtype))
        raise

    if np.random.choice(['var', 'const']) == 'var':  # our choice is variable
        if terminals_type:  # Is there an entry in the list?
            return np.random.choice(terminals_type)  # ...so we return one
    return tree_build_type_constant_get(term_type=the_type)  # otherwise: constant (There are always constants :P)


def tree_build_type_constant_get(term_type='', mode='float-1to1', uniform_range=None):
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
        # term_type = np.random.choice(self.variables_dict['types'])
        const = tree_build_type_constant_get(term_type=term_type)
    return str(const)


def xtype_choose_func_v2(op_array, xtype=None):
    """
    This fills in a function that fits the type of the function/terminal before.
    terminal  '2f' -> '_2f', arity
    function 'f2f' -> '_2f', arity
    function 'b2f2f' -> '_2f', arity
    > ->
    """
    if xtype:
        if '2f' in xtype:
            choose_func = sum(op_array[f2f] + op_array[b2f] + op_array[b2f2f], [])
        elif '2b' in xtype:
            choose_func = sum(op_array[f2b] + op_array[b2b], [])
        else:
            raise
    else:
        choose_func = sum(op_array[f2f] +
                          op_array[b2f] +
                          op_array[b2f2f] +
                          op_array[f2b] +
                          op_array[b2b], [])

    # Attention! do not choose out of an dictionary.
    # every function is inside there only once, so no higher chance for functions that are more often in the list
    # label = np.random.choice(self.xype_func_dict['2f'])

    # choose out of a list, or add another way. maybe automatic?

    label = np.random.choice(choose_func)

    return label, op[str(label)]['arity']


def xtype_choose_func_pointmutation(op_type_arity_array, xtype=None, arity=None):
    """
    returns a function for a function in point mutation
    This only accepts functions as inputs. (point mutation)
    No need to handle terminals
    """

    if arity:

        if xtype == 'f2f':
            return np.random.choice(op_type_arity_array[f2f][arity])
        elif xtype == 'f2b':
            return np.random.choice(op_type_arity_array[f2b][arity])
        elif xtype == 'b2b':
            return np.random.choice(op_type_arity_array[b2b][arity])
        elif xtype == 'b2f':
            return np.random.choice(op_type_arity_array[b2f][arity])
        elif xtype == 'b2f2f':
            return np.random.choice(op_type_arity_array[b2f2f][arity])  # sfeh okay that does not make sense tbh
        else:
            print_e('Function was not found in function_types_dict {}'.format(xtype))
            raise

    else:
        raise


def xtype_get_v2(label, variables_dict=None, action_dict=None, node_arity=None):
    """
    returns xtype for a label
    variables_dict and action_dict MUST be set
    if you are not 100% sure that it is a function.
    """
    if not node_arity:
        node_arity = label_get_arity(label)

    if node_arity == 0:  # arity=0 -> terminal
        if 'True' in label or 'False' in label:
            node_xtype = '2b'
        elif 'observation' in label:
            term_position = variables_dict['all'].index(label)
            node_xtype = op[variables_dict['types'][term_position]]['xtype']
        elif 'action' in label:
            print_warning('w', 'Does this happen? Test it!')
            node_xtype = action_dict[label]

        else:  # only 'float' left
            node_xtype = '2f'
    elif node_arity > 0:
        node_xtype = op[label]['xtype']
    else:
        print_e('This arity is not known: {}'.format(node_arity))
        raise

    return node_xtype
