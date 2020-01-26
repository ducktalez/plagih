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


def xtype_get_v2(label, variables_dict, action_dict, node_arity=None):
    """
    returns xtype for a label
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
