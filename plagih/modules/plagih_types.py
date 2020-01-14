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


def op_label_get_basictype(node_label):
    """
    return terminal or function according to the label
    """

    if node_label in op:
        return 'func'
    else:
        return 'term'


def op_label_get_arity(node_label):
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
