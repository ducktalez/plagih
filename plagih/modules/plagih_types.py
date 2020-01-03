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


def xtype_outcome_equi_test(a_xtype, b_xtype):
    """
    Dummy. Returns, whether two xtypes are equal
    """
    return a_xtype in b_xtype or b_xtype in a_xtype


def xtype_get_converter(a_xtype, b_xtype):
    """
    convert b-to-a dummy
    """
    if '2b' in a_xtype and '2f' in b_xtype:
        return 'Ftob'
    if '2f' in a_xtype and '2b' in b_xtype:
        return 'Btof'
    else:
        print('e', 'One of those two cases should happen', a_xtype, b_xtype)
        raise
