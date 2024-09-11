"""
Tree nodes that do not require fixed arity.
Separate file to NOT confuse anything, even though there might be some redundancy
"""
import os

import sympy

from plagih.tree_labels import OperatorChained
from plagih.util import get_subclasses, FLOAT_PRECISION, DEBUG_DUMMY  # noqa

os.environ["KMP_WARNINGS"] = "FALSE"


class AddChain(OperatorChained):
    symfun = sympy.Add
    expr_dmy = 'Add'
    showme = '+'
    xtype = ((float,), float)
    xtype_chain = float


class MulChain(OperatorChained):
    # sfeh:discuss: if sympy is
    expr_dmy = 'Mul'
    showme = '*'
    symfun = sympy.Mul
    xtype = ((float,), float)
    xtype_chain = float


class MinChained(OperatorChained):
    symfun = sympy.Min
    expr_dmy = 'Min'
    showme = 'Min'
    xtype = ((float, float), float)
    chain_xtype = float


class MaxChained(OperatorChained):
    symfun = sympy.Max
    expr_dmy = 'Max'
    showme = 'Max'
    xtype = ((float, float), float)
    chain_xtype = float


# class OrderedSelector(ChainOp):
#     """sfeh:Orders Elements with < and picks the (1, -1 or even -2, 'median')-element?
#     sfeh:idea using a function for selecting the n-th is an option!"""
#     xtype = ((float,), float)
#     chain_xtype = float
#     symfun = lambda *a: sympy.Order(a)


class Piecewise(OperatorChained):
    """sfeh:discuss: the only Operator, which has tuples as input

    sfeh:open force a (foo, True) option, restrict mutating "True"
            CAUTION! Notimplemented?
        A method to determine whether a multivariate conditional is consistent
        with a complete coverage of all variables has not been implemented so
        the rewrite is being stopped after encountering `cartPos >=
        Max(cartPos, 10.4*cartPos*cartVel)`. This error would not occur if a
        default expression like `(foo, True)` were given.
"""
    # ogclass = Ifte
    # xtype = ((float, bool), float)
    expr_dmy = 'Piecewise'
    showme = 'Piecewise'
    # these must be handeled differently, so commented out
    # xtype = ((ExprCondPair,), float)
    # xtype_chain = ExprCondPair_Dummy
    # symfun = sympy.Piecewise
    symfun = lambda *a: sympy.Piecewise(a)


class AndChained(OperatorChained):
    expr_dmy = 'And'
    showme = 'and'
    xtype = ((bool,), bool)
    xtype_chain = bool
    symfun = sympy.And


class OrChained(OperatorChained):
    xtype = ((bool,), bool)
    xtype_chain = bool
    expr_dmy = 'Or'
    showme = 'or'
    symfun = sympy.Or

