"""
Tree nodes that do not require fixed arity.
Separate file to NOT confuse anything, even though there might be some redundancy
"""
import os

import numpy as np
import sympy

from plagih.tree_labels import OperatorChained
from plagih.util import get_subclasses, FLOAT_PRECISION, DEBUG_DUMMY  # noqa

os.environ["KMP_WARNINGS"] = "FALSE"


class AddChain(OperatorChained):
    """It is Sum, but we call it chain, as vector also is taken"""
    symfun = sympy.Add
    np_fun = np.add
    np_fun = lambda x: np.sum(np.vstack(x), axis=0)
    showme = 'Add'
    sy_str = 'Add({})'
    formulae_str = 'Add({})'
    repr_str = 'AddChain{},[{}]'
    inline_sep = ' + '
    xtype = ((float,), float)
    xtype_chain = float


class MulChain(OperatorChained):
    showme = 'Mul'
    symfun = sympy.Mul
    np_fun = lambda x: np.prod(np.vstack(x), axis=0)
    sy_str = 'Mul({})'
    formulae_str = 'Mul({})'
    repr_str = 'MulChain{},[{}]'
    inline_sep = ' * '
    xtype = ((float,), float)
    xtype_chain = float


class MinChain(OperatorChained):
    symfun = sympy.Min
    np_fun = lambda x: np.min(np.vstack(x), axis=0)
    showme = 'Min'
    sy_str = 'Min({})'
    formulae_str = 'Min({})'
    repr_str = 'MinChain{},[{}]'
    xtype = ((float, float), float)
    chain_xtype = float


class MaxChain(OperatorChained):
    symfun = sympy.Max
    np_fun = lambda x: np.max(np.vstack(x), axis=0)
    showme = 'Max'
    sy_str = 'Max({})'
    formulae_str = 'Max({})'
    repr_str = 'MaxChain{},[{}]'
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
    symfun = lambda *a: sympy.Piecewise(a)
    # (e1, c1), (e2, c2) -> ex, (c1, c2), (e1, e2)
    # np_fun = lambda *a: np.piecewise(a, a)
    showme = 'Piecewise'
    sy_str = 'Piecewise({})'
    formulae_str = 'Piecewise({})'
    repr_str = 'Piecewise{},[{}]'
    # these must be handeled differently, so commented out
    # xtype = ((ExprCondPair,), float)
    # xtype_chain = ExprCondPair_Dummy
    # symfun = sympy.Piecewise


class AndChain(OperatorChained):
    expr_dmy = 'And'
    sy_str = 'And({})'
    formulae_str = 'And({})'
    repr_str = 'And{},[{}]'
    inline_sep = ' & '
    xtype = ((bool,), bool)
    xtype_chain = bool
    symfun = sympy.And


class OrChain(OperatorChained):
    xtype = ((bool,), bool)
    xtype_chain = bool
    showme = 'Or'
    sy_str = 'Or({})'
    formulae_str = 'Or({})'
    inline_sep = ' | '
    repr_str = 'OrChain{},[{}]'
    symfun = sympy.Or

