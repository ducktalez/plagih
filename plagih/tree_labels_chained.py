"""
Tree nodes that do not require fixed arity.
Separate file to NOT confuse anything, even though there might be some redundancy
"""
import os

import sympy
# import sympy.functions.elementary.piecewise  # sfeh: needs separate import?
from sympy.functions.elementary.piecewise import ExprCondPair

from plagih.tree_labels import OperatorChained
from plagih.util import get_subclasses, FLOAT_PRECISION, DEBUG_DUMMY  # noqa

os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://github.com/tensorflow/tensorflow/issues/27023
import tensorflow as tf  # noqa check if ignoring warnings still required (tensorflow sends endless warnings)

tf.compat.v1.disable_eager_execution()


# tf.compat.v1.enable_eager_execution()  # sfeh possibly faster with disable


# sfeh:discuss: Min/Max is just a ordeded list. ->taking element 1, -1, ...


class AddChain(OperatorChained):
    symfun = sympy.Add
    # symfun = lambda *a: sympy.Add(a)
    expr_dmy = 'Add'
    xtype = ((float,), float)
    xtype_chain = float


class MulChain(OperatorChained):
    # sfeh:discuss: if sympy is
    expr_dmy = 'Mul'
    symfun = sympy.Mul
    xtype = ((float,), float)
    xtype_chain = float


class MinChained(OperatorChained):
    symfun = sympy.Min
    expr_dmy = 'Min'
    tflow = tf.minimum
    xtype = ((float, float), float)
    chain_xtype = float


class MaxChained(OperatorChained):
    symfun = sympy.Max
    # symfun = lambda *a: sympy.Max(a)
    expr_dmy = 'Max'
    tflow = tf.maximum
    xtype = ((float, float), float)
    chain_xtype = float


# class OrderedSelector(ChainOp):
#     """sfeh:Orders Elements with < and picks the (1, -1 or even -2, 'median')-element?"""
#     xtype = ((float,), float)
#     chain_xtype = float
#     symfun = lambda *a: sympy.Order(a)


class Piecewise(OperatorChained):
    """sfeh:discuss: the only Operator, which has tuples as input"""
    # ogclass = Ifte
    # xtype = ((float, bool), float)
    expr_dmy = 'Piecewise'
    xtype = ((ExprCondPair,), float)
    xtype_chain = ExprCondPair
    # symfun = sympy.Piecewise
    symfun = lambda *a: sympy.Piecewise(a)


class AndChained(OperatorChained):
    expr_dmy = 'And'
    xtype = ((bool,), bool)
    xtype_chain = bool
    symfun = lambda *a: sympy.And(a)


class OrChained(OperatorChained):
    xtype = ((bool,), bool)
    xtype_chain = bool
    expr_dmy = 'Or'
    symfun = lambda *a: sympy.Or(a)

# def sympy_to_tensorflow(expr_sy, d_tensors):
#     pass
#     # todo


###
