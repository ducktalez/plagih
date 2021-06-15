"""
todo umbenennen in was neues
"""

import ast
import numpy as np
import re
import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # sfeh wasd wtf

# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling
# import tensorflow.python.util.deprecation as deprecation  # not possible on python 3.6
# deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.disable_eager_execution()  # sfeh damn what was this line good for?

"""
sfeh: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
sfeh: use function-types (-> 'kommuttative'?)
"""


class Plabel:
    label = 'None'
    arity = 0
    coolxtype = (tuple([None]), None)
    # tf = None
    # sym_str = 'None'
    # pycode = 'None'
    # latex1 = 'None'
    # latexF = 'None'

    def __str__(self):
        return str(self.label)


class Observation(Plabel):
    """
    aka input values
    """
    arity = 0

    def mutate_filter(self):
        """
        was filter_new_index
         # as default, return own index
        """
        if self.index_minmax is None:
            return
        else:
            new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
            self.obs_index = new_index
            self.name = f'{self.family}_{new_index}'

    def __init__(self, value, coolxtype_out=float, obs_indizes=None):
        fam, obs_index, prelabel = observation_get_family_and_time(value, none_return=None)

        self.label = value
        self.name = value if value[0] != '-' else value[1:]  # sfeh delete todo

        self.family = fam
        self.obs_index = obs_index  # is None when no index but 0 when

        self.xtype = coolxtype_out
        self.tf_type = tf.float32

        self.index_minmax = None

        self.obs_indizes = obs_indizes
        if obs_index is None:
            self.latex1 = f'{prelabel}\\text{{{fam}}}'
            self.latexF = f'{prelabel}\\text{{{fam}}}'
        else:
            self.latex1 = f'{prelabel}\\text{{{fam}}}_{{{obs_index}}}'
            self.latexF = f'{prelabel}\\text{{{fam}}}_{{{obs_index}}}'
        self.sym_str = value  # sfeh delete?
        self.pycode = value  # sfeh delete?


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    todo put this function somewhere where it can actually help
    """

    core_label = re.split(re_pattern, name)[0]
    if core_label[0] == '-':
        core_label = core_label[1::]
        prelabel = '-'
    else:
        prelabel = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception:
        temp_diff = none_return
    return core_label, temp_diff, prelabel


class FloatConstant(Plabel):
    """

    """
    arity = 0
    otype = float
    coolxtype = (tuple([]), float)

    def __init__(self, value):
        self.label = value
        self.latex1 = f'{value:.3f}'
        self.latexF = f'{value:.3f}'
        self.sym_str = value
        self.pycode = value
        # self.name = value if value[0] != '-' else value[1:]  # sfeh delete todo

    def mutate_filter(self, filter_type='gaussian_filter', float_decimals=6):  # todo
        if filter_type == 'gaussian_filter':
            if random.choice(['v1', 'v2']) == 'v1' or self.label == 0:
                self.label += np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.label, 0.1)  # sfeh better adjustments?
                self.label = round(constant, float_decimals)  # sfeh be careful, might create zero sometimes


class BoolConstant(Plabel):
    """

    """
    arity = 0
    otype = bool
    coolxtype = (tuple([]), bool)

    def __init__(self, value):
        self.latex1 = f'{value}'
        self.latexF = f'{value}'
        self.sym_str = value
        self.pycode = value


class EvalAction(Plabel):
    """
    - minmax for histograms
    - minmax for regression-bounded
    """
    tf_type = tf.float32  # sfeh especiall when the type is integer
    iotype = float  # sfeh todo
    coolxtype = [None, float]

    def __init__(self, name):
        self.plabel = name
        self.name = name  # delete this


class Add(Plabel):
    label = '+'
    arity = 2
    tf = tf.add
    latex1 = '+'
    latexF = '{}+{}'
    sym_str = '({} + {})'
    pycode = '({}+{})'
    coolxtype = (tuple([float, float]), float)


class Subtract(Plabel):
    """

    """
    label = '-'
    arity = 2
    tf = tf.subtract
    latex1 = '-'
    latexF = '{}-{}'
    sym_str = '({} - {})'
    pycode = '({}-{})'
    coolxtype = (tuple([float, float]), float)


class Usub(Plabel):
    label = 'Usub'
    arity = 1
    tf = tf.negative
    latex1 = '-'
    latexF = '-{}'
    sym_str = '(-{})'
    pycode = '(-{})'
    coolxtype = (tuple([float]), float)


class Multiply(Plabel):
    label = '*'
    arity = 2
    tf = tf.multiply
    latex1 = '\\cdot '
    latexF = '{}\\cdot {}'
    sym_str = '({} * {})'
    pycode = '({}*{})'
    coolxtype = (tuple([float, float]), float)


class Divide_no_nan(Plabel):
    label = '/'
    arity = 2
    tf = tf.math.divide_no_nan
    latex1 = '\\div '
    latexF = '\\frac{}{}'
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    coolxtype = (tuple([float, float]), float)


class Power(Plabel):
    label = '**'
    arity = 2
    tf = tf.pow
    latex1 = '{{x}}^{{y}}'
    latexF = '{}^{}'
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'
    coolxtype = (tuple([float, float]), float)


class Abs(Plabel):
    label = 'abs'
    arity = 1
    tf = tf.abs
    latex1 = 'abs'
    latexF = '|{}|'
    sym_str = 'abs({})'
    pycode = 'abs({})'
    coolxtype = (tuple([float]), float)


class Sign(Plabel):
    label = 'sign'
    arity = 1
    tf = tf.sign
    latex1 = 'sign'
    latexF = 'sign({})'
    sym_str = 'sign({})'
    pycode = 'np.sign({})'
    coolxtype = (tuple([float]), float)


class Round(Plabel):
    label = 'Round'
    arity = 1
    tf = tf.round
    latex1 = 'round'
    latexF = 'round({})'
    sym_str = 'Round({})'
    pycode = 'round({})'
    coolxtype = (tuple([float]), float)


class Square(Plabel):
    label = 'Square'
    arity = 1
    tf = tf.square
    latex1 = 'x^2'
    latexF = '{}^2'
    sym_str = 'Square({})'
    pycode = '({})**2'
    coolxtype = (tuple([float]), float)


class Sqrt(Plabel):
    label = 'sqrt'
    arity = 1
    tf = tf.sqrt
    latex1 = '\\sqrt{x}'
    latexF = '\\sqrt{}'
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'
    coolxtype = (tuple([float]), float)


class Log(Plabel):
    label = 'log'
    arity = 1
    tf = tf.math.log
    latex1 = '\\log()'
    latexF = '\\log{}'
    sym_str = 'log({})'
    pycode = 'math.log({})'
    coolxtype = (tuple([float]), float)


class Log1p(Plabel):
    label = 'log1p'
    arity = 1
    tf = tf.math.log1p
    latex1 = '\\log(1+x)'
    latexF = '\\log(1+{})'
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'
    coolxtype = (tuple([float]), float)


class Cos(Plabel):
    label = 'cos'
    arity = 1
    tf = tf.cos
    latex1 = '\\cos '
    latexF = '\\cos({})'
    sym_str = 'cos({})'
    pycode = 'math.cos({})'
    coolxtype = (tuple([float]), float)


class Sin(Plabel):
    label = 'sin'
    arity = 1
    tf = tf.sin
    latex1 = '\\sin '
    latexF = '\\sin({})'
    sym_str = 'sin({})'
    pycode = 'math.sin({})'
    coolxtype = (tuple([float]), float)


class Tan(Plabel):
    label = 'tan'
    arity = 1
    tf = tf.tan
    latex1 = '\\tan '
    latexF = '\\tan({})'
    sym_str = 'tan({})'
    pycode = 'math.tan({})'
    coolxtype = (tuple([float]), float)


class Acos(Plabel):
    label = 'acos'
    arity = 1
    tf = tf.acos
    latex1 = '\\acos '
    latexF = '\\acos({})'
    sym_str = 'acos({})'
    pycode = 'math.acos({})'
    coolxtype = (tuple([float]), float)


class Asin(Plabel):
    label = 'asin'
    arity = 1
    tf = tf.asin
    latex1 = '\\asin '
    latexF = '\\asin({})'
    sym_str = 'asin({})'
    pycode = 'math.asin({})'
    coolxtype = (tuple([float]), float)


class Atan(Plabel):
    label = 'atan'
    arity = 1
    tf = tf.atan
    latex1 = '\\atan '
    latexF = '\\atan({})'
    sym_str = 'atan({})'
    pycode = 'math.atan({})'
    coolxtype = (tuple([float]), float)


class Tanh(Plabel):
    label = 'tanh'
    arity = 1
    tf = tf.tanh
    latex1 = '\\tanh '
    latexF = '\\tanh({})'
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'
    coolxtype = (tuple([float]), float)


class And(Plabel):
    label = 'Andb'
    arity = 2
    tf = tf.logical_and
    latex1 = 'and'
    latexF = '({}\\wedge{})'
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'
    coolxtype = (tuple([bool, bool]), bool)


class Or(Plabel):
    label = 'Orb'
    arity = 2
    tf = tf.logical_or
    latex1 = 'or'
    latexF = '({}\\vee{})'
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'
    coolxtype = (tuple([bool, bool]), bool)


class Xor(Plabel):
    label = 'Xor'
    arity = 2
    tf = tf.math.logical_xor
    latex1 = '\\oplus'
    latexF = '({}\\oplus{})'
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'
    coolxtype = (tuple([bool, bool]), bool)


class Not(Plabel):
    label = 'Notb'
    arity = 1
    tf = tf.logical_not
    latex1 = '\\neg'
    latexF = '\\neg{}'
    sym_str = 'Notb({})'
    pycode = 'not({})'
    coolxtype = (tuple([bool]), bool)


class Eq(Plabel):
    label = '=='
    arity = 2
    tf = tf.equal
    latex1 = '='
    latexF = '({}={})'
    sym_str = '({} == {})'
    pycode = '({}=={})'
    coolxtype = (tuple([bool, bool]), bool)


class Neq(Plabel):
    label = '!='
    arity = 2
    tf = tf.not_equal
    latex1 = '\\neq'
    latexF = '({}\\neq{})'
    sym_str = '({} != {})'
    pycode = '({}!={})'
    coolxtype = (tuple([bool, bool]), bool)


class Lt(Plabel):
    label = '<'
    arity = 2
    tf = tf.less
    latex1 = '<'
    latexF = '{}<{}'
    sym_str = '({} < {})'
    pycode = '({}<{})'
    coolxtype = (tuple([float, float]), bool)


class Le(Plabel):
    label = '<='
    arity = 2
    tf = tf.less_equal
    latex1 = '\\leq'
    latexF = '{}\\leq{}'
    sym_str = '({} <= {})'
    pycode = '({}<={})'
    coolxtype = (tuple([float, float]), bool)


class Gt(Plabel):
    label = '>'
    arity = 2
    tf = tf.greater
    latex1 = '>'
    latexF = '{}>{}'
    sym_str = '({} > {})'
    pycode = '({}>{})'
    coolxtype = (tuple([float, float]), bool)


class Ge(Plabel):
    label = '>='
    arity = 2
    tf = tf.greater_equal
    latex1 = '\\geq'
    latexF = '{}\\geq{}'
    sym_str = '({} >= {})'
    pycode = '({}>={})'
    coolxtype = (tuple([float, float]), bool)


class Ifte(Plabel):
    label = 'Ifte'
    arity = 3
    tf = tf.where
    latex1 = '\\text{if-then-else}'
    latexF = '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})'  # 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    coolxtype = (tuple([bool, float, float]), float)


class Min(Plabel):
    label = 'Mini'
    arity = 2
    tf = tf.minimum
    latex1 = '\\min'
    latexF = '\\min({}, {})'
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'
    coolxtype = (tuple([float, float]), float)


class Max(Plabel):
    label = 'Maxi'
    arity = 2
    tf = tf.maximum
    latex1 = '\\max'
    latexF = '\\max({}, {})'
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    coolxtype = (tuple([float, float]), float)


op = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': Add,
    ast.Add: Add,
    '-': Subtract,
    ast.Sub: Subtract,
    'Usub': Usub,
    ast.USub: Usub,
    '*': Multiply,
    ast.Mult: Multiply,
    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': Divide_no_nan,
    ast.Div: Divide_no_nan,
    '**': Power,
    ast.Pow: Power,
    'Abs': Abs,
    'sign': Sign,
    'Round': Round,
    'Square': Square,
    'sqrt': Sqrt,
    'log': Log,  # sfeh log/ln?
    'log1p': Log1p,
    'cos': Cos,
    'sin': Sin,
    'tan': Tan,
    'acos': Acos,
    'asin': Asin,
    'atan': Atan,
    'tanh': Tanh,

    # bool->bool
    # DON'T USE tf.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': And,
    ast.And: And,
    'Orb': Or,
    ast.Or: Or,
    'Xor': Xor,
    # ast.BitXor: Xor,
    'Notb': Not,
    ast.Not: Not,

    # float->bool
    '==': Eq,
    ast.Eq: Eq,
    '!=': Neq,
    ast.NotEq: Neq,
    '<': Lt,  # a < b
    ast.Lt: Lt,
    '<=': Le,
    ast.LtE: Le,
    '>': Gt,  # a > b
    ast.Gt: Gt,
    '>=': Ge,  # a >= 1
    ast.GtE: Ge,

    'Ifte': Ifte,  # sfeh essential for evaluation
    'Mini': Min,  # with forced arity-2
    'Maxi': Max,  # with forced arity-2
}

latex_inline = ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']

# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['coolxtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


if __name__ == "__main__":
    for k, v in op.items():
        print(f'{k}\t: {v}')
