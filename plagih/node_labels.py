"""

"""

from plagih.fitness_kernel import *

import random
import tensorflow
import ast
import re

tensorflow.compat.v1.disable_eager_execution()  # sfeh damn what was this line good for?


# @dataclass  # sfeh maybe
class NodeLabel:  # todo
    """
    Kind of abstract class; Dummy-node that holds a expr
    """
    expr = None
    arity = None
    xtype = None

    tflow = None  # not tf, might be confusing
    sym_expr = None
    pycode = None
    latex = None

    expr_sym = None

    # def __init__(self, *args, **kwargs):
    #     # expr=None, arity=0, xtype=(tuple([None]), None), tflow=None, expr_sym=None, pycode=None, latex=None
    #     # self.expr = expr
    #     # self.arity = arity
    #     # self.xtype = xtype
    #     #
    #     # self.tflow = tflow  # not tf, might be confusing
    #     # self.sym_expr = expr_sym
    #     # self.pycode = pycode
    #     # self.latex = latex
    #
    #     # new version?
    #     self.expr = kwargs.get('expr', None)
    #     self.arity = kwargs.get('arity', None)
    #     self.xtype = kwargs.get('xtype', None)
    #
    #     self.tflow = kwargs.get('tflow', None)  # not tf, might be confusing']
    #     self.sym_expr = kwargs.get('expr_sym', None)
    #     self.pycode = kwargs.get('pycode', (tuple([None]), None))
    #     self.latex = kwargs.get('latex', None)

    def mutate_filter(self):
        """
        was filter_new_index
         # as default, return own index
        """
        # if self.index_minmax is None:
        pass

    def mutate_point(self):
        """

        """
        # if self.index_minmax is None:
        pass


class BuildNode(NodeLabel):
    """

    """
    pass


class Operator(NodeLabel):
    """
    operator nodes (+, +, *, /, sin(), sign(), ...)
    inner nodes of a tree
    """
    pass


class Terminal(NodeLabel):

    def mutate_filter(self):
        # todo? ...only for terminal nodes
        pass


class Constant(Terminal):
    arity = 0

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)


class Observation(Terminal):
    """

    todo discuss: labels should not have a sign (-pos); just pos
    # self.name = expr if expr[0] != '-' else expr[1:]  # sfeh delete?
    """
    tf_type = tensorflow.float32  # todo yeah...

    def __init__(self, expr, *args, **kwargs):
        # todo xtype_out=float
        # super().__init__(*args, **kwargs)  # todo maybe?
        self.expr = expr
        self.fam, self.obs_index, _ = observation_get_family_and_time(expr, none_return=None)  # remove this self.preexpr
        self.xtype = float  # todo
        self.sym_str = expr  # sfeh delete?
        self.index_minmax = None

        latex = f'\\text{{{self.fam}}}'  # remove this {self.preexpr}
        self.latex = (latex, latex)


# class ObservationIndex(Observation):
#     """
#     todo
#     """
#
#     def __init__(self, expr, xtype=float, obs_indizes=None):
#         # super().__init__(expr, xtype)
#         self.obs_indizes = obs_indizes
#         latex = f'\\text{{{self.fam}}}_{{{self.obs_index}}}'  # remove this {self.preexpr}
#         self.latex = (latex, latex)  # remove this {self.preexpr}
#
#     def mutate_filter(self):
#         new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
#         self.obs_index = new_index
#         self.name = f'{self.fam}_{new_index}'


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    todo put this function somewhere where it can actually help
    """

    core_expr = re.split(re_pattern, name)[0]
    if core_expr[0] == '-':
        core_expr = core_expr[1::]
        preexpr = '-'
    else:
        preexpr = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception:
        temp_diff = none_return
    return core_expr, temp_diff, preexpr


class FloatConstant(Constant):
    """
    discuss: how to deal with sign of observations?
    """
    arity = 0
    otype = float
    xtype = (tuple([]), float)

    def __init__(self, expr):
        # super().__init__(expr)
        self.expr = expr
        self.latex = (f'{expr:.3f}', f'{expr:.3f}')
        self.sym_str = expr
        self.pycode = expr

    def mutate_filter(self, filter_type='gaussian_filter', precision=6):  # todo
        if filter_type == 'gaussian_filter':
            if random.choice(['v1', 'v2']) == 'v1' or self.expr == 0:
                self.expr += np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.expr, 0.1)  # sfeh better adjustments?
                self.expr = round(constant, precision)  # sfeh be careful, might create zero sometimes


class BoolConstant(Constant):
    """

    """
    xtype = (tuple([]), bool)
    tf_type = tensorflow.bool

    def __init__(self, expr):
        # super().__init__(expr)
        self.expr = expr
        self.latex = (f'{expr}', f'{expr}')
        self.sym_str = expr
        self.pycode = expr


class EvalAction:
    """
    todo plabel/labelNode? daut it.
    - minmax for histograms
    - minmax for regression-bounded
    """
    tf_type = tensorflow.float32  # sfeh especiall when the type is integer
    xtype = (None, float)

    def __init__(self, name):
        self.expr = name
        self.name = name  # delete this


class Add(Operator):
    expr = '+'
    arity = 2
    tflow = tensorflow.add
    latex = ('+', '{}+{}')
    sym_str = '({} + {})'
    pycode = '({}+{})'
    xtype = (tuple([float, float]), float)
    #
    # def __init__(self, *args, **kwargs):
    #     pass
    #     # super().__init__(*args, **kwargs)


class Subtract(Operator):
    """

    """
    expr = '-'
    arity = 2
    tflow = tensorflow.subtract
    latex = ('-', '{}-{}')
    sym_str = '({} - {})'
    pycode = '({}-{})'
    xtype = (tuple([float, float]), float)


class Usub(Operator):
    expr = 'Usub'
    arity = 1
    tflow = tensorflow.negative
    latex = ('-', '-{}')
    sym_str = '(-{})'
    pycode = '(-{})'
    xtype = (tuple([float]), float)


class Multiply(Operator):
    expr = '*'
    arity = 2
    tflow = tensorflow.multiply
    latex = ('\\cdot ', '{}\\cdot {}')
    sym_str = '({} * {})'
    pycode = '({}*{})'
    xtype = (tuple([float, float]), float)


class Divide_no_nan(Operator):
    expr = '/'
    arity = 2
    tflow = tensorflow.math.divide_no_nan
    latex = ('\\div ', '\\frac{}{}')
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    xtype = (tuple([float, float]), float)


class Power(Operator):
    expr = '**'
    arity = 2
    tflow = tensorflow.pow
    latex = ('{{x}}^{{y}}', '{}^{}')
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'
    xtype = (tuple([float, float]), float)


class Abs(Operator):
    expr = 'abs'
    arity = 1
    tflow = tensorflow.abs
    latex = ('abs', '|{}|')
    sym_expr = 'abs({})'
    pycode = 'abs({})'
    xtype = (tuple([float]), float)


class Sign(Operator):
    expr = 'sign'
    arity = 1
    tflow = tensorflow.sign
    latex = ('sign', 'sign({})')
    sym_str = 'sign({})'
    pycode = 'np.sign({})'
    xtype = (tuple([float]), float)


class Round(Operator):
    expr = 'Round'
    arity = 1
    tflow = tensorflow.round
    latex = ('round', 'round({})')
    sym_str = 'Round({})'
    pycode = 'round({})'
    xtype = (tuple([float]), float)


class Square(Operator):
    expr = 'Square'
    arity = 1
    tflow = tensorflow.square
    latex = ('x^2', '{}^2')
    sym_str = 'Square({})'
    pycode = '({})**2'
    xtype = (tuple([float]), float)


class Sqrt(Operator):
    expr = 'sqrt'
    arity = 1
    tflow = tensorflow.sqrt
    latex = ('\\sqrt{x}', '\\sqrt{}')
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'
    xtype = (tuple([float]), float)


class Log(Operator):
    expr = 'log'
    arity = 1
    tflow = tensorflow.math.log
    latex = ('\\log()', '\\log{}')
    sym_str = 'log({})'
    pycode = 'math.log({})'
    xtype = (tuple([float]), float)


class Log1p(Operator):
    expr = 'log1p'
    arity = 1
    tflow = tensorflow.math.log1p
    latex = ('\\log(1+x)', '\\log(1+{})')
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'
    xtype = (tuple([float]), float)


class Cos(Operator):
    expr = 'cos'
    arity = 1
    tflow = tensorflow.cos
    latex = ('\\cos ', '\\cos({})')
    sym_str = 'cos({})'
    pycode = 'math.cos({})'
    xtype = (tuple([float]), float)


class Sin(Operator):
    expr = 'sin'
    arity = 1
    tflow = tensorflow.sin
    latex = ('\\sin ', '\\sin({})')
    sym_str = 'sin({})'
    pycode = 'math.sin({})'
    xtype = (tuple([float]), float)


class Tan(Operator):
    expr = 'tan'
    arity = 1
    tflow = tensorflow.tan
    latex = ('\\tan ', '\\tan({})')
    sym_str = 'tan({})'
    pycode = 'math.tan({})'
    xtype = (tuple([float]), float)


class Acos(Operator):
    expr = 'acos'
    arity = 1
    tflow = tensorflow.acos
    latex = ('\\acos ', '\\acos({})')
    sym_str = 'acos({})'
    pycode = 'math.acos({})'
    xtype = (tuple([float]), float)


class Asin(Operator):
    expr = 'asin'
    arity = 1
    tflow = tensorflow.asin
    latex = ('\\asin ', '\\asin({})')
    sym_str = 'asin({})'
    pycode = 'math.asin({})'
    xtype = (tuple([float]), float)


class Atan(Operator):
    expr = 'atan'
    arity = 1
    tflow = tensorflow.atan
    latex = ('\\atan ', '\\atan({})')
    sym_str = 'atan({})'
    pycode = 'math.atan({})'
    xtype = (tuple([float]), float)


class Tanh(Operator):
    expr = 'tanh'
    arity = 1
    tflow = tensorflow.tanh
    latex = ('\\tanh ', '\\tanh({})')
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'
    xtype = (tuple([float]), float)


class And(Operator):
    expr = 'Andb'
    arity = 2
    tflow = tensorflow.logical_and
    latex = ('and', '({}\\wedge{})')
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'
    xtype = (tuple([bool, bool]), bool)


class Or(Operator):
    expr = 'Orb'
    arity = 2
    tflow = tensorflow.logical_or
    latex = ('or', '({}\\vee{})')
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'
    xtype = (tuple([bool, bool]), bool)


class Xor(Operator):
    expr = 'Xor'
    arity = 2
    tflow = tensorflow.math.logical_xor
    latex = ('\\oplus', '({}\\oplus{})')
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'
    xtype = (tuple([bool, bool]), bool)


class Not(Operator):
    expr = 'Notb'
    arity = 1
    tflow = tensorflow.logical_not
    latex = ('\\neg', '\\neg{}')
    sym_str = 'Notb({})'
    pycode = 'not({})'
    xtype = (tuple([bool]), bool)


class Eq(Operator):
    expr = '=='
    arity = 2
    tflow = tensorflow.equal
    latex = ('=', '({}={})')
    sym_str = '({} == {})'
    pycode = '({}=={})'
    xtype = (tuple([bool, bool]), bool)


class Neq(Operator):
    expr = '!='
    arity = 2
    tflow = tensorflow.not_equal
    latex = ('\\neq', '({}\\neq{})')
    sym_str = '({} != {})'
    pycode = '({}!={})'
    xtype = (tuple([bool, bool]), bool)


class Lt(Operator):
    expr = '<'
    arity = 2
    tflow = tensorflow.less
    latex = ('<', '{}<{}')
    sym_str = '({} < {})'
    pycode = '({}<{})'
    xtype = (tuple([float, float]), bool)


class Le(Operator):
    expr = '<='
    arity = 2
    tflow = tensorflow.less_equal
    latex = ('\\leq', '{}\\leq{}')
    sym_str = '({} <= {})'
    pycode = '({}<={})'
    xtype = (tuple([float, float]), bool)


class Gt(Operator):
    expr = '>'
    arity = 2
    tflow = tensorflow.greater
    latex = ('>', '{}>{}')
    sym_str = '({} > {})'
    pycode = '({}>{})'
    xtype = (tuple([float, float]), bool)


class Ge(Operator):
    expr = '>='
    arity = 2
    tflow = tensorflow.greater_equal
    latex = ('\\geq', '{}\\geq {}')  # sfeh check inserted space
    sym_str = '({} >= {})'
    pycode = '({}>={})'
    xtype = (tuple([float, float]), bool)


class Ifte(Operator):
    expr = 'Ifte'
    arity = 3
    tflow = tensorflow.where
    latex = ('\\text{if-then-else}', '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})')  # 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    xtype = (tuple([bool, float, float]), float)


class Min(Operator):
    expr = 'Mini'
    arity = 2
    tflow = tensorflow.minimum
    latex = ('\\min', '\\min({}, {})')
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'
    xtype = (tuple([float, float]), float)


class Max(Operator):
    expr = 'Maxi'
    arity = 2
    tflow = tensorflow.maximum
    latex = ('\\max', '\\max({}, {})')
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    xtype = (tuple([float, float]), float)


op = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': Add,
    ast.Add: Add,
    '-': Subtract,
    ast.Sub: Subtract,
    'Usub': Usub,
    ast.USub: Usub,
    '*': Multiply,
    ast.Mult: Multiply,
    # Division: SAFE division by zero! -->tensorflow.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
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
    # DON'T USE tensorflow.bitwise.bitwise_and
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


if __name__ == '__main__':
    for oo, ocls in op.items():
        print('operators:', oo, ocls.expr, ocls.xtype)

    for x in [FloatConstant(0.44), BoolConstant(True)]:
        print(f'x: {x}: {x.expr}, {x.xtype}')
