"""

"""
import re

import plagih.util
import tensorflow
import ast
import numpy as np


class NodeLabel:  # todo
    """
    Kind of abstract class; Dummy-node that holds a nlabel
    """
    nlabel = None
    arity = None
    xtype = None

    tflow = None  # not tf, might be confusing
    pycode = None
    latex = None

    expr_sym = None

    # def __init__(self, *args, **kwargs):
    #     # nlabel=None, arity=0, xtype=(tuple([None]), None), tflow=None, expr_sym=None, pycode=None, latex=None
    #     # self.nlabel = nlabel
    #     # self.arity = arity
    #     # self.xtype = xtype
    #     #
    #     # self.tflow = tflow  # not tf, might be confusing
    #     # self.nlabel = expr_sym
    #     # self.pycode = pycode
    #     # self.latex = latex
    #
    #     # new version?
    #     self.nlabel = kwargs.get('nlabel', None)
    #     self.arity = kwargs.get('arity', None)
    #     self.xtype = kwargs.get('xtype', None)
    #
    #     self.tflow = kwargs.get('tflow', None)  # not tf, might be confusing']
    #     self.nlabel = kwargs.get('expr_sym', None)
    #     self.pycode = kwargs.get('pycode', (tuple([None]), None))
    #     self.latex = kwargs.get('latex', None)

    def mutate_filter(self, *args, **kwargs):
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


class Operator(NodeLabel):
    """
    operator nodes (+, +, *, /, sin(), sign(), ...)
    inner nodes of a tree
    """
    pass

    def eval(self, *args):
        return args[self.arity:]


class Terminal(NodeLabel):
    arity = 0

    def mutate_filter(self, *args, **kwargs):
        # todo? ...only for terminal nodes
        pass


class Constant(Terminal):
    arity = 0

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def mutate_filter(self, *args, **kwargs):
        pass


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


class Observation(Terminal):
    """
    todo discuss: labels should not have a sign (-pos); just pos
    # self.name = nlabel if nlabel[0] != '-' else nlabel[1:]  # sfeh delete?
    """

    # tf_type = tensorflow.float32  # todo yeah...

    def __init__(self, nlabel):
        # todo xtype_out=float
        self.nlabel = nlabel
        self.fam, self.timeindex, _ = observation_get_family_and_time(self.nlabel, none_return=None)  # remove this self.preexpr
        self.xtype = (tuple([]), float)  # todo
        self.expr_sym = self.nlabel  # sfeh delete?
        self.index_minmax = None

        latex = f'\\text{{{self.fam}}}'  # remove this {self.preexpr}
        self.latex = (latex, latex)


class FloatConstant(Constant):
    """
    discuss: how to deal with sign of observations?
    """
    arity = 0
    otype = float
    xtype = (tuple([]), float)

    def __init__(self, nlabel):
        # super().__init__(nlabel)
        self.nlabel = nlabel
        self.latex = (f'{self.nlabel:.3f}', f'{self.nlabel:.3f}')
        self.expr_sym = self.nlabel
        self.pycode = self.nlabel

    def mutate_filter(self, filter_type='gaussian_filter', precision=6, *args, **kwargs):  # todo
        """

        """
        if filter_type == 'gaussian_filter':
            if np.random.choice(['v1', 'v2']) == 'v1' or self.nlabel == 0:
                constant = self.nlabel + np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.nlabel, 0.1)  # sfeh better adjustments?
            self.nlabel = round(constant, precision)  # todo sfeh be careful, might create zero sometimes


class BoolConstant(Constant):
    """
    True/False
    """
    xtype = (tuple([]), bool)
    tf_type = tensorflow.bool

    def __init__(self, expr):
        # super().__init__(nlabel)
        self.nlabel = expr
        self.latex = (f'{self.nlabel}', f'{self.nlabel}')
        self.expr_sym = str(self.nlabel)
        self.pycode = str(self.nlabel)
        self.regex = self.nlabel

    def mutate_filter(self, *args, **kwargs):
        """
        sfeh: filtering these is kind of nonsense
        """
        pass


class Add(Operator):
    nlabel = '+'
    arity = 2
    tflow = tensorflow.add
    latex = ('+', '{}+{}')
    expr_sym = '({} + {})'
    pycode = '({}+{})'
    regex = '\\+'
    xtype = (tuple([float, float]), float)
    #
    # def __init__(self, *args, **kwargs):
    #     pass
    #     # super().__init__(*args, **kwargs)


class Subtract(Operator):
    """

    """
    nlabel = '-'
    arity = 2
    tflow = tensorflow.subtract
    latex = ('-', '{}-{}')
    expr_sym = '({} - {})'
    regex = ''
    pycode = '({}-{})'
    xtype = (tuple([float, float]), float)


class Usub(Operator):
    nlabel = 'Usub'
    arity = 1
    tflow = tensorflow.negative
    latex = ('-', '-{}')
    expr_sym = '(-{})'
    regex = ''
    pycode = '(-{})'
    xtype = (tuple([float]), float)


class Multiply(Operator):
    nlabel = '*'
    arity = 2
    tflow = tensorflow.multiply
    latex = ('\\cdot ', '{}\\cdot {}')
    expr_sym = '({} * {})'
    pycode = '({}*{})'
    xtype = (tuple([float, float]), float)


class Divide_no_nan(Operator):
    nlabel = '/'
    arity = 2
    tflow = tensorflow.math.divide_no_nan
    latex = ('\\div ', '\\frac{}{}')
    expr_sym = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    xtype = (tuple([float, float]), float)

    def eval(self, a, b):
        return a/b


class Power(Operator):
    nlabel = '**'
    arity = 2
    tflow = tensorflow.pow
    latex = ('{{x}}^{{y}}', '{}^{}')
    expr_sym = '({}**Round({}))'
    pycode = '({}**round({}))'
    xtype = (tuple([float, float]), float)

    def eval(self, a, b):
        return a**b


class Abs(Operator):
    nlabel = 'abs'
    arity = 1
    tflow = tensorflow.abs
    latex = ('abs', '|{}|')
    expr_sym = 'abs({})'
    pycode = 'abs({})'
    xtype = (tuple([float]), float)


class Sign(Operator):
    nlabel = 'sign'
    arity = 1
    tflow = tensorflow.sign
    latex = ('sign', 'sign({})')
    expr_sym = 'sign({})'
    pycode = 'np.sign({})'
    xtype = (tuple([float]), float)


class Round(Operator):
    nlabel = 'Round'
    arity = 1
    tflow = tensorflow.round
    latex = ('round', 'round({})')
    expr_sym = 'Round({})'
    pycode = 'round({})'
    xtype = (tuple([float]), float)


class Square(Operator):
    nlabel = 'Square'
    arity = 1
    tflow = tensorflow.square
    latex = ('x^2', '{}^2')
    expr_sym = 'Square({})'
    pycode = '({})**2'
    xtype = (tuple([float]), float)


class Sqrt(Operator):
    nlabel = 'sqrt'
    arity = 1
    tflow = tensorflow.sqrt
    latex = ('\\sqrt{x}', '\\sqrt{}')
    expr_sym = 'sqrt({})'
    pycode = 'math.sqrt({})'
    xtype = (tuple([float]), float)


class Log(Operator):
    nlabel = 'log'
    arity = 1
    tflow = tensorflow.math.log
    latex = ('\\log()', '\\log{}')
    expr_sym = 'log({})'
    pycode = 'math.log({})'
    xtype = (tuple([float]), float)


class Log1p(Operator):
    nlabel = 'log1p'
    arity = 1
    tflow = tensorflow.math.log1p
    latex = ('\\log(1+x)', '\\log(1+{})')
    expr_sym = 'log1p({})'
    pycode = 'math.log1p({})'
    xtype = (tuple([float]), float)


class Cos(Operator):
    nlabel = 'cos'
    arity = 1
    tflow = tensorflow.cos
    latex = ('\\cos ', '\\cos({})')
    expr_sym = 'cos({})'
    pycode = 'math.cos({})'
    xtype = (tuple([float]), float)


class Sin(Operator):
    nlabel = 'sin'
    arity = 1
    tflow = tensorflow.sin
    latex = ('\\sin ', '\\sin({})')
    expr_sym = 'sin({})'
    pycode = 'math.sin({})'
    xtype = (tuple([float]), float)


class Tan(Operator):
    nlabel = 'tan'
    arity = 1
    tflow = tensorflow.tan
    latex = ('\\tan ', '\\tan({})')
    expr_sym = 'tan({})'
    pycode = 'math.tan({})'
    xtype = (tuple([float]), float)


class Acos(Operator):
    nlabel = 'acos'
    arity = 1
    tflow = tensorflow.acos
    latex = ('\\acos ', '\\acos({})')
    expr_sym = 'acos({})'
    pycode = 'math.acos({})'
    xtype = (tuple([float]), float)


class Asin(Operator):
    nlabel = 'asin'
    arity = 1
    tflow = tensorflow.asin
    latex = ('\\asin ', '\\asin({})')
    expr_sym = 'asin({})'
    pycode = 'math.asin({})'
    xtype = (tuple([float]), float)


class Atan(Operator):
    nlabel = 'atan'
    arity = 1
    tflow = tensorflow.atan
    latex = ('\\atan ', '\\atan({})')
    expr_sym = 'atan({})'
    pycode = 'math.atan({})'
    xtype = (tuple([float]), float)


class Tanh(Operator):
    nlabel = 'tanh'
    arity = 1
    tflow = tensorflow.tanh
    latex = ('\\tanh ', '\\tanh({})')
    expr_sym = 'tanh({})'
    pycode = 'math.tanh({})'
    xtype = (tuple([float]), float)


class And(Operator):
    nlabel = 'Andb'
    arity = 2
    tflow = tensorflow.logical_and
    latex = ('and', '({}\\wedge{})')
    expr_sym = 'Andb({}, {})'
    pycode = '({} and {})'
    xtype = (tuple([bool, bool]), bool)


class Or(Operator):
    nlabel = 'Orb'
    arity = 2
    tflow = tensorflow.logical_or
    latex = ('or', '({}\\vee{})')
    expr_sym = 'Orb({}, {})'
    pycode = '({} or {})'
    xtype = (tuple([bool, bool]), bool)


class Xor(Operator):
    nlabel = 'Xor'
    arity = 2
    tflow = tensorflow.math.logical_xor
    latex = ('\\oplus', '({}\\oplus{})')
    expr_sym = 'Xor({}, {})'
    pycode = '({} ^ {})'
    xtype = (tuple([bool, bool]), bool)


class Not(Operator):
    nlabel = 'Notb'
    arity = 1
    tflow = tensorflow.logical_not
    latex = ('\\neg', '\\neg{}')
    expr_sym = 'Notb({})'
    pycode = 'not({})'
    xtype = (tuple([bool]), bool)


class Eq(Operator):
    nlabel = '=='
    arity = 2
    tflow = tensorflow.equal
    latex = ('=', '({}={})')
    expr_sym = '({} == {})'
    pycode = '({}=={})'
    xtype = (tuple([bool, bool]), bool)


class Neq(Operator):
    nlabel = '!='
    arity = 2
    tflow = tensorflow.not_equal
    latex = ('\\neq', '({}\\neq{})')
    expr_sym = '({} != {})'
    pycode = '({}!={})'
    xtype = (tuple([bool, bool]), bool)


class Lt(Operator):
    nlabel = '<'
    arity = 2
    tflow = tensorflow.less
    latex = ('<', '{}<{}')
    expr_sym = '({} < {})'
    pycode = '({}<{})'
    xtype = (tuple([float, float]), bool)


class Le(Operator):
    nlabel = '<='
    arity = 2
    tflow = tensorflow.less_equal
    latex = ('\\leq', '{}\\leq{}')
    expr_sym = '({} <= {})'
    pycode = '({}<={})'
    xtype = (tuple([float, float]), bool)


class Gt(Operator):
    nlabel = '>'
    arity = 2
    tflow = tensorflow.greater
    latex = ('>', '{}>{}')
    expr_sym = '({} > {})'
    pycode = '({}>{})'
    xtype = (tuple([float, float]), bool)


class Ge(Operator):
    nlabel = '>='
    arity = 2
    tflow = tensorflow.greater_equal
    latex = ('\\geq', '{}\\geq {}')  # sfeh check inserted space
    expr_sym = '({} >= {})'
    pycode = '({}>={})'
    xtype = (tuple([float, float]), bool)


class Ifte(Operator):
    nlabel = 'Ifte'
    arity = 3
    tflow = tensorflow.where
    latex = ('\\text{if-then-else}', '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})')  # 'if({} then {} else {})'
    expr_sym = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    xtype = (tuple([bool, float, float]), float)


class Min(Operator):
    nlabel = 'Mini'
    arity = 2
    tflow = tensorflow.minimum
    latex = ('\\min', '\\min({}, {})')
    expr_sym = 'Mini({}, {})'
    pycode = 'min({}, {})'
    xtype = (tuple([float, float]), float)


class Max(Operator):
    nlabel = 'Maxi'
    arity = 2
    tflow = tensorflow.maximum
    latex = ('\\max', '\\max({}, {})')
    expr_sym = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    xtype = (tuple([float, float]), float)


ops = {  # 'f2f': Classical mathematical operators, evaluate from float to float
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
    'abs': Abs,  # sfeh: required... y though?
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

latex_inline = ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']


if __name__ == '__main__':
    for oo, ocls in ops.items():
        print('operators:', oo, ocls.nlabel, ocls.xtype)

    for x in [FloatConstant(0.44), BoolConstant(True)]:
        print(f'x: {x.nlabel}, {x.xtype}')

