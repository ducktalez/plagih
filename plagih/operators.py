import ast
from plagih.plagih_data import *
from plagih.plagih_data import observation_get_family_and_time

# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling
# import tensorflow.python.util.deprecation as deprecation  # not possible on python 3.6
# deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # sfeh wasd wtf

"""
op: Dict to work as 'Database' for every expression-bit and its features
- KEY: expression-bit: can occur in various forms, which group into following uses:
    - ast.KEY: Found by pythons 'ast' when inline op like [+, -, *, ...] occurs
    - 'KEY': Found by pythons 'ast' when non-inline functions are found as string
    -> some operators are identical, but occur several times, e. g. ('And', '&', ast.BitAnd)
- VALUE: several features that have to be regarded. Some are irrelevant or have to be done.
    - name: (irrelevant) information what this function actually does. identical operators can be found like this.
    - arity: Amount of inputs an operator has to have. This must always be constant (reason why pythons min/max does not work)
    - tf: Tensorflow graph representation for the key. Was separated earlier. (!): [tf.bool, tf.float] are just variables of the used tf.cast function.
    - gpbp: Genetic Programming Backpropagation: An open idea from sfeh to introduce backpropagation for genetic operators.
    - latex: For visualizing trees with latex, these representations might look better

Features should always be defined, even though they might not occur at all. If not used, they CAN be filled with dummy values like äöü
Some, which are known of not being used yet are commented with '# not tested' or '# not used'
sfeh: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
sfeh: use function-types (-> 'kommuttative'?)
"""


class Plabel:
    nlabel = 'None'
    arity = 0
    tf = None
    sym_str = 'None'
    coolxtype = (tuple([None]), None)
    pycode = 'None'
    latex1 = 'None'
    latexF = 'None'

    def __str__(self):
        return str(self.nlabel)


class Add(Plabel):
    nlabel = '+'
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
    nlabel = '-'
    arity = 2
    tf = tf.subtract
    latex1 = '-'
    latexF = '{}-{}'
    sym_str = '({} - {})'
    pycode = '({}-{})'
    coolxtype = (tuple([float, float]), float)


class Usub(Plabel):
    nlabel = 'Usub'
    arity = 1
    tf = tf.negative
    latex1 = '-'
    latexF = '-{}'
    sym_str = '(-{})'
    pycode = '(-{})'
    coolxtype = (tuple([float]), float)


class Multiply(Plabel):
    nlabel = '*'
    arity = 2
    tf = tf.multiply
    latex1 = '\\cdot '
    latexF = '{}\\cdot {}'
    sym_str = '({} * {})'
    pycode = '({}*{})'
    coolxtype = (tuple([float, float]), float)


class Divide_no_nan(Plabel):
    nlabel = '/'
    arity = 2
    tf = tf.math.divide_no_nan
    latex1 = '\\div '
    latexF = '\\frac{}{}'
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    coolxtype = (tuple([float, float]), float)


class Power(Plabel):
    nlabel = '**'
    arity = 2
    tf = tf.pow
    latex1 = '{{x}}^{{y}}'
    latexF = '{}^{}'
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'
    coolxtype = (tuple([float, float]), float)


class Abs(Plabel):
    nlabel = 'abs'
    arity = 1
    tf = tf.abs
    latex1 = 'abs'
    latexF = '|{}|'
    sym_str = 'abs({})'
    pycode = 'abs({})'
    coolxtype = (tuple([float]), float)


class Sign(Plabel):
    nlabel = 'sign'
    arity = 1
    tf = tf.sign
    latex1 = 'sign'
    latexF = 'sign({})'
    sym_str = 'sign({})'
    pycode = 'np.sign({})'
    coolxtype = (tuple([float]), float)


class Round(Plabel):
    nlabel = 'Round'
    arity = 1
    tf = tf.round
    latex1 = 'round'
    latexF = 'round({})'
    sym_str = 'Round({})'
    pycode = 'round({})'
    coolxtype = (tuple([float]), float)


class Square(Plabel):
    nlabel = 'Square'
    arity = 1
    tf = tf.square
    latex1 = 'x^2'
    latexF = '{}^2'
    sym_str = 'Square({})'
    pycode = '({})**2'
    coolxtype = (tuple([float]), float)


class Sqrt(Plabel):
    nlabel = 'sqrt'
    arity = 1
    tf = tf.sqrt
    latex1 = '\\sqrt{x}'
    latexF = '\\sqrt{}'
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'
    coolxtype = (tuple([float]), float)


class Log(Plabel):
    nlabel = 'log'
    arity = 1
    tf = tf.math.log
    latex1 = '\\log()'
    latexF = '\\log{}'
    sym_str = 'log({})'
    pycode = 'math.log({})'
    coolxtype = (tuple([float]), float)


class Log1p(Plabel):
    nlabel = 'log1p'
    arity = 1
    tf = tf.math.log1p
    latex1 = '\\log(1+x)'
    latexF = '\\log(1+{})'
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'
    coolxtype = (tuple([float]), float)


class Cos(Plabel):
    nlabel = 'cos'
    arity = 1
    tf = tf.cos
    latex1 = '\\cos '
    latexF = '\\cos({})'
    sym_str = 'cos({})'
    pycode = 'math.cos({})'
    coolxtype = (tuple([float]), float)


class Sin(Plabel):
    nlabel = 'sin'
    arity = 1
    tf = tf.sin
    latex1 = '\\sin '
    latexF = '\\sin({})'
    sym_str = 'sin({})'
    pycode = 'math.sin({})'
    coolxtype = (tuple([float]), float)


class Tan(Plabel):
    nlabel = 'tan'
    arity = 1
    tf = tf.tan
    latex1 = '\\tan '
    latexF = '\\tan({})'
    sym_str = 'tan({})'
    pycode = 'math.tan({})'
    coolxtype = (tuple([float]), float)


class Acos(Plabel):
    nlabel = 'acos'
    arity = 1
    tf = tf.acos
    latex1 = '\\acos '
    latexF = '\\acos({})'
    sym_str = 'acos({})'
    pycode = 'math.acos({})'
    coolxtype = (tuple([float]), float)


class Asin(Plabel):
    nlabel = 'asin'
    arity = 1
    tf = tf.asin
    latex1 = '\\asin '
    latexF = '\\asin({})'
    sym_str = 'asin({})'
    pycode = 'math.asin({})'
    coolxtype = (tuple([float]), float)


class Atan(Plabel):
    nlabel = 'atan'
    arity = 1
    tf = tf.atan
    latex1 = '\\atan '
    latexF = '\\atan({})'
    sym_str = 'atan({})'
    pycode = 'math.atan({})'
    coolxtype = (tuple([float]), float)


class Tanh(Plabel):
    nlabel = 'tanh'
    arity = 1
    tf = tf.tanh
    latex1 = '\\tanh '
    latexF = '\\tanh({})'
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'
    coolxtype = (tuple([float]), float)


class And(Plabel):
    nlabel = 'Andb'
    arity = 2
    tf = tf.logical_and
    latex1 = 'and'
    latexF = '({}\\wedge{})'
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'
    coolxtype = (tuple([bool, bool]), bool)


class Or(Plabel):
    nlabel = 'Orb'
    arity = 2
    tf = tf.logical_or
    latex1 = 'or'
    latexF = '({}\\vee{})'
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'
    coolxtype = (tuple([bool, bool]), bool)


class Xor(Plabel):
    nlabel = 'Xor'
    arity = 2
    tf = tf.math.logical_xor
    latex1 = '\\oplus'
    latexF = '({}\\oplus{})'
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'
    coolxtype = (tuple([bool, bool]), bool)


class Not(Plabel):
    nlabel = 'Notb'
    arity = 1
    tf = tf.logical_not
    latex1 = '\\neg'
    latexF = '\\neg{}'
    sym_str = 'Notb({})'
    pycode = 'not({})'
    coolxtype = (tuple([bool]), bool)


class Eq(Plabel):
    nlabel = '=='
    arity = 2
    tf = tf.equal
    latex1 = '='
    latexF = '({}={})'
    sym_str = '({} == {})'
    pycode = '({}=={})'
    coolxtype = (tuple([bool, bool]), bool)


class Neq(Plabel):
    nlabel = '!='
    arity = 2
    tf = tf.not_equal
    latex1 = '\\neq'
    latexF = '({}\\neq{})'
    sym_str = '({} != {})'
    pycode = '({}!={})'
    coolxtype = (tuple([bool, bool]), bool)


class Lt(Plabel):
    nlabel = '<'
    arity = 2
    tf = tf.less
    latex1 = '<'
    latexF = '{}<{}'
    sym_str = '({} < {})'
    pycode = '({}<{})'
    coolxtype = (tuple([float, float]), bool)


class Le(Plabel):
    nlabel = '<='
    arity = 2
    tf = tf.less_equal
    latex1 = '\\leq'
    latexF = '{}\\leq{}'
    sym_str = '({} <= {})'
    pycode = '({}<={})'
    coolxtype = (tuple([float, float]), bool)


class Gt(Plabel):
    nlabel = '>'
    arity = 2
    tf = tf.greater
    latex1 = '>'
    latexF = '{}>{}'
    sym_str = '({} > {})'
    pycode = '({}>{})'
    coolxtype = (tuple([float, float]), bool)


class Ge(Plabel):
    nlabel = '>='
    arity = 2
    tf = tf.greater_equal
    latex1 = '\\geq'
    latexF = '{}\\geq{}'
    sym_str = '({} >= {})'
    pycode = '({}>={})'
    coolxtype = (tuple([float, float]), bool)


class Ifte(Plabel):
    nlabel = 'Ifte'
    arity = 3
    tf = tf.where
    latex1 = '\\text{if-then-else}'
    latexF = '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})'  # 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    coolxtype = ([bool, float, float], float)


class Min(Plabel):
    nlabel = 'Mini'
    arity = 2
    tf = tf.minimum
    latex1 = '\\min'
    latexF = '\\min({}, {})'
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'
    coolxtype = (tuple([float, float]), float)


class Max(Plabel):
    nlabel = 'Maxi'
    arity = 2
    tf = tf.maximum
    latex1 = '\\max'
    latexF = '\\max({}, {})'
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    coolxtype = (tuple([float, float]), float)


class Ploperator(Plabel):
    pass


class Observation(Plabel):
    """

    """
    arity = 0

    def filter_new_index(self):
        if self.index_minmax is None:
            return
        else:
            new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
            self.obs_index = new_index
            self.name = f'{self.family}_{new_index}'

    def __init__(self, name, iotype=float, obs_indizes=None):
        self.label = name
        self.name = name  # sfeh delete
        self.nlabel = name

        obs_family, obs_index, prelabel = observation_get_family_and_time(name, none_return=None)
        self.family = obs_family
        self.obs_index = obs_index  # is None when no index but 0 when

        self.iotype = iotype
        self.tf_type = tf.float32

        self.index_minmax = None
        self.fun_filter_index = lambda: None  # as default, return own index

        self.obs_indizes = obs_indizes
        if obs_index is None:
            self.latex1 = f'{prelabel}\\text{{{obs_family}}}'
            self.latexF = f'{prelabel}\\text{{{obs_family}}}'
        else:
            self.latex1 = f'{prelabel}\\text{{{obs_family}}}_{{{obs_index}}}'
            self.latexF = f'{prelabel}\\text{{{obs_family}}}_{{{obs_index}}}'
        self.sym_str = name  # sfeh delete?
        self.pycode = name  # sfeh delete?


class FloatConstant(Plabel):
    """
    
    """
    arity = 0
    otype = float
    coolxtype = (tuple([]), float)

    def __init__(self, value):
        self.n_label = value
        self.latex1 = f'{value:.3f}'
        self.latexF = f'{value:.3f}'
        self.sym_str = value
        self.pycode = value


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


op_what = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': Add,
    '-': Subtract,
    'Usub': Usub,
    '*': Multiply,
    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': Divide_no_nan,
    '**': Power,
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
    # 'Integer': {'fun_class': '', 'nlabel': 'Integer', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 0.5, 'tf_name': '', 'tf': tf.cast({}, tf.int32), 'latex1': None, 'latexF': '{}',
    # 'sym_reduce': None, 'sym_str': 'N({}, )', 'pycode': 'math.tanh({})'},

    # 'b2b' Classical logical operators, evaluate from bool to bool
    # DON'T USE tf.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': And,
    'Orb': Or,
    'Xor': Xor,
    'Notb': Not,

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==': Eq,
    '!=': Neq,
    '<': Lt,  # a < b
    '<=': Le,
    '>': Gt,  # a > b
    '>=': Ge,  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte': Ifte,  # sfeh essential for evaluation
    # long version of Ifte-'pycode': 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))
    'Mini': Min,  # with forced arity-2
    'Maxi': Max,  # with forced arity-2
}

## Currently not in use

op = {
    '+': op_what['+'],
    ast.Add: op_what['+'],
    '-': op_what['-'],
    ast.Sub: op_what['-'],
    '~': op_what['Usub'],
    'Usub': op_what['Usub'],
    'usub': op_what['Usub'],  # delete this sfeh
    ast.USub: op_what['Usub'],
    'Round': op_what['Round'],
    '*': op_what['*'],
    ast.Mult: op_what['*'],
    '/': op_what['/'],
    ast.Div: op_what['/'],
    '**': op_what['**'],
    ast.Pow: op_what['**'],
    'abs': op_what['Abs'],  # delete this
    'Abs': op_what['Abs'],
    'sign': op_what['sign'],
    'Square': op_what['Square'],
    'sqrt': op_what['sqrt'],
    'log': op_what['log'],
    'log1p': op_what['log1p'],
    'cos': op_what['cos'],
    'sin': op_what['sin'],
    'tan': op_what['tan'],
    'acos': op_what['acos'],
    'asin': op_what['asin'],
    'atan': op_what['atan'],
    'tanh': op_what['tanh'],
    'Andb': op_what['Andb'],
    'And': op_what['Andb'],
    ast.And: op_what['Andb'],
    '&': op_what['Orb'],
    ast.BitAnd: op_what['Andb'],
    'Orb': op_what['Orb'],
    'Or': op_what['Orb'],
    ast.Or: op_what['Orb'],
    'Xor': op_what['Xor'],
    'Notb': op_what['Notb'],
    ast.Not: op_what['Notb'],
    '==': op_what['=='],
    ast.Eq: op_what['=='],
    '!=': op_what['!='],
    ast.NotEq: op_what['!='],
    '<': op_what['<'],
    ast.Lt: op_what['<'],
    '<=': op_what['<='],
    ast.LtE: op_what['<='],
    '>': op_what['>'],
    ast.Gt: op_what['>'],
    '>=': op_what['>='],
    ast.GtE: op_what['>='],
    'Ifte': op_what['Ifte'],
    'Mini': op_what['Mini'],
    'Maxi': op_what['Maxi'],
}

latex_inline = ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']

op_test = {
    '&': {'fun_class': '', 'nlabel': '&', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 0.5, 'tf_name': '', 'tf': tf.logical_and, 'latex1': '\\land', 'latexF': '({}\\wedge{})',
          'sym_reduce': None, 'sym_str': '({} & {})', 'pycode': '({} and {})'},
    'Power3': {'fun_class': '', 'nlabel': '', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 3, 'tf_name': '', 'tf': tf.pow, 'latex1': None, 'latexF': '{}',
               'sym_reduce': None, 'sym_str': '({}**2)', 'pycode': '({}**2)'},
    'Nand': {'fun_class': '', 'nlabel': 'Nand', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'Nand({}, {})', 'pycode': 'Notb({} and {})'},
    'Xand': {'fun_class': '', 'nlabel': 'Xand', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'Xand({}, {})', 'pycode': 'Notb({} ^ {})'},
    'Nor': {'fun_class': '', 'nlabel': 'Nor', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_reduce': None, 'sym_str': 'Nor({}, {})', 'pycode': None},
    'Xnor': {'fun_class': '', 'nlabel': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'Xnor({}, {})', 'pycode': None},

    'log2': {'fun_class': '', 'nlabel': 'log2', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 3, 'tf_name': '', 'tf': False, 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'log({})', 'pycode': 'math.log({})'},
    'log10': {'fun_class': '', 'nlabel': 'log2', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 3, 'tf_name': '', 'tf': False, 'latex1': None, 'latexF': '{}',
              'sym_reduce': None, 'sym_str': 'log({})', 'pycode': 'math.log({})'},

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float': {'fun_class': '', 'nlabel': 'float', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': 'float({})'},  # not tested
    'int': {'fun_class': '', 'nlabel': 'int', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_reduce': None, 'sym_str': 'Integer({})', 'pycode': 'int({})'},  # not tested
    'bool': {'fun_class': '', 'nlabel': 'bool', 'arity': 1, 'xtype': 'f2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': '', 'pycode': 'bool({})'},  # not tested

    # sfeh sqrt, is only 2nd root, also 3rd-root?

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while': {'fun_class': '', 'nlabel': 'while', 'arity': 2, 'xtype': 'b2?2?', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': None},  # sfeh: not working # Condition must change in loop

    # sfeh: not working # repeat n time, specify n (int) by user?
}

# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['nlabel'], 1/v['coolxtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


if __name__ == "__main__":
    for k, v in op.items():
        print(f'Operators: {k}, {v}')
