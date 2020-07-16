
import ast
from plagih.modules.import_variables import *
from plagih.modules.printing import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf


class Plus:
    """
    +
    '+': {'fun_class': '', 'fun_label': '+', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': '', 'tf': tf.add, 'latex1': '+', 'latexF': '{}', 'sym_str': '({} + {})', 'pycode': '({}+{})'},
    """
    fun_label = '+'
    fun_arity = 2
    xtype = 'f2f'

    fun_tf = tf.add
    fun_latex = '+'
    fun_sym = '({})+({})'

    def tf_code(self, *args):
        codetf = tf.add(*args)
        return codetf

    def fun_pycode(self, *args):
        codepy = '({}+{})'
        return codepy


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
op_what = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': {'fun_class': 'Add', 'fun_label': '+', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': 'add', 'tf': tf.add, 'opgroup': ['aritygroup'], 'latex1': '+', 'latexF': '{}+{}',
          'sym_str': '({} + {})', 'pycode': '({}+{})'},
    '-': {'fun_class': 'Subtract', 'fun_label': '-', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': 'subtract', 'tf': tf.subtract, 'opgroup': ['aritygroup'], 'latex1': '-', 'latexF': '{}-{}',
          'sym_str': '({} - {})', 'pycode': '({}-{})'},
    'usub': {'fun_class': 'Usub', 'fun_label': 'usub', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf_name': 'negative', 'tf': tf.negative, 'opgroup': [], 'latex1': '-', 'latexF': '-{}',
             'sym_str': '(-{})', 'pycode': '(-{})'},
    '*': {'fun_class': 'Multiply', 'fun_label': '*', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': 'multiply', 'tf': tf.multiply, 'opgroup': ['aritygroup'], 'latex1': '\\cdot ', 'latexF': '{}\\cdot {}',
          'sym_str': '({} * {})', 'pycode': '({}*{})'},
    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': {'fun_class': 'Divide_no_nan', 'fun_label': '/', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': 'math.divide_no_nan', 'tf': tf.math.divide_no_nan, 'opgroup': [], 'latex1': '\\div ', 'latexF': '\\frac{}{}',
          'sym_str': '({} / {})',
          'pycode': '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'},
    '**': {'fun_class': 'Power', 'fun_label': '**', 'arity': 2, 'xtype': 'f2f', 'c-weight': 2, 'tf_name': 'pow', 'tf': tf.pow, 'opgroup': [], 'latex1': '{{x}}^{{y}}', 'latexF': '{}^{}',  # sfeh latexf requires some testing...
           'sym_str': '({} ** {})', 'pycode': '({}*{})'},

    'abs': {'fun_class': 'Abs', 'fun_label': 'abs', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf_name': 'abs', 'tf': tf.abs, 'opgroup': [], 'latex1': 'abs', 'latexF': '|{}|',
            'sym_str': 'abs({})', 'pycode': 'abs({})'},
    'sign': {'fun_class': 'Sign', 'fun_label': 'sign', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf_name': 'sign', 'tf': tf.sign, 'opgroup': [], 'latex1': 'sign', 'latexF': 'sign({})',
             'sym_str': 'sign({})', 'pycode': 'np.sign({})'},

    'Square': {'fun_class': 'Square', 'fun_label': 'Square', 'arity': 1, 'xtype': 'f2f', 'c-weight': 2, 'tf_name': 'square', 'tf': tf.square, 'opgroup': [], 'latex1': 'x^2', 'latexF': '{}^2',
               'sym_str': 'Square({})', 'pycode': '({}**2)'},
    'sqrt': {'fun_class': 'Sqrt', 'fun_label': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'sqrt', 'tf': tf.sqrt, 'opgroup': [], 'latex1': '\\sqrt{x}', 'latexF': '\\sqrt{}',
             'sym_str': 'sqrt({})', 'pycode': 'math.sqrt({})'},

    'log': {'fun_class': 'Log', 'fun_label': 'log', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'log', 'tf': tf.log, 'opgroup': ['logs'], 'latex1': '\\log()', 'latexF': '\\log{}',
            'sym_str': 'log({})', 'pycode': 'math.log({})'},  # sfeh log/ln?
    'log1p': {'fun_class': 'Log1p', 'fun_label': 'log1p', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'log1p', 'tf': tf.log1p, 'opgroup': ['logs'], 'latex1': '\\log(1+x)', 'latexF': '\\log(1+{})',
              'sym_str': 'log1p({})', 'pycode': 'math.log1p({})'},

    'cos': {'fun_class': 'Cos', 'fun_label': 'cos', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'cos', 'tf': tf.cos, 'opgroup': ['angle'], 'latex1':  '\\cos ', 'latexF': '\\cos({})',
            'sym_str': 'cos({})', 'pycode': 'math.cos({})'},
    'sin': {'fun_class': 'Sin', 'fun_label': 'sin', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'sin', 'tf': tf.sin, 'opgroup': ['angle'], 'latex1': '\\sin ', 'latexF': '\\sin({})',
            'sym_str': 'sin({})', 'pycode': 'math.sin({})'},
    'tan': {'fun_class': 'Tan', 'fun_label': 'tan', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'tan', 'tf': tf.tan, 'opgroup': ['angle'], 'latex1': '\\tan ', 'latexF': '\\tan({})',
            'sym_str': 'tan({})', 'pycode': 'math.tan({})'},
    'acos': {'fun_class': 'Acos', 'fun_label': 'acos', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'acos', 'tf': tf.acos, 'opgroup': ['angle'], 'latex1': '\\acos ', 'latexF': '\\acos({})',
             'sym_str': 'acos({})', 'pycode': 'math.acos({})'},
    'asin': {'fun_class': 'Asin', 'fun_label': 'asin', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'asin', 'tf': tf.asin, 'opgroup': ['angle'], 'latex1': '\\asin ', 'latexF': '\\asin({})',
             'sym_str': 'asin({})', 'pycode': 'math.asin({})'},
    'atan': {'fun_class': 'Atan', 'fun_label': 'atan', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'atan', 'tf': tf.atan, 'opgroup': ['angle'], 'latex1':  '\\atan ', 'latexF': '\\atan({})',
             'sym_str': 'atan({})', 'pycode': 'math.atan({})'},
    'tanh': {'fun_class': 'Tanh', 'fun_label': 'tanh', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': 'tanh', 'tf': tf.tanh, 'opgroup': ['angle'], 'latex1': '\\tanh ', 'latexF': '\\tanh({})',
             'sym_str': 'tanh({})', 'pycode': 'math.tanh({})'},
    # 'Integer': {'fun_class': '', 'fun_label': 'Integer', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf_name': '', 'tf': tf.cast({}, tf.int32), 'latex1': None, 'latexF': '{}',
    # 'sym_str': 'N({}, )', 'pycode': 'math.tanh({})'},

    # 'b2b' Classical logical operators, evaluate from bool to bool
    # DON'T USE tf.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': {'fun_class': 'And', 'fun_label': 'Andb', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf_name': 'logical_and', 'tf': tf.logical_and, 'latex1': 'and', 'latexF': '({}\\wedge{})',
             'sym_str': 'Andb({}, {})', 'pycode': '({} and {})'},
    'Orb': {'fun_class': 'Or', 'fun_label': 'Orb', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf_name': 'logical_or', 'tf': tf.logical_or, 'latex1': 'or', 'latexF': '({}\\vee{})',
            'sym_str': 'Orb({}, {})', 'pycode': '({} or {})'},
    'Xor': {'fun_class': 'Xor', 'fun_label': 'Xor', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf_name': 'logical_xor', 'tf': tf.math.logical_xor, 'latex1': '\\oplus', 'latexF': '({}\\oplus{})',
            'sym_str': 'Xor({}, {})', 'pycode': '({} ^ {})'},
    'Notb': {'fun_class': 'Not', 'fun_label': 'Notb', 'arity': 1, 'xtype': 'b2b', 'c-weight': 0.5, 'tf_name': 'logical_not', 'tf': tf.logical_not, 'latex1': '\\neg', 'latexF': '\\neg{}',
             'sym_str': 'Notb({})', 'pycode': 'not({})'},  # not a
    '|': {'fun_class': 'SKIP', 'fun_label': '|', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf_name': 'logical_or', 'tf': tf.logical_or, 'latex1': '\\lor', 'latexF': '({}\\vee{})',
          'sym_str': '({} | {})', 'pycode': '({} or {})'},

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==': {'fun_class': 'Eq', 'fun_label': '==', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': 'equal', 'tf': tf.equal, 'latex1': '=', 'latexF': '({}={})',
           'sym_str': '({} == {})', 'pycode': '({}=={})'},
    '!=': {'fun_class': 'Neq', 'fun_label': '!=', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': 'not_equal', 'tf': tf.not_equal, 'latex1': '\\neq', 'latexF': '({}\\neq{})',
           'sym_str': '({} != {})', 'pycode': '({}!={})'},
    '<': {'fun_class': 'Lt', 'fun_label': '<', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': 'less', 'tf': tf.less, 'latex1': '<', 'latexF': '{}<{}',
          'sym_str': '({} < {})', 'pycode': '({}<{})'},  # a < b
    '<=': {'fun_class': 'Le', 'fun_label': '<=', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': 'less_equal', 'tf': tf.less_equal, 'latex1': '\\leq', 'latexF': '{}\leq{}',
           'sym_str': '({} <= {})', 'pycode': '({}<={})'},  # a <= b
    '>': {'fun_class': 'Gt', 'fun_label': '>', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': 'greater', 'tf': tf.greater, 'latex1': '>', 'latexF': '{}>{}',
          'sym_str': '({} > {})', 'pycode': '({}>{})'},  # a > b
    '>=': {'fun_class': 'Ge', 'fun_label': '>=', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': 'greater_equal', 'tf': tf.greater_equal, 'latex1': '\\geq', 'latexF': '{}\geq{}',
           'sym_str': '({} >= {})', 'pycode': '({}>={})'},  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte': {'fun_class': 'Ifte', 'fun_label': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'c-weight': 0.1, 'tf_name': 'where', 'tf': tf.compat.v2.where, 'latex1': '\\text{if-then-else}', 'latexF': 'if({} then {} else {})',
             'sym_str': 'Ifte({}, {}, {})', 'pycode': '({} if {} else {})'},
    # long version of Ifte-'pycode': 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))
    'Mini': {'fun_class': 'Min', 'fun_label': 'Mini', 'arity': 2, 'xtype': 'f2f', 'c-weight': 0.5, 'tf_name': 'minimum', 'tf': tf.minimum, 'latex1': '\\min', 'latexF': '\\min({}, {})',
             'sym_str': 'Mini({}, {})', 'pycode': 'min({}, {})'},  # with forced arity-2
    'Maxi': {'fun_class': 'Max', 'fun_label': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'c-weight': 0.5, 'tf_name': 'maximum', 'tf': tf.maximum, 'latex1': '\\max', 'latexF': '\\max({}, {})',
             'sym_str': 'Maxi({}, {})', 'pycode': 'max({}, {})'},  # with forced arity-2
}

## Currently not in use

op = {
    '+': op_what['+'],
    ast.Add: op_what['+'],
    '-': op_what['-'],
    ast.Sub: op_what['-'],
    '~': op_what['usub'],
    'usub': op_what['usub'],
    ast.USub: op_what['usub'],
    '*': op_what['*'],
    ast.Mult: op_what['*'],
    '/': op_what['/'],
    ast.Div: op_what['/'],
    '**': op_what['**'],
    ast.Pow: op_what['**'],
    'abs': op_what['abs'],
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
    '|': op_what['Orb'],
    ast.BitOr: op_what['Orb'],
    'Xor': op_what['Xor'],
    ast.BitXor: op_what['Xor'],
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

op_test = {
    # ast.BitOr
    '&': {'fun_class': '', 'fun_label': '&', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf_name': '', 'tf': tf.logical_and, 'latex1': '\\land', 'latexF': '({}\\wedge{})',
          'sym_str': '({} & {})', 'pycode': '({} and {})'},
    'Power3': {'fun_class': '', 'fun_label': '', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': '', 'tf': tf.pow, 'latex1': None, 'latexF': '{}',
               'sym_str': '({}**2)', 'pycode': '({}**2)'},
    'Nand': {'fun_class': '', 'fun_label': 'Nand', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': 'Nand({}, {})', 'pycode': 'Notb({} and {})'},
    'Xand': {'fun_class': '', 'fun_label': 'Xand', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': 'Xand({}, {})', 'pycode': 'Notb({} ^ {})'},
    'Nor': {'fun_class': '', 'fun_label': 'Nor', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_str': 'Nor({}, {})', 'pycode': None},
    'Xnor': {'fun_class': '', 'fun_label': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': 'Xnor({}, {})', 'pycode': None},

    'log2': {'fun_class': '', 'fun_label': 'log2', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': '', 'tf': False, 'latex1': None, 'latexF': '{}',
             'sym_str': 'log({})', 'pycode': 'math.log({})'},
    'log10': {'fun_class': '', 'fun_label': 'log2', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf_name': '', 'tf': False, 'latex1': None, 'latexF': '{}',
              'sym_str': 'log({})', 'pycode': 'math.log({})'},

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float': {'fun_class': '', 'fun_label': 'float', 'arity': 1, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': 'float({})'},  # not tested
    'int': {'fun_class': '', 'fun_label': 'int', 'arity': 1, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_str': 'Integer({})', 'pycode': 'int({})'},  # not tested
    'bool': {'fun_class': '', 'fun_label': 'bool', 'arity': 1, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': '', 'pycode': 'bool({})'},  # not tested

    # Not tested: Converters: Dummy operators that convert between float and bool
    'Ftob': {'fun_class': '', 'fun_label': 'Ftob', 'arity': 1, 'xtype': 'f2b', 'c-weight': 1, 'tf_name': '', 'tf': tf.bool, 'latex1': None, 'latexF': '{}',
             'sym_str': 'bool', 'pycode': 'bool({})'},  # not tested, same as bool
    'Btof': {'fun_class': '', 'fun_label': 'Btof', 'arity': 1, 'xtype': 'b2f', 'c-weight': 1, 'tf_name': '', 'tf': tf.float32, 'latex1': None, 'latexF': '{}',
             'sym_str': 'float', 'pycode': 'float({})'},  # not tested

    # Never used yet, trying to get rid of the ** function
    'Power': {'fun_class': '', 'fun_label': 'Power', 'arity': 1, 'xtype': 'f2f', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': '({})**({})', 'pycode': '({}**{})'},  # sfeh: round the exponent
    # sfeh sqrt, is only 2nd root, also 3rd-root?

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while': {'fun_class': '', 'fun_label': 'while', 'arity': 2, 'xtype': 'b2?2?', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': None},  # sfeh: not working # Condition must change in loop
    'repeat_n': {'fun_class': '', 'fun_label': 'repeat_n', 'arity': 2, 'xtype': 'b2?', 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
                 'sym_str': None, 'pycode': None},
    # sfeh: not working # repeat n time, specify n (int) by user?
}


# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['fun_label'], 1/v['c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


def xtype_get_func_list(choose_oparray, xtype=None, arity=None):
    """
    returns a function list out of the given 2d-op array randomly
    This fills in a function that fits the type of the function/terminal before.
    terminal  '2f' -> '_2f', arity
    function 'f2f' -> '_2f', arity
    function 'b2f2f' -> '_2f', arity

    Note: arity-0 functions (e.g. dummies, that calculate a problem specific value) are terminals!
    todo most of this is irrelevant. just make a better array
    """

    func_tuple_list = []
    funcs_float = [f2f, b2f, b2f2f]
    funcs_bool = [f2b, b2b]

    # arity and xtype
    if arity is not None and xtype:
        try:
            xtype_row = ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f'].index(xtype)
            func_tuple_list.extend(choose_oparray[xtype_row][arity])
        except ValueError:
            all_xfuncs = funcs_float if '2f' in xtype else funcs_bool
            func_tuple_list = sum([choose_oparray[funcs][arity] for funcs in all_xfuncs], [])

    # arity
    if arity is not None and xtype is None:
        func_tuple_list = sum([xtype_row[arity] for xtype_row in choose_oparray], [])

    # xtype
    if arity is None and xtype is not None:
        if '2f' in xtype:
            func_tuple_list = sum([sum(choose_oparray[funcs], []) for funcs in funcs_float], [])
        elif '2b' in xtype:
            func_tuple_list = sum([sum(choose_oparray[funcs], []) for funcs in funcs_bool], [])
        else:
            print_e('xtype {} is not accepted. Must be \'2f\' or \'2b\'.'.format(xtype))
            raise

    # return all functions
    if arity is None and xtype is None:
        func_tuple_list = sum(sum(choose_oparray, []), [])

    try:
        func_list = [x[0] for x in func_tuple_list]
        probability_list = [x[1] for x in func_tuple_list]
        # probability_normalised = [x/sum(probability_list) for x in probability_list]  # only required if using np-random version. can be deleted

        return func_list, probability_list

    except:
        return [], []


def oparray_from_list(functions):
    """
    Load all operators ready-to-use from a file
    """

    # rows are the function types (f2f)
    # columns are the arity
    choose_oparray = [[[], [], [], []],
                      [[], [], [], []],
                      [[], [], [], []],
                      [[], [], [], []],
                      [[], [], [], []]]

    # sfeh make this a np.array
    # p /= p.sum()

    for fun in functions:
        label = fun[0]
        try:
            probability = float(fun[1])
        except Exception:
            probability = 0

        labael_and_prob = [label, probability]
        op_info = op[label]
        arity = op_info['arity']  # arity = int(fun[1])
        xtype = op_info['xtype']

        if xtype == 'f2f':
            choose_oparray[f2f][arity].append(labael_and_prob)
        elif xtype == 'f2b':
            choose_oparray[f2b][arity].append(labael_and_prob)
        elif xtype == 'b2b':
            choose_oparray[b2b][arity].append(labael_and_prob)
        elif xtype == 'b2f':
            choose_oparray[b2f][arity].append(labael_and_prob)
        elif xtype == 'b2f2f':
            choose_oparray[b2f2f][arity].append(labael_and_prob)

    # sfeh all the above was from an old version
    choose_oparray2 = {

        # # all operators (not needed (?))
        # None: {0: [], 1: [], 2: [], 3: [], None: []},

        # all operators with a certain xtype-result
        '2f': {0: [], 1: [], 2: [], 3: [], None: []},
        '2b': {0: [], 1: [], 2: [], None: []},

        # all operators for point mutation
        'f2f': {1: [], 2: [], None: []},
        'f2b': {1: [], 2: [], None: []},
        'b2b': {1: [], 2: [], None: []},
        'b2f': {1: []},
        'b2f2f': {3: [], None: []}
    }

    for xtype_dummy, vals in choose_oparray2.items():
        for ari_dummy in vals.keys():
            choose_oparray2[xtype_dummy][ari_dummy] = xtype_get_func_list(choose_oparray, xtype=xtype_dummy, arity=ari_dummy)

    return choose_oparray2


# def get_all_oparrays():
#     oplist = []
#     for key in op_what.keys():
#         oplist.append((key, None))
#     oparray = oparray_from_list(oplist)
#     return oparray


expr_raw_infix = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=', '&', '|']  # sfeh / is removed for
