
import ast
from plagih.modules.import_variables import *
from plagih.modules.printing import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf


class P_Plus:
    """
    +
    '+': {'fun': '+', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf': tf.add, 'latex1': '$+$', 'latexF': '{}', 'sym_str': '({} + {})', 'pycode': lambda *args: '({}+{})'.format(args[0], args[1])},
    """
    fun_label = '+'
    fun_arity = 2
    xtype = 'f2f'

    fun_tf = tf.add
    fun_latex = '$+$'
    fun_sym = '({})+({})'

    def tf_code(self, a, b):
        codetf = tf.add(args[0], args[1])
        return codetf

    def fun_pycode(self, a, b):
        codepy = '({}+{})'.format(args[0], args[1])
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
    '+': {'fun': '+', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf': tf.add, 'opgroup': ['aritygroup'], 'latex1': '$+$', 'latexF': '{}+{}',
          'sym_str': '({} + {})', 'pycode': lambda *args: '({}+{})'.format(args[0], args[1])},
    '-': {'fun': '-', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf': tf.subtract, 'opgroup': ['aritygroup'], 'latex1': '$-$', 'latexF': '{}-{}',
          'sym_str': '({} - {})', 'pycode': lambda *args: '({}-{})'.format(args[0], args[1])},
    'usub': {'fun': 'usub', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf': tf.negative, 'opgroup': [], 'latex1': '$-$', 'latexF': '-{}',
             'sym_str': '(-{})', 'pycode': lambda a: '(-{})'.format(a)},
    '*': {'fun': '*', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf': tf.multiply, 'opgroup': ['aritygroup'], 'latex1': '$\\cdot$', 'latexF': '{}\\cdot{}',
          'sym_str': '({} * {})', 'pycode': lambda *args: '({}*{})'.format(args[0], args[1])},
    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': {'fun': '/', 'arity': 2, 'xtype': 'f2f', 'c-weight': 1, 'tf': tf.math.divide_no_nan, 'opgroup': [], 'latex1': '$\\div$', 'latexF': '\\frac{{{}}}{{{}}}',
          'sym_str': '({} / {})',
          'pycode': lambda *args: '(lambda x, y: x/y if y!=0 else 0)({}, {})'.format(args[0], args[1])},
    '**': {'fun': '**', 'arity': 2, 'xtype': 'f2f', 'c-weight': 2, 'tf': tf.pow, 'opgroup': [], 'latex1': '${x}^{y}$', 'latexF': '{{{}}}^{{{}}}',  # sfeh latexf requires some testing...
           'sym_str': '({} ** {})', 'pycode': lambda *args: '({}*{})'.format(args[0], args[1])},

    'abs': {'fun': 'abs', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf': tf.math.abs, 'opgroup': [], 'latex1': '$abs$', 'latexF': '|{{{}}}|',
            'sym_str': 'abs({})', 'pycode': lambda a: 'abs({})'.format(a)},
    'sign': {'fun': 'sign', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf': tf.math.sign, 'opgroup': [], 'latex1': '$sign$', 'latexF': 'sign({{{}}})',
             'sym_str': 'sign({})', 'pycode': lambda a: 'np.sign({})'.format(a)},

    'Square': {'fun': 'Square', 'arity': 1, 'xtype': 'f2f', 'c-weight': 2, 'tf': tf.math.square, 'opgroup': [], 'latex1': '$x^2$', 'latexF': '{{{}}}^2',
               'sym_str': 'Square({})', 'pycode': lambda a: '({}**2)'.format(a)},
    'sqrt': {'fun': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.sqrt, 'opgroup': [], 'latex1': '$\\sqrt{x}$', 'latexF': '\\sqrt{{{}}}',
             'sym_str': 'sqrt({})', 'pycode': lambda a: 'math.sqrt({})'.format(a)},

    'log': {'fun': 'log', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.math.log, 'opgroup': ['logs'], 'latex1': '$\\ln$', 'latexF': '\\ln{{{}}}',
            'sym_str': 'log({})', 'pycode': lambda a: 'math.log({})'.format(a)},
    'log1p': {'fun': 'log1p', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.math.log1p, 'opgroup': ['logs'], 'latex1': '$\\log(1+x)$', 'latexF': '\\log(1+{{{}}})',
              'sym_str': 'log1p({})', 'pycode': lambda a: 'math.log1p({})'.format(a)},

    'cos': {'fun': 'cos', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.cos, 'opgroup': ['angle'], 'latex1':  '$\\cos$', 'latexF': '\\cos({{{}}})',
            'sym_str': 'cos({})', 'pycode': lambda a: 'math.cos({})'.format(a)},
    'sin': {'fun': 'sin', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.sin, 'opgroup': ['angle'], 'latex1': '$\\sin$', 'latexF': '\\sin({{{}}})',
            'sym_str': 'sin({})', 'pycode': lambda a: 'math.sin({})'.format(a)},
    'tan': {'fun': 'tan', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.atan, 'opgroup': ['angle'], 'latex1': '$\\tan$', 'latexF': '\\tan({{{}}})',
            'sym_str': 'tan({})', 'pycode': lambda a: 'math.tan({})'.format(a)},
    'acos': {'fun': 'acos', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.acos, 'opgroup': ['angle'], 'latex1': '$\\acos$', 'latexF': '\\acos({{{}}})',
             'sym_str': 'acos({})', 'pycode': lambda a: 'math.acos({})'.format(a)},
    'asin': {'fun': 'asin', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.asin, 'opgroup': ['angle'], 'latex1': '$\\asin$', 'latexF': '\\asin({{{}}})',
             'sym_str': 'asin({})', 'pycode': lambda a: 'math.asin({})'.format(a)},
    'atan': {'fun': 'atan', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.atan, 'opgroup': ['angle'], 'latex1':  '$\\atan$', 'latexF': '\\atan({{{}}})',
             'sym_str': 'atan({})', 'pycode': lambda a: 'math.atan({})'.format(a)},
    'tanh': {'fun': 'tanh', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.tanh, 'opgroup': ['angle'], 'latex1': '$\\tanh$', 'latexF': '\\tanh({{{}}})',
             'sym_str': 'tanh({})', 'pycode': lambda a: 'math.tanh({})'.format(a)},
    # 'Integer': {'fun': 'Integer', 'arity': 1, 'xtype': 'f2f', 'c-weight': 0.5, 'tf': tf.cast({}, tf.int32), 'latex1': None, 'latexF': '{}',
    # 'sym_str': 'N({}, )', 'pycode': lambda a: 'math.tanh({})'.format(a)},

    # 'b2b' Classical logical operators, evaluate from bool to bool
    # DON'T USE tf.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': {'fun': 'Andb', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf': tf.logical_and, 'latex1': 'and', 'latexF': '({{{}}}\\wedge{{{}}})',
             'sym_str': 'Andb({}, {})', 'pycode': lambda *args: '({} and {})'.format(args[0], args[1])},
    'Orb': {'fun': 'Orb', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf': tf.logical_or, 'latex1': 'or', 'latexF': '({{{}}}\\vee{{{}}})',
            'sym_str': 'Orb({}, {})', 'pycode': lambda *args: '({} or {})'.format(args[0], args[1])},
    'Xor': {'fun': 'Xor', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf': tf.math.logical_xor, 'latex1': '$\\oplus$', 'latexF': '({{{}}}\\oplus{{{}}})',
            'sym_str': 'Xor({}, {})', 'pycode': lambda *args: '({} ^ {})'.format(args[0], args[1])},
    'Notb': {'fun': 'Notb', 'arity': 1, 'xtype': 'b2b', 'c-weight': 0.5, 'tf': tf.logical_not, 'latex1': '$\\neg$', 'latexF': '\\neg{{{}}}',
             'sym_str': 'Notb({})', 'pycode': lambda a: 'not({})'.format(a)},  # not a
    '|': {'fun': '|', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf': tf.logical_or, 'latex1': '$\\lor$', 'latexF': '({{{}}}\\vee{{{}}})',
          'sym_str': '({} | {})', 'pycode': lambda *args: '({} or {})'.format(args[0], args[1])},

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==': {'fun': '==', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.equal, 'latex1': '$=$', 'latexF': '({{{}}}={{{}}})',
           'sym_str': '({} == {})', 'pycode': lambda *args: '({}=={})'.format(args[0], args[1])},
    '!=': {'fun': '!=', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.not_equal, 'latex1': '$\\neq$', 'latexF': '({{{}}}\\neq{{{}}})',
           'sym_str': '({} != {})', 'pycode': lambda *args: '({}!={})'.format(args[0], args[1])},
    '<': {'fun': '<', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.less, 'latex1': '$<$', 'latexF': '{{{}}}<{{{}}}',
          'sym_str': '({} < {})', 'pycode': lambda *args: '({}<{})'.format(args[0], args[1])},  # a < b
    '<=': {'fun': '<=', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.less_equal, 'latex1': '$\\leq$', 'latexF': '{{{}}}\leq{{{}}}',
           'sym_str': '({} <= {})', 'pycode': lambda *args: '({}<={})'.format(args[0], args[1])},  # a <= b
    '>': {'fun': '>', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.greater, 'latex1': '$>$', 'latexF': '{{{}}}>{{{}}}',
          'sym_str': '({} > {})', 'pycode': lambda *args: '({}>{})'.format(args[0], args[1])},  # a > b
    '>=': {'fun': '>=', 'arity': 2, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.greater_equal, 'latex1': '$\\geq$', 'latexF': '{{{}}}\geq{{{}}}',
           'sym_str': '({} >= {})', 'pycode': lambda *args: '({}>={})'.format(args[0], args[1])},  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte': {'fun': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'c-weight': 0.1, 'tf': tf.compat.v2.where, 'latex1': 'if-then-else', 'latexF': 'if({{{}}} then {{{}}} else {{{}}})',
             'sym_str': 'Ifte({}, {}, {})', 'pycode': lambda *args: f'{args[0]} if {args[1]} else {args[2]}'},
    # long version of Ifte-'pycode': lambda *args: 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))
    'Mini': {'fun': 'Mini', 'arity': 2, 'xtype': 'f2f', 'c-weight': 0.5, 'tf': tf.math.minimum, 'latex1': '$\\min$', 'latexF': '\\min({{{}}}, {{{}}})',
             'sym_str': 'Mini({}, {})', 'pycode': lambda *args: 'min({}, {})'.format(args[0], args[1])},  # with forced arity-2
    'Maxi': {'fun': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'c-weight': 0.5, 'tf': tf.math.maximum, 'latex1': '$\\max$', 'latexF': '\\max({{{}}}, {{{}}})',
             'sym_str': 'Maxi({}, {})', 'pycode': lambda *args: 'max({}, {})'.format(args[0], args[1])},  # with forced arity-2
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
    '&': {'fun': '&', 'arity': 2, 'xtype': 'b2b', 'c-weight': 0.5, 'tf': tf.logical_and, 'latex1': '$\\land$', 'latexF': '({{{}}}\\wedge{{{}}})',
          'sym_str': '({} & {})', 'pycode': lambda *args: '({} and {})'.format(args[0], args[1])},
    'Power3': {'fun': '', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': tf.math.pow, 'latex1': None, 'latexF': '{}',
               'sym_str': '({}**2)', 'pycode': lambda a: '({}**2)'.format(a)},
    'Nand': {'fun': 'Nand', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': 'Nand({}, {})', 'pycode': lambda *args: 'Notb({} and {})'.format(args[0], args[1])},
    'Xand': {'fun': 'Xand', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': 'Xand({}, {})', 'pycode': lambda *args: 'Notb({} ^ {})'.format(args[0], args[1])},
    'Nor': {'fun': 'Nor', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_str': 'Nor({}, {})', 'pycode': None},
    'Xnor': {'fun': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': 'Xnor({}, {})', 'pycode': None},

    'log2': {'fun': 'log2', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': False, 'latex1': None, 'latexF': '{}',
             'sym_str': 'log({})', 'pycode': lambda a: 'math.log({})'.format(a)},
    'log10': {'fun': 'log2', 'arity': 1, 'xtype': 'f2f', 'c-weight': 3, 'tf': False, 'latex1': None, 'latexF': '{}',
              'sym_str': 'log({})', 'pycode': lambda a: 'math.log({})'.format(a)},

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float': {'fun': 'float', 'arity': 1, 'xtype': 'f2f', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': lambda a: 'float({})'.format(a)},  # not tested
    'int': {'fun': 'int', 'arity': 1, 'xtype': 'f2f', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_str': 'Integer({})', 'pycode': lambda a: 'int({})'.format(a)},  # not tested
    'bool': {'fun': 'bool', 'arity': 1, 'xtype': 'f2b', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_str': '', 'pycode': lambda a: 'bool({})'.format(a)},  # not tested

    # Not tested: Converters: Dummy operators that convert between float and bool
    'Ftob': {'fun': 'Ftob', 'arity': 1, 'xtype': 'f2b', 'c-weight': 1, 'tf': tf.bool, 'latex1': None, 'latexF': '{}',
             'sym_str': 'bool', 'pycode': lambda a: 'bool({})'.format(a)},  # not tested, same as bool
    'Btof': {'fun': 'Btof', 'arity': 1, 'xtype': 'b2f', 'c-weight': 1, 'tf': tf.float32, 'latex1': None, 'latexF': '{}',
             'sym_str': 'float', 'pycode': lambda a: 'float({})'.format(a)},  # not tested

    # Never used yet, trying to get rid of the ** function
    'Power': {'fun': 'Power', 'arity': 1, 'xtype': 'f2f', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': '({})**({})', 'pycode': lambda *args: '({}**{})'.format(args[0], args[1])},  # sfeh: round the exponent
    # sfeh sqrt, is only 2nd root, also 3rd-root?

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while': {'fun': 'while', 'arity': 2, 'xtype': 'b2?2?', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': None},  # sfeh: not working # Condition must change in loop
    'repeat_n': {'fun': 'repeat_n', 'arity': 2, 'xtype': 'b2?', 'c-weight': 1, 'tf': 'ä', 'latex1': None, 'latexF': '{}',
                 'sym_str': None, 'pycode': None},
    # sfeh: not working # repeat n time, specify n (int) by user?
}


# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['fun'], 1/v['c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


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
