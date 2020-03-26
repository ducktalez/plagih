import tensorflow as tf
import ast
from pathlib import Path
import textwrap

fitt_dict = {'classification': 'max',
             'regression': 'min',
             'regression bounded': 'min',
             'match': 'max'}

FIRST_TREE = 0
name_action = 'action'
first_action = name_action + str(0)
name_observation = 'observation'
first_gen_id = 1  # maybe take care to make this 0 for base gen

delete_this = True

# ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f'].index('f2f')
f2f, f2b, b2b, b2f, b2f2f = 0, 1, 2, 3, 4


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
todo: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
"""
op_what = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'latex': '$+$', 'call': 'inline', 'pycode': lambda a, b: '({}+{})'.format(a, b)},
    '-':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract, 'latex': '$-$', 'call': 'inline', 'pycode': lambda a, b: '({}-{})'.format(a, b)},
    '~':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.negative, 'latex': '$-$', 'call': 'inline', 'pycode': lambda a: '(-{})'.format(a)},  # todo this is minus again
    '*':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'latex': '$\\cdot$', 'call': 'inline', 'pycode': lambda a, b: '({}*{})'.format(a, b)},
    '/':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.math.divide_no_nan, 'latex': '$\\div$', 'call': 'inline', 'pycode': lambda a, b: '({}/{})'.format(a, b)},  # a / b  # todo try tf.math.divide_no_nan
    '**':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.pow, 'latex': '$**$', 'call': 'inline', 'pycode': lambda a, b: '({}*{})'.format(a, b)},
    'abs':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.abs, 'latex': None, 'call': None, 'pycode': lambda a: 'abs({})'.format(a)},
    'sign':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.sign, 'latex': None, 'call': None, 'pycode': None},
    'square':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.square, 'latex': None, 'call': None, 'pycode': lambda a: '({}**2)'.format(a)},
    'sqrt':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt, 'latex': None, 'call': None, 'pycode': lambda a: 'math.sqrt({})'.format(a)},
    'log':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log, 'latex': None, 'call': None, 'pycode': lambda a: 'math.log({})'.format(a)},
    'log1p':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p, 'latex': None, 'call': None, 'pycode': lambda a: 'math.log1p({})'.format(a)},
    'cos':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.cos, 'latex': None, 'call': None, 'pycode': lambda a: 'math.cos({})'.format(a)},
    'sin':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.sin, 'latex': '$\\sin$', 'call': None, 'pycode': lambda a: 'math.sin({})'.format(a)},
    'tan':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'latex': None, 'call': None, 'pycode': lambda a: 'math.tan({})'.format(a)},
    'acos':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.acos, 'latex': None, 'call': None, 'pycode': lambda a: 'math.acos({})'.format(a)},
    'asin':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.asin, 'latex': None, 'call': None, 'pycode': lambda a: 'math.asin({})'.format(a)},
    'atan':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'latex': None, 'call': None, 'pycode': lambda a: 'math.atan({})'.format(a)},
    'tanh':	{'arity': 1, 'xtype': 'f2f', 'tf': tf.tanh, 'latex': None, 'call': None, 'pycode': lambda a: 'math.tanh({})'.format(a)},
    # todo round operation! sympify: N(1.2345, decimals). e.g. Int

    # 'b2b' Classical logical operators, evaluate from bool to bool
    'And':	{'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': None, 'call': None, 'pycode': lambda a, b: '({} and {})'.format(a, b)},
    '&':	{'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': '$\\land$', 'call': 'inline', 'pycode': lambda a, b: '({} & {})'.format(a, b)},  # DON'T USE tf.bitwise.bitwise_and
    'Or':	{'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': '$\\lor$', 'call': None, 'pycode': lambda a, b: '({} or {})'.format(a, b)},
    '|':	{'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': None, 'call': 'inline', 'pycode': lambda a, b: '({} | {})'.format(a, b)},  # a or b
    'Xor':	{'arity': 2, 'xtype': 'b2b', 'tf': tf.math.logical_xor, 'latex': None, 'call': None, 'pycode': lambda a, b: '({} ^ {})'.format(a, b)},
    'Not':	{'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not, 'latex': '$\\neg$', 'call': None, 'pycode': None},  # not a

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==':	{'arity': 2, 'xtype': 'f2b', 'tf': tf.equal, 'latex': '$==$', 'call': 'inline', 'pycode': lambda a, b: '({}=={})'.format(a, b)},
    '!=':	{'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal, 'latex': '$\\neg$', 'call': 'inline', 'pycode': lambda a, b: '({}!={})'.format(a, b)},
    '<':	{'arity': 2, 'xtype': 'f2b', 'tf': tf.less, 'latex': None, 'call': 'inline', 'pycode': lambda a, b: '({}<{})'.format(a, b)},  # a < b
    '<=':	{'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal, 'latex': None, 'call': 'inline', 'pycode': lambda a, b: '({}<={})'.format(a, b)},  # a <= b
    '>':	{'arity': 2, 'xtype': 'f2b', 'tf': tf.greater, 'latex': None, 'call': 'inline', 'pycode': lambda a, b: '({}>{})'.format(a, b)},  # a > b
    '>=':	{'arity': 2, 'xtype': 'f2b', 'tf': tf.greater_equal, 'latex': None, 'call': 'inline', 'pycode': lambda a, b: '({}{}{})'.format(a, '>=', b)},  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte':	{'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'latex': 'if', 'call': None, 'pycode': lambda a, b, c: '{1} if {0} else {2}'.format(a, b, c)},
    # 'Ifte':	{'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'latex': 'if', 'call': None,    'pycode': lambda a, b, c: 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))},
    'Mini':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum, 'latex': '$\\min$', 'call': None, 'pycode': lambda a, b: 'min({}, {})'.format(a, b)},  # maximum
    'Maxi':	{'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum, 'latex': '$\\max$', 'call': None, 'pycode': lambda a, b: 'max({}, {})'.format(a, b)},  # minimum
}

op = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+':{'fun': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'latex': '$+$', 'call': 'inline',                    'pycode': lambda a, b: '({}+{})'.format(a, b)},
    ast.Add:{'fun': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'latex': None, 'call': 'inline',                'pycode': None},
    '-':{'fun': '-', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'latex': '$-$', 'call': 'inline',                        'pycode': lambda a, b: '({}-{})'.format(a, b)},
    ast.Sub:{'fun': '-', 'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract, 'latex': None, 'call': 'inline',            'pycode': None},
    '~':{'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': '$-$', 'call': 'inline',                        'pycode': lambda a: '(-{})'.format(a)},  # todo this is minus again
    'usub':{'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': None},
    ast.USub:{'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': tf.negative, 'latex': None, 'call': None,                'pycode': None},
    '*':{'fun': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'latex': '$\\cdot$', 'call': 'inline',            'pycode': lambda a, b: '({}*{})'.format(a, b)},
    ast.Mult:{'fun': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'latex': None, 'call': 'inline',            'pycode': None},  # a * b
    '/':{'fun': '/', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'latex': '$\\div$', 'call': 'inline',                    'pycode': lambda a, b: '({}/{})'.format(a, b)},
    ast.Div:{'fun': '/', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.divide_no_nan, 'latex': None, 'call': 'inline',    'pycode': None},  # a / b  # todo try tf.math.divide_no_nan
    '**':{'fun': '**', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'latex': '$**$', 'call': 'inline',                    'pycode': lambda a, b: '({}*{})'.format(a, b)},
    ast.Pow:{'fun': '**', 'arity': 2, 'xtype': 'f2f', 'tf': tf.pow, 'latex': None, 'call': 'inline',                'pycode': None},  # a ** 2
    'abs':{'fun': 'abs', 'arity': 1, 'xtype': 'f2f', 'tf': tf.abs, 'latex': None, 'call': None,                    'pycode': lambda a: 'abs({})'.format(a)},
    'sign':{'fun': 'sign', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sign, 'latex': None, 'call': None,                    'pycode': None},
    'square':{'fun': 'square', 'arity': 1, 'xtype': 'f2f', 'tf': tf.square, 'latex': None, 'call': None,            'pycode': lambda a: '({}**2)'.format(a)},
    'sqrt':{'fun': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.sqrt({})'.format(a)},
    'log':{'fun': 'log', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log, 'latex': None, 'call': None,                'pycode': lambda a: 'math.log({})'.format(a)},
    'log1p':{'fun': 'log1p', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p, 'latex': None, 'call': None,            'pycode': lambda a: 'math.log1p({})'.format(a)},
    'cos':{'fun': 'cos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.cos, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.cos({})'.format(a)},
    'sin':{'fun': 'sin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sin, 'latex': '$\\sin$', 'call': None,                'pycode': lambda a: 'math.sin({})'.format(a)},
    'tan':{'fun': 'tan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.tan({})'.format(a)},
    'acos':{'fun': 'acos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.acos, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.acos({})'.format(a)},
    'asin':{'fun': 'asin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.asin, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.asin({})'.format(a)},
    'atan':{'fun': 'atan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.atan({})'.format(a)},
    'tanh':{'fun': 'tanh', 'arity': 1, 'xtype': 'f2f', 'tf': tf.tanh, 'latex': None, 'call': None,                    'pycode': lambda a: 'math.tanh({})'.format(a)},
    # todo round operation! sympify: N(1.2345, decimals). e.g. Int

    # 'b2b' Classical logical operators, evaluate from bool to bool
    'And':{'fun': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': lambda a, b: '({} and {})'.format(a, b)},
    ast.And:{'fun': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': None, 'call': None,            'pycode': None},  # a and b
    '&':{'fun': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': '$\\land$', 'call': 'inline', 'pycode': lambda a, b: '({} & {})'.format(a, b)},
    ast.BitAnd:{'fun': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': None, 'call': 'inline',        'pycode': None},  # DON'T USE tf.bitwise.bitwise_and
    'Or':{'fun': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': '$\\lor$', 'call': None,                    'pycode': lambda a, b: '({} or {})'.format(a, b)},
    ast.Or:{'fun': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': None, 'call': None,                'pycode': None},  # a or b
    '|':{'fun': '|', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': None, 'call': 'inline',                'pycode': lambda a, b: '({} | {})'.format(a, b)},  # a or b
    ast.BitOr:{'fun': '|', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': None, 'call': 'inline',        'pycode': None},  # a or b
    'Not':{'fun': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': None},
    ast.Not:{'fun': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not, 'latex': '$\\neg$', 'call': None,        'pycode': None},  # not a

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==':{'fun': '==', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'latex': '$==$', 'call': 'inline',                    'pycode': lambda a, b: '({}=={})'.format(a, b)},
    ast.Eq:{'fun': '==', 'arity': 2, 'xtype': 'f2b', 'tf': tf.equal, 'latex': None, 'call': 'inline',                'pycode': lambda a, b: '({}=={})'.format(a, b)},  # a == b
    '!=':{'fun': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'latex': '$\\neg$', 'call': 'inline',                'pycode': lambda a, b: '({}!={})'.format(a, b)},
    ast.NotEq:{'fun': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal, 'latex': None, 'call': 'inline',        'pycode': lambda a, b: '({}!={})'.format(a, b)},  # a != b
    '<':{'fun': '<', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'latex': '$<$', 'call': 'inline',                        'pycode': lambda a, b: '({}<{})'.format(a, b)},
    ast.Lt:{'fun': '<', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less, 'latex': None, 'call': 'inline',                'pycode': lambda a, b: '({}<{})'.format(a, b)},  # a < b
    '<=':{'fun': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'latex': '$<=$', 'call': 'inline',                    'pycode': lambda a, b: '({}<={})'.format(a, b)},
    ast.LtE:{'fun': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal, 'latex': None, 'call': 'inline',        'pycode': lambda a, b: '({}<={})'.format(a, b)},  # a <= b
    '>':{'fun': '>', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'latex': '$>$', 'call': 'inline',                        'pycode': lambda a, b: '({}>{})'.format(a, b)},
    ast.Gt:{'fun': '>', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater, 'latex': None, 'call': 'inline',                'pycode': lambda a, b: '({}>{})'.format(a, b)},  # a > b
    '>=':{'fun': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'latex': '$>=$', 'call': 'inline',                    'pycode': lambda a, b: '({}>={})'.format(a, b)},
    ast.GtE:{'fun': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater_equal, 'latex': None, 'call': 'inline',        'pycode': lambda a, b: '({}{}{})'.format(a, '>=', b)},  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte':{'fun': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'latex': 'if', 'call': None,    'pycode': lambda a, b, c: '{1} if {0} else {2}'.format(a, b, c)},
    # 'Ifte':{'fun': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'latex': 'if', 'call': None,    'pycode': lambda a, b, c: 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))},
    'Mini':{'fun': 'Mini', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum, 'latex': '$\\min$', 'call': None,    'pycode': lambda a, b: 'min({}, {})'.format(a, b)},  # maximum
    'Maxi':{'fun': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum, 'latex': '$\\max$', 'call': None,    'pycode': lambda a, b: 'max({}, {})'.format(a, b)},  # minimum

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float':{'fun': 'float', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'latex': None, 'call': None,                    'pycode': lambda a: 'float({})'.format(a)},  # not tested
    'int':{'fun': 'int', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': lambda a: 'int({})'.format(a)},  # not tested
    'bool':{'fun': 'bool', 'arity': 0, 'xtype': '2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': lambda a: 'bool({})'.format(a)},  # not tested
}
op_v2 = {
    '+': op_what['+'],
    ast.Add: op_what['+'],
    '-': op_what['-'],
    ast.Sub: op_what['-'],
    '~': op_what['~'],
    'usub': op_what['~'],
    ast.USub: op_what['~'],
    '*': op_what['*'],
    ast.Mult: op_what['*'],
    '/': op_what['/'],
    ast.Div: op_what['/'],
    '**': op_what['**'],
    ast.Pow: op_what['**'],
    'abs': op_what['abs'],
    'sign': op_what['sign'],
    'square': op_what['square'],
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
    'And': op_what['And'],
    ast.And: op_what['And'],
    '&': op_what['&'],
    ast.BitAnd: op_what['&'],
    'Or': op_what['Or'],
    ast.Or: op_what['Or'],
    '|': op_what['|'],
    ast.BitOr: op_what['|'],
    'Not': op_what['Not'],
    ast.Not: op_what['Not'],
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
    # no (easy-to use) tensorflow-operations available
    # ast.BitOr
    'Xor':{'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': lambda a, b: '({} ^ {})'.format(a, b)},
    'Nand':{'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': lambda a, b: 'not({} and {})'.format(a, b)},
    'Xand':{'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': lambda a, b: 'not({} ^ {})'.format(a, b)},
    'Nor':{'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': None},
    'Xnor': {'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'call': None,                        'pycode': None},

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float':{'arity': 0, 'xtype': '2f', 'tf': 'ä', 'latex': None, 'call': None, 'pycode': lambda a: 'float({})'.format(a)},  # not tested
    'int':	{'arity': 0, 'xtype': '2f', 'tf': 'ä', 'latex': None, 'call': None, 'pycode': lambda a: 'int({})'.format(a)},  # not tested
    'bool':	{'arity': 0, 'xtype': '2b', 'tf': 'ä', 'latex': None, 'call': None, 'pycode': lambda a: 'bool({})'.format(a)},  # not tested

    # Not tested: Converters: Dummy operators that convert between float and bool
    'Ftob':{'arity': 1, 'xtype': 'f2b', 'tf': tf.bool, 'latex': None, 'call': 'bool',                'pycode': lambda a: 'bool({})'.format(a)},  # not tested, same as bool
    'Btof':{'arity': 1, 'xtype': 'b2f', 'tf': tf.float32, 'latex': None, 'call': 'float',            'pycode': lambda a: 'float({})'.format(a)},  # not tested

    # Never used yet, trying to get rid of the ** function
    'Power':{'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'call': '(({})**{})',            'pycode': lambda a, b: '({}**{})'.format(a, b)},  # not used # todo # a*a*a -> a**3
    'inverse':{'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'call': None,                'pycode': None},  # not used # 1/float. important for squareroots. todo

    # Never used yet, todo. Trying to introduce multi-dimensional outputs that should
    'vector':{'fun': 'vector', 'arity': 1, 'xtype': 'ü', 'tf': 'ä', 'latex': None, 'call': None,                    'pycode': None},  # not tested # sfeh: not working

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while':{'fun': 'while', 'arity': 2, 'xtype': 'b2?2?', 'tf': 'ä', 'latex': None, 'call': None,                    'pycode': None},  # sfeh: not working # Condition must change in loop
    'repeat_n':{'fun': 'repeat_n', 'arity': 2, 'xtype': 'b2?', 'tf': 'ä', 'latex': None, 'call': None,                'pycode': None},  # sfeh: not working # repeat n time, specify n (int) by user?
}
# import tensorflow as tf; import ast; import textwrap
# print([x for x in op.keys() if type(x) == type('asd')])  # retreive a list with all non-ast ops:

functions_infix_dict = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=', '&', '|']

function_infix_to_prefix = {  # currently obsolete
    '+': 'add',
    '-': 'sub',
    '*': 'mult',
    '/': 'div',
    '**': 'power',
    '==': 'eq',
    '!=': 'neq',
    '<': 'lt',
    '<=': 'leq',
    '>': 'gt',
    '>=': 'geq',
    '&': 'And',
}
