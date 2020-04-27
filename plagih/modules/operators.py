import tensorflow as tf
import ast
from pathlib import Path
import textwrap

FIRST_TREE = 0
name_action = 'action'
first_action = name_action + str(0)
name_observation = 'observation'
first_gen_id = 1  # maybe take care to make this 0 for base gen
karoo_skip = 1

delete_this = True
debug_this_please = False

# ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f'].index('f2f')
f2f, f2b, b2b, b2f, b2f2f = 0, 1, 2, 3, 4


class Plagih_Plus:
    fun_name = '+'
    fun_arity = 2
    xtype = 'f2f'
    fun_tf = tf.add
    fun_latex = '$+$'
    fun_sym = '({})+({})'
    fun_pycode = lambda a, b: '({}+{})'.format(a, b)


"""
op: Dict to work as 'Database' for every expression-bit and its features
- KEY: expression-bit: can occur in various forms, which group into following uses:
    - ast.KEY: Found by pythons 'ast' when inline op like [+, -, *, ...] occurs
    - 'KEY': Found by pythons 'ast' when non-inline functions are found as string
    -> some operators_csv are identical, but occur several times, e. g. ('And', '&', ast.BitAnd)
- VALUE: several features that have to be regarded. Some are irrelevant or have to be done.
    - name: (irrelevant) information what this function actually does. identical operators_csv can be found like this.
    - arity: Amount of inputs an operator has to have. This must always be constant (reason why pythons min/max does not work)
    - tf: Tensorflow graph representation for the key. Was separated earlier. (!): [tf.bool, tf.float] are just variables of the used tf.cast function.
    - gpbp: Genetic Programming Backpropagation: An open idea from sfeh to introduce backpropagation for genetic operators_csv.
    - latex: For visualizing trees with latex, these representations might look better

Features should always be defined, even though they might not occur at all. If not used, they CAN be filled with dummy values like äöü
Some, which are known of not being used yet are commented with '# not tested' or '# not used'
sfeh: write test that checks all operators_csv for sympificytion (...+branch-combinations, and more?)
"""
op_what = {  # 'f2f': Classical mathematical operators_csv, evaluate from float to float
    '+': {'fun': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'latex': '$+$', 'sym_str': '({} + {})', 'pycode': lambda a, b: '({}+{})'.format(a, b)},
    '-': {'fun': '-', 'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract, 'latex': '$-$', 'sym_str': '({} - {})', 'pycode': lambda a, b: '({}-{})'.format(a, b)},
    '~': {'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': tf.negative, 'latex': '$-$', 'sym_str': '(-{})', 'pycode': lambda a: '(-{})'.format(a)},
    '*': {'fun': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'latex': '$\\cdot$', 'sym_str': '({} * {})', 'pycode': lambda a, b: '({}*{})'.format(a, b)},

    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': {'fun': '/', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.divide_no_nan, 'latex': '$\\div$', 'sym_str': '({} / {})',
          'pycode': lambda a, b: '(lambda x, y: x/y if y!=0 else 0)({}, {})'.format(a, b)},
    '**': {'fun': '**', 'arity': 2, 'xtype': 'f2f', 'tf': tf.pow, 'latex': '$**$', 'sym_str': '({} ** {})', 'pycode': lambda a, b: '({}*{})'.format(a, b)},
    'abs': {'fun': 'abs', 'arity': 1, 'xtype': 'f2f', 'tf': tf.abs, 'latex': None, 'sym_str': 'abs({})', 'pycode': lambda a: 'abs({})'.format(a)},
    'sign': {'fun': 'sign', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sign, 'latex': None, 'sym_str': 'sign({})', 'pycode': lambda a: 'sign({})'.format(a)},
    'square': {'fun': 'square', 'arity': 1, 'xtype': 'f2f', 'tf': tf.square, 'latex': None, 'sym_str': '({}**2)', 'pycode': lambda a: '({}**2)'.format(a)},
    'sqrt': {'fun': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt, 'latex': None, 'sym_str': 'sqrt({})', 'pycode': lambda a: 'math.sqrt({})'.format(a)},
    'log': {'fun': 'log', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log, 'latex': None, 'sym_str': 'log({})', 'pycode': lambda a: 'math.log({})'.format(a)},
    'log1p': {'fun': 'log1p', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p, 'latex': None, 'sym_str': 'log1p({})', 'pycode': lambda a: 'math.log1p({})'.format(a)},
    'cos': {'fun': 'cos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.cos, 'latex': '$\\cos$', 'sym_str': 'cos({})', 'pycode': lambda a: 'math.cos({})'.format(a)},
    'sin': {'fun': 'sin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sin, 'latex': '$\\sin$', 'sym_str': 'sin({})', 'pycode': lambda a: 'math.sin({})'.format(a)},
    'tan': {'fun': 'tan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'latex': '$\\tan$', 'sym_str': 'tan({})', 'pycode': lambda a: 'math.tan({})'.format(a)},
    'acos': {'fun': 'acos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.acos, 'latex': None, 'sym_str': 'acos({})', 'pycode': lambda a: 'math.acos({})'.format(a)},
    'asin': {'fun': 'asin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.asin, 'latex': None, 'sym_str': 'asin({})', 'pycode': lambda a: 'math.asin({})'.format(a)},
    'atan': {'fun': 'atan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'latex': None, 'sym_str': 'atan({})', 'pycode': lambda a: 'math.atan({})'.format(a)},
    'tanh': {'fun': 'tanh', 'arity': 1, 'xtype': 'f2f', 'tf': tf.tanh, 'latex': None, 'sym_str': 'tanh({})', 'pycode': lambda a: 'math.tanh({})'.format(a)},

    # todo round operation! sympify: N(1.2345, decimals). e.g. Int

    # 'b2b' Classical logical operators_csv, evaluate from bool to bool
    'Andb': {'fun': 'Andb', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': 'and', 'sym_str': 'Andb({}, {})', 'pycode': lambda a, b: '({} and {})'.format(a, b)},
    # sfeh Andbm and + & (delete this)
    '&': {'fun': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'latex': '$\\land$', 'sym_str': '({} & {})', 'pycode': lambda a, b: '({} and {})'.format(a, b)},
    # DON'T USE tf.bitwise.bitwise_and
    'Orb': {'fun': 'Orb', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': 'or', 'sym_str': 'Orb({}, {})', 'pycode': lambda a, b: '({} or {})'.format(a, b)},
    '|': {'fun': '|', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'latex': '$\\lor$', 'sym_str': '({} | {})', 'pycode': lambda a, b: '({} or {})'.format(a, b)},
    # a or b, sfeh python-'|' not used here

    'Notb': {'fun': 'Notb', 'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not, 'latex': '$\\neg$', 'sym_str': 'Notb({})', 'pycode': None},  # not a

    # 'f2b' Classical comparative operators_csv, evaluate from float to bool
    '==': {'fun': '==', 'arity': 2, 'xtype': 'f2b', 'tf': tf.equal, 'latex': '$==$', 'sym_str': '({} == {})', 'pycode': lambda a, b: '({}=={})'.format(a, b)},
    '!=': {'fun': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal, 'latex': '$\\neg$', 'sym_str': '({} != {})', 'pycode': lambda a, b: '({}!={})'.format(a, b)},
    '<': {'fun': '<', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less, 'latex': '$<$', 'sym_str': '({} < {})', 'pycode': lambda a, b: '({}<{})'.format(a, b)},  # a < b
    '<=': {'fun': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal, 'latex': '$<=$', 'sym_str': '({} <= {})', 'pycode': lambda a, b: '({}<={})'.format(a, b)},  # a <= b
    '>': {'fun': '>', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater, 'latex': '$>$', 'sym_str': '({} > {})', 'pycode': lambda a, b: '({}>{})'.format(a, b)},  # a > b
    '>=': {'fun': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater_equal, 'latex': '$>=$', 'sym_str': '({} >= {})', 'pycode': lambda a, b: '({}{}{})'.format(a, '>=', b)},  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte': {'fun': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'latex': 'if', 'sym_str': 'Ifte({}, {}, {})', 'pycode': lambda a, b, c: '{1} if {0} else {2}'.format(a, b, c)},
    # long version of Ifte-'pycode': lambda a, b, c: 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))
    'Mini': {'fun': 'Mini', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum, 'latex': '$\\min$', 'sym_str': 'Mini({}, {})', 'pycode': lambda a, b: 'min({}, {})'.format(a, b)},  # maximum
    'Maxi': {'fun': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum, 'latex': '$\\max$', 'sym_str': 'Maxi({}, {})', 'pycode': lambda a, b: 'max({}, {})'.format(a, b)},  # minimum
}

op = {
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
    # no (easy-to use) tensorflow-operations available
    # ast.BitOr
    'Xor': {'fun': 'Xor', 'arity': 2, 'xtype': 'b2b', 'tf': tf.math.logical_xor, 'latex': None, 'sym_str': 'Xor({}, {})', 'pycode': lambda a, b: '({} ^ {})'.format(a, b)},
    'Nand': {'fun': 'Nand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'sym_str': 'Nand({}, {})', 'pycode': lambda a, b: 'Notb({} and {})'.format(a, b)},
    'Xand': {'fun': 'Xand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'sym_str': 'Xand({}, {})', 'pycode': lambda a, b: 'Notb({} ^ {})'.format(a, b)},
    'Nor': {'fun': 'Nor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'sym_str': 'Nor({}, {})', 'pycode': None},
    'Xnor': {'fun': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'latex': None, 'sym_str': 'Xnor({}, {})', 'pycode': None},

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float': {'fun': 'float', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'sym_str': None, 'pycode': lambda a: 'float({})'.format(a)},  # not tested
    'int': {'fun': 'int', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'sym_str': 'Integer({})', 'pycode': lambda a: 'int({})'.format(a)},  # not tested
    'bool': {'fun': 'bool', 'arity': 1, 'xtype': 'f2b', 'tf': 'ä', 'latex': None, 'sym_str': '', 'pycode': lambda a: 'bool({})'.format(a)},  # not tested

    # Not tested: Converters: Dummy operators_csv that convert between float and bool
    'Ftob': {'fun': 'Ftob', 'arity': 1, 'xtype': 'f2b', 'tf': tf.bool, 'latex': None, 'sym_str': 'bool', 'pycode': lambda a: 'bool({})'.format(a)},  # not tested, same as bool
    'Btof': {'fun': 'Btof', 'arity': 1, 'xtype': 'b2f', 'tf': tf.float32, 'latex': None, 'sym_str': 'float', 'pycode': lambda a: 'float({})'.format(a)},  # not tested

    # Never used yet, trying to get rid of the ** function
    'Power': {'fun': 'Power', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'sym_str': '({})**({})', 'pycode': lambda a, b: '({}**{})'.format(a, b)},  # not used # todo # a*a*a -> a**3
    'Sqrt': {'fun': 'inverse', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'latex': None, 'sym_str': None, 'pycode': None},  # not used # 1/float. important for squareroots. todo sqrt, not inverse

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while': {'fun': 'while', 'arity': 2, 'xtype': 'b2?2?', 'tf': 'ä', 'latex': None, 'sym_str': None, 'pycode': None},  # sfeh: not working # Condition must change in loop
    'repeat_n': {'fun': 'repeat_n', 'arity': 2, 'xtype': 'b2?', 'tf': 'ä', 'latex': None, 'sym_str': None, 'pycode': None},  # sfeh: not working # repeat n time, specify n (int) by user?
}

# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print([x for x in op.keys() if type(x) == type('q')])  # retreive a list with all non-ast ops:

def get_example_distribution_dict():
    distributions_as_string = {'2f': ['lambda: np.random.normal(1,2)',
                                      'lambda: np.random.normal(1,1)',
                                      'lambda: np.random.randint(0, 10)'],
                               '2b': ['lambda: np.random.choice([True, False])']}
    return distributions_as_string


def oparray_from_list(functions):
    """
    Load all operators_csv ready-to-use from a file
    """

    # rows are the function types (f2f)
    # columns are the arity
    choose_oparray = [[[], [], [], []],
                      [[], [], [], []],
                      [[], [], [], []],
                      [[], [], [], []],
                      [[], [], [], []]]

    # sfeh make this a np.array

    for fun in functions:
        label = fun[0]
        arity = op[label]['arity']  # arity = int(fun[1])
        xtype = op[label]['xtype']

        if xtype == 'f2f':
            choose_oparray[f2f][arity].append(label)
        elif xtype == 'f2b':
            choose_oparray[f2b][arity].append(label)
        elif xtype == 'b2b':
            choose_oparray[b2b][arity].append(label)
        elif xtype == 'b2f':
            choose_oparray[b2f][arity].append(label)
        elif xtype == 'b2f2f':
            choose_oparray[b2f2f][arity].append(label)

    return choose_oparray


def get_all_oparrays():
    oplist = []
    for key in op_what.keys():
        oplist.append((key, None))
    oparray = oparray_from_list(oplist)
    return oparray


expr_raw_infix = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=', '&', '|']  # sfeh / is removed for
