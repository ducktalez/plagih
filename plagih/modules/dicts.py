import tensorflow as tf
import ast
from pathlib import Path


fitt_dict = {'classification': 'max',
             'regression': 'min',
             'regression bounded': 'min',
             'match': 'max'}

FIRST_TREE = 0
first_action = 'action0'
first_gen_id = 1  # maybe take care to make this 0 for base gen
input_name = 'observation'

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
op = {      # 'f2f': Classical mathematical operators, evaluate from float to float
      '+':      {'name': '+',       'arity': 2,     'xtype': 'f2f', 	'tf': tf.add, 				'gpbp': ['ö'], 	'latex': '$+$'},  # not tested
      ast.Add:  {'name': '+',       'arity': 2,     'xtype': 'f2f', 	'tf': tf.add, 				'gpbp': ['ö'], 	'latex': None},  # e.g., a + b
      '-':      {'name': '-',       'arity': 2,     'xtype': 'f2f', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$-$'},
      ast.Sub:  {'name': '-',       'arity': 2,     'xtype': 'f2f', 	'tf': tf.subtract, 			'gpbp': ['ö'], 	'latex': None},  # e.g., a - b
      '~':      {'name': '~',       'arity': 1,     'xtype': 'f2f', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$-$'},  # todo this is minus again
      'usub':   {'name': '~',       'arity': 1,     'xtype': 'f2f', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      ast.USub: {'name': '~',       'arity': 1,     'xtype': 'f2f', 	'tf': tf.negative, 			'gpbp': ['ö'], 	'latex': None},  # e.g., -a
      '*':      {'name': '*',       'arity': 2,     'xtype': 'f2f', 	'tf': tf.multiply, 			'gpbp': ['ö'], 	'latex': '$\\cdot$'},
      ast.Mult: {'name': '*',       'arity': 2,     'xtype': 'f2f', 	'tf': tf.multiply, 			'gpbp': ['ö'], 	'latex': None},  # e.g., a * b
      '/':      {'name': '/',       'arity': 2,     'xtype': 'f2f', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$\\div$'},
      ast.Div:  {'name': '/',       'arity': 2,     'xtype': 'f2f', 	'tf': tf.math.divide_no_nan, 'gpbp': ['ö'], 'latex': None},  # e.g., a / b  # todo try tf.math.divide_no_nan
      '**':     {'name': '**',      'arity': 2,     'xtype': 'f2f', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$**$'},
      ast.Pow:  {'name': '**',      'arity': 2,     'xtype': 'f2f', 	'tf': tf.pow, 				'gpbp': ['ö'], 	'latex': None},  # e.g., a ** 2
      'abs':    {'name': 'abs',     'arity': 1,     'xtype': 'f2f', 	'tf': tf.abs, 				'gpbp': ['ö'], 	'latex': None},
      'sign':   {'name': 'sign',    'arity': 1,     'xtype': 'f2f', 	'tf': tf.sign, 				'gpbp': ['ö'], 	'latex': None},
      'square': {'name': 'square',  'arity': 1,     'xtype': 'f2f', 	'tf': tf.square, 			'gpbp': ['ö'], 	'latex': None},
      'sqrt':   {'name': 'sqrt',    'arity': 1,     'xtype': 'f2f', 	'tf': tf.sqrt, 				'gpbp': ['ö'], 	'latex': None},
      'log':    {'name': 'log',     'arity': 1,     'xtype': 'f2f', 	'tf': tf.math.log, 			'gpbp': ['ö'], 	'latex': None},
      'log1p':  {'name': 'log1p',   'arity': 1,     'xtype': 'f2f', 	'tf': tf.math.log1p, 		'gpbp': ['ö'], 	'latex': None},
      'cos':    {'name': 'cos',     'arity': 1,     'xtype': 'f2f', 	'tf': tf.cos, 				'gpbp': ['ö'], 	'latex': None},
      'sin':    {'name': 'sin',     'arity': 1,     'xtype': 'f2f', 	'tf': tf.sin, 				'gpbp': ['ö'], 	'latex': '$\\sin$'},
      'tan':    {'name': 'tan',     'arity': 1,     'xtype': 'f2f', 	'tf': tf.atan, 				'gpbp': ['ö'], 	'latex': None},
      'acos':   {'name': 'acos',    'arity': 1,     'xtype': 'f2f', 	'tf': tf.acos, 				'gpbp': ['ö'], 	'latex': None},
      'asin':   {'name': 'asin',    'arity': 1,     'xtype': 'f2f', 	'tf': tf.asin, 				'gpbp': ['ö'], 	'latex': None},
      'atan':   {'name': 'atan',    'arity': 1,     'xtype': 'f2f', 	'tf': tf.atan, 				'gpbp': [1], 	'latex': None},
      # todo round operation! sympify: N(1.2345, decimals). e.g. Int

            # 'b2b' Classical logical operators, evaluate from bool to bool
      'And':    {'name': 'And',     'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      ast.And:  {'name': 'And',     'arity': 2,     'xtype': 'b2b', 	'tf': tf.logical_and, 		'gpbp': ['ö'], 	'latex': None},  # e.g., a and b
      '&':      {'name': '&',       'arity': 2,     'xtype': 'b2b', 	'tf': tf.logical_and, 		'gpbp': ['ö'], 	'latex': '$\\land$'},
      ast.BitAnd: {'name': '&',     'arity': 2,     'xtype': 'b2b', 	'tf': tf.logical_and, 		'gpbp': ['ö'], 	'latex': None},  # DON'T USE tf.bitwise.bitwise_and
      'Or':     {'name': 'Or',      'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$\\lor$'},
      ast.Or:   {'name': 'Or',      'arity': 2,     'xtype': 'b2b', 	'tf': tf.logical_or, 		'gpbp': ['ö'], 	'latex': None},  # e.g., a or b
      '|':       {'name': '|',      'arity': 2,     'xtype': 'b2b', 	'tf': tf.logical_or, 		'gpbp': ['ö'], 	'latex': None},  # e.g., a or b
      ast.BitOr: {'name': '|',      'arity': 2,     'xtype': 'b2b', 	'tf': tf.logical_or, 		'gpbp': ['ö'], 	'latex': None},  # e.g., a or b
# ast.BitOr
      'Xor':    {'name': 'Xor',     'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      'Nand':   {'name': 'Nand',    'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      'Xand':   {'name': 'Xand',    'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      'Nor':    {'name': 'Nor',     'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      'Xnor':   {'name': 'Xnor',    'arity': 2,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      'Not':    {'name': 'Not',     'arity': 1,     'xtype': 'b2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},
      ast.Not:  {'name': 'Not',     'arity': 1,     'xtype': 'b2b', 	'tf': tf.logical_not, 		'gpbp': ['ö'], 	'latex': '$\\neg$'},  # e.g., not a

            # 'f2b' Classical comparative operators, evaluate from float to bool
      '==':     {'name': '==',      'arity': 2,     'xtype': 'f2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$==$'},
      ast.Eq:   {'name': '==',      'arity': 2,     'xtype': 'f2b', 	'tf': tf.equal, 			'gpbp': ['ö'], 	'latex': None},  # e.g., a == b
      '!=':     {'name': '!=',      'arity': 2,     'xtype': 'f2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$\\neg$'},
      ast.NotEq: {'name': '!=',     'arity': 2,     'xtype': 'f2b', 	'tf': tf.not_equal, 		'gpbp': ['ö'], 	'latex': None},  # e.g., a != b
      '<':      {'name': '<',       'arity': 2,     'xtype': 'f2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$<$'},
      ast.Lt:   {'name': '<',       'arity': 2,     'xtype': 'f2b', 	'tf': tf.less, 				'gpbp': ['ö'], 	'latex': None},  # e.g., a < b
      '<=':     {'name': '<=',      'arity': 2,     'xtype': 'f2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$<=$'},
      ast.LtE:  {'name': '<=',      'arity': 2,     'xtype': 'f2b', 	'tf': tf.less_equal, 		'gpbp': ['ö'], 	'latex': None},  # e.g., a <= b
      '>':      {'name': '>',       'arity': 2,     'xtype': 'f2b', 	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$>$'},
      ast.Gt:   {'name': '>',       'arity': 2,     'xtype': 'f2b', 	'tf': tf.greater, 			'gpbp': ['ö'], 	'latex': None},  # e.g., a > b
      '>=':     {'name': '>=',      'arity': 2,     'xtype': 'f2b',   	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': '$>=$'},
      ast.GtE:  {'name': '>=',      'arity': 2,     'xtype': 'f2b',   	'tf': tf.greater_equal, 	'gpbp': ['ö'], 	'latex': None},  # e.g., a >= 1

      # Functions which need separate handling in sympify
      'Ifte': {'name': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'gpbp': ['ö'], 'latex': 'if'},
      'Mini': {'name': 'Mini', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum, 'gpbp': ['ö'], 'latex': '$\\min$'},  # e.g. Mini(a, b)
      'Maxi': {'name': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum, 'gpbp': ['ö'], 'latex': '$\\max$'},  # e.g. Maxi(a, b)

            # Not tested: Converters: Dummy operators that convert between float and bool
      'Ftob':   {'name': 'Ftob',    'arity': 1,     'xtype': 'f2b',   	'tf': tf.bool, 				'gpbp': ['ö'], 	'latex': None},  # not tested
      'Btof':   {'name': 'Btof',    'arity': 1,     'xtype': 'b2f',   	'tf': tf.float32, 			'gpbp': ['ö'], 	'latex': None},  # not tested

            # Never used yet, trying to get rid of the ** function
      'Power':  {'name': 'Power',   'arity': 1,     'xtype': 'f2f',   	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},  # not used # todo # a*a*a -> a**3
      'inverse': {'name': 'inverse', 'arity': 1,    'xtype': 'f2f',  	'tf': 'ä', 					'gpbp': ['ö'], 	'latex': None},  # not used # 1/float. important for squareroots. todo

            # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
      'float':  {'name': 'float',   'arity': 0,     'xtype': '2f',      'tf': 'ä',                  'gpbp': ['ö'], 'latex': None},  # not tested
      'int':    {'name': 'int',     'arity': 0,     'xtype': '2f',      'tf': 'ä',                  'gpbp': ['ö'], 'latex': None},  # not tested
      'bool':   {'name': 'bool',    'arity': 0,     'xtype': '2b',      'tf': 'ä',                  'gpbp': ['ö'], 'latex': None},  # not tested

            # Never used yet, todo. Trying to introduce multi-dimensional outputs that should
      'vector': {'name': 'vector',  'arity': 1, 'xtype': 'ü', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None},  # not tested # sfeh: not working

            # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
            # Also very unclear how they should work.
            # - Let user specify them completely. Limit use to 1 (or 2) per tree.
            # - temporary variable(s) must be introduced that change within loop
            # - Condition must be change within loop
      'while':  {'name': 'while',   'arity': 2, 'xtype': 'b2?2?',        'tf': 'ä',                 'gpbp': ['ö'], 'latex': None},  # sfeh: not working # Condition must change in loop
      'repeat_n': {'name': 'repeat_n', 'arity': 2, 'xtype': 'b2?',       'tf': 'ä',                 'gpbp': ['ö'], 'latex': None},  # sfeh: not working # repeat n time, specify n (int) by user?
      }

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
