import tensorflow as tf
import ast
from pathlib import Path

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
op = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': 	{'fun': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'gpbp': ['ö'], 'latex': '$+$', 'call': 'inline', 				'pycode': '({}+{})'},
    ast.Add: 	{'fun': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 				    'pycode': None},
    '-': 	{'fun': '-', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$-$', 'call': 'inline', 			    	'pycode': '({}-{})'},
    ast.Sub: 	{'fun': '-', 'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 			'pycode': None},
    '~': 	{'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$-$', 'call': 'inline', 				    'pycode': '(-{})'},  # todo this is minus again
    'usub': 	{'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 			      	'pycode': None},
    ast.USub: 	{'fun': '~', 'arity': 1, 'xtype': 'f2f', 'tf': tf.negative, 'gpbp': ['ö'], 'latex': None, 'call': None, 	    	'pycode': None},
    '*': 	{'fun': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'gpbp': ['ö'], 'latex': '$\\cdot$', 'call': 'inline',   	'pycode': '({}*{})'},
    ast.Mult: 	{'fun': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 	    'pycode': None},  # a * b
    '/': 	{'fun': '/', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$\\div$', 'call': 'inline', 		    	'pycode': '({}/{})'},
    ast.Div: 	{'fun': '/', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.divide_no_nan, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 'pycode': None},  # a / b  # todo try tf.math.divide_no_nan
    '**': 	{'fun': '**', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$**$', 'call': 'inline', 			    	'pycode': '({}*{})'},
    ast.Pow: 	{'fun': '**', 'arity': 2, 'xtype': 'f2f', 'tf': tf.pow, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 			'pycode': None},  # a ** 2
    'abs': 	{'fun': 'abs', 'arity': 1, 'xtype': 'f2f', 'tf': tf.abs, 'gpbp': ['ö'], 'latex': None, 'call': None, 			    	'pycode': 'abs({})'},
    'sign': 	{'fun': 'sign', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sign, 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': None},
    'square': 	{'fun': 'square', 'arity': 1, 'xtype': 'f2f', 'tf': tf.square, 'gpbp': ['ö'], 'latex': None, 'call': None, 			'pycode': '(({})**2)'},
    'sqrt': 	{'fun': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt, 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'math.sqrt({})'},
    'log': 	{'fun': 'log', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log, 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'math.log({})'},
    'log1p': 	{'fun': 'log1p', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p, 'gpbp': ['ö'], 'latex': None, 'call': None, 		'pycode': 'math.log1p()'},
    'cos': 	{'fun': 'cos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.cos, 'gpbp': ['ö'], 'latex': None, 'call': None, 			    	'pycode': 'math.cos({})'},
    'sin': 	{'fun': 'sin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sin, 'gpbp': ['ö'], 'latex': '$\\sin$', 'call': None, 				'pycode': 'math.sin({})'},
    'tan': 	{'fun': 'tan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'gpbp': ['ö'], 'latex': None, 'call': None, 			    	'pycode': 'math.tan({})'},
    'acos':     {'fun': 'acos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.acos, 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'math.acos({})'},
    'asin': 	{'fun': 'asin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.asin, 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'math.asin({})'},
    'atan': 	{'fun': 'atan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'gpbp': [1], 'latex': None, 'call': None, 				'pycode': 'math.atan({})'},
    'tanh': 	{'fun': 'tanh', 'arity': 1, 'xtype': 'f2f', 'tf': tf.tanh, 'gpbp': [1], 'latex': None, 'call': None, 				'pycode': 'math.tanh({})'},
    # todo round operation! sympify: N(1.2345, decimals). e.g. Int

    # 'b2b' Classical logical operators, evaluate from bool to bool
    'And': 	{'fun': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				        'pycode': '({} and {})'},
    ast.And: 	{'fun': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö'], 'latex': None, 'call': None, 		'pycode': None},  # a and b
    '&': 	{'fun': 'bitwise', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö'], 'latex': '$\\land$', 'call': 'inline', 	'pycode': '({} & {})'},
    ast.BitAnd: 	{'fun': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 	'pycode': None},  # DON'T USE tf.bitwise.bitwise_and
    'Or': 	{'fun': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$\\lor$', 'call': None, 			    	'pycode': '({} or {})'},
    ast.Or: 	{'fun': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'gpbp': ['ö'], 'latex': None, 'call': None, 			'pycode': None},  # a or b
    '|': 	{'fun': '|', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 			'pycode': '({} | {})'},  # a or b
    ast.BitOr: 	{'fun': '|', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 		'pycode': None},  # a or b
    # ast.BitOr
    'Xor': 	{'fun': 'Xor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 			    	'pycode': '({} ^ {})'},
    'Nand': 	{'fun': 'Nand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'not({} and {})'},
    'Xand': 	{'fun': 'Xand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'not({} ^ {})'},
    'Nor': 	{'fun': 'Nor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				    'pycode': None},
    'Xnor': 	{'fun': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': None},
    'Not': 	{'fun': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				    'pycode': None},
    ast.Not: 	{'fun': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not, 'gpbp': ['ö'], 'latex': '$\\neg$', 'call': None, 				'pycode': None},  # not a

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==': 	{'fun': '==', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$==$', 'call': 'inline', 				'pycode': '({}=={})'},
    ast.Eq: 	{'fun': '==', 'arity': 2, 'xtype': 'f2b', 'tf': tf.equal, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 		'pycode': '({}=={})'},  # a == b
    '!=': 	{'fun': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$\\neg$', 'call': 'inline', 			'pycode': '({}!={})'},
    ast.NotEq: 	{'fun': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal, 'gpbp': ['ö'], 'latex': None, 'call': 'inline',	'pycode': '({}!={})'},  # a != b
    '<': 	{'fun': '<', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$<$', 'call': 'inline', 				'pycode': '({}<{})'},
    ast.Lt: 	{'fun': '<', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 		'pycode': '({}<{})'},  # a < b
    '<=': 	{'fun': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$<=$', 'call': 'inline', 				'pycode': '({}<={})'},
    ast.LtE: 	{'fun': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 	'pycode': '({}<={})'},  # a <= b
    '>': 	{'fun': '>', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$>$', 'call': 'inline', 				'pycode': '({}>{})'},
    ast.Gt: 	{'fun': '>', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 		'pycode': '({}>{})'},  # a > b
    '>=': 	{'fun': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': '$>=$', 'call': 'inline', 				'pycode': '({}>={})'},
    ast.GtE: 	{'fun': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater_equal, 'gpbp': ['ö'], 'latex': None, 'call': 'inline', 	'pycode': '({}>={})'},  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte': 	{'fun': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'gpbp': ['ö'], 'latex': 'if', 'call': None, 'pycode': 'if {}:\n{}\nelse:{}'},
    'Mini': 	{'fun': 'Mini', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum, 'gpbp': ['ö'], 'latex': '$\\min$', 'call': None, 'pycode': 'min({})'},  # maximum
    'Maxi': 	{'fun': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum, 'gpbp': ['ö'], 'latex': '$\\max$', 'call': None, 'pycode': 'max({})'},  # minimum

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float': 	{'fun': 'float', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'float({})'},  # not tested
    'int': 	{'fun': 'int', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				    'pycode': 'int({})'},  # not tested
    'bool': 	{'fun': 'bool', 'arity': 0, 'xtype': '2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': 'bool({})'},  # not tested
}

op_test = {

    # Not tested: Converters: Dummy operators that convert between float and bool
    'Ftob': 	{'fun': 'Ftob', 'arity': 1, 'xtype': 'f2b', 'tf': tf.bool, 'gpbp': ['ö'], 'latex': None, 'call': 'bool', 			'pycode': None},  # not tested, same as bool
    'Btof': 	{'fun': 'Btof', 'arity': 1, 'xtype': 'b2f', 'tf': tf.float32, 'gpbp': ['ö'], 'latex': None, 'call': 'float', 		'pycode': None},  # not tested

    # Never used yet, trying to get rid of the ** function
    'Power': 	{'fun': 'Power', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': '(({})**{})', 			'pycode': None},  # not used # todo # a*a*a -> a**3
    'inverse': 	{'fun': 'inverse', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 			'pycode': None},  # not used # 1/float. important for squareroots. todo

    # Never used yet, todo. Trying to introduce multi-dimensional outputs that should
    'vector': 	{'fun': 'vector', 'arity': 1, 'xtype': 'ü', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 				'pycode': None},  # not tested # sfeh: not working

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while': 	{'fun': 'while', 'arity': 2, 'xtype': 'b2?2?', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 			'pycode': None},  # sfeh: not working # Condition must change in loop
    'repeat_n': 	{'fun': 'repeat_n', 'arity': 2, 'xtype': 'b2?', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None, 'call': None, 		'pycode': None},  # sfeh: not working # repeat n time, specify n (int) by user?
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
