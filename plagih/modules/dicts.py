import tensorflow as tf
import ast
from pathlib import Path


fitt_dict = {'classification': 'max',
             'regression': 'min',
             'regression bounded': 'min',
             'match': 'max'}

FIRST_TREE = 1
first_action = 'action0'


delete_this = True

# ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f'].index('f2f')
f2f, f2b, b2b, b2f, b2f2f = 0, 1, 2, 3, 4

op = {'float':  {'name': 'float', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None},
      'int':    {'name': 'int', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None},
      'bool':   {'name': 'bool', 'arity': 0, 'xtype': '2b', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None},

      'vector':  {'name': 'vec_a', 'arity': 1, 'xtype': 'ü', 'tf': 'ä', 'gpbp': ['ö'], 'latex': None},  # sfeh: not working

      '+':      {'name': '+',       'arity': 2, 'xtype': 'f2f', 'tf': tf.add,       'gpbp': ['ö'], 'latex': '$+$'},
      ast.Add:  {'name': '+',       'arity': 2, 'xtype': 'f2f', 'tf': tf.add,       'gpbp': ['ö'], 'latex': None},  # e.g., a + b
      '-':      {'name': '-',       'arity': 2, 'xtype': 'f2f', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$-$'},
      ast.Sub:  {'name': '-',       'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract, 'gpbp': ['ö'], 'latex': None},  # e.g., a - b
      '~':      {'name': '~',       'arity': 1, 'xtype': 'f2f', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$-$'},  # todo this is minus again
      'usub':   {'name': '~',       'arity': 1, 'xtype': 'f2f', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      ast.USub: {'name': '~',       'arity': 1, 'xtype': 'f2f', 'tf': tf.negative, 'gpbp': ['ö'], 'latex': None},  # e.g., -a
      '*':      {'name': '*',       'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'gpbp': ['ö'], 'latex': '$\\cdot$'},
      ast.Mult: {'name': '*',       'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'gpbp': ['ö'], 'latex': None},  # e.g., a * b
      '/':      {'name': '/',       'arity': 2, 'xtype': 'f2f', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$\\div$'},
      ast.Div:  {'name': '/',       'arity': 2, 'xtype': 'f2f', 'tf': tf.divide,    'gpbp': ['ö'], 'latex': None},  # e.g., a / b
      '**':     {'name': '**',      'arity': 2, 'xtype': 'f2f', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$**$'},
      ast.Pow:  {'name': '**',      'arity': 2, 'xtype': 'f2f', 'tf': tf.pow,       'gpbp': ['ö'], 'latex': None},  # e.g., a ** 2
      'abs':    {'name': 'abs',     'arity': 1, 'xtype': 'f2f', 'tf': tf.abs,       'gpbp': ['ö'], 'latex': None},
      'sign':   {'name': 'sign',    'arity': 1, 'xtype': 'f2f', 'tf': tf.sign,      'gpbp': ['ö'], 'latex': None},
      'square': {'name': 'square',  'arity': 1, 'xtype': 'f2f', 'tf': tf.square,    'gpbp': ['ö'], 'latex': None},
      'sqrt':   {'name': 'sqrt',    'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt,      'gpbp': ['ö'], 'latex': None},
      'log':    {'name': 'log',     'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log,  'gpbp': ['ö'], 'latex': None},
      'log1p':  {'name': 'log1p',   'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p, 'gpbp': ['ö'], 'latex': None},
      'cos':    {'name': 'cos',     'arity': 1, 'xtype': 'f2f', 'tf': tf.cos,       'gpbp': ['ö'], 'latex': None},
      'sin':    {'name': 'sin',     'arity': 1, 'xtype': 'f2f', 'tf': tf.sin,       'gpbp': ['ö'], 'latex': '$\\sin$'},
      'tan':    {'name': 'tan',     'arity': 1, 'xtype': 'f2f', 'tf': tf.atan,      'gpbp': ['ö'], 'latex': None},
      'acos':   {'name': 'acos',    'arity': 1, 'xtype': 'f2f', 'tf': tf.acos,      'gpbp': ['ö'], 'latex': None},
      'asin':   {'name': 'asin',    'arity': 1, 'xtype': 'f2f', 'tf': tf.asin,      'gpbp': ['ö'], 'latex': None},
      'atan':   {'name': 'atan',    'arity': 1, 'xtype': 'f2f', 'tf': tf.atan,      'gpbp': [1], 'latex': None},

      'And':    {'name': 'And',     'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      ast.And:  {'name': 'And',     'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö'], 'latex': None},  # e.g., a and b
      '&':      {'name': '&',       'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö'], 'latex': '$\\land$'},
      ast.BitAnd: {'name': '&',     'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö'], 'latex': None},  # DON'T USE tf.bitwise.bitwise_and
      'Or':     {'name': 'Or',      'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$\\lor$'},
      ast.Or:   {'name': 'Or',      'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'gpbp': ['ö'], 'latex': None},  # e.g., a or b
      'Xor':    {'name': 'Xor',     'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      'Nand':   {'name': 'Nand',    'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      'Xand':   {'name': 'Xand',    'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      'Nor':    {'name': 'Nor',     'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      'Xnor':   {'name': 'Xnor',    'arity': 2, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      'Not':    {'name': 'Not',     'arity': 1, 'xtype': 'b2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': None},
      ast.Not:  {'name': 'Not',     'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not, 'gpbp': ['ö'], 'latex': '$\\neg$'},  # e.g., not a

      '==':     {'name': '==',      'arity': 2, 'xtype': 'f2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$==$'},
      ast.Eq:   {'name': '==',      'arity': 2, 'xtype': 'f2b', 'tf': tf.equal,     'gpbp': ['ö'], 'latex': None},  # e.g., a == b
      '!=':     {'name': '!=',      'arity': 2, 'xtype': 'f2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$\\neg$'},
      ast.NotEq: {'name': '!=',     'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal, 'gpbp': ['ö'], 'latex': None},  # e.g., a != b
      '<':      {'name': '<',       'arity': 2, 'xtype': 'f2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$<$'},
      ast.Lt:   {'name': '<',       'arity': 2, 'xtype': 'f2b', 'tf': tf.less,      'gpbp': ['ö'], 'latex': None},  # e.g., a < b
      '<=':     {'name': '<=',      'arity': 2, 'xtype': 'f2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$<=$'},
      ast.LtE:  {'name': '<=',      'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal, 'gpbp': ['ö'], 'latex': None},  # e.g., a <= b
      '>':      {'name': '>',       'arity': 2, 'xtype': 'f2b', 'tf': 'ä',          'gpbp': ['ö'], 'latex': '$>$'},
      ast.Gt:   {'name': '>',       'arity': 2, 'xtype': 'f2b', 'tf': tf.greater,   'gpbp': ['ö'], 'latex': None},  # e.g., a > b
      '>=':     {'name': '>=',      'arity': 2, 'xtype': 'f2b',   'tf': 'ä',        'gpbp': ['ö'], 'latex': '$>=$'},
      ast.GtE:  {'name': '>=',      'arity': 2, 'xtype': 'f2b',   'tf': tf.greater_equal, 'gpbp': ['ö'], 'latex': None},  # e.g., a >= 1

      'Ftob':   {'name': 'Ftob',    'arity': 1, 'xtype': 'f2b',   'tf': tf.bool,    'gpbp': ['ö'], 'latex': None},
      'Btof':   {'name': 'Btof',    'arity': 1, 'xtype': 'b2f',   'tf': tf.float32, 'gpbp': ['ö'], 'latex': None},

      'Power':  {'name': 'Power',   'arity': 1, 'xtype': 'f2f',   'tf': 'ä',        'gpbp': ['ö'], 'latex': None},  # todo
      'inverse': {'name': 'inverse', 'arity': 1, 'xtype': 'f2f',  'tf': 'ä',        'gpbp': ['ö'], 'latex': None},  # 1/float. important for squareroots. todo

      'Ifte':   {'name': 'Ifte',    'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'gpbp': ['ö'], 'latex': 'if'},
      'Mini':   {'name': 'Mini',    'arity': 2, 'xtype': 'f2f',   'tf': tf.math.minimum, 'gpbp': ['ö'], 'latex': '$\\min$'},
      'Maxi':   {'name': 'Maxi',    'arity': 2, 'xtype': 'f2f',   'tf': tf.math.maximum, 'gpbp': ['ö'], 'latex': '$\\max$'},
      }

# print([x for x in op.keys() if type(x) == type('asd')])  # retreive a list with all non-ast ops:

functions_infix_dict = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=', '&']

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
