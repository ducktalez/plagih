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

op = {'float':  {'name': 'float', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'gpbp': ['ö']},
      'int':    {'name': 'int', 'arity': 0, 'xtype': '2f', 'tf': 'ä', 'gpbp': ['ö']},
      'bool':   {'name': 'bool', 'arity': 0, 'xtype': '2b', 'tf': 'ä', 'gpbp': ['ö']},

      'vector':  {'name': 'vec_a', 'arity': 1, 'xtype': 'ü', 'tf': 'ä', 'gpbp': ['ö']},  # sfeh: not working

      '+':      {'name': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'gpbp': ['ö']},
      ast.Add:  {'name': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add, 'gpbp': ['ö']},  # e.g., a + b
      '-':      {'name': '-', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Sub:  {'name': '-', 'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract, 'gpbp': ['ö']},  # e.g., a - b
      '~':      {'name': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},
      'usub':   {'name': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},
      ast.USub: {'name': '~', 'arity': 1, 'xtype': 'f2f', 'tf': tf.negative, 'gpbp': ['ö']},  # e.g., -a
      '*':      {'name': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'gpbp': ['ö']},
      ast.Mult: {'name': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply, 'gpbp': ['ö']},  # e.g., a * b
      '/':      {'name': '/', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Div:  {'name': '/', 'arity': 2, 'xtype': 'f2f', 'tf': tf.divide, 'gpbp': ['ö']},  # e.g., a / b
      '**':     {'name': '**', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Pow:  {'name': '**', 'arity': 2, 'xtype': 'f2f', 'tf': tf.pow, 'gpbp': ['ö']},  # e.g., a ** 2
      'abs':    {'name': 'abs', 'arity': 1, 'xtype': 'f2f', 'tf': tf.abs, 'gpbp': ['ö']},
      'sign':   {'name': 'sign', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sign, 'gpbp': ['ö']},
      'square': {'name': 'square', 'arity': 1, 'xtype': 'f2f', 'tf': tf.square, 'gpbp': ['ö']},
      'sqrt':   {'name': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt, 'gpbp': ['ö']},
      'log':    {'name': 'log', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log, 'gpbp': ['ö']},
      'log1p':  {'name': 'log1p', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p, 'gpbp': ['ö']},
      'cos':    {'name': 'cos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.cos, 'gpbp': ['ö']},
      'sin':    {'name': 'sin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sin, 'gpbp': ['ö']},
      'tan':    {'name': 'tan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'gpbp': ['ö']},
      'acos':   {'name': 'acos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.acos, 'gpbp': ['ö']},
      'asin':   {'name': 'asin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.asin, 'gpbp': ['ö']},
      'atan':   {'name': 'atan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan, 'gpbp': [1]},

      'And':    {'name': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.And:  {'name': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö']},  # e.g., a and b
      '&':      {'name': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö']},
      ast.BitAnd: {'name': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and, 'gpbp': ['ö']},  # DON'T USE tf.bitwise.bitwise_and
      'Or':     {'name': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Or:   {'name': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or, 'gpbp': ['ö']},  # e.g., a or b
      'Xor':    {'name': 'Xor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      'Nand':   {'name': 'Nand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      'Xand':   {'name': 'Xand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      'Nor':    {'name': 'Nor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      'Xnor':   {'name': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      'Not':    {'name': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Not:  {'name': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not, 'gpbp': ['ö']},  # e.g., not a

      '==':     {'name': '==', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Eq:   {'name': '==', 'arity': 2, 'xtype': 'f2b', 'tf': tf.equal, 'gpbp': ['ö']},  # e.g., a == b
      '!=':     {'name': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.NotEq: {'name': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal, 'gpbp': ['ö']},  # e.g., a != b
      '<':      {'name': '<', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Lt:   {'name': '<', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less, 'gpbp': ['ö']},  # e.g., a < b
      '<=':     {'name': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.LtE:  {'name': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal, 'gpbp': ['ö']},  # e.g., a <= b
      '>':      {'name': '>', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.Gt:   {'name': '>', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater, 'gpbp': ['ö']},  # e.g., a > b
      '>=':     {'name': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä', 'gpbp': ['ö']},
      ast.GtE:  {'name': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater_equal, 'gpbp': ['ö']},  # e.g., a >= 1

      'Ftob':   {'name': 'Ftob', 'arity': 1, 'xtype': 'f2b', 'tf': tf.bool, 'gpbp': ['ö']},
      'Btof':   {'name': 'Btof', 'arity': 1, 'xtype': 'b2f', 'tf': tf.float32, 'gpbp': ['ö']},

      'Power':  {'name': 'Power', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},  # todo
      'inverse': {'name': 'inverse', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä', 'gpbp': ['ö']},  # 1/float. important for squareroots. todo

      'Ifte':   {'name': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where, 'gpbp': ['ö']},
      'Mini':   {'name': 'Mini', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum, 'gpbp': ['ö']},
      'Maxi':   {'name': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum, 'gpbp': ['ö']},
      }
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
