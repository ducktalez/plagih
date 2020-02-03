import tensorflow as tf
import ast


fitt_dict = {'classification': 'max',
             'regression': 'min',
             'match': 'max'}

FIRST_TREE = 1

TR_ID = 0
TR_type = 1
# TR_depth = 2
N_id = 3
N_depth = 4
N_type = 5
N_label = 6
N_parent = 7
N_arity = 8
N_c1 = 9
N_c2 = 10
N_c3 = 11
T_fitness = 12
N_modify = 13
T_parsimony = 14
T_num_lines = 15
P_first_node = 1
root_id = 1

f2f, f2b, b2b, b2f, b2f2f = 0, 1, 2, 3, 4

ast_tensor_dict = {ast.Add: tf.add,  # e.g., a + b
                   ast.Sub: tf.subtract,  # e.g., a - b
                   ast.Mult: tf.multiply,  # e.g., a * b
                   ast.Div: tf.divide,  # e.g., a / b
                   ast.Pow: tf.pow,  # e.g., a ** 2
                   ast.USub: tf.negative,  # e.g., -a
                   ast.And: tf.logical_and,  # e.g., a and b
                   ast.Or: tf.logical_or,  # e.g., a or b
                   ast.Not: tf.logical_not,  # e.g., not a
                   ast.Eq: tf.equal,  # e.g., a == b
                   ast.NotEq: tf.not_equal,  # e.g., a != b
                   ast.Lt: tf.less,  # e.g., a < b
                   ast.LtE: tf.less_equal,  # e.g., a <= b
                   ast.Gt: tf.greater,  # e.g., a > b
                   ast.GtE: tf.greater_equal,  # e.g., a >= 1
                   ast.BitAnd: tf.logical_and,  # DON'T USE tf.bitwise.bitwise_and
                   'abs': tf.abs,  # e.g., abs(a)
                   'sign': tf.sign,  # e.g., sign(a)
                   'square': tf.square,  # e.g., square(a)
                   'sqrt': tf.sqrt,  # e.g., sqrt(a)
                   'pow': tf.pow,  # e.g., pow(a, b)
                   'log': tf.math.log,  # e.g., log(a)
                   'log1p': tf.math.log1p,  # e.g., log1p(a)
                   'cos': tf.cos,  # e.g., cos(a)
                   'sin': tf.sin,  # e.g., sin(a)
                   'tan': tf.tan,  # e.g., tan(a)
                   'acos': tf.acos,  # e.g., acos(a)
                   'asin': tf.asin,  # e.g., asin(a)
                   'atan': tf.atan,  # e.g., atan(a)
                   'Ifte': tf.compat.v2.where,  # e.g., Ifte(a, b, c)
                   'Mini': tf.math.minimum,  # if reduce_min does not work...
                   'Maxi': tf.math.maximum}

op = {'float': {'name': 'float', 'arity': 0, 'xtype': '2f', 'tf': 'ä'},
      'int': {'name': 'int', 'arity': 0, 'xtype': '2f', 'tf': 'ä'},
      'bool': {'name': 'bool', 'arity': 0, 'xtype': '2b', 'tf': 'ä'},

      '+': {'name': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add},
      ast.Add: {'name': '+', 'arity': 2, 'xtype': 'f2f', 'tf': tf.add},  # e.g., a + b
      '-': {'name': '-', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä'},
      ast.Sub: {'name': '-', 'arity': 2, 'xtype': 'f2f', 'tf': tf.subtract},  # e.g., a - b
      '~': {'name': '~', 'arity': 1, 'xtype': 'f2f', 'tf': 'ä'},
      ast.USub: {'name': '~', 'arity': 1, 'xtype': 'f2f', 'tf': tf.negative},  # e.g., -a
      '*': {'name': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply},
      ast.Mult: {'name': '*', 'arity': 2, 'xtype': 'f2f', 'tf': tf.multiply},  # e.g., a * b
      '/': {'name': '/', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä'},
      ast.Div: {'name': '/', 'arity': 2, 'xtype': 'f2f', 'tf': tf.divide},  # e.g., a / b
      '**': {'name': '**', 'arity': 2, 'xtype': 'f2f', 'tf': 'ä'},
      ast.Pow: {'name': '**', 'arity': 2, 'xtype': 'f2f', 'tf': tf.pow},  # e.g., a ** 2
      'abs': {'name': 'abs', 'arity': 1, 'xtype': 'f2f', 'tf': tf.abs},
      'sign': {'name': 'sign', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sign},
      'square': {'name': 'square', 'arity': 1, 'xtype': 'f2f', 'tf': tf.square},
      'sqrt': {'name': 'sqrt', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt},
      'log': {'name': 'log', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log},
      'log1p': {'name': 'log1p', 'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p},
      'cos': {'name': 'cos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.cos},
      'sin': {'name': 'sin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.sin},
      'tan': {'name': 'tan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan},
      'acos': {'name': 'acos', 'arity': 1, 'xtype': 'f2f', 'tf': tf.acos},
      'asin': {'name': 'asin', 'arity': 1, 'xtype': 'f2f', 'tf': tf.asin},
      'atan': {'name': 'atan', 'arity': 1, 'xtype': 'f2f', 'tf': tf.atan},

      'And': {'name': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      ast.And: {'name': 'And', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and},  # e.g., a and b
      '&': {'name': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and},
      ast.BitAnd: {'name': '&', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_and},  # DON'T USE tf.bitwise.bitwise_and
      'Or': {'name': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      ast.Or: {'name': 'Or', 'arity': 2, 'xtype': 'b2b', 'tf': tf.logical_or},  # e.g., a or b
      'Xor': {'name': 'Xor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      'Nand': {'name': 'Nand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      'Xand': {'name': 'Xand', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      'Nor': {'name': 'Nor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      'Xnor': {'name': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'tf': 'ä'},
      'Not': {'name': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': 'ä'},
      ast.Not: {'name': 'Not', 'arity': 1, 'xtype': 'b2b', 'tf': tf.logical_not},  # e.g., not a

      '==': {'name': '==', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä'},
      ast.Eq: {'name': '==', 'arity': 2, 'xtype': 'f2b', 'tf': tf.equal},  # e.g., a == b
      '!=': {'name': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä'},
      ast.NotEq: {'name': '!=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.not_equal},  # e.g., a != b
      '<': {'name': '<', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä'},
      ast.Lt: {'name': '<', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less},  # e.g., a < b
      '<=': {'name': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä'},
      ast.LtE: {'name': '<=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.less_equal},  # e.g., a <= b
      '>': {'name': '>', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä'},
      ast.Gt: {'name': '>', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater},  # e.g., a > b
      '>=': {'name': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': 'ä'},
      ast.GtE: {'name': '>=', 'arity': 2, 'xtype': 'f2b', 'tf': tf.greater_equal},  # e.g., a >= 1

      'Ftob': {'name': 'Ftob', 'arity': 1, 'xtype': 'f2b', 'tf': tf.bool},
      'Btof': {'name': 'Btof', 'arity': 1, 'xtype': 'b2f', 'tf': tf.float32},

      'Ifte': {'name': 'Ifte', 'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where},  # Note that boolean if's can be realized with boolean operators. (Or ITE())
      'Mini': {'name': 'Mini', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.minimum},
      'Maxi': {'name': 'Maxi', 'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum},
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
