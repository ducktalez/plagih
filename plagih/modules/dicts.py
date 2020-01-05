import tensorflow as tf
import ast


class BColors:  # sfeh can be deleted
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[39m'

    BLACK2 = '\033[40m'
    RED2 = '\033[41m'


fitt_dict = {'classification': 'max',
             'regression': 'min',
             'match': 'max'}


TR_ID = 0
TR_type = 1
TR_depth = 2
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
                   'Maxi': tf.math.maximum,
                   }
op = {
    'float': {'arity': 0, 'xtype': '2f', 'tf': ''},
    'int': {'arity': 0, 'xtype': '2f', 'tf': ''},
    'bool': {'arity': 0, 'xtype': '2b', 'tf': ''},

    '+': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '-': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '*': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '/': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '**': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    'abs': {'arity': 1, 'xtype': 'f2f', 'tf': tf.abs},
    'sign': {'arity': 1, 'xtype': 'f2f', 'tf': tf.sign},
    'square': {'arity': 1, 'xtype': 'f2f', 'tf': tf.square},
    'sqrt': {'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt},
    'log': {'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log},
    'log1p': {'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p},
    'cos': {'arity': 1, 'xtype': 'f2f', 'tf': tf.cos},
    'sin': {'arity': 1, 'xtype': 'f2f', 'tf': tf.sin},
    'tan': {'arity': 1, 'xtype': 'f2f', 'tf': tf.atan},
    'acos': {'arity': 1, 'xtype': 'f2f', 'tf': tf.acos},
    'asin': {'arity': 1, 'xtype': 'f2f', 'tf': tf.asin},
    'atan': {'arity': 1, 'xtype': 'f2f', 'tf': tf.atan},

    'And': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Or': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Xor': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Nand': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Xand': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Nor': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Xnor': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Not': {'arity': 1, 'xtype': 'b2b', 'tf': ''},

    '==': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '!=': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '<': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '<=': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '>': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '>=': {'arity': 2, 'xtype': 'f2b', 'tf': ''},

    'Ftob': {'arity': 1, 'xtype': 'f2b', 'tf': ''},
    'Btof': {'arity': 1, 'xtype': 'b2f', 'tf': ''},

    'Ifte': {'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where},  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Mini': {'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum},
    'Maxi': {'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum},
}

function_arity_dict = {  # Needs A LOT OF further testing
    'float': 0,  # these three are dummies
    'int': 0,  # neede to use the dict for function types aswell
    'bool': 0,  # so we can "workarounded" use them

    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    '**': 2,
    'abs': 1,
    'sign': 1,
    'square': 1,
    'sqrt': 1,
    'log': 1,
    'log1p': 1,
    'cos': 1,
    'sin': 1,
    'tan': 1,
    'acos': 1,
    'asin': 1,
    'atan': 1,

    'And': 2,
    'Or': 2,
    'Xor': 2,
    'Nand': 2,
    'Xand': 2,
    'Nor': 2,
    'Xnor': 2,
    'Not': 1,

    '==': 2,
    '!=': 2,
    '<': 2,
    '<=': 2,
    '>': 2,
    '>=': 2,

    'Ftob': 1,
    'Btof': 1,  # False->0, True->1, dummy-function
    'Btof_extreme': 1,  # False->-1, True->1. Does that make sense?

    'Ifte': 3,  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Mini': 2,
    'Maxi': 2,
}

functions_wrap_dict = ['Mini', 'Maxi', 'abs', 'sign', 'square', 'sqrt', 'log', 'log1p', 'cos', 'sin', 'tan', 'acos', 'asin', 'atan']
functions_infix_dict = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=']

# op_xtype_dict = {  # Needs A LOT OF further testing
#     'float': '2f',  # these three are dummies
#     'int': '2f',  # needed to use the dict for function types as well
#     'bool': '2b',  # so we can "work around" use them
#
#     '+': 'f2f',
#     '-': 'f2f',
#     '*': 'f2f',
#     '/': 'f2f',
#     '**': 'f2f',
#     'abs': 'f2f',
#     'sign': 'f2f',
#     'square': 'f2f',
#     'sqrt': 'f2f',
#     'log': 'f2f',
#     'log1p': 'f2f',
#     'cos': 'f2f',
#     'sin': 'f2f',
#     'tan': 'f2f',
#     'acos': 'f2f',
#     'asin': 'f2f',
#     'atan': 'f2f',
#
#     'And': 'b2b',
#     'Or': 'b2b',
#     'Xor': 'b2b',
#     'Nand': 'b2b',
#     'Xand': 'b2b',
#     'Nor': 'b2b',
#     'Xnor': 'b2b',
#     'Not': 'b2b',
#     'ITE': 'b2b',
#
#     '==': 'f2b',
#     '!=': 'f2b',
#     '<': 'f2b',
#     '<=': 'f2b',
#     '>': 'f2b',
#     '>=': 'f2b',
#
#     'Ftob': 'f2b',
#     'Btof': 'b2f',  # False->0, True->1, dummy-function
#     'Btof_extreme': 'b2f',  # False->-1, True->1. Does that make sense?
#
#     'Ifte': 'b2f2f',  # Note that boolean if's can be realized with boolean operators. (Or ITE())
#     'Mini': 'f2f',
#     'Maxi': 'f2f',
# }
