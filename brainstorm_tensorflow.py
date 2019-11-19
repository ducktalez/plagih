from karoo.modules.plagih_sympy_extras import plagih_sympify
from karoo.modules.plagih_gp_base_class_xai import operators, non_inline_multielem_functions
import tensorflow as tf
import ast


def fx_fitness_chain_bool(values, operation, tensors):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

    Called by: plagih_fitness_node_parse

    Arguments required: values, operation, tensors
    """

    x = tf.dtypes.cast(self.plagih_fitness_node_parse(values[0], tensors), tf.bool)
    if len(values) > 1:
        return operation(x, fx_fitness_chain_bool(values[1:], operation, tensors))
    else:
        return x


def fx_fitness_chain_compare(comparators, ops, tensors):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    Called by: plagih_fitness_node_parse

    Arguments required: comparators, ops, tensors
    """

    x = plagih_fitness_node_parse(comparators[0], tensors)
    y = plagih_fitness_node_parse(comparators[1], tensors)

    if len(comparators) > 2:
        return tf.logical_and(operators[type(ops[0])](x, y), fx_fitness_chain_compare(comparators[1:], ops[1:], tensors))
    else:
        return operators[type(ops[0])](x, y)
    # sfeh idea: note: we have to convert all values to the action space if not discrete


def plagih_fitness_node_parse(node, tensors):
    """
    Recursively transforms parsed expression tree into TensorFlow (TF) graph.
    """

    if isinstance(node, ast.Name):  # <tensor_name>
        return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        shape = tensors[list(tensors.keys())[0]].get_shape()
        return tf.constant(node.n, shape=shape, dtype=tf.float32)

    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>, e.g., x + y
        return operators[type(node.op)](plagih_fitness_node_parse(node.left, tensors), plagih_fitness_node_parse(node.right, tensors))

    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](plagih_fitness_node_parse(node.operand, tensors))

    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or ftob(a)
        if node.func.id == 'Ifte':
            return operators[node.func.id](
                tf.dtypes.cast(plagih_fitness_node_parse(node.args[0], tensors), tf.bool),
                plagih_fitness_node_parse(node.args[1], tensors),
                plagih_fitness_node_parse(node.args[2], tensors))

        if node.func.id in non_inline_multielem_functions:  # Min, Max Goddamn. yeah, min and max need the same type, apparently. TODO? Does this work now?
            print('Here', [plagih_fitness_node_parse(arg, tensors) for arg in node.args])
            return operators[node.func.id]([plagih_fitness_node_parse(arg, tensors) for arg in node.args])  # the star '*' makes the difference

        if node.func.id == 'ftob':
            return tf.dtypes.cast(*[plagih_fitness_node_parse(arg, tensors) for arg in node.args], dtype=tf.bool)
        elif node.func.id == 'btof':
            return tf.dtypes.cast(*[plagih_fitness_node_parse(arg, tensors) for arg in node.args], dtype=tf.float32)

        return operators[node.func.id](*[plagih_fitness_node_parse(arg, tensors) for arg in node.args])

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        return fx_fitness_chain_bool(node.values, operators[type(node.op)], tensors)

    elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
        return fx_fitness_chain_compare([node.left] + node.comparators, node.ops, tensors)

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if type(node.value) is not type(True):
            print('This True/False name constant is something else', node.value)
            raise
        try:
            return tf.constant(node.value)
        except:
            print('e', 'Oh no this was not True or False')
            raise
    else:
        raise TypeError(node)


expr = plagih_sympify('Ifte(Min(-a - 0.83, a + Min(a, Min(0.07, -2*a - 0.04)/Min(2, -0.13*a - 0.21))) < 0, 0, 2)')
print('Expression:')
print(expr)

tree = ast.parse(expr, mode='eval').body
print('Tree:')
print(tree)

# Examples for one terminal tensor set
tensors = {}
tensors['observation0'] = tf.constant(0.5, dtype=tf.float32)
tensors['observation1'] = tf.constant(0.8, dtype=tf.float32)
tensors['action0'] = tf.constant(2, dtype=tf.float32)

plagih_fitness_node_parse(tree, tensors)

print(min_1)

