from plagih.modules.printing import *
from plagih.modules.dicts import *
from plagih.modules.plagih_types import xtype_get_v2


def eval_tf(expr, data, eval_parameters, get_pred_labels=False):

    """
    computes config-tree results and fitness scores.
    - Computes tree expression using TensorFlow (TF)
    - parsing input string 'expression' and converting it into a TF operation graph
    - processing tf graph in an isolated TF session (results and corresponding fitness)

        'self.tf_device' - controls which device will be used for computations (CPU or GPU).
        'self.tf_device_log' - controls device placement logging (debug only).

    Args:
        'expr' - a string expression to be computed on the data_csv_path. Variable -> 'self.terminals'
        'data_csv_path' - an 'n by m' matrix of the data_csv_path points containing n observations like 'self.terminals'.
        'get_pred_labels' - (Classify Kernel) a boolean flag which controls whether the predicted labels should be
        extracted from the evolved results.

    Returns:
        A dict mapping keys to the following outputs:
            'result'            - array of the results of applying given expression to the data_csv_path
            'pred_labels'       - (Classify) an array of the predicted labels extracted from the results
            'solution'          - array of the solution values extracted from the data_csv_path (variable 's' in the dataset)
            'pairwise_fitness'  - array of the element-wise results of applying the fitness kernel function
            'fitness'           - aggregated scalar fitness score

    """
    # eval_parameters = self.eval_parameters
    kernel = eval_parameters['kernel']
    action_dict = eval_parameters['action_dict']
    variables_dict = eval_parameters['variables_dict']
    tf_device_log = eval_parameters['tf_device_log']
    tf_device = eval_parameters['tf_device']
    unique_outputs_num = eval_parameters['unique_outputs_num']
    tf_classify_labels_map = eval_parameters['tf_classify_labels_map']

    # Initialize TensorFlow session
    tf.compat.v1.reset_default_graph()  # tf.reset_default_graph()
    config = tf.compat.v1.ConfigProto(log_device_placement=tf_device_log, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        with sess.graph.device(tf_device):
            # 1. data_csv_path (observations, actions) to tensors
            tensors = {}

            tensors = tensors_leaves(tensors, data, variables_dict, action_dict)

            # 2- Transform string expression into TF operation graph
            tf_result = tf_from_ast_expr(expr, tensors)
            pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

            solution = tensors['action0']

            pairwise_fitness = tf_get_pairwise_fitness(kernel, solution, tf_result, action_dict, unique_outputs_num)
            fitness = tf.reduce_sum(pairwise_fitness)

            if get_pred_labels:
                pred_labels = tf.map_fn(tf_classify_labels_map, tf_result, dtype=(tf.int32, tf.string), swap_memory=True)

            tf_result, pred_labels, solution, fitness, pairwise_fitness = sess.run([tf_result, pred_labels, solution, fitness, pairwise_fitness])

    return {'result': tf_result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness),  # this was changed
            'pairwise_fitness': pairwise_fitness}


def tensors_leaves(tensors, data, variables_dict, action_dict):
    """
    All the tensors in leaf nodes, aka
    """
    num_terminals = len(variables_dict['all'])

    for i in range(num_terminals):
        var = variables_dict['all'][i]
        if '2f' in xtype_get_v2(var, variables_dict, action_dict, node_arity=0):
            tensors[var] = tf.constant(data[:, i], dtype=tf.float32)  # converts data_csv_path into vectors
        else:  # '2b'
            tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

    for i, action in enumerate(action_dict):
        py_type = action_dict[action]
        if 'float' in py_type:
            tensors['action' + str(i)] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data_csv_path into vectors
        else:
            print_e('action_dict type for {} is: {}.'.format(action, py_type))
    return tensors


def tf_get_pairwise_fitness(kernel, solution, tf_result, action_dict, unique_outputs_num):
    # 3- Add fitness computation into TF graph
    if kernel == 'classification':  # CLASSIFY kernel

        """
        This multiclass classifer compares each row of a given Tree to the known solution.
        The left-most (class?) bin includes -inf. The right-most bin includes +inf. Those inbetween are 
        by default confined to the spacing of 1.0 each, as defined by:

            (solution - 1) < result <= solution

        The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the 
        origin. At the time of this writing, an odd number of class labels will generate an extra bin on the positive 
        side of origin as it has not yet been determined the effect of enabling the middle bin to include both a 
        negative and positive result.
        """

        if len(action_dict) > 1:
            print_e('TODO multidimensional input. To be done, there is no solution yet.')

        skew = (unique_outputs_num / 2) - 1

        rule11 = tf.equal(solution, 0)
        rule12 = tf.less_equal(tf_result, 0 - skew)
        rule13 = tf.logical_and(rule11, rule12)

        rule21 = tf.equal(solution, unique_outputs_num - 1)
        rule22 = tf.greater(tf_result, solution - 1 - skew)
        rule23 = tf.logical_and(rule21, rule22)

        rule31 = tf.less(solution - 1 - skew, tf_result)
        rule32 = tf.less_equal(tf_result, solution - skew)
        rule33 = tf.logical_and(rule31, rule32)

        pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule13, rule23), rule33), tf.int32)

    elif kernel == 'regression':  # REGRESSION kernel

        """
        A very, very basic REGRESSION kernel which is not designed to perform well in the real world. It requires
        that you raise the minimum node count to keep it from converging on the c1 of '1'. Consider writing or 
        integrating a more sophisticated kernel.
        """

        pairwise_fitness = tf.abs(solution - tf_result)

    elif kernel == 'match':  # MATCH kernel

        """
        This is used for demonstration purposes only.
        """

        # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
        rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
        pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - tf_result), atol + rtol * tf.abs(tf_result)), tf.int32)

    else:
        raise Exception('Kernel type is wrong or missing. You entered {}'.format(kernel))

    return pairwise_fitness


def tf_from_ast_expr(expr, tensors, prnt=None, build=None):
    """
    Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.

    """
    # print('Current expr:', expr)  # importantprint for debugging failed expressions
    tree = ast.parse(expr, mode='eval').body

    return tf_graph_from_expr_recursive(tree, tensors, prnt=prnt, build=build)


def tf_graph_from_expr_recursive(node, tensors, prnt=None, build=None):
    """
    Recursively transforms parsed expression tree into TensorFlow (TF) graph.

    """

    # Arity 0
    if isinstance(node, ast.Name):  # <tensor_name>
        if prnt:
            return '{}'.format(node.id)
        elif build:
            return [node.id]
        else:
            return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        if prnt:
            return '{}'.format(node.n)
        if build:
            return [node.n]
        else:
            shape = tensors[list(tensors.keys())[0]].get_shape()
            return tf.constant(node.n, shape=shape, dtype=tf.float32)

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if prnt:
            return '{}'.format(node.value)
        if build:
            return [node.value]
        else:
            return tf.constant(node.value)

    # Arity 1
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., sin(1)
        if prnt:
            return '({}{})'.format(
                op[type(node.op)]['name'],
                tf_graph_from_expr_recursive(node.operand, tensors, prnt=prnt))
        if build:
            return [op[type(node.op)]['name'], [tf_graph_from_expr_recursive(node.operand, tensors, build=build)]]
        else:
            return ast_tensor_dict[type(node.op)](
                tf_graph_from_expr_recursive(node.operand, tensors))

    # Arity 2
    elif isinstance(node, ast.BinOp) or isinstance(node, ast.BitAnd):  # <left> <operator> <right>, e.g., (x + y), (a & True)
        if prnt:
            return '({} {} {})'.format(
                tf_graph_from_expr_recursive(node.left, tensors, prnt=prnt),
                op[type(node.op)]['name'],
                tf_graph_from_expr_recursive(node.right, tensors, prnt=prnt))
        if build:
            return [op[type(node.op)]['name'],
                    [tf_graph_from_expr_recursive(node.left, tensors, build=build),
                     tf_graph_from_expr_recursive(node.right, tensors, build=build)]]
        else:
            return ast_tensor_dict[type(node.op)](
                tf_graph_from_expr_recursive(node.left, tensors),
                tf_graph_from_expr_recursive(node.right, tensors))

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        if prnt:
            return tf_chain_bool(node.values, op[type(node.op)]['name'], tensors, prnt=True)
        if build:
            return tf_chain_bool(node.values, op[type(node.op)]['name'], tensors, build=build)
        else:
            return tf_chain_bool(node.values, ast_tensor_dict[type(node.op)], tensors)

    elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
        if prnt:
            return tf_chain_compare([node.left] + node.comparators, node.ops, tensors, prnt=prnt)
        if build:
            return tf_chain_compare([node.left] + node.comparators, node.ops, tensors, build=build)
        else:
            return tf_chain_compare([node.left] + node.comparators, node.ops, tensors)

    # Arity x, all custom functions
    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)

        if node.func.id == 'Ifte':
            if prnt:
                return '(If ({}) then ({}) else ({}))'.format(
                    tf_graph_from_expr_recursive(node.args[0], tensors, prnt=prnt),
                    tf_graph_from_expr_recursive(node.args[1], tensors, prnt=prnt),
                    tf_graph_from_expr_recursive(node.args[2], tensors, prnt=prnt))
            if build:
                return ['Ifte',
                        [tf_graph_from_expr_recursive(node.args[0], tensors, build=build),
                         tf_graph_from_expr_recursive(node.args[1], tensors, build=build),
                         tf_graph_from_expr_recursive(node.args[2], tensors, build=build)]]
            else:
                return ast_tensor_dict[node.func.id](tf.dtypes.cast(
                    tf_graph_from_expr_recursive(node.args[0], tensors), tf.bool),
                    tf_graph_from_expr_recursive(node.args[1], tensors),
                    tf_graph_from_expr_recursive(node.args[2], tensors))

        elif node.func.id == 'Ftob' or node.func.id == 'Btof':
            if prnt:
                return '({} {})'.format(node.func.id, tf_graph_from_expr_recursive(node.args[0], tensors, prnt=prnt))
            if build:
                return [node.func.id,
                        [tf_graph_from_expr_recursive(node.args[0], tensors, build=build)]]
            else:
                return tf.dtypes.cast(*[tf_graph_from_expr_recursive(arg, tensors) for arg in node.args], dtype=ast_tensor_dict[node.func.id])

        elif len(node.args) <= 2:
            if prnt:
                if len(node.args) == 1:
                    return '({} {})'.format(
                        op[node.func.id]['name'],
                        tf_graph_from_expr_recursive(node.args[0], tensors, prnt=prnt))
                elif len(node.args) == 2:
                    return '({} ({}, {}))'.format(
                        op[node.func.id]['name'],
                        tf_graph_from_expr_recursive(node.args[0], tensors, prnt=prnt),
                        tf_graph_from_expr_recursive(node.args[1], tensors, prnt=prnt))
                else:
                    raise Exception('This arity is not supported')
            if build:
                if len(node.args) == 1:
                    return [op[node.func.id]['name'],
                            [tf_graph_from_expr_recursive(node.args[0], tensors, build=build)]]
                elif len(node.args) == 2:
                    return [op[node.func.id]['name'],
                            [tf_graph_from_expr_recursive(node.args[0], tensors, build=build),
                            tf_graph_from_expr_recursive(node.args[1], tensors, build=build)]]
                else:
                    raise Exception('This arity is not supported')
            else:
                return ast_tensor_dict[node.func.id](*[tf_graph_from_expr_recursive(arg, tensors) for arg in node.args])

            # If nothing matched
        else:
            raise Exception('Failed to identify the function')

    else:
        raise TypeError(node)


def tf_chain_bool(values, operation, tensors, prnt=False, build=False):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

    """

    x = tf.dtypes.cast(tf_graph_from_expr_recursive(values[0], tensors), tf.bool)
    if len(values) > 1:
        if prnt:
            if len(values) == 2:
                return '({} {} {})'.format(
                    values[0],
                    operation,
                    values[1])
            else:
                print('FUCK')
                raise
        if build:
            if len(values) == 2:
                return [operation,
                        [values[0],
                         values[1]]]
            else:
                print('FUCK')
                raise
        return operation(x, tf_chain_bool(values[1:], operation, tensors))
    else:
        if prnt:
            print_warning('w', 'Whats x? {}'.format(x))
            return str(x)
        return x


def tf_chain_compare(comparators, ops, tensors, prnt=False, build=False):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    """

    x = tf_graph_from_expr_recursive(comparators[0], tensors, prnt=prnt, build=build)
    y = tf_graph_from_expr_recursive(comparators[1], tensors, prnt=prnt, build=build)

    if len(comparators) > 2:
        print_warning('e', 'This is usually not used, and-concatenation of multiple chain compares')
        return tf.logical_and(ast_tensor_dict[type(ops[0])](x, y), tf_chain_compare(comparators[1:], ops[1:], tensors))
    else:
        if prnt:
            return '({} {} {})'.format(x, op[type(ops[0])]['name'], y)
        if build:
            return [op[type(ops[0])]['name'], [x, y]]
        else:
            return ast_tensor_dict[type(ops[0])](x, y)

# # x = 'Ifte(1.019*(-0.09)**b*(0.98 - 0.13) + Mini(b, observation0) > -0.97, 0.0, 2.0)'
# fix_labels = ['Ifte',
#               '&', '2', '0',
#               '<=', '<=',
#               'Mini', 'observation1', 'observation1', '+',
#               '+', '-', '*', '0.7',
#               '*', '0.03', '*', '0.008', '-0.07', '**',
#               '-0.09', '**', '0.3', '**', '+', '2',
#               '+', '2', '+', '4', 'observation0', '0.38',
#               'observation0', '0.25', 'pos', '0.9']
# fix_tree = karoo_tree_from_labellist(fix_labels)
# fix_expr_raw = tree_expr_raw(fix_tree, root_id)
#
# fix_expr_sym = tree_expr_sympify(tree=fix_tree)
#
# fake_tensors = {'observation0': tf.constant(1.1, dtype=tf.float32),
#                 'observation1': tf.constant(2.2, dtype=tf.float32),
#                 'bl': tf.constant(True, dtype=tf.bool)}
# # graph = tf_from_ast_expr('Ifte(Or(b & b, b), Mini(a,2), a+a)', fake_tensors, print_string=True)
# graph = tf_from_ast_expr(fix_expr_sym, fake_tensors, build=True)
# # test = tf_from_ast_expr(graph, fake_tensors, build=1)
# print(graph)
# expr = labels_from_algo(graph, [])
# print(expr)
