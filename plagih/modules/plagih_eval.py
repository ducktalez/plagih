from plagih.modules.printing import *
from plagih.modules.dicts import *
from plagih.modules.plagih_types import xtype_get_from_label

import sklearn.metrics as skm


class FitnessKernel:

    def __init__(self, kernel_name):
        self.kernel = kernel_name

    def fitness_compare(self, fitness1, fitness2, mode='better'):
        """
        Compares the fitness of two candidates according to the kernel

        Example:
            >
            fitness_compare
        """
        if fitness2 is None:
            return True
        elif self.kernel == 'regression' and fitness1 < fitness2:
            return True
        elif self.kernel == 'regression bounded' and fitness1 < fitness2:
            return True
        elif self.kernel == 'classification' and fitness1 > fitness2:
            return True
        elif self.kernel == 'match' and fitness1 > fitness2:
            return True
        elif fitness1 == fitness2 and mode == 'better_or_equal':
            return True
        else:
            return False

    def conclusion_get_text(self, result, fitness_control_best):
        result_str = ''

        if self.kernel == 'classification':
            result_str += ('\n\n Classification fitness score: {}'.format(fitness_control_best))
            result_str += ('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution'], result['pred_labels'][0])))
            result_str += ('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution'], result['pred_labels'][0])))

        elif self.kernel == 'regression':
            mse = skm.mean_squared_error(result['tf_result'], result['solution'])
            result_str += ('\n\n Regression fitness score: {}'.format(result['fitness']))
            result_str += ('\n Mean Squared Error: {}'.format(mse))

        elif self.kernel == 'regression bounded':
            mse = skm.mean_squared_error(result['tf_result'], result['solution'])
            result_str += ('\n\n Regression bounded fitness score: {}'.format(result['fitness']))
            result_str += ('\n Mean Squared Error: {}'.format(mse))

        elif self.kernel == 'match':
            result_str += ('\n\n Matching fitness score: {}'.format(result['fitness']))
        return result_str

    def tf_get_pairwise_fitness(self, solution, tf_result, action_dict, unique_outputs_num, action_min_max=None):
        # 3- Add fitness computation into TF graph

        if self.kernel == 'classification':  # CLASSIFY kernel

            """
            This multiclass classifer compares each row of a given Tree to the known solution.
            The left-most (class?) bin includes -inf. The right-most bin includes +inf. Those inbetween are 
            by default confined to the spacing of 1.0 each, as defined by:

                (solution - 1) < result <= solution

            The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the 
            origin_meta. At the time of this writing, an odd number of class labels will generate an extra bin on the positive 
            side of origin_meta as it has not yet been determined the effect of enabling the middle bin to include both a 
            negative and positive result.
            """

            if len(action_dict) > 1:
                print_e('TODO multidimensional input. To be custom_done, there is no solution yet.')

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

        elif self.kernel == 'regression':

            """
            A very, very basic  kernel which is not designed to perform well in the real world. It requires
            that you raise the minimum node count to keep it from converging on the c1 of '1'. Consider writing or 
            integrating a more sophisticated kernel.
            """

            pairwise_fitness = tf.abs(solution - tf_result)

        elif self.kernel == 'regression bounded':
            """
            special regression for float solutions and discrete labels
            - if the solution is between labels, the difference is added
            - if the solution > bound_pop (the highest possible label), difference 0 is added
            ---
            suitable for:
            - not fitting labels beforehand
            - small amount of labels
            - orderable amount of labels
            """

            act_min = tf.constant(action_min_max[0], dtype=tf.float32)
            act_max = tf.constant(action_min_max[1], dtype=tf.float32)
            new_result = tf.math.minimum(tf.math.maximum(tf_result, act_min), act_max)
            pairwise_fitness = tf.abs(solution - new_result)
            # todo fitness anpassen

        elif self.kernel == 'match':  # MATCH kernel

            """
            This is used for demonstration purposes only.
            """

            # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
            rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
            pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - tf_result), atol + rtol * tf.abs(tf_result)), tf.int32)

        else:
            raise Exception('Kernel type is wrong or missing. You entered {}'.format(self.kernel))

        return pairwise_fitness


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
            'tf_result'         - array of the results of applying given expression to the data_csv_path
            'pred_labels'       - (Classify) an array of the predicted labels extracted from the results
            'solution'          - array of the solution values extracted from the data_csv_path (variable 's' in the dataset)
            'pairwise_fitness'  - array of the element-wise results of applying the fitness kernel function
            'fitness'           - aggregated scalar fitness score

    """
    # eval_parameters = self.eval_parameters
    kernel = eval_parameters['kernel_name']
    action_dict = eval_parameters['action_dict']
    variables_dict = eval_parameters['variables_dict']
    tf_device_log = eval_parameters['tf_device_log']
    tf_device = eval_parameters['tf_device']
    unique_outputs_num = eval_parameters['unique_outputs_num']
    tf_classify_labels_map = eval_parameters['tf_classify_labels_map']
    action_min_max = eval_parameters['action_min_max']

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
            tf_result = ast_convert_from_expr(expr, tensors=tensors)
            pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

            solution = tensors[first_action]  # todo

            pairwise_fitness = kernel.tf_get_pairwise_fitness(solution, tf_result, action_dict, unique_outputs_num, action_min_max=action_min_max)
            fitness = tf.reduce_sum(pairwise_fitness)

            if get_pred_labels:
                pred_labels = tf.map_fn(tf_classify_labels_map, tf_result, dtype=(tf.int32, tf.string), swap_memory=True)

            tf_result, pred_labels, solution, fitness, pairwise_fitness = sess.run([tf_result, pred_labels, solution, fitness, pairwise_fitness])

    return {'tf_result': tf_result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness),  # this was changed
            'pairwise_fitness': pairwise_fitness}


def tensors_leaves(tensors, data, variables_dict, action_dict):
    """
    All the tensors in leaf nodes
    - variables (observation0, ...)
    - constants (True, False, 1.234, ...)
    """
    num_columns = len(variables_dict['all'])  # sfeh not when entries are not used. which does not happen.

    for i in range(num_columns):
        obs = variables_dict['all'][i]['label']
        xtype = variables_dict['all'][i]['xtype']
        # xtype = xtype_get_from_label(obs, variables_dict=variables_dict, node_arity=0)

        if '2f' in xtype:
            tensors[obs] = tf.constant(data[:, i], dtype=tf.float32)  # converts data_csv_path into vectors
        elif '2b' in xtype:
            tensors[obs] = tf.constant(data[:, i], dtype=tf.bool)
        else:
            raise Exception('The xtype of your variable does not exist: {}'.format(xtype))

    # sfeh: if more than one action is provided...
    action_for_regression = [0]
    for action_name, action_info in action_dict.items():
        action_type = action_info['type']
        action_label = action_info['label']
        column = action_info['pos']
        if 'float' in action_type:
            tensors[action_label] = tf.constant(data[:, column], dtype=tf.float32)  # converts data_csv_path into vectors
        else:
            print_e('action {} has these infos: {}.'.format(action_name, action_info))
    return tensors


def ast_convert_from_expr(expr, tensors=None, prnt=None, build=None):
    """
    Extract expression tree from the string algo_sym.
    Please provide ONE of the following if you want to get...
    ...tensors: All variables (observation0, ...) as tensors.
    ...prnt: True
    ...build: True
    More information in ast_convert_from_expr_recursive()

    """
    # print('Current expr:', expr)  # importantprint for debugging failed expressions
    tree = ast.parse(expr, mode='eval').body

    graph = ast_convert_from_expr_recursive(tree, tensors=tensors, prnt=prnt, build=build)

    if build:
        # print('before:', graph)
        graph = labels_from_nestedexpr(graph, [])
        # graph = [str(x).replace('~', '-') for x in graph]

    return graph


def labels_from_nestedexpr(expr_array, expr):
    """
    Returns a label list from the nested list which ast_convert_from_expr_recursive() created
    [+, [a], [/, [b, c]]]]  -> [+, a, /, b, c]
    """

    for x in expr_array:  # all elements, that are not lists themselves
        if type(x) is not list:
            # x = str(x).replace('~', '-')  # workaround for usub/sub problem
            expr.append(x)

    only_lists = [x for x in expr_array if (type(x) == list)]
    if only_lists:
        from itertools import chain
        lists_removed = list(chain(*only_lists))
        expr = labels_from_nestedexpr(lists_removed, expr)
    return expr


def ast_convert_from_expr_recursive(node, tensors=None, prnt=None, build=None):
    """
    Returns (recursively) a (tensorflow) graph from a (raw or sympified) math expression.
    please use by calling labels_from_graphlist()

    Used to be for tensorflow only, but was modified to save 'sympified' trees.

    One of [tensors, prnt, build] must be set
    -> tensors: Creates a tensorflow graph for evaluation
    -> prnt: creates a string expression of the tree (I think I tried this before 'build' worked)
    -> build: creates a nested expr-list, e.g. a+(b/c) -> [+, [a], [/, [b, c]]]] (at least I think so)
    """

    # Arity 0
    if isinstance(node, ast.Name):  # <tensor_name>
        if prnt:
            return '{}'.format(node.id)
        elif build:
            return [node.id]
            # sfeh, what is better?
            # return node.id
        else:
            return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        if prnt:
            return '{}'.format(node.n)
        if build:
            # return node.n
            return [node.n]
        else:
            shape = tensors[list(tensors.keys())[0]].get_shape()
            return tf.constant(node.n, shape=shape, dtype=tf.float32)

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if prnt:
            return '{}'.format(node.value)
        if build:
            return [node.value]
            # return node.value
        else:
            return tf.constant(node.value)
#
    # Arity 1
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., sin(1), -1
        if prnt:
            return '({}{})'.format(
                op[type(node.op)]['fun'],
                ast_convert_from_expr_recursive(node.operand, prnt=prnt))
        if build:
            if type(node.op) == ast.USub:
                return ['~', [ast_convert_from_expr_recursive(node.operand, build=True)]]
                # return ['-', ['0', ast_convert_from_expr_recursive(node.operand, build=True)]]
            return [op[type(node.op)]['fun'], [ast_convert_from_expr_recursive(node.operand, build=True)]]
        else:
            return op[type(node.op)]['tf'](
                ast_convert_from_expr_recursive(node.operand, tensors=tensors))

    # Arity 2
    elif isinstance(node, ast.BinOp) or isinstance(node, ast.BitAnd):  # <left> <operator> <right>, e.g., (x + y), (a & True)
        if prnt:
            return '({} {} {})'.format(
                ast_convert_from_expr_recursive(node.left, prnt=True),
                op[type(node.op)]['fun'],
                ast_convert_from_expr_recursive(node.right, prnt=True))
        if build:
            return [op[type(node.op)]['fun'],
                    [ast_convert_from_expr_recursive(node.left, build=True),
                     ast_convert_from_expr_recursive(node.right, build=True)]]
        else:
            return op[type(node.op)]['tf'](
                ast_convert_from_expr_recursive(node.left, tensors=tensors),
                ast_convert_from_expr_recursive(node.right, tensors=tensors))

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        if prnt:
            return ast_chain_bool(node.values, op[type(node.op)]['fun'], prnt=True)
        if build:
            return ast_chain_bool(node.values, op[type(node.op)]['fun'], build=True)
        else:
            return ast_chain_bool(node.values, op[type(node.op)]['tf'], tensors=tensors)

    elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
        if prnt:
            return ast_chain_compare([node.left] + node.comparators, node.ops, prnt=True)
        if build:
            return ast_chain_compare([node.left] + node.comparators, node.ops, build=True)
        else:
            return ast_chain_compare([node.left] + node.comparators, node.ops, tensors=tensors)

    # Arity x, all custom functions
    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)

        if node.func.id == 'Ifte':
            if prnt:
                return '(If ({}) then ({}) else ({}))'.format(
                    ast_convert_from_expr_recursive(node.args[0], prnt=True),
                    ast_convert_from_expr_recursive(node.args[1], prnt=True),
                    ast_convert_from_expr_recursive(node.args[2], prnt=True))
            if build:
                return ['Ifte',
                        [ast_convert_from_expr_recursive(node.args[0], build=True),
                         ast_convert_from_expr_recursive(node.args[1], build=True),
                         ast_convert_from_expr_recursive(node.args[2], build=True)]]
            else:
                return op[node.func.id]['tf'](tf.dtypes.cast(
                    ast_convert_from_expr_recursive(node.args[0], tensors=tensors), tf.bool),
                    ast_convert_from_expr_recursive(node.args[1], tensors=tensors),
                    ast_convert_from_expr_recursive(node.args[2], tensors=tensors))

        elif node.func.id == 'Ftob' or node.func.id == 'Btof':
            if prnt:
                return '({} {})'.format(node.func.id, ast_convert_from_expr_recursive(node.args[0], prnt=prnt))
            if build:
                return [node.func.id, [ast_convert_from_expr_recursive(node.args[0], build=True)]]
            else:
                return tf.dtypes.cast(*[ast_convert_from_expr_recursive(arg, tensors=tensors) for arg in node.args], dtype=op[node.func.id]['tf'])

        elif len(node.args) <= 2:
            if prnt:
                if len(node.args) == 1:
                    return '({} {})'.format(
                        op[node.func.id]['fun'],
                        ast_convert_from_expr_recursive(node.args[0], prnt=True))
                elif len(node.args) == 2:
                    return '({} ({}, {}))'.format(
                        op[node.func.id]['fun'],
                        ast_convert_from_expr_recursive(node.args[0], prnt=True),
                        ast_convert_from_expr_recursive(node.args[1], prnt=True))
                else:
                    raise Exception('This arity is not supported')
            if build:
                if len(node.args) == 1:
                    return [op[node.func.id]['fun'],
                            [ast_convert_from_expr_recursive(node.args[0], build=True)]]
                elif len(node.args) == 2:
                    return [op[node.func.id]['fun'],
                            [ast_convert_from_expr_recursive(node.args[0], build=True),
                             ast_convert_from_expr_recursive(node.args[1], build=True)]]
                else:
                    raise Exception('This arity is not supported')
            else:
                return op[node.func.id]['tf'](*[ast_convert_from_expr_recursive(arg, tensors=tensors) for arg in node.args])

            # If nothing matched
        else:
            raise Exception('Failed to identify the function')

    else:
        raise TypeError(node)


def ast_chain_bool(values, operation, tensors=None, prnt=False, build=False):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.
        a & b
    --> values[0] operation values[1]
    """
    if prnt:
        x = ast_convert_from_expr_recursive(values[0], prnt=True)
        if len(values) == 2:
            return '({} {} {})'.format(values[0], operation, values[1])
        elif len(values) == 1:
            return x
        else:
            raise
    elif build:
        x = ast_convert_from_expr_recursive(values[0], build=True)
        if len(values) == 2:
            return [operation, [values[0], values[1]]]
        elif len(values) == 1:
            return x
        else:
            raise
    elif tensors:
        x = tf.dtypes.cast(ast_convert_from_expr_recursive(values[0], tensors=tensors), tf.bool)
        if len(values) > 1:
            return operation(x, ast_chain_bool(values[1:], operation, tensors=tensors))
        else:
            return x


def ast_chain_compare(comparators, ops, tensors=None, prnt=False, build=False):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    """

    x = ast_convert_from_expr_recursive(comparators[0], tensors=tensors, prnt=prnt, build=build)
    y = ast_convert_from_expr_recursive(comparators[1], tensors=tensors, prnt=prnt, build=build)

    if len(comparators) > 2:
        print_warning('e', 'This is usually not used, and-concatenation of multiple chain compares')
        return tf.logical_and(op[type(ops[0])]['tf'](x, y), ast_chain_compare(comparators[1:], ops[1:], tensors=tensors))
    else:
        if prnt:
            return '({} {} {})'.format(x, op[type(ops[0])]['fun'], y)
        if build:
            return [op[type(ops[0])]['fun'], [x, y]]
        else:
            return op[type(ops[0])]['tf'](x, y)
