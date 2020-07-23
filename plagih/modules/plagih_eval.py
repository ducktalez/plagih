from plagih.modules.printing import *
from plagih.modules.operators import *
import numpy as np
import sklearn.metrics as skm
from sys import getsizeof


class DummyKernel:

    def __init__(self, name, bounded=False, discrete=False):
        self.type = name
        self.bounded = bounded
        self.discrete = discrete

    def fitness_compare(self, fit1, fit2):
        if fit2 is None:
            return True

    def conclusion_text(self):
        pass

    def tf_wrap_result(self, *args):
        pass

    def tf_get_pairwise_fitness(self):
        pass


class RegressionKernel(DummyKernel):

    def __init__(self, bounded=False, discrete=False):
        pass

    def tf_wrap_result(self, tf_result, action_min_max):
        # regression that fits the outputs to a discrete set of actions defined by min and max
        act_min = tf.constant(action_min_max[0], dtype=tf.float32)
        act_max = tf.constant(action_min_max[1], dtype=tf.float32)
        customised_result = tf.math.minimum(tf.math.maximum(tf.math.round(tf_result), act_min), act_max)
        return customised_result

    def fitness_compare(self, fitness1, fitness2):

        if fitness2 is None:  # try block?
            return True

        return fitness1 < fitness2

    def conclusion_text(self):
        pass

    def tf_get_pairwise_fitness(self):
        pass


class FitnessKernel:

    def __init__(self, kernel_name):
        self.kname = kernel_name

        self.regression, self.classification, self.match = False, False, False

        if 'regression' in kernel_name:
            self.regression = True
            self.bounded = 'bounded' in kernel_name
            self.discrete = 'discrete' in kernel_name
            self.tanhpenalize = 'tanhpenalize' in kernel_name
            self.squared_error = 'L2_squared' in kernel_name  # 'L1_absolute' not required

            self.relative_regression_fun = 'relative_regression_fun' in kernel_name
            self.origin_pairwise_fitness = None  # TODO
            sfeh_help = {'pen_explorate(1)': 1.0,
                         'pen_explorate(0.5)': 0.5}

            self.pen_explorate = None
            for k, v in sfeh_help.items():
                if k in kernel_name:
                    self.pen_explorate = v
        else:
            self.regression = False

    def fitness_compare(self, fitness1, fitness2, only_better=True):
        """
        Compares the fitness of two candidates according to the kernel

        Example:
            >
            fitness_compare
        """
        if fitness2 is None:
            return True
        elif self.regression and fitness1 < fitness2:
            return True
        elif self.classification and fitness1 > fitness2:
            return True
        elif self.match and fitness1 > fitness2:
            return True
        elif not only_better:
            return fitness1 == fitness2
        else:
            return False

    def np_best_fitness(self, fitness_list):
        """
        """
        if self.regression:
            return np.min(fitness_list)

        elif any([self.classification, self.match]):
            return np.max(fitness_list)
        else:
            raise

    def best_fitness(self, fit1, fit2):
        """
        """
        if self.regression:
            return min(fit1, fit2)
        elif any([self.classification, self.match]):
            return max(fit1, fit2)
        else:
            raise

    def best_fitness_function(self):
        """
        Returning either min or max
        """
        if self.regression:
            return min
        elif any([self.classification, self.match]):
            return max
        else:
            raise

    def best_fitness_function_TRUTH(self):
        """
        Returning either min or max
        """
        if self.regression:
            return lambda y, x: x < y
        elif any([self.classification, self.match]):
            return lambda y, x: x > y
        else:
            raise

    # def conclusion_text(self, result, fitness_control_best):
    #     """
    #
    #     """
    #     result_str = ''
    #
    #     if self.kernel == 'classification':
    #         result_str += f'\n\n Classification fitness score: {fitness_control_best}'
    #         result_str += ('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution_goal'], result['predicted_labels'][0])))
    #         result_str += ('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution_goal'], result['predicted_labels'][0])))
    #
    #     elif self.kernel == 'regression':
    #         mse = skm.mean_squared_error(result['agent_result'], result['solution_goal'])
    #         result_str += ('\n\n Regression fitness score: {}'.format(result['fitness']))
    #         result_str += ('\n Mean Squared Error: {}'.format(mse))
    #
    #     elif self.kernel == 'regression bounded':
    #         mse = skm.mean_squared_error(result['agent_result'], result['solution_goal'])
    #         result_str += f"\n\n Regression bounded fitness score: {result['fitness']}"
    #         result_str += f'\n Mean Squared Error: {mse}'
    #
    #     elif self.kernel == 'match':
    #         result_str += f"\n\n Matching fitness score: {result['fitness']}"
    #
    #     else:  # 'regression discrete':
    #         result_str = 'No summary provided for this kernel'
    #
    #     return result_str

    def tf_wrap_result(self, tf_result, action_min_max):

        if 'discrete' in self.kname:
            # regression that fits the outputs to a discrete set of actions defined by min and max
            tf_result = tf.math.round(tf_result)

        # if 'bounded_tanh' in self.kernel:
        #     tf_result = todo

        if 'bounded' in self.kname:
            act_min = tf.constant(action_min_max[0])
            act_max = tf.constant(action_min_max[1])
            tf_result = tf.math.minimum(tf.math.maximum(tf_result, act_min), act_max)

        return tf_result

    def pycode_wrap_result(self, action_min_max):
        wrap = '{}'
        if 'discrete' in self.kname:
            # regression that fits the outputs to a discrete set of actions defined by min and max
            wrap = f'math.round({wrap})'

        # if 'bounded_tanh' in self.kernel:
        #     tf_result = todo

        elif 'bounded' in self.kname:
            wrap = f'min(max({action_min_max[0]}, {wrap}), {action_min_max[1]})'

        return wrap

    def tf_get_pairwise_fitness(self, solution, kernel_result, uniques_num, agent_result, origin_pairwise_fitness=None):  #sfeh explorate
        """
        Calculates the kernel-specific fitness for the solution.
        - classification: dummy
        """

        if self.classification:  # CLASSIFY kernel
            """
            """

            skew = (uniques_num / 2) - 1

            rule1 = tf.logical_and(
                tf.equal(solution, 0),
                tf.less_equal(kernel_result, 0 - skew))

            rule2 = tf.logical_and(
                tf.equal(solution, uniques_num - 1),
                tf.greater(kernel_result, solution - 1 - skew))

            rule3 = tf.logical_and(
                tf.less(solution - 1 - skew, kernel_result),
                tf.less_equal(kernel_result, solution - skew))

            pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule1, rule2), rule3), tf.int32)

        elif self.regression:
            """
            """

            pairwise_diff = solution - kernel_result

            tf_error = tf.square if self.squared_error else tf.abs
            pairwise_fitness = tf_error(pairwise_diff)

            if self.relative_regression_fun and origin_pairwise_fitness is not None:
                tf_error = tf.abs  # todo this is required (??)
                # regression_goal = tf.abs(solution - tf_result)  # double the penalty
                exploration_diff = (origin_pairwise_fitness - kernel_result)  # NO abs value
                paretodiff = tf_error(solution - origin_pairwise_fitness)
                pairwise_fitness = tf_error((2 * pairwise_fitness) - self.pen_explorate*(paretodiff - exploration_diff))  # faster version

            if self.tanhpenalize:
                tanhpenalize = 0.02*tf.tanh(tf.square(agent_result-kernel_result)*0.1)  # todo amplitude, stretch, squared
                pairwise_fitness = pairwise_fitness + tanhpenalize

        elif self.match:  # MATCH kernel
            """
            This is used for demonstration purposes only.
            """
            # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
            rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
            pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - kernel_result), atol + rtol * tf.abs(kernel_result)), tf.int32)

        else:
            raise Exception('Kernel type is wrong or missing. You entered {}'.format(self.kname))

        return pairwise_fitness

    # todo garbage collector?


def eval_tf(expr_sym, used_observations, pd_data, kernel, eval_action, tf_config, tf_device, tf_classify_labels_map, get_predicted_labels=False, complete=False, origin_pairwise_fitness=None):
    """
    Evaluates an expression using TensorFlow (TF)
    The is usually extracted from a tree and is sympified
    - parsing input string 'expression' and converting it into a TF operation graph
    - processing tf graph in an isolated TF session (results and corresponding fitness)

        'self.tf_device' - controls which device will be used for computations (CPU or GPU).
        'self.tf_device_log' - controls device placement logging (debug only).

    'get_predicted_labels' - (Classify Kernel) a boolean flag which controls whether the predicted labels should be extracted from the evolved results.
    todo make this a kernel option
    """

    tf.compat.v1.reset_default_graph()
    tensors = {eval_action.name: tf.constant(pd_data[eval_action.name])}  # converts data_csv_path into vectors

    for obs_x_name in used_observations:
        tensors[obs_x_name] = tf.constant(pd_data[obs_x_name])  # , dtype=tf.float32 todo: neverask.jpg

    # print('size of tensors', getsizeof(tensors), getsizeof(pd_data))

    with tf.compat.v1.Session(config=tf_config) as sess:  # starting a tf-session
        with sess.graph.device(tf_device):  # device can be the gpu  # todo check if gpu is used

            agent_result = ast_convert_from_expr(expr_sym, tensors=tensors)  # the actual result from the expression in the agent
            kernel_result = kernel.tf_wrap_result(agent_result, eval_action.minmax)  # if the result should be discrete or has a min/max, this is done here
            act_solution = tensors[eval_action.name]
            pairwise_fitness = kernel.tf_get_pairwise_fitness(act_solution, kernel_result, eval_action.uniques, agent_result, origin_pairwise_fitness=origin_pairwise_fitness)
            fitness = tf.reduce_sum(pairwise_fitness)

            if complete:
                if get_predicted_labels:
                    predicted_labels = tf.map_fn(tf_classify_labels_map, kernel_result, dtype=(tf.int32, tf.string), swap_memory=True)
                else:
                    predicted_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

                agent_result, kernel_result, predicted_labels, act_solution, fitness, pairwise_fitness = sess.run([agent_result, kernel_result, predicted_labels, act_solution, fitness, pairwise_fitness])
                return {'agent_result': agent_result, 'kernel_result': kernel_result, 'predicted_labels': predicted_labels,
                        'solution_goal': act_solution, 'fitness': float(fitness), 'pairwise_fitness': pairwise_fitness}
            else:  # reduced evaluation, only fitness is evaluated
                fitness = sess.run(fitness)
                return float(fitness)


def get_env_tensors(pd_data, eval_action, obs_infos):
    """
    return tensors-dictionary with all the terminals/leaf nodes
    - variables (observation0, ...)
    - distributions_file (True, False, 1.234, ...)
    """
    tensors = {}  # todo dauerhafte tensoren?

    return tensors


def ast_convert_from_expr(expr, tensors=None, build=None):
    """
    Starts the recursive ast-analysis of the expression

    Extract expression tree from the string algo_sym.
    Please provide ONE of the following if you want to get...
    - tensorflow-graph: All variables (observation0, ...) as tensors.
    - build: True
    More information in ast_expr_to()

    """
    # print('Current expr:', expr)  # importantprint for debugging failed expressions

    ast_tree = ast.parse(expr, mode='eval').body
    graph = ast_expr_to(ast_tree, tensors=tensors, build=build)

    if build:
        graph = labels_from_nestedexpr(graph, [])

    return graph


def labels_from_nestedexpr(labels_nested_list, result_accum):
    """
    Returns a label list from the nested list which ast_expr_to() created
    [+, [a], [/, [b, c]]]]  -> [+, a, /, b, c]
    """

    for x in labels_nested_list:  # all elements, that are not lists themselves
        if type(x) is not list:
            x = str(x)  # labels must be string!
            result_accum.append(x)

    only_lists = [x for x in labels_nested_list if (type(x) == list)]
    if only_lists:
        from itertools import chain
        lists_removed = list(chain(*only_lists))
        result_accum = labels_from_nestedexpr(lists_removed, result_accum)

    return result_accum


def ast_expr_to(node, tensors=None, build=None):
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
        if build:
            return [node.id]
        else:
            try:
                return tensors[node.id]
            except:
                raise  # TODO

    elif isinstance(node, ast.Num):  # <number>
        if build:
            return [node.n]
        else:
            shape = tensors[list(tensors.keys())[0]].get_shape()
            return tf.constant(node.n, shape=shape, dtype=tf.float32)

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if build:
            return [node.value]
            # return node.value
        else:
            return tf.constant(node.value)
    #
    # Arity 1
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., sin(1), -1
        if build:
            if type(node.op) == ast.USub:
                if isinstance(node.operand, ast.Name) or isinstance(node.operand, ast.Num) or isinstance(node.operand, ast.NameConstant):
                    # return ['~', [ast_expr_to(node.operand, build=True)]]  # todo wait what. sympify uses ~. also,
                    return [f'-{ast_expr_to(node.operand, build=True)[0]}']
                else:
                    return ['usub', [ast_expr_to(node.operand, build=True)]]
            return [op[type(node.op)]['fun_label'], [ast_expr_to(node.operand, build=True)]]
        else:
            return op[type(node.op)]['tf'](
                ast_expr_to(node.operand, tensors=tensors))

    # Arity 2
    elif isinstance(node, ast.BinOp) or isinstance(node, ast.BitAnd):  # <left> <operator> <right>, e.g., (x + y), (a & True)
        if build:
            return [op[type(node.op)]['fun_label'],
                    [ast_expr_to(node.left, build=True),
                     ast_expr_to(node.right, build=True)]]
        else:
            return op[type(node.op)]['tf'](
                ast_expr_to(node.left, tensors=tensors),
                ast_expr_to(node.right, tensors=tensors))

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        if build:
            return ast_chain_bool(node.values, op[type(node.op)]['fun_label'], build=True)
        else:
            return ast_chain_bool(node.values, op[type(node.op)]['tf'], tensors=tensors)

    elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
        if build:
            return ast_chain_compare([node.left] + node.comparators, node.ops, build=True)
        else:
            return ast_chain_compare([node.left] + node.comparators, node.ops, tensors=tensors)

    # Arity x, all custom functions
    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)

        if node.func.id == 'Ifte':
            if build:
                return ['Ifte',
                        [ast_expr_to(node.args[0], build=True),
                         ast_expr_to(node.args[1], build=True),
                         ast_expr_to(node.args[2], build=True)]]
            else:
                return op[node.func.id]['tf'](tf.dtypes.cast(
                    ast_expr_to(node.args[0], tensors=tensors), tf.bool),
                    ast_expr_to(node.args[1], tensors=tensors),
                    ast_expr_to(node.args[2], tensors=tensors))

        elif node.func.id == 'Ftob' or node.func.id == 'Btof':
            if build:
                return [node.func.id, [ast_expr_to(node.args[0], build=True)]]
            else:
                return tf.dtypes.cast(*[ast_expr_to(arg, tensors=tensors) for arg in node.args], dtype=op[node.func.id]['tf'])

        elif len(node.args) <= 2:
            if build:
                if len(node.args) == 1:
                    return [op[node.func.id]['fun_label'],
                            [ast_expr_to(node.args[0], build=True)]]
                elif len(node.args) == 2:
                    return [op[node.func.id]['fun_label'],
                            [ast_expr_to(node.args[0], build=True),
                             ast_expr_to(node.args[1], build=True)]]
                else:
                    raise Exception('This arity is not supported')
            else:
                return op[node.func.id]['tf'](*[ast_expr_to(arg, tensors=tensors) for arg in node.args])

            # If nothing matched
        else:
            raise Exception('Failed to identify the function. {}'.format(type(node)))

    else:
        raise TypeError('Node type could not be handeled in ast-evaluation: {}'.format(node))


def ast_chain_bool(values, operation, tensors=None, build=False):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.
        a & b
    --> values[0] operation values[1]
    """
    if build:
        x = ast_expr_to(values[0], build=True)
        if len(values) == 2:
            return [operation, [values[0], values[1]]]
        elif len(values) == 1:
            return x
        else:
            raise
    elif tensors:
        x = tf.dtypes.cast(ast_expr_to(values[0], tensors=tensors), tf.bool)
        if len(values) > 1:
            return operation(x, ast_chain_bool(values[1:], operation, tensors=tensors))
        else:
            return x


def ast_chain_compare(comparators, ops, tensors=None, build=False):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    """

    x = ast_expr_to(comparators[0], tensors=tensors, build=build)
    y = ast_expr_to(comparators[1], tensors=tensors, build=build)

    if len(comparators) > 2:
        print_warning('e', 'This is usually not used, and-concatenation of multiple chain compares')
        return tf.logical_and(op[type(ops[0])]['tf'](x, y), ast_chain_compare(comparators[1:], ops[1:], tensors=tensors))
    else:
        if build:
            return [op[type(ops[0])]['fun_label'], [x, y]]
        else:
            return op[type(ops[0])]['tf'](x, y)
