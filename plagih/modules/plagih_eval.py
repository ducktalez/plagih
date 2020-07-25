from plagih.modules.printing import *
from plagih.modules.operators import *
import numpy as np
import sklearn.metrics as skm
from sys import getsizeof
import copy


class DummyKernel:

    def __init__(self, *args):
        pass

    def fitness_compare(self, fitness1, fitness2):
        if fitness2 is None:
            return True
        elif self.classification and fitness1 > fitness2:
            return True

        elif self.match and fitness1 > fitness2:
            return True

    def np_best_fitness(self, fitness_list):
        """
        """
        return np.min(fitness_list)
        # elif any([self.classification, self.match]):
        # return np.max(fitness_list)


    def best_fitness(self, fit1, fit2):
        """
        """
        if self.regression:
            return min(fit1, fit2)
        elif any([self.classification, self.match]):
            return max(fit1, fit2)
        else:
            raise


    def tf_wrap_result(self, *args):
        pass

    def tf_get_pairwise_fitness(self):
        """
        Calculates the kernel-specific fitness for the solution.
        - classification: dummy
        """
        pass

        # if self.classification:  # CLASSIFY kernel
        #     """
        #     """
        #
        #     skew = (self.eval_action.uniques / 2) - 1
        #
        #     rule1 = tf.logical_and(
        #         tf.equal(solution, 0),
        #         tf.less_equal(kernel_result, 0 - skew))
        #
        #     rule2 = tf.logical_and(
        #         tf.equal(solution, self.eval_action.uniques - 1),
        #         tf.greater(kernel_result, solution - 1 - skew))
        #
        #     rule3 = tf.logical_and(
        #         tf.less(solution - 1 - skew, kernel_result),
        #         tf.less_equal(kernel_result, solution - skew))
        #
        #     pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule1, rule2), rule3), tf.int32)
        #
        # elif self.match:  # MATCH kernel
        #     """
        #     This is used for demonstration purposes only.
        #     """
        #     # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
        #     rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
        #     pairwise_fitness = tf.dtypes.cast(
        #         tf.less_equal(tf.abs(solution - kernel_result), atol + rtol * tf.abs(kernel_result)), tf.int32)
        #
        # else:
        #     raise Exception('Kernel type is wrong or missing. You entered {}'.format(self.kname))

    def eval_tf(self):
        pass
        # if self.get_predicted_labels:
        #     predicted_labels = tf.map_fn(self.tf_classify_labels_map, kernel_result, dtype=(tf.int32, tf.string), swap_memory=True)
        # else:
        #     predicted_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel
        #  # , 'predicted_labels': predicted_labels    # predicted_labels


    # def conclusion_text(self, result, fitness_control_best):
    #     """
    #
    #     """
    #     elif self.kernel == 'regression':
    #         mse = skm.mean_squared_error(result['agent_result'], result['solution_goal'])
    #         result_str += ('\n\n Regression fitness score: {}'.format(result['fitness']))
    #         result_str += ('\n Mean Squared Error: {}'.format(mse))
    #
    #     result_str = ''
    #
    #     if self.kernel == 'classification':
    #         result_str += f'\n\n Classification fitness score: {fitness_control_best}'
    #         result_str += ('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution_goal'], result['predicted_labels'][0])))
    #         result_str += ('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution_goal'], result['predicted_labels'][0])))
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


class RegressionKernel:

    def __init__(self, kernel_name, data_train, tf_config, tf_device, eval_action, origin_pairwise_fitness):

        self.best_fitness_function = min  # todo check
        self.np_best_fitness = np.min
        # , self.tf_classify_labels_map  # todo
        self.kname = kernel_name
        self.tf_config = tf_config
        self.tf_device = tf_device
        self.eval_action = eval_action
        self.origin_results = None
        self.data_train = data_train  # todo where is the best?

        self.regression, self.classification, self.match = False, False, False

        self.regression = True
        self.bounded = 'bounded' in kernel_name
        self.discrete = 'discrete' in kernel_name
        self.tanhpenalize = 'tanhpenalize' in kernel_name
        self.squared_error = 'MSE' in kernel_name  # 'L1_absolute' not required

        self.relative_regression_fun = 'relative_regression_fun' in kernel_name
        self.origin_results = origin_pairwise_fitness
        sfeh_help = {'pen_explorate(1)': 1.0,
                     'pen_explorate(0.5)': 0.5}

        self.pen_explorate = 0.5  # todo
        for k, v in sfeh_help.items():
            if k in kernel_name:
                self.pen_explorate = v
        return

    def fitness_compare(self, fitness1, fitness2, only_better=True):
        """
        Compares the fitness of two candidates according to the kernel
        """
        if fitness2 is None:
            return True
        elif fitness1 < fitness2:
            return True
        elif not only_better:
            return fitness1 == fitness2
        else:
            return False

    # def np_best_fitness(self, fitness_list):
    #     """
    #     """
    #     return np.min(fitness_list)

    # def best_fitness(self, *args):
    #     """
    #     """
    #     return min(*args)

    def tf_wrap_result(self, tf_result):

        return tf_result

    def pycode_wrap_result(self, action_min_max):
        wrap = '{}'
        if 'discrete' in self.kname:
            # regression that fits the outputs to a discrete set of actions defined by min and max
            wrap = f'math.round({wrap})'

        elif 'bounded' in self.kname:
            wrap = f'min(max({action_min_max[0]}, {wrap}), {action_min_max[1]})'

        return wrap

    def tf_get_pairwise_fitness(self, solution, kernel_result, agent_result):
        """

        """

        return pairwise_fitness

    def eval_tf(self, expr_sym, used_observations, complete=False):
        """
        Evaluates an expression using TensorFlow (TF)
        - receives a (string) expression in numpy-style that was reduced with pythons "sympy" (for simplification)
        - uses "ast" to generate a, kind of, python-intern-executable-tree
        - creating a tensorflow graph that is evaluated in an isolated TF session
        """

        tf.compat.v1.reset_default_graph()
        # tensors = {self.eval_action.name: }  # converts data_csv_path into vectors , dtype=tf.float32
        solution = tf.constant(self.data_train[self.eval_action.name])  # tensors[self.eval_action.name]
        tensors = {tf.constant(self.data_train[obs]) for obs in used_observations}  # do not assign dtype here, do this in the pandas df aka data

        agent_result = ast_convert_from_expr(expr_sym, tensors=tensors)  # the actual result from the expression in the agent

        kernel_results = agent_result.copy()

        if self.discrete:
            kernel_results = tf.math.round(kernel_results)

        if self.bounded:
            act_min = tf.constant(self.eval_action.minmax[0], dtype=tf.float32)
            act_max = tf.constant(self.eval_action.minmax[1], dtype=tf.float32)
            kernel_results = tf.math.minimum(tf.math.maximum(kernel_results, act_min), act_max)

        # pairwise_fitness = self.tf_get_pairwise_fitness(solution, kernel_result, agent_result)
        pairwise_diff = solution - kernel_results


        pairwise_error = tf_error(pairwise_diff)



        if self.relative_regression_fun and self.origin_results is not None:
            # tf_error = tf.abs  # sfeh this is required (??)
            # (1 * tf.abs(pairwise_diff)) - # 1 * abs, as the other one is within the error. usually 2*
            improvement_range = tf.abs(solution - self.origin_results)
            exploration = (self.origin_results - kernel_results)

            add_penalty = (improvement_range - exploration)
            smaller_penalty = self.pen_explorate*add_penalty
            pairwise_fitness = tf_error(add_penalty)  # faster version

        if self.tanhpenalize:
            tanhpenalize = 0.02*tf.tanh(tf.square(agent_result-kernel_results)*0.1)  # sfeh amplitude, stretch, squared
            pairwise_fitness = pairwise_fitness + tanhpenalize

        if self.squared_error:
            tf_error = tf.square
            regression_error = tf.keras.losses.mean_squared_error(solution, kernel_results)

        else:
            tf_error = tf.abs
            regression_error = tf.keras.losses.mean_absolute_error(solution, kernel_results)
            # todo huber loss! mse, mae, huber, (log)




        fitness = tf.reduce_sum(pairwise_fitness)

        with tf.compat.v1.Session(config=self.tf_config) as sess:  # tensorflow evaluation must be done in a "session". funfact: debugging is not ez
            with sess.graph.device(self.tf_device):  # GPU evaluation in tensorflow

                if complete:
                    agent_result, kernel_results, solution, fitness, pairwise_fitness = sess.run([agent_result, kernel_results, solution, fitness, pairwise_fitness])
                    return {'agent_result': agent_result, 'kernel_result': kernel_results,
                            'solution_goal': solution, 'fitness': float(fitness), 'pairwise_fitness': pairwise_fitness}
                else:  # reduced evaluation, only fitness is evaluated
                    fitness = sess.run(fitness)
                    return float(fitness)


def ast_convert_from_expr(expr, tensors=None, build=None):
    """
    Starts the recursive ast-analysis of the expression

    Extract expression tree from the string algo_sym.
    Please provide ONE of the following if you want to get...
    - tensorflow-graph: All variables (observation0, ...) as tensors.
    - build: True
    More information in ast_expr_to()

    """

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
            return tensors[node.id]

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
            if type(node.op) == ast.USub:   # workaround for ~-problem
                if isinstance(node.operand, ast.Name) or isinstance(node.operand, ast.Num) or isinstance(node.operand, ast.NameConstant):
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
        print_e('This is usually not used, and-concatenation of multiple chain compares')
        return tf.logical_and(op[type(ops[0])]['tf'](x, y), ast_chain_compare(comparators[1:], ops[1:], tensors=tensors))
    else:
        if build:
            return [op[type(ops[0])]['fun_label'], [x, y]]
        else:
            return op[type(ops[0])]['tf'](x, y)
