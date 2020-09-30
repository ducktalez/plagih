from plagih.printing import *
from plagih.operators import *
import numpy as np
import sklearn.metrics as skm
# from sys import getsizeof


class GPKernel:

    # def __init__(self, *args):
    #     self.eval_action = None
    #     pass

    def fitness_compare(self, fitness1, fitness2):
        pass

    def better_fitness_relation(self, x, y):
        pass

    def best_fitness_function(self, *args, **kwargs):
        pass

    def tf_get_pairwise_fitness(self, *arg):
        pass

    def conclusion(self, *arg):
        pass


class RegressionKernel(GPKernel):

    def best_fitness_function(self, *args, **kwargs):
        return min(*args, **kwargs)

    def better_fitness_relation(self, x, y):
        return x < y

    def __init__(self, kernel_name, data_train, tf_config, tf_device, eval_action):
        self.np_best_fitness = np.min
        self.kname = kernel_name
        self.tf_config = tf_config
        self.tf_device = tf_device
        self.eval_action = eval_action
        self.origin_results = None
        self.data_train = data_train  # sfeh where is the best?

        self.bounded = 'bounded' in kernel_name
        self.discrete = 'discrete' in kernel_name
        self.tanhpenalize = 'tanhpenalize' in kernel_name  # sfeh only makes sense when bounded

        self.MSE = 'MSE' in kernel_name
        self.RMSE = 'RMSE' in kernel_name
        self.MAE = 'MAE' in kernel_name

        self.exploration_risk = 'explun' in kernel_name
        self.origin_results = None  # can only be set after the evaluation of the origin...
        sfeh_help = {'explorate01': 0.1,
                     'explorate05': 0.5}

        self.pen_explorate = 0.1
        for k, v in sfeh_help.items():
            if k in kernel_name:
                self.pen_explorate = v
        return

    def pycode_wrap_result(self, action_min_max):
        wrap = '{}'

        if self.bounded:
            wrap = f'min(max({action_min_max[0]}, {wrap}), {action_min_max[1]})'

        if self.discrete:
            # regression that fits the outputs to a discrete set of actions defined by min and max
            wrap = f'int(math.round({wrap}))'

        return wrap

    def eval_tf(self, expr_sym, used_observations, only_fitness=False):
        """
        Evaluates an expression using TensorFlow (TF)
        - receives a (string) expression in numpy-style that was reduced with pythons "sympy" (for simplification)
        - uses "ast" to generate a, kind of, python-intern-executable-tree
        - creating a tensorflow graph that is evaluated in an isolated TF session
        """
        tf.compat.v1.reset_default_graph()
        solution = tf.constant(self.data_train[self.eval_action.name])  # tensors[self.eval_action.name]
        tensors = {obs_name: tf.constant(self.data_train[obs_name]) for obs_name in used_observations}  # do not assign dtype here, do this in the pandas df aka data

        results_agent = ast_convert_from_expr(expr_sym, tensors=tensors)  # the actual result from the expression in the agent

        # fit the agents to the possible outcome
        results_kernel = results_agent

        if self.discrete:
            results_kernel = tf.math.round(results_kernel)
        if self.bounded:
            act_min = tf.constant(self.eval_action.minmax[0], dtype=tf.float32)
            act_max = tf.constant(self.eval_action.minmax[1], dtype=tf.float32)
            results_kernel = tf.math.minimum(tf.math.maximum(results_kernel, act_min), act_max)

        # pairwise_fitness = self.tf_get_pairwise_fitness(solution, kernel_result, results_agent)
        pairwise_diff = solution - results_kernel

        if self.MSE or self.RMSE:  # sfeh huber loss! mse, mae, rmse, huber, (log)
            tf_error = tf.square
        else:
            tf_error = tf.abs

        regression_errors = tf_error(pairwise_diff)
        # improved_errors = regression_errors  # sfeh not yet required... only one error

        if self.exploration_risk and self.origin_results is not None:
            # tf_error = tf.abs  # sfeh this is required (??)
            # (1 * tf.abs(pairwise_diff)) - # 1 * abs, as the other one is within the error. usually 2*  # sfeh not sure
            exploration = (self.origin_results - results_kernel)  # the difference to the origin - which we want to "penalize" here
            required_improvement = tf.abs(solution - self.origin_results)  # the complete range that is 'okay' to actually explore here.
            explore_penalize = tf.maximum(required_improvement-exploration, 0)  # removes the above mentioned expected exploration from the penalize process
            penalize_exploration = self.pen_explorate*(tf_error(explore_penalize))  # this should not be weighted as much as the regular expression (0 to 1).
            # Although, even more extreme penalisations are possible. Also, ideas about dummy pen (for no exploration, but no easy improvement) or values >1 for sticking to the origin policy
            # use factor before or after squaring the distance?
            regression_errors += penalize_exploration

            """
            sfeh idea: process is markov chain, but logic seems to correct until the first wrong decision.
            This point could be of large interest, as ir marks the moment where the good policy is lost.
            'correct' is not known, though. (MTC - yes, but IB may be very close)
            """
        else:
            penalize_exploration = tf.no_op()

        mean_error = tf.reduce_mean(regression_errors)
        if self.RMSE:
            mean_error = tf.sqrt(mean_error)

        if self.tanhpenalize:
            """
            for the bounded kernel.
            Values, that are far too high, which get assigned to the action range, should be slightly punished.
            This should hopefully make improvements towards smaller numbers possible without affecting the parsimony.
            (e.g. results_agent = 33.6, but actionminmax[-1, 1] --> kernel_result = +1)

            tanh: closer to 0 is better, but rising steadyly without exceeding max value of 1 (outliers like single points inf become irrelevant)
            factor 1 (0.02) the amplitude. should be small enough to not significantly influence the gp process
            factor 2 (0.1) stretches the tanh function. the largest improvement should be at the points we want to get rid of
            squared distance? -> smooth transition from the area that is considered okay
            """
            penalized_bounds = 0.02 * tf.tanh(tf.square(results_agent - results_kernel) * 0.1)  # sfeh amplitude, stretch, squared
            mean_boundpen = tf.reduce_mean(penalized_bounds)  # sfeh could easily be a reduce_sum
            mean_error += mean_boundpen
        else:
            penalized_bounds = tf.no_op()

        with tf.compat.v1.Session(config=self.tf_config) as sess:  # tensorflow evaluation must be done in a "session". funfact: debugging is not ez
            with sess.graph.device(self.tf_device):  # GPU evaluation in tensorflow
                tf_results = sess.run({'pairwise_diff': pairwise_diff, 'results_kernel': results_kernel, 'regression_errors': regression_errors, 'mean_error': mean_error, 'penalize_exploration': penalize_exploration})
                # sfeh attention: the dict above returns np-type results, not real floats
        if only_fitness:  # reduced evaluation, only mean_error is returned... (may save memory as only one value gets returned)
            return tf_results['mean_error']
        else:
            return tf_results

    def conclusion(self, result):
        """
        sfeh this is baad
        """
        mse = skm.mean_squared_error(result['agent_result'], result['solution'])
        return f"\n\n Regression bounded fitness score: {result['fitness']}\n Mean Squared Error: {mse}"


class ClassificationKernel(GPKernel):

    def __init__(self, *args):
        pass

    def tf_classify_labels_map(self, result, env_vars):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the samples-csv.
        Outputs an array of tuples containing the predicted labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.uniques_num / 2) - 1 # '-1' keeps a binary classification splitting over the
            if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
            elif solution == self.uniques_num - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
            else: fitness = 0 # no class match
        sfeh remove

        """
        uniques_num = env_vars.eval_action.uniques
        skew = (uniques_num / 2) - 1
        label_rules = {uniques_num - 1: (
            tf.constant(uniques_num - 1), tf.constant(f' > {uniques_num - 2 - skew}'))}

        for class_label in range(uniques_num - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tf.cond(cond, lambda: (
                tf.constant(class_label), tf.constant(f' <= {class_label - skew}')), lambda: label_rules[class_label + 1])

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(f' <= {0 - skew}')), lambda: label_rules[1])

        return pred_label

    def fitness_compare(self, fitness1, fitness2):
        if fitness2 is None:
            return True
        else:
            return fitness1 > fitness2

    def best_fitness_function(self, *args, **kwargs):
        return max(*args, **kwargs)

    def tf_wrap_result(self, *args):
        pass

    def tf_get_pairwise_fitness(self, solution, kernel_result, eval_action):
        """
        Calculates the kernel-specific fitness for the solution.
        - classification: dummy
        """

        skew = (eval_action.uniques / 2) - 1

        rule1 = tf.logical_and(
            tf.equal(solution, 0),
            tf.less_equal(kernel_result, 0 - skew))

        rule2 = tf.logical_and(
            tf.equal(solution, eval_action.uniques - 1),
            tf.greater(kernel_result, solution - 1 - skew))

        rule3 = tf.logical_and(
            tf.less(solution - 1 - skew, kernel_result),
            tf.less_equal(kernel_result, solution - skew))

        pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule1, rule2), rule3), tf.int32)
        return pairwise_fitness


class MatchKernel(GPKernel):

    def __init__(self, *args):
        super().__init__(*args)

    def fitness_compare(self, fitness1, fitness2):
        if fitness2 is None:
            return True
        else:
            return fitness1 > fitness2

    def best_fitness_function(self, *args, **kwargs):
        return max(*args, **kwargs)

    def tf_get_pairwise_fitness(self, solution, kernel_result):
        """
        Calculates the kernel-specific fitness for the solution.
        - classification: dummy
        """
        """
        This is used for demonstration purposes only.
        """
        # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
        rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
        pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - kernel_result), atol + rtol * tf.abs(kernel_result)), tf.int32)

        return pairwise_fitness

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
    #
    #     elif self.kernel == 'match':
    #         result_str += f"\n\n Matching fitness score: {result['fitness']}"
    #
    #     else:  # 'regression discrete':
    #         result_str = 'No summary provided for this kernel'
    #
    #     return result_str


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
            # shape = tensors[list(tensors.keys())[0]].get_shape()
            return tf.constant(node.n, dtype=tf.float32)  # , shape=shape

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if build:
            return [node.value]
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
                    return ['Usub', [ast_expr_to(node.operand, build=True)]]
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
