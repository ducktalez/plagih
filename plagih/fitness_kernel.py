from plagih.node_labels import op_dict

import tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split

from plagih.util import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast


# def activate_dataset(path_data, action):
#     """
#     loading the data which the GP will be working on.
#     The .csv-file is prepared (loading correct data-type, splitting data, ...)
#     and saved as pickle-file for reloading runs.
#     This is especially important, as the split in training and test-data must be the same.
#
#     separate loading the prepared data into the main class.
#     Why like this? I needed to find a bug in the data_from_csv file and
#     did not want to start the whole stuff everytime
#
#         # self.data_train_panda, self.data_control_panda
#     """
#
#     if path_data.suffix == '.p':
#         data_prepared = pickle_load(path_data)
#     elif path_data.suffix == '.csv':
#         data_prepared = data_from_csv(path_data, action=action)
#     else:
#         raise FileNotFoundError(f'No data provided? File must be a pickle (.p) or csv (.csv) file. Loaded file: {path_data}')
#
#     env_vars, data_train, data_control = data_prepared  # data_control is data_test
#
#     return env_vars, data_train, data_control


def ast_expr_to(node, tensors=None, build=None):
    """
    Returns (recursively) a (tensorflow) graph from a (raw or sympified) math expression.
    please use by calling labels_from_graphlist()

    Used to be for tensorflow only, but was modified to save 'sympified' trees.

    One of [tensors, prnt, build] must be set
    -> tensors: Creates a tensorflow graph for evaluation
    -> prnt: creates a string expression of the fintree (I think I tried this before 'build' worked)
    -> build: creates a nested nlabel-list, e.g. a+(b/c) -> [+, [a], [/, [b, c]]]] (at least I think so)
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
            try:
                shape = tensors[list(tensors.keys())[0]].get_shape()  # todo
                return tensorflow.constant(node.n, dtype=tensorflow.float32, shape=shape)  # , shape=shape
                # ^ValueError: Shapes must be equal rank, but are 0 and 1 for 'Select_1' (op: 'Select') with input shapes
                # => sfeh: in some tf-versions, the constants have to match their shape. data has shape [3423, 0]
                # constants have shape []
            except Exception as ex:
                return tensorflow.constant(node.n, dtype=tensorflow.float32)  # , shape=shape
                # # ^Problem occurs, when no real constants/variables are in the tree
                # # Could not append fintree to population because: eval-ex: list index out of range
                # #   =>fintree: [Ifte, [<, [0.0], [0.0]], [0.0], [2.0]]

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if build:
            return [node.value]
        else:
            return tensorflow.constant(node.value)
    #
    # Arity 1
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., sin(1), -1
        if build:
            if type(node.op) == ast.USub:  # workaround for ~-problem
                if isinstance(node.operand, ast.Name) or isinstance(node.operand, ast.Num) or isinstance(node.operand, ast.NameConstant):
                    return [f'-{ast_expr_to(node.operand, build=True)[0]}']
                else:
                    return ['Usub', [ast_expr_to(node.operand, build=True)]]
            return [op_dict[type(node.op)].nlabel, [ast_expr_to(node.operand, build=True)]]
            # was: return [op_dict[type(node.op_dict)].nlabel, [ast_expr_to(node.operand, build=True)]]
        else:
            return op_dict[type(node.op)].tflow(ast_expr_to(node.operand, tensors=tensors))

    # Arity 2
    elif isinstance(node, ast.BinOp) or isinstance(node, ast.BitAnd):  # <left> <operator> <right>, e.g., (x + y), (a & True)
        if build:
            return [op_dict[type(node.op)].nlabel,
                    [ast_expr_to(node.left, build=True),
                     ast_expr_to(node.right, build=True)]]
        else:
            return op_dict[type(node.op)].tflow(
                ast_expr_to(node.left, tensors=tensors),
                ast_expr_to(node.right, tensors=tensors))

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        if build:
            return ast_chain_bool(node.values, op_dict[type(node.op)].nlabel, build=True)
        else:
            return ast_chain_bool(node.values, op_dict[type(node.op)].tflow, tensors=tensors)

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
                return op_dict[node.func.id].tflow(tensorflow.dtypes.cast(
                    ast_expr_to(node.args[0], tensors=tensors), tensorflow.bool),
                    ast_expr_to(node.args[1], tensors=tensors),
                    ast_expr_to(node.args[2], tensors=tensors))

        elif len(node.args) <= 2:
            if build:
                if len(node.args) == 1:
                    return [op_dict[node.func.id].nlabel,
                            [ast_expr_to(node.args[0], build=True)]]
                    # return [op_dict[node.func.id],  # sfeh:check: remove .nlabel?
                    #         [ast_expr_to(node.args[0], build=True)]]
                elif len(node.args) == 2:
                    return [op_dict[node.func.id].nlabel,
                            [ast_expr_to(node.args[0], build=True),
                             ast_expr_to(node.args[1], build=True)]]
                else:
                    raise Exception('This arity is not supported')
            else:
                return op_dict[node.func.id].tflow(*[ast_expr_to(arg, tensors=tensors) for arg in node.args])

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
        x = tensorflow.dtypes.cast(ast_expr_to(values[0], tensors=tensors), tensorflow.bool)
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
        print_e('This is usually not used, and-concatenation of multiple chain compares. sfeh, bring this back?')
        return tensorflow.logical_and(op_dict[type(ops[0])].tflow(x, y), ast_chain_compare(comparators[1:], ops[1:], tensors=tensors))
    else:
        if build:
            return [op_dict[type(ops[0])].nlabel, [x, y]]
        else:
            return op_dict[type(ops[0])].tflow(x, y)


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


def ast_convert_from_expr(expr, tensors=None, build=None):
    """
    Starts the recursive ast-analysis of the expression

    Extract expression fintree from the string algo_sym.
    Please provide ONE of the following if you want to get...
    - tensorflow-graph: All variables (observation0, ...) as tensors.
    - build: True
    More information in ast_expr_to()

    """
    try:
        ast_tree = ast.parse(expr, mode='eval').body
        graph = ast_expr_to(ast_tree, tensors=tensors, build=build)
    except Exception as ex:
        # todo debug/remove
        ast_tree = ast.parse(expr, mode='eval').body
        graph = ast_expr_to(ast_tree, tensors=tensors, build=build)
    if build:
        graph = labels_from_nestedexpr(graph, [])

    return graph


class EvalAction:
    """
    no nlabel; the action is not part of the evotree
    - minmax for histograms
    - minmax for regression-bounded
    """
    tf_type = tensorflow.float32  # sfeh especiall when the type is integer
    xtype = (None, float)

    def __init__(self, name, minmax=None):
        # self.nlabel = name
        self.minmax = minmax
        self.data_column_name = name


class Kernel:
    """
    The "abstract" Kernel class for the GP process.
    optional: creating another
    """

    def __init__(self, path_data_csv, conf, *args, **kwargs):
        """
        sfeh: tf_device_log is set automatically to false
        """
        # Evaluating kernel (that uses tensorflow)
        self.tf_config = tensorflow.compat.v1.ConfigProto(log_device_placement=conf.tf_device_log,
                                                          allow_soft_placement=True)  # TF device usage logging (for debugging) (default false. I lately used it to check if the GPU is used)
        self.tf_config.gpu_options.allow_growth = conf.tf_gpu_allow_growth
        self.tf_device = conf.tf_device

        # with Path.open(path_data_csv, 'r') as file:
        df = pd.read_csv(path_data_csv)
        # sfeh it is float64, float64, int64 with MTC.. does it work with Tensorflow?
        # sfeh Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device. Is cpu otherwise

        action_name = conf.action_name or df.columns[len(df.columns) - 1]  # sfeh conf & kernel
        self.action = EvalAction(action_name, minmax=(0, 2))  # todo conf.action_minmax

        printez('i', f'Ignoring columns: {conf.dc}')  # , print_type=print_type print_type sfeh
        df = df.drop(conf.dc, axis=1)  # no need to keep other actions

        self.obs_names = list(df.columns)
        self.obs_names.remove(action_name)

        df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P Following design pattern #YOLO
        self.data_train, self.data_control = train_test_split(df, test_size=0.2, random_state=0)  # discussion: random state 0 okay? test_size 0.2?

    def fitness_compare(self, fitness1, fitness2):
        """
        if fitness_compared is None:
            return True
        else:
            return True/False, DEPEND
        sfeh: replace with pythonic way (__gt__ function above)
        """
        pass

    def eval_tf(self, *args, **kwargs):
        return float('nan')

    def relation(self, x, y):
        pass

    def get_fitness_extreme_function(self, *args, **kwargs):
        """
        probably return min(*args, **kwargs)
        e.g. the regression kernel
        """
        return min(*args, **kwargs)

    def tf_get_pairwise_fitness(self, *arg):
        pass

    def conclusion(self, *arg):
        pass


class RegressionKernel(Kernel):

    def __init__(self, path_data_csv, conf, *args, **kwargs):
        super().__init__(path_data_csv, conf, *args, **kwargs)
        self.np_best_fitness = np.min

        # self.kernel_version_plot_yaxis = f"regression error"
        # for option, plot_axis_string in {'discrete': ', discrete', 'bounded': ', bounded', 'tanhpenalize': ', penalize (tanh)'}.items():
        #     if option in kernel_name:
        #         self.kernel_version_plot_yaxis += plot_axis_string
        kernel_name = conf.kernel_name
        self.precision = conf.precision or 6
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

        self.fitness_sign = -1

        self.pen_explorate = 0.1
        for k, v in sfeh_help.items():
            if k in kernel_name:
                self.pen_explorate = v
        return

    def fitness_compare(self, fitness, fitness_compared):
        """
        """
        if fitness_compared is None:
            return True
        else:
            return fitness < fitness_compared

    def histogram_bins(self, action_minmax):
        act_min, act_max = action_minmax
        act_range = act_max - act_min
        if self.discrete:  # [0, 1, 2] -> 2
            # sfehfun make kernel histogram function?
            action_bins = np.linspace(-0.5 - act_range, 0.5 + act_range, 2 * act_range + 1 + 1)  # for +-0.5 and 0
        else:
            num_bins = 16 + 1  # +1 is extra bin for 0
            breite = 0.5 * (act_range * 2) / num_bins
            action_bins = np.linspace(-(breite + act_range), + (breite + act_range), num_bins + 1)  # sfeh 10 bins?
        return action_bins

    def get_fitness_extreme_function(self, *args, **kwargs):
        return min(*args, **kwargs)

    def relation(self, x, y):
        return lambda x, y: x < y

    def __gt__(self, other):
        """
        gt means better?
        """
        pass  # no, greater=better is ambiguous

    def pycode_wrap_result(self, action_min_max):
        wrap = '{}'

        if self.bounded:
            wrap = f'min(max({action_min_max[0]}, {wrap}), {action_min_max[1]})'

        if self.discrete:
            # regression that fits the outputs to a discrete set of actions defined by min and max
            wrap = f'int(math.round({wrap}))'

        return wrap

    def plot_agent_histogram(self, parsim, tree, path_hist):
        """
        Make histograms for all paretofront-efficient candidates
        sfeh: based on training data- maybe use test data...

        useful code?
        # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
        # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
        # histogram_data = np.multiply.reduce(histogram_data, axis=1)
        # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
        """

        action_bins = self.histogram_bins(self.action.minmax)
        expr_sym = tree.eval_expr()
        used_observations = tree.get_observation_list()
        pairwise_diff = self.eval_tf(expr_sym, used_observations)['pairwise_diff']

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            ax.hist(pairwise_diff, bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')
            ax.set(ylim=(0, len(self.data_train)), ylabel='frequency', xlabel='deviation')
            histpath = path_hist / f'acthist_{parsim}.pdf'
            fig.savefig(histpath)
            plt.close('all')

        return histpath

    def eval_tf(self, expr, used_observations, only_fitness=False):
        """
        Evaluates an expression using TensorFlow (TF)
        - receives a (string) expression in numpy-style that was reduced with pythons "sympy" (for simplification)
        - uses "ast" to generate a, kind of, python-intern-executable-fintree
        - creating a tensorflow graph that is evaluated in an isolated TF session
        """
        tensorflow.compat.v1.reset_default_graph()
        solution = tensorflow.constant(self.data_train[self.action.data_column_name])  # tensors[self.action.name]
        tensors = {obs_name: tensorflow.constant(self.data_train[obs_name]) for obs_name in used_observations}  # do not assign dtype here, do this in the pandas df aka data

        results_agent = ast_convert_from_expr(expr, tensors=tensors)  # the actual result from the expression in the agent

        # fit the agents to the possible outcome
        results_kernel = results_agent  # The ids change in the next lines! {id(results_kernel)} vs. {id(results_agent)}
        if self.discrete:
            results_kernel = tensorflow.math.round(results_kernel)
        if self.bounded:
            act_min = tensorflow.constant(self.action.minmax[0], dtype=tensorflow.float32)
            act_max = tensorflow.constant(self.action.minmax[1], dtype=tensorflow.float32)
            results_kernel = tensorflow.math.minimum(tensorflow.math.maximum(results_kernel, act_min), act_max)

        # pairwise_fitness = self.tf_get_pairwise_fitness(solution, kernel_result, results_agent)
        pairwise_diff = solution - results_kernel

        if self.MSE or self.RMSE:  # sfeh huber loss! mse, mae, rmse, huber, (log)
            # sfeh remove the fucking RMSE?^^
            tf_error = tensorflow.square
        else:
            tf_error = tensorflow.abs

        regression_errors = tf_error(pairwise_diff)
        # improved_errors = regression_errors  # sfeh not yet required... only one error

        if self.exploration_risk and self.origin_results is not None:
            # tf_error = tensorflow.abs  # sfeh this is required (??)
            # (1 * tensorflow.abs(pairwise_diff)) - # 1 * abs, as the other one is within the error. usually 2*  # sfeh not sure
            exploration_korridor = tensorflow.abs(solution - self.origin_results)  # the complete range that is 'okay' to actually explore here.
            exploration = (self.origin_results - results_kernel)  # the difference to the origin - which we want to "penalize" here
            explore_penalize = tensorflow.maximum(exploration_korridor - exploration, 0)  # removes the above mentioned expected exploration from the penalize process
            penalize_exploration = self.pen_explorate * (tf_error(explore_penalize))  # this should not be weighted as much as the regular expression (0 to 1).
            # Although, even more extreme penalisations are possible. Also, ideas about dummy pen (for no exploration, but no easy improvement) or values >1 for sticking to the origin policy
            # use factor before or after squaring the distance?
            regression_errors += penalize_exploration

            """
            sfeh idea: process is markov chain, but logic seems to correct until the first wrong decision.
            This point could be of large interest, as ir marks the moment where the good policy is lost.
            'correct' is not known, though. (MTC - yes, but IB may be very close)
            """
        else:
            penalize_exploration = tensorflow.no_op()

        mean_error = tensorflow.reduce_mean(regression_errors)
        if self.RMSE:
            mean_error = tensorflow.sqrt(mean_error)

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
            penalized_bounds = 0.02 * tensorflow.tanh(tensorflow.square(results_agent - results_kernel) * 0.1)  # sfeh amplitude, stretch, squared
            mean_boundpen = tensorflow.reduce_mean(penalized_bounds)  # sfeh could easily be a reduce_sum
            mean_error += mean_boundpen
        else:
            penalized_bounds = tensorflow.no_op()

        with tensorflow.compat.v1.Session(config=self.tf_config) as sess:  # tensorflow evaluation must be done in a "session". funfact: debugging is not ez
            with sess.graph.device(self.tf_device):  # GPU evaluation in tensorflow
                tf_results = sess.run(
                    {'pairwise_diff': pairwise_diff,
                     'results_kernel': results_kernel,
                     'regression_errors': regression_errors,
                     'mean_error': mean_error,
                     'penalize_exploration': penalize_exploration})
                # sfeh attention: the dict above returns np-type results, not real floats

        fitness = tf_results['mean_error']

        if fitness != fitness or fitness == float('inf'):
            raise ValueError(f"fitness is: '{fitness}'")  # sfeh or Exception happens, eg when values are soo wrong that it leaves the float-range

        tf_results['mean_error'] = round(float(tf_results['mean_error']), self.precision)  # sfeh option: round directly in tensorflow

        if only_fitness:  # reduced evaluation, only mean_error is returned... (may save memory as only one value gets returned)
            return tf_results['mean_error']
        else:
            return tf_results

    def conclusion(self, result):
        """
        sfeh this is baad
        """
        # return f"\n\n Regression bounded fitness_train score: {result['fitness_train']}\n Mean Squared Error: {}"
        return


class ClassificationKernel(Kernel):

    def __init__(self, path_data_csv, conf, *args, **kwargs):
        super().__init__(path_data_csv, conf, *args, **kwargs)

        self.fitness_sign = +1

    def eval_tf(self):
        pass

    def tf_classify_labels_map(self, result, action):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the samples-csv.
        Outputs an array of tuples containing the predicted labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.uniques_num / 2) - 1 # '-1' keeps a binary classification splitting over the
            if solution == 0 and result <= 0 - skew; fitness_train = 1: # check for first class (the left-most bin)
            elif solution == self.uniques_num - 1 and result > solution - 1 - skew; fitness_train = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness_train = 1: # check for class bins between first and last
            else: fitness_train = 0 # no class match
        sfeh remove

        """
        uniques_num = action.uniques
        skew = (uniques_num / 2) - 1
        label_rules = {uniques_num - 1: (
            tensorflow.constant(uniques_num - 1), tensorflow.constant(f' > {uniques_num - 2 - skew}'))}

        for class_label in range(uniques_num - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tensorflow.cond(cond, lambda: (
                tensorflow.constant(class_label), tensorflow.constant(f' <= {class_label - skew}')), lambda: label_rules[class_label + 1])

        pred_label = tensorflow.cond(result <= 0 - skew, lambda: (tensorflow.constant(0), tensorflow.constant(f' <= {0 - skew}')), lambda: label_rules[1])

        return pred_label

    def fitness_compare(self, fitness1, fitness2):
        """

        """
        if fitness2 is None:
            return True
        else:
            return fitness1 > fitness2

    def get_fitness_extreme_function(self, *args, **kwargs):
        return max(*args, **kwargs)

    def tf_get_pairwise_fitness(self, solution, kernel_result, eval_action):
        """
        Calculates the kernel-specific fitness_train for the solution.
        - classification: dummy
        """

        skew = (eval_action.uniques / 2) - 1

        rule1 = tensorflow.logical_and(
            tensorflow.equal(solution, 0),
            tensorflow.less_equal(kernel_result, 0 - skew))

        rule2 = tensorflow.logical_and(
            tensorflow.equal(solution, eval_action.uniques - 1),
            tensorflow.greater(kernel_result, solution - 1 - skew))

        rule3 = tensorflow.logical_and(
            tensorflow.less(solution - 1 - skew, kernel_result),
            tensorflow.less_equal(kernel_result, solution - skew))

        pairwise_fitness = tensorflow.dtypes.cast(tensorflow.logical_or(tensorflow.logical_or(rule1, rule2), rule3), tensorflow.int32)
        return pairwise_fitness


class MatchKernel(Kernel):
    """
    The match kernel does
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fitness_sign = +1

    def eval_tf(self):
        pass

    def fitness_compare(self, fitness1, fitness2):
        """

        """
        if fitness2 is None:
            return True
        else:
            return fitness1 > fitness2

    def get_fitness_extreme_function(self, *args, **kwargs):
        """

        """
        return max(*args, **kwargs)

    def tf_get_pairwise_fitness(self, solution, kernel_result):
        """
        Calculates the kernel-specific fitness_train for the solution.
        - classification: dummy
        """
        """
        This is used for demonstration purposes only.
        """
        # pairwise_fitness = tensorflow.dtypes.cast(tensorflow.equal(solution, result), tensorflow.int32) # breaks due to floating points
        rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
        pairwise_fitness = tensorflow.dtypes.cast(tensorflow.less_equal(tensorflow.abs(solution - kernel_result), atol + rtol * tensorflow.abs(kernel_result)), tensorflow.int32)

        return pairwise_fitness

    def eval_tf(self, *args, **kwargs):
        # if self.get_predicted_labels:
        #     predicted_labels = tensorflow.map_fn(self.tf_classify_labels_map, kernel_result, dtype=(tensorflow.int32, tensorflow.string), swap_memory=True)
        # else:
        #     predicted_labels = tensorflow.no_op()  # a placeholder, applies only to CLASSIFY kernel
        #  # , 'predicted_labels': predicted_labels    # predicted_labels
        pass

    # def conclusion_text(self, result, fitness_control_best):
    #     """
    #
    #     """
    #     elif self.kernel == 'regression':
    #         mse = skm.mean_squared_error(result['agent_result'], result['solution_goal'])
    #         result_str += ('\n\n Regression fitness_train score: {}'.format(result['fitness_train']))
    #         result_str += ('\n Mean Squared Error: {}'.format(mse))
    #
    #     result_str = ''
    #
    #     if self.kernel == 'classification':
    #         result_str += f'\n\n Classification fitness_train score: {fitness_control_best}'
    #         result_str += ('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution_goal'], result['predicted_labels'][0])))
    #         result_str += ('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution_goal'], result['predicted_labels'][0])))
    #
    #     elif self.kernel == 'regression bounded':
    #
    #     elif self.kernel == 'match':
    #         result_str += f"\n\n Matching fitness_train score: {result['fitness_train']}"
    #
    #     else:  # 'regression discrete':
    #         result_str = 'No summary provided for this kernel'
    #
    #     return result_str
