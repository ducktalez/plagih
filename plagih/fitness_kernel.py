import copy
import os
from abc import ABC

import pandas as pd
import sympy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from plagih.plagih_tree import sympy_to_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from plagih.util import *

import tensorflow as tf
import numpy as np

# Eager execution used to be not possible in tf v1, in tf v2, this is the standard
# however, we need to build a complete graph before inserting the data in the leaf nodes.
tf.compat.v1.disable_eager_execution()


class EvalAction:
    """
    no nlabel; the action is not part of the evotree
    - minmax for histograms
    - minmax for regression-bounded
    """
    xtype = (None, float)

    def __init__(self, name, minmax=None):
        self.minmax = minmax
        self.data_column_name = name


class Kernel(ABC):
    pass


class OnlineKernel(Kernel, ABC):
    pass


class OfflineKernel(Kernel, ABC):
    """
    sfeh:OfflineKernel? Also: online kernels here?
    The "abstract" Kernel class for the GP process.
    idea: supporting multiple Kernels?
    """

    def __init__(self, data_train, action_name, tf_error_metric, tf_sanitize_results,
                 tf_gpu_allow_growth=True, tf_device='/gpu:0', tf_device_log=False, *args, **kwargs):
        """
        sfeh: tf_device_log is set automatically to false
        """
        self.solution_train = tf.constant(data_train[action_name])  # tensors[self.action.name],
        self.tf_error_metric = tf_error_metric
        self.tf_sanitize_results = tf_sanitize_results

        self.tf_config = tf.compat.v1.ConfigProto(log_device_placement=tf_device_log,
                                                  allow_soft_placement=True)  # check if GPU is actually used
        self.tf_config.gpu_options.allow_growth = tf_gpu_allow_growth  # sfeh:use optional args?
        self.tf_device = tf_device

        self.data_dict = data_train.to_dict('list')
        # self.data_control = data_control  # sfeh: currently not used

    def eval_tf(self, *args, **kwargs):
        return float('nan')


class Regression(OfflineKernel):

    def __init__(self, data_train, action_name, tf_error_metric, tf_sanitize_results, *args, **kwargs):
        super().__init__(data_train, action_name, tf_error_metric, tf_sanitize_results, *args, **kwargs)

        return

    def eval_tf(self, expr):
        """
        Evaluates an expression using TensorFlow (TF)
        - receives a (string) expression in numpy-style that was reduced with pythons "sympy" (for simplification)
        - uses "ast" to generate a, kind of, python-intern-executable-fintree
        - creating a tensorflow graph that is evaluated in an isolated TF session

        sfeh: data_control missing
        """
        # tf.compat.v1.reset_default_graph()  # sfeh:discuss is this really required to do? legacy code

        results_raw = sympy_to_tensorflow(expr, self.data_dict)  # self.data_train
        results = results_raw  # The ids change in the next lines! {id(results)} vs. {id(results_raw)}

        results = self.tf_sanitize_results(results)

        pairwise_diff = self.solution_train - results

        mean_error = self.tf_error_metric(pairwise_diff)

        with tf.compat.v1.Session(config=self.tf_config) as sess:
            with sess.graph.device(self.tf_device):  # GPU evaluation in tensorflow
                tf_results = sess.run(
                    {'pairwise_diff': pairwise_diff,
                     'results': results,
                     'mean_error': mean_error})  # no feed_dict

        fitness = tf_results['mean_error']

        if fitness != fitness or fitness == float('inf'):
            raise ValueError(f"fitness is: '{fitness}'")  # sfeh: so bad, they leave the float-range. delete this?

        tf_results['mean_error'] = round(float(tf_results['mean_error']), PRECISION)

        return tf_results

    # def histogram_bins(self):
    #     act_range = self.action_clip[1] - self.action_clip[0]  # max value - min value
    #     if self.action_round:  # [0, 1, 2] -> 2
    #         # sfehfun make kernel histogram function?
    #         action_bins = np.linspace(-0.5 - act_range, 0.5 + act_range, 2 * act_range + 1 + 1)  # for +-0.5 and 0
    #     else:
    #         num_bins = 16 + 1  # +1 is extra bin for 0
    #         breite = 0.5 * (act_range * 2) / num_bins
    #         action_bins = np.linspace(-(breite + act_range), + (breite + act_range), num_bins + 1)  # sfeh 10 bins?
    #     return action_bins
    #
    # def plot_agent_histogram(self, parsim, tree, path_hist):
    #     """
    #     Make histograms for all paretofront-efficient candidates
    #     sfeh: based on training data- maybe use test data...
    #
    #     useful code?
    #     # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
    #     # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
    #     # histogram_data = np.multiply.reduce(histogram_data, axis=1)
    #     # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
    #     """
    #
    #     action_bins = self.histogram_bins()
    #     expr_raw = tree.eval_expr_str()
    #     # used_observations = tree.get_observation_list()  sfeh:delete all the cases where this was used
    #     pairwise_diff = self.eval_tf(expr_raw)['pairwise_diff']
    #
    #     with plt.rc_context(rc=pyplot_rc_tex):
    #         fig, ax = plt.subplots()
    #         ax.hist(pairwise_diff, bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')
    #         ax.set(ylim=(0, len(self.data_train)), ylabel='frequency', xlabel='deviation')
    #         histpath = path_hist / f'acthist_{parsim}.pdf'
    #         fig.savefig(histpath)
    #         plt.close('all')
    #
    #     return histpath

    def eval_sym_experimental(self, expr, return_results=False):
        """
        Not stable and only working with mountaincar
        """
        _inputs = self.data_dict

        cartVel, cartPos = sympy.symbols('cartVel cartPos')
        ex = sympy.sympify(str(expr))
        f = sympy.lambdify([cartVel, cartPos], ex, 'numpy')
        cartVel = np.array(_inputs['cartVel'])
        cartPos = np.array(_inputs['cartPos'])
        action = np.array(_inputs['action'])
        raw_results = f(cartVel, cartPos)
        results = np.round(np.clip(raw_results, 0, 2), 0)

        if not return_results:
            fitness = np.sqrt(np.mean((results-action)**2))
            return np.round(fitness, PRECISION)
        else:
            return results


def sfeh_open():
    pass

    #################################################
    # if self.exploration_risk and self.origin_results is not None:
    #     # tf_error = tf.abs  # sfeh this is required (??)
    #     # (1 * tf.abs(pairwise_diff)) - # 1 * abs, as the other one is within the error. usually 2*  # sfeh not sure
    #     exploration_korridor = tf.abs(
    #         self.solution - self.origin_results)  # the complete range that is 'okay' to actually explore here.
    #     exploration = (self.origin_results - results)  # the difference to the origin - which to "penalize" here
    #     explore_penalize = tf.maximum(exploration_korridor - exploration,
    #                                           0)  # removes the above mentioned expected exploration from penalize
    #     penalize_exploration = self.pen_explorate * (
    #         tf_error(explore_penalize))  # this should not be weighted as much as the regular expression (0 to 1).
    #     # Although, even more extreme penalisations are possible. Also, ideas about dummy pen
    #     # (for no exploration, but no easy improvement) or values >1 for sticking to the origin policy
    #     # use factor before or after squaring the distance?
    #     regression_errors += penalize_exploration
    #
    #     """
    #     sfeh idea: process is markov chain, but logic seems to correct until the first wrong decision.
    #     This point could be of large interest, as ir marks the moment where the good policy is lost.
    #     'correct' is not known, though. (MTC - yes, but IB may be very close)
    #     """
    # else:
    #     penalize_exploration = tf.no_op()


# class ClassificationKernel(Kernel):
#
#     def __init__(self, path_data_csv, conf, *args, **kwargs):
#         super().__init__(path_data_csv, conf, *args, **kwargs)
#
#     def eval_tf(self):
#         pass
#
#     def tf_classify_labels_map(self, result, action):
#         """
#         For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions
#         based upon the quantity of true class labels provided in the samples-csv.
#         Outputs an array of tuples containing the predicted labels based upon the result and corresponding boolean
#         condition triggered.
#
#         For comparison, the original (pre-TensorFlow) cod follows:
#
# skew = (self.uniques_num / 2) - 1 # '-1' keeps a binary classification splitting over the if solution == 0 and
# result <= 0 - skew; fitness_train = 1: # check for first class (the left-most bin) elif solution ==
# self.uniques_num - 1 and result > solution - 1 - skew; fitness_train = 1: # check for last class (the right-most
# bin) elif solution - 1 - skew < result <= solution - skew; fitness_train = 1: # check for class bins between first
# and last else: fitness_train = 0 # no class match sfeh remove
#
#         """
#         uniques_num = action.uniques
#         skew = (uniques_num / 2) - 1
#         label_rules = {uniques_num - 1: (
#             tf.constant(uniques_num - 1), tf.constant(f' > {uniques_num - 2 - skew}'))}
#
#         for class_label in range(uniques_num - 2, 0, -1):
#             cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
#             label_rules[class_label] = tf.cond(cond, lambda: (
#                 tf.constant(class_label), tf.constant(f' <= {class_label - skew}')),
#                                                lambda: label_rules[class_label + 1])
#
#         pred_label = tf.cond(result <= 0 - skew,
#                              lambda: (tf.constant(0), tf.constant(f' <= {0 - skew}')),
#                              lambda: label_rules[1])
#
#         return pred_label
#
#     def get_fitness_extreme_function(self, *args, **kwargs):
#         return min(*args, **kwargs)
#
#     def tf_get_pairwise_fitness(self, solution, kernel_result, eval_action):
#         """
#         Calculates the kernel-specific fitness_train for the solution.
#         - classification: dummy
#         """
#
#         skew = (eval_action.uniques / 2) - 1
#
#         rule1 = tf.logical_and(
#             tf.equal(solution, 0),
#             tf.less_equal(kernel_result, 0 - skew))
#
#         rule2 = tf.logical_and(
#             tf.equal(solution, eval_action.uniques - 1),
#             tf.greater(kernel_result, solution - 1 - skew))
#
#         rule3 = tf.logical_and(
#             tf.less(solution - 1 - skew, kernel_result),
#             tf.less_equal(kernel_result, solution - skew))
#
#         pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule1, rule2), rule3),
#                                           tf.int32)
#         return pairwise_fitness
#
#
# class MatchKernel(Kernel):
#     """
#     The match kernel does
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def eval_tf(self):
#         pass
#
#     def tf_get_pairwise_fitness(self, solution, kernel_result):
#         """
#         Calculates the kernel-specific fitness_train for the solution.
#         - classification: dummy
#         """
#         """
#         This is used for demonstration purposes only.
#         """
#         # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
#         rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
#         pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - kernel_result),
#                                                         atol + rtol * tf.abs(kernel_result)),
#                                           tf.int32)
#
#         return pairwise_fitness
#
# def eval_tf(self, *args, **kwargs): # if self.get_predicted_labels: #     predicted_labels = tf.map_fn(
# self.tf_classify_labels_map, kernel_result, dtype=(tf.int32, tf.string), swap_memory=True) # else: #
# predicted_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel #  # , 'predicted_labels':
# predicted_labels    # predicted_labels pass


if __name__ == "__main__":

    # ['cos(1.69234081*Max(0.024077, cartPos + 12.726454)**2) + sign(cartVel)',
    #  '1.527033*cartVel + sign(cartVel) + Max(cartPos, cartVel)',
    #  'sign(cartVel) + 1.476507']
    df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv')
    df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P design pattern #YOLO
    data_train, data_control = train_test_split(df, test_size=0.2, random_state=0)
