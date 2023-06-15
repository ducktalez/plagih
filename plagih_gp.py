"""
This starts the whole genetic programming.
This (extra) file was added to have a file in the root directory that can be started.
"""
import itertools
import sys
import random

# import multiprocessing as mp
import sympy
from sklearn.model_selection import train_test_split

from plagih.fitness_kernel import Regression
from plagih.plagih_gp_base_class_xai import *
from plagih.random_nodes_generator import NodeCreatorBase, norm_choices, operatorpool_to_picks
from plagih.util import *


def _test_random_pop():
    name = 'MTC200_RMSE_scratch'
    rootdir = Path.cwd() / f'{name}'

    INPUT_NAMES = ['cartVel', 'cartPos']
    action_name = 'action'

    # ## Load the training data into Kernel-class(...only offline training in this run).
    df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv')
    df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P design pattern #YOLO
    data_train, data_control = train_test_split(df, test_size=0.2, random_state=0)
    origin_root = RootNode_Dummy(Number)
    outcome = sympy.Symbol('outcome')
    tree_base = Clip(Round(outcome), 0, 2)  # sfeh:open

    tf_sanitize_results = lambda res: tf.round(tf.clip_by_value(res, 0, 2))
    tf_error_metric = lambda pw_diffs: tf.sqrt(tf.reduce_mean(tf.square(pw_diffs)))
    # tf.reduce_mean(tf.abs(pairwise_diff))  # sfeh:open
    kernel = Regression(data_train, action_name, tf_error_metric, tf_sanitize_results)

    # sfeh:idea track total trees in lut and matches, maybe even check diversity?

    # ## Run/Computation restrictions
    pop_max = 100
    gen_max = 100
    nodes_max = 50
    depth_max = 10

    # GP Evolution
    period = {'gen_plots': 5, 'gen_save': 5}
    mp_cores = 1

    # ### A simple tree and a simple tree with fixed nodes
    # '["Ifte",["<",["cartVel"],["0"]],["0"],["2"]]'
    # '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]'
    # origin_tree = Ifte(Le(Symbol('cartVel'), Float(0)), Float(0), Float(2))

    class Node_creator(NodeCreatorBase):
        def __init__(self):
            """make all probabilities sum to 1 for each categoray (Add: 2, Mul: 1, Tan: 0.5) in"""

            self.pick_op, self.pick_op_match = operatorpool_to_picks(
                {Add: 2, Sub: 1, Mul: 2, Div: 1, Square: 0.75, Abs: 0.5, Sign: 0.5, Sqrt: 0.1, Log: 0.1,
                 Sin: 0.5, Tan: 0.1, Cos: 0.33, Min: 1, Max: 1, And: 1, Or: 1, Not: 0.5, Xor: 1, Lt: 0.5,
                 Le: 0.5, Ifte: 2,
                 Powrounded: 0.5})  # sfeh: Acos: 0.33, Asin: 0.33, Atan: 0.33, Tanh: 0.5, Usub: 1,
            # Round: 0.5, Eq: 1,  # Ne: 0.5, #  # Log1p: 0.1, Gt: 0.1, Ge: 0.1,

            self.pick_symbol = {float: norm_choices([[sympy.Symbol(ii, real=True, imaginary=False), 1] for ii in INPUT_NAMES]),
                                bool: []}  # NotImplementedError

            # -> Choosing 50 random numeric values from the dataset for building trees ...just not zeros)
            samples = [ii for ii in itertools.chain.from_iterable(df[INPUT_NAMES].sample(n=50).values) if ii != 0]
            self.pick_constant = {float: norm_choices([
                [lambda: round(random.normalvariate(0, 1), FLOAT_PRECISION), 0.2],
                [lambda: round(random.normalvariate(1, 1), FLOAT_PRECISION), 0.1],
                [lambda: round(random.normalvariate(10, 5), FLOAT_PRECISION), 0.1],
                [lambda: round(random.randint(1, 20), FLOAT_PRECISION), 0.1],
                [lambda: round(random.choice(samples), FLOAT_PRECISION), 0.5]]),
                                  bool: norm_choices([[lambda: random.choice((True, False)), 1]])}

        def choose_operator(self, xt):
            # todo allow_chain
            op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
            return op

        def choose_operator_match(self, xtype):
            # todo allow_chain
            op = np.random.choice(self.pick_op_match[xtype][0], p=self.pick_op_match[xtype][1])
            return op

        def choose_terminal(self, xt, p_observation=0.5):
            if np.random.random() > p_observation:
                try:
                    _v = self.choose_symbol(xt)
                    return Node(Symbol, [_v])
                except (TypeError, IndexError):
                    pass  # return a constant (E.g. because there are no boolean observations)

            _v = self.choose_constant(xt)
            # sfeh expected str|int|long|float|Decimal|Number object but got 'Node'

            return _v

        def choose_constant(self, xt):
            _v = np.random.choice(self.pick_constant[xt][0], p=self.pick_constant[xt][1])()  # only dist. must be ()
            if xt == float:
                _v = sympy.Float(_v)
                # todo allow rational
                return Node(Number, [_v])  # round FLOAT_PRECISION was here
            else:
                _v = sympy.logic.boolalg.BooleanAtom(_v)  # sfeh:discuss: vs. Boolean
                return Node(Boolean, [_v])

        def choose_symbol(self, xt):
            _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
            return _v

    nc = Node_creator()

    build_restrictions = {'depth_max': 7, 'nodes_max': 50}
    tb = TreeBuildRestrictions(origin_root, nc, build_restrictions, 'tree_node_count')

    gp = ExplainableGP(name, pop_max, gen_max, rootdir, kernel, tb)
    # gp.pop_kill()  # optional, maybe restart pop between runs?
    try:
        gp.backup_load(path_load_custom_backup=rootdir)
    except FileNotFoundError as ex:
        printpl('i', f'No backup file found at {ex}. Starting a new run.')

    period_plots = 10
    period_save = 10
    gp.evoloop(period_plots, period_save)
    gp.evoloop_monitoring_plots()

    print('***Program ending***\n'
          '********************\n\n')
    sys.exit()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    _test_random_pop()

# class ObservationIndex(Observation):
#     """
#       sfeh:open
#     """
#
#     def __init__(self, nlabel, xtype_out=float, obs_indizes=None):
#         # super().__init__(nlabel, xtype_out)
#         self.obs_indizes = obs_indizes
#         latex = f'\\text{{{self.fam}}}_{{{self.time_index}}}'  # remove this {self.preexpr}
#         self.latex = (latex, latex)  # remove this {self.preexpr}
#
