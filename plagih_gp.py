"""
This starts the whole genetic programming.
This (extra) file was added to have a file in the root directory that can be started.
"""
# import copy
# import itertools
import sys

import numpy as np

# import random

# import multiprocessing as mp
# import sympy
# from sklearn.model_selection import train_test_split

from plagih.fitness_kernel import *
from plagih.plagih_gp_base_class_xai import *
from plagih.random_nodes_generator import norm_choices, operatorpool_to_picks
from plagih.util import *


INPUT_NAMES = ['cartVel', 'cartPos']


class NodeRandomizer:

    def __init__(self, build_operator_dict, build_variables_list):
        """make all probabilities sum to 1 for each categoray (Add: 2, Mul: 1, Tan: 0.5) in"""

        self.pick_op, self.pick_op_match = operatorpool_to_picks(build_operator_dict)
        # sfeh: Acos: 0.33, Asin: 0.33, Atan: 0.33, Tanh: 0.5, Usub: 1, Xor: 1
        # Round: 0.5, Eq: 1,  # Ne: 0.5, #  # Log1p: 0.1, Gt: 0.1, Ge: 0.1,, Tan: 0.1, Sub: 1, Cos: 0.33
        # Powrounded: 0.5

        self.pick_symbol = {
            float: norm_choices([[sympy.Symbol(ii, real=True, imaginary=False), 1] for ii in build_variables_list]),
            bool: []}  # NotImplementedError

        # -> Choosing 50 random numeric values from the dataset for building trees ...just not zeros)
        # samples = [ii for ii in itertools.chain.from_iterable(df[build_variables_list].sample(n=50).values) if ii != 0]
        self.pick_constant = {float: norm_choices([
            [lambda: round(random.normalvariate(1, 1), FLOAT_PRECISION), 0.1],
            [lambda: round(random.randint(1, 20), FLOAT_PRECISION), 0.1],
            # [lambda: round(random.choice(samples), FLOAT_PRECISION), 0.5]
        ]),
            bool: norm_choices([[lambda: random.choice((True, False)), 1]])}

    def choose_operator(self, xt):
        # sfehxxxx allow_chain
        op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
        return op

    def choose_operator_match(self, xtype):
        # sfehxxx allow_chain
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
        _v = np.random.choice(self.pick_constant[xt][0], p=self.pick_constant[xt][1])()  # just dist. must be ()
        if xt == float:
            _v = sympy.Float(_v)  # sfeh:discuss allow "rational" inputs? 1/3, 3/4, ...
            # _v = sympy.Rational(_v)  # sfeh:discuss allow "rational" inputs? 1/3, 3/4, ...
            return Node(Number, [_v])  # round FLOAT_PRECISION was here
        else:
            # _v = sympy.logic.boolalg.BooleanAtom(_v)  # sfeh:discuss: vs. Boolean
            # -> sympy.sympify('And(True, BooleanAtom(False))')
            _v = _v  # BooleanAtom was here - why? Any purpose?
            return Node(Boolean, [_v])

    def choose_symbol(self, xt):
        _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
        return _v


def kernel_for_mtc():

    # ## Load the training data into Kernel-class(...only offline training in this run).
    df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv')
    df = df.astype('float32')  # sfeh: this will NOT work with bool or int data :P design pattern #YOLO
    data_train, data_control = train_test_split(df, test_size=0.2, random_state=0)
    origin_xtype = float
    outcome = sympy.Symbol('outcome')
    tree_base = Clip(Round(outcome), 0, 2)  # sfeh:open

    # tf_sanitize_results = lambda res: tf.round(tf.clip_by_value(res, 0, 2))
    # tf_error_metric = lambda pw_diffs: tf.sqrt(tf.reduce_mean(tf.square(pw_diffs)))
    # todo
    sanitize_results = lambda res: np.round(np.clip(res, 0, 2))
    error_metric = lambda pw_diffs: np.sqrt(np.mean(np.square(pw_diffs)))
    # tf.reduce_mean(tf.abs(pairwise_diff))  # sfeh:open
    kernel = Regression(data_train, 'action', error_metric, sanitize_results)

    return df, kernel


def selection_tournament(pop, tournsize=3):
    """
    SFEH's tournament selection
    sfeh: discuss extracting & deepcopying the inner tree
        -> Use complete candidate for saving the last evolution?
    """
    tree_list = [np.random.choice(pop) for _ in range(tournsize)]
    fintree: 'Candidate' = min(tree_list, key=lambda tree: tree.get_fitness())
    evotree = fintree.get_evotree()
    evotree = copy.deepcopy(evotree)
    return evotree


def _test_simple():
    """SIMPLE"""
    df, kernel = kernel_for_mtc()

    build_operator_dict = {Add: 2, Mul: 2, Div: 1, Square: 0.75, Sqrt: 0.1, Log: 0.1, Abs: 0.5, Sign: 0.5,
                     Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}
    node_selector = NodeRandomizer(build_operator_dict, INPUT_NAMES)
    tb = Evolution(None, None, node_selector, {'depth_max': 7, 'nodes_max': 50}, 'tree_node_count')
    gp = ExplainableGP('TEST', 100, 10, Path.cwd() / f'MTC200_RMSE_scratch', kernel, tb)

    gp.gen_create_initial()
    for _ in range(1):
        @gp.create_trees(rate=1)
        def rand2():
            return gp.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)
        gp.end_generation()

    for _ in range(2):
        @gp.create_trees(rate=1)
        def rand2_CHAIN():
            tree = gp.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)
            tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
            tree.repair_depth(depth=0)  # sfeh repairing depth should be part of any evolution
            return tree
        gp.end_generation()

    for _ in range(10):
        @gp.create_trees(rate=1)
        def mx_branch_n1():
            tree = selection_tournament(gp.pop_genepool, tournsize=3)
            n = np.clip(int(random.normalvariate(12, 4)), 0, 20)
            tree = gp.tb.evolve_mutate_branch_nodes(tree, n, p_term=0.2)
            return tree
        gp.end_generation()

    for _ in range(10):
        @gp.create_trees(rate=1, crossover=True)
        def xover_CHAINA():
            tree_a = selection_tournament(gp.pop_genepool, tournsize=3)
            tree_b = selection_tournament(gp.pop_genepool, tournsize=3)
            evo1, evo2 = gp.tb.evolve_crossover(tree_a, tree_b)
            evo1 = tree_simplification(evo1, allow_chain=CHAINED_VERION)
            evo2 = tree_simplification(evo2, allow_chain=CHAINED_VERION)
            return evo1, evo2
        gp.end_generation()
    gp.evoloop_monitoring_plots()

    print('***Program ending***\n'
          '********************\n\n')
    sys.exit()


def _test_random_pop():
    """Testrun"""
    name = 'MTC200_RMSE_scratch'
    rootdir = Path.cwd() / f'{name}'

    df, kernel = kernel_for_mtc()
    # sfeh:idea track total trees in lut and matches, maybe even check diversity?

    # ## Run/Computation restrictions
    pop_max = 100
    gen_max = 100
    nodes_max = 50
    depth_max = 10

    # class Fixed(Node):
    #     def __init__(self, *args):
    #         super(Fixed, self).__init__(*args, is_fix=True)
    #
    # class N(Node):
    #     def __init__(self, *args):
    #         super(N, self).__init__(*args, is_fix=True)

    build_operator_dict = {Add: 2, Mul: 2, Div: 1, Square: 0.75, Abs: 0.5, Sign: 0.5, Sqrt: 0.1, Log: 0.1,
                     Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}
    ops_arity = {Ifte: 2}  # operators that contradict with fixed arity
    build_operator_dict.update(ops_arity)
    node_selector = NodeRandomizer(build_operator_dict, INPUT_NAMES)

    build_restrictions = {'depth_max': 7, 'nodes_max': 50}

    # ### A simple tree and a simple tree with fixed nodes
    # '["Ifte",["<",["cartVel"],["0"]],["0"],["2"]]'
    # '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]'
    # origin_tree = Node(Ifte, [Node(Le, [(Symbol('cartVel'), Number(0))], Number(0), Number(2)])
    # origin_tree = Node(Add, [Node(Number, [sympy.Float(1)]), Node(Symbol, [sympy.Symbol('cartVel')])], is_fix=True)
    # origin_tree = '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]'

    # origin_tree = Node(Ifte,
    #                    [Node(Lt,
    #                          [Node(Symbol, [sympy.Symbol('cartVel')]),
    #                           Node(Number, [0])]),
    #                     Node(Number, [0], is_fix=True),
    #                     Node(Number, [2], is_fix=True)], is_fix=True)
    # origin_tree.repair_depth()  # sfeh:discuss
    # print(origin_tree)
    # print(origin_tree.get_sympy_expr())

    origin_tree = None

    # tb = TreeBuildRestrictions(origin_xtype, None, nc, build_restrictions, 'tree_node_count')
    tb = Evolution(None, origin_tree, node_selector, build_restrictions, 'tree_node_count')

    gp = ExplainableGP(name, pop_max, gen_max, rootdir, kernel, tb)
    try:
        # gp.backup_load()
        printpl('i', 'Ignore loading backup!')
    except FileNotFoundError as ex:
        printpl('i', f'No backup file found at {ex}. Starting a new run.')

    if gp.gen_id == 0:
        gp.gen_create_initial()  # sfeh check if last pop is empty? +info/warnung: neue generation?

    gp.time_genstart = time.perf_counter()  # sfeh here?

    while gp.gen_id <= gp.gen_max and not gp.run_custom_exit_condition():

        @gp.create_trees(rate=0.1)
        def repro1():
            tree = selection_tournament(gp.pop_genepool, tournsize=3)
            return tree

        if CHAINED_VERION:

            @gp.create_trees(rate=0.30)
            def mx_branch_d_CHAIN():
                tree = selection_tournament(gp.pop_genepool, tournsize=3)
                tree = gp.tb.evolve_mutate_branch_depth(tree, 4, p_term=0.5)
                tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                return tree

            @gp.create_trees(rate=0.1)
            def rand2_CHAIN():
                # sfeh float? nope
                # sfeh:discuss: deep random trees have a tendency to also allow weird-ass looking nonsense
                tree = gp.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)
                tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                return tree

            @gp.create_trees(rate=0.3, crossover=True)
            def xover_CHAIN():
                tree_a = selection_tournament(gp.pop_genepool, tournsize=3)
                tree_b = selection_tournament(gp.pop_genepool, tournsize=3)
                evo1, evo2 = gp.tb.evolve_crossover(tree_a, tree_b)
                evo1 = tree_simplification(evo1, allow_chain=CHAINED_VERION)
                evo2 = tree_simplification(evo2, allow_chain=CHAINED_VERION)
                return evo1, evo2

        else:
            @gp.create_trees(rate=0.05)
            def re_sym_all():
                tree = selection_tournament(gp.pop_genepool, tournsize=3)
                return evolve_reduce_simplify(tree, completely=True)

            @gp.create_trees(rate=0.10)
            def mx_branch_d():
                tree = selection_tournament(gp.pop_genepool, tournsize=3)
                return gp.tb.evolve_mutate_branch_depth(tree, 4, p_term=0.5)

            @gp.create_trees(rate=0.15)
            def mx_branch_n2():
                tree = selection_tournament(gp.pop_genepool, tournsize=3)
                n = np.clip(int(random.normalvariate(12, 4)), 0, 20)
                tree = gp.tb.evolve_mutate_branch_nodes(tree, n, p_term=0.2)
                return tree

            @gp.create_trees(rate=0.1)  # was error source?
            def mut_br():
                tree = selection_tournament(gp.pop_genepool, tournsize=3)
                tree = gp.tb.evolve_mutate_branch_nodes(tree, 4, p_term=0)
                return tree

            @gp.create_trees(rate=0.1)
            def filter_optimize():
                tree = selection_tournament(gp.pop_genepool, tournsize=3)
                return gp.tb.evolve_mutate_filter(tree)

            @gp.create_trees(rate=0.1)
            def rand2():
                # sfeh float? nope
                # sfeh:discuss: deep random trees have a tendency to also allow weird-ass looking nonsense
                return gp.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)

            @gp.create_trees(rate=0.3, crossover=True)
            def xover():
                tree_a = selection_tournament(gp.pop_genepool, tournsize=3)
                tree_b = selection_tournament(gp.pop_genepool, tournsize=3)
                evo1, evo2 = gp.tb.evolve_crossover(tree_a, tree_b)
                return evo1, evo2

            # @gp.create_trees(rate=0.15)
            # def rand1():
            #     return gp.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3, 1)), 2, 4), float, p_term=0)

            # @self.create_trees(rate=0.01)
            # def pareto_revive():
            #     fintree = np.random.choice(self.paretofront)
            #     return fintree.tree

            # @gp.create_trees(rate=0.1)
            # def re_sym():
            #     tree = selection_tournament(gp.population, tournsize=3)
            #     return evolve_reduce_simplify(tree, completely=False)
            #
            # @gp.create_trees(rate=0.1)
            # def mut_pt():
            #     tree = selection_tournament(gp.population, tournsize=3)
            #     return gp.tb.evolve_mutate_point(tree)
            #
            # @gp.create_trees(rate=0.1)
            # def mxPointXXX():
            #     evotree = selection_tournament(gp.pop_base, tournsize=3)
            #     return gp.tb.evolve_mutate_pointxxx(evotree)

        # tmp_pareto = pareto_from_pop(gp.pop_next)  # sfeh:idea paretofront in each generation?

        gp.end_generation()

    printpl('g', f'Done after Generation {gp.gen_id}.\nTime since start: {time.perf_counter() - gp.time_start:4.2f}s')

    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print('***Program ending***\n'
          '********************\n\n')
    sys.exit()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    _test_simple()
    # _test_random_pop()

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
