"""
This starts the whole genetic programming.
This (extra) file was added to have a file in the root directory that can be started.
"""

from plagih.fitness_kernel import *
from plagih.plagih_gp_base_class_xai import *
from plagih.util import *

INPUT_NAMES = ['cartVel', 'cartPos']
df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv')
df = df.astype('float32')
df_train, df_control = train_test_split(df, test_size=0.2, random_state=0)


def _test_simple():
    """SIMPLE"""
    build_operator_dict = {Add: 2, Mul: 2, Div: 1, Square: 0.75, Sqrt: 0.1, Log: 0.1, Abs: 0.5, Sign: 0.5,
                           Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}
    node_selector = NodeRandomizer(build_operator_dict, INPUT_NAMES)
    evolve = Evolution(None, None, node_selector, {'depth_max': 7, 'nodes_max': 50}, 'tree_node_count')
    gp = ExplainableGP('TEST', 100, 10, Path.cwd() / f'MTC200_RMSE_scratch', df_train, df_control, evolve)

    gp.gen_create_initial()
    for _ in range(1):
        @gp.create_trees(rate=1)
        def rand2():
            return gp.evolve.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)

        gp.end_generation()

    for _ in range(2):
        @gp.create_trees(rate=1)
        def rand2_CHAINA():
            tree = gp.evolve.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)
            tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
            tree.repair_depth(depth=0)  # sfeh repairing depth should be part of any evolution
            return tree

        gp.end_generation()

    for _ in range(10):
        @gp.create_trees(rate=1)
        def mx_branch_n1():
            tree = selection_tournament(gp.pop_genepool, n=3)
            n = np.clip(int(random.normalvariate(12, 4)), 0, 20)
            tree = gp.evolve.evolve_mutate_branch_nodes(tree, n, p_term=0.2)
            return tree

        gp.end_generation()

    for _ in range(10):
        @gp.create_trees(rate=1, crossover=True)
        def xover_CHAINA():
            tree_a = selection_tournament(gp.pop_genepool, n=3)
            tree_b = selection_tournament(gp.pop_genepool, n=3)
            evo1, evo2 = gp.evolve.evolve_crossover(tree_a, tree_b)
            evo1 = tree_simplification(evo1, allow_chain=CHAINED_VERION)
            evo2 = tree_simplification(evo2, allow_chain=CHAINED_VERION)
            return evo1, evo2

        gp.end_generation()
    gp.evoloop_monitoring_plots()

    print('***Program ending***\n'
          '********************\n\n')
    # sys.exit()


def _test_random_pop():
    """Testrun"""
    rootdir = Path.cwd() / 'MTC200_RMSE_scratch'
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
    evolve = Evolution(None, origin_tree, node_selector, build_restrictions, 'tree_node_count')
    gp = ExplainableGP('MTC200_RMSE_scratch', pop_max, gen_max, rootdir, df_train, df_control, evolve)
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
            tree = selection_tournament(gp.pop_genepool, n=3)
            return tree

        if CHAINED_VERION:

            @gp.create_trees(rate=0.30)
            def mx_branch_d_CHAIN():
                tree = selection_tournament(gp.pop_genepool, n=3)
                tree = gp.evolve.evolve_mutate_branch_depth(tree, 4, p_term=0.5)
                tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                return tree

            @gp.create_trees(rate=0.1)
            def rand2_CHAINB():
                # sfeh:discuss: deep random trees have a tendency to also allow weird-ass looking nonsense
                tree = gp.evolve.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)
                tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                return tree

            @gp.create_trees(rate=0.3, crossover=True)
            def xover_CHAIN():
                tree_a = selection_tournament(gp.pop_genepool, n=3)
                tree_b = selection_tournament(gp.pop_genepool, n=3)
                evo1, evo2 = gp.evolve.evolve_crossover(tree_a, tree_b)
                evo1 = tree_simplification(evo1, allow_chain=CHAINED_VERION)
                evo2 = tree_simplification(evo2, allow_chain=CHAINED_VERION)
                return evo1, evo2

        else:
            @gp.create_trees(rate=0.05)
            def re_sym_all():
                tree = selection_tournament(gp.pop_genepool, n=3)
                return evolve_reduce_simplify(tree, completely=True)

            @gp.create_trees(rate=0.10)
            def mx_branch_d():
                tree = selection_tournament(gp.pop_genepool, n=3)
                return gp.evolve.evolve_mutate_branch_depth(tree, 4, p_term=0.5)

            @gp.create_trees(rate=0.15)
            def mx_branch_n2():
                tree = selection_tournament(gp.pop_genepool, n=3)
                n = np.clip(int(random.normalvariate(12, 4)), 0, 20)
                tree = gp.evolve.evolve_mutate_branch_nodes(tree, n, p_term=0.2)
                return tree

            @gp.create_trees(rate=0.1)  # was error source?
            def mut_br():
                tree = selection_tournament(gp.pop_genepool, n=3)
                tree = gp.evolve.evolve_mutate_branch_nodes(tree, 4, p_term=0)
                return tree

            @gp.create_trees(rate=0.1)
            def filter_optimize():
                tree = selection_tournament(gp.pop_genepool, n=3)
                return gp.evolve.evolve_mutate_filter(tree)

            @gp.create_trees(rate=0.1)
            def rand2():
                # sfeh float? nope
                # sfeh:discuss: deep random trees have a tendency to also allow weird-ass looking nonsense
                return gp.evolve.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 3, 5), float, p_term=0)

            @gp.create_trees(rate=0.3, crossover=True)
            def xover():
                tree_a = selection_tournament(gp.pop_genepool, n=3)
                tree_b = selection_tournament(gp.pop_genepool, n=3)
                evo1, evo2 = gp.evolve.evolve_crossover(tree_a, tree_b)
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

    printpl('g', f'Done after Generation {gp.gen_id}.\n'
                 f'Time since start: {time.perf_counter() - gp.time_start:4.2f}s')

    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print('***Program ending***\n'
          '********************\n\n')
    # sys.exit()


if __name__ == "__main__":
    _test_simple()
    _test_random_pop()
