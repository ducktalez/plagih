"""
This starts the whole genetic programming.
This (extra) file was added to have a file in the root directory that can be started.
"""
from sklearn.model_selection import train_test_split
from plagih.trees import *
from plagih.util import *
import pandas as pd
import sympy

"""Mountain Car dataset experiment setup"""
col_names = ['cartVel', 'cartPos']
df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
df_train, df_control = train_test_split(df, test_size=0.2, random_state=0)
DATA_SYMBOLS = sympy.symbols(df[col_names].columns, real=True)  # sfeh input_names dirty here
operator_dict = Evolution.operator_presets['math_simple']
normalize_numpy = lambda x: np.round(np.clip(x, 0, 2), 0)

rootdir = Path.cwd() / '.testruns'


def _test_simple(dir_name, chained_on=True):
    """SIMPLE"""

    evolve = Evolution(symbol_list=sympy.symbols(['cartVel', 'cartPos']), operators=operator_dict, allow_chain=chained_on)
    gp = ExplainableGP(evolve, df_train, rootdir=rootdir / dir_name, pop_max_size=50, gen_end=20, normalize_numpy=normalize_numpy, allow_chain=chained_on)

    gp.gen_create_initial()

    for _ in range(1):
        @gp.create_trees(rate=1)
        def rand2():
            d = np.clip(int(random.normalvariate(3.5, 1)), 3, 5)
            return gp.evolve.evolve_new_tree_depth(float, d, p_term=0)

        gp.end_generation()

    for _ in range(2):
        @gp.create_trees(rate=1)
        def rand2_CHAINA():  # noqa
            d = np.clip(int(random.normalvariate(4.5, 1)), 3, gp.evolve.depth_max)
            tree = gp.evolve.evolve_new_tree_depth(float, d, p_term=0)
            tree = tree_simplification(tree, allow_chain=chained_on)
            tree.repair_depth(depth=0)  # sfeh repairing depth should be part of any evolution
            return tree

        gp.end_generation()

    for _ in range(10):
        @gp.create_trees(rate=1)
        def mx_branch_n1():
            tree = selection_tournament(gp.pop_genepool, n=3)
            n = np.clip(int(random.normalvariate(16, 4)), 0, gp.evolve.nodes_max)
            tree = gp.evolve.evolve_mutate_branch_nodes(tree, n, p_term=0.2)
            return tree

        gp.end_generation()

    for _ in range(10):
        @gp.create_trees(rate=1, crossover=True)
        def xover_CHAINA():
            tree_a = selection_tournament(gp.pop_genepool, n=3)
            tree_b = selection_tournament(gp.pop_genepool, n=3)
            tree_a = tree_simplification(tree_a, allow_chain=chained_on)
            tree_b = tree_simplification(tree_b, allow_chain=chained_on)
            evo1, evo2 = gp.evolve.evolve_crossover(tree_a, tree_b)
            return evo1, evo2

        gp.end_generation()
    gp.evoloop_monitoring_plots()

    print('***Program ending***\n'
          '********************\n\n')
    # sys.exit()


def _test_random_pop(dir_name, chained_on=True, simplicate=False):
    """Testrun"""

    operator_dict.update({Ifte: 2, PowRounded: 1, Round: 1})

    evolve = Evolution(symbol_list=['cartVel', 'cartPos'], operators=operator_dict, depth_max=9, nodes_max=50, allow_chain=chained_on)
    gp = ExplainableGP(evolve, df_train, rootdir=rootdir / dir_name, pop_max_size=50, gen_end=20, normalize_numpy=normalize_numpy, allow_chain=chained_on)
    try:
        # gp.backup_load()
        printpl('i', 'Ignore loading backup!')
    except FileNotFoundError as ex:
        printpl('i', f'No backup file found at {ex}. Starting a new run.')

    if gp.gen_id == 0:
        gp.gen_create_initial()  # sfeh check if last pop is empty? +info/warnung: neue generation?

    gp.time_genstart = time.perf_counter()  # sfeh here?

    while gp.gen_id <= gp.gen_end and not gp.run_custom_exit_condition():

        @gp.create_trees(rate=0.1)
        def repro1():
            tree = selection_tournament(gp.pop_genepool, n=3)
            return tree

        if chained_on:

            @gp.create_trees(rate=0.30, simplicate=simplicate)
            def mx_branch_d_CHAIN():
                tree = selection_tournament(gp.pop_genepool, n=3)
                tree = gp.evolve.evolve_mutate_branch_depth(tree, 4, chained_on, p_term=0.5)
                return tree

            @gp.create_trees(rate=0.1, simplicate=simplicate)
            def rand2_CHAINB():
                tree = gp.evolve.evolve_new_tree_depth(float, np.clip(int(random.normalvariate(3.5, 1)), 3, 5),
                                                       p_term=0)
                # tree = tree_simplification(tree, allow_chain=gp.allow_chain)
                return tree

            @gp.create_trees(rate=0.3, crossover=True)
            def xover_CHAIN(simplicate=simplicate):
                tree_a = selection_tournament(gp.pop_genepool, n=3)
                tree_b = selection_tournament(gp.pop_genepool, n=3)
                evo1, evo2 = gp.evolve.evolve_crossover(tree_a, tree_b)
                # evo1 = tree_simplification(evo1, allow_chain=gp.allow_chain)
                # evo2 = tree_simplification(evo2, allow_chain=gp.allow_chain)
                return evo1, evo2

        else:
            @gp.create_trees(rate=0.05)
            def re_sym_all():
                tree = selection_tournament(gp.pop_genepool, n=3)
                return evolve_reduce_simplicate(tree, gp.allow_chain, completely=True)

            @gp.create_trees(rate=0.10)
            def mx_branch_d():
                tree = selection_tournament(gp.pop_genepool, n=3)
                return gp.evolve.evolve_mutate_branch_depth(tree, 4, chained_on, p_term=0.5)

            @gp.create_trees(rate=0.15)
            def mx_branch_n2():
                tree = selection_tournament(gp.pop_genepool, n=3)
                # n = np.clip(int(random.normalvariate(12, 4)), 0, 20)
                n = np.random.choice([1, 5, 10, 15, 17, 19, 20, 30, 40, 50, 60])
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
                return gp.evolve.evolve_mutate_filter(tree, allow_chain=chained_on)

            @gp.create_trees(rate=0.1)
            def rand2b():
                tree = gp.evolve.evolve_new_tree_depth(float, np.random.choice([3, 4, 4, 5]), p_term=0)
                return tree

            @gp.create_trees(rate=0.3, crossover=True)
            def xover():
                tree_a = selection_tournament(gp.pop_genepool, n=3)
                tree_b = selection_tournament(gp.pop_genepool, n=3)
                evo1, evo2 = gp.evolve.evolve_crossover(tree_a, tree_b)
                return evo1, evo2

            @gp.create_trees(rate=0.01)
            def pareto_revive():
                fintree = np.random.choice(gp.paretofront)
                return fintree.tree

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
    # _test_simple(dir_name='simple-MTC200_RMSE_scratch', chained_on=False)
    # _test_simple(dir_name='simple-MTC200_RMSE_scratch_chained', chained_on=True)
    _test_random_pop(dir_name='MTC200_RMSE_scratch_chained', chained_on=True)
    _test_random_pop(dir_name='MTC200_RMSE_scratch', chained_on=False)
