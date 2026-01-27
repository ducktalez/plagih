"""
Plagih GP - Explainable Genetic Programming Framework

This file demonstrates how to use the plagih GP framework.
Contains:
1. demo_minimal() - A quick, well-documented example (~30s runtime)
2. _test_simple() - Basic test run with various evolution strategies
3. _test_random_pop() - Complete test run with all features

Run this file to see the GP in action:
    python plagih_gp.py
"""

from sklearn.model_selection import train_test_split
from plagih.trees import *
from plagih.util import *
import pandas as pd
import sympy
import logging


# =============================================================================
# MINIMAL DEMO - Start here to understand the framework
# =============================================================================

def demo_minimal():
    """
    Minimal demonstration of the plagih GP framework.

    This demo shows the core concepts in ~30 seconds:
    1. Setting up the evolution (operators, symbols, constraints)
    2. Creating the GP system with fitness evaluation
    3. Running evolution: create, mutate, crossover
    4. Inspecting the Pareto front (trade-off between fitness and complexity)

    The goal: Find a symbolic expression that predicts 'action' from 'cartPos' and 'cartVel'.
    """
    # Setup logging (optional, but recommended)
    setup_logging(
        log_file=Path('./logs/demo_minimal.log'),
        console_level=logging.INFO,
        verbose=False
    )

    log_info("Starting minimal demo")

    print("\n" + "="*60)
    print("PLAGIH GP - Minimal Demo")
    print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # STEP 1: Load data
    # -------------------------------------------------------------------------
    # Data has input features (cartPos, cartVel) and target (action)
    df = pd.read_csv(Path(__file__).parent.absolute() / 'benchmarks/mc/gp_files/samples200.csv')
    df = df.astype('float32')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Training data: {len(df_train)} samples")
    print(f"Features: {list(df_train.columns[:-1])}")
    print(f"Target: action (values: {df_train['action'].unique()})\n")

    log_debug(f"Training data shape: {df_train.shape}, Test data shape: {df_test.shape}")

    # -------------------------------------------------------------------------
    # STEP 2: Define operators (optional - can also use presets)
    # -------------------------------------------------------------------------
    # Operators: mathematical functions available to build trees
    operator_dict = {
        Add: 2,      # Addition (weight 2 = more likely to be selected)
        Mul: 2,      # Multiplication
        Div: 1,      # Division
        Sub: 1,      # Subtraction
        Abs: 1,      # Absolute value
        Square: 1,   # x^2
        Sin: 0.5,    # Sine (weight 0.5 = less likely)
        Cos: 0.5,    # Cosine
        Min: 1,      # Minimum
        Max: 1,      # Maximum
        # Boolean operators for conditions
        Lt: 1,       # Less than (<)
        Le: 1,       # Less or equal (<=)
        And: 1,      # Logical AND
        Or: 1,       # Logical OR
        Not: 1,      # Logical NOT
        Ifte: 1,     # If-then-else
    }

    print(f"Available operators: {len(operator_dict)}")
    print(f"Input symbols: ['cartPos', 'cartVel']\n")

    # -------------------------------------------------------------------------
    # STEP 3: Create GP system (simplified with factory method)
    # -------------------------------------------------------------------------
    # Create output directory
    output_dir = Path.cwd() / '.testruns' / 'demo_minimal'

    # NEW: Much simpler initialization with ExplainableGP.create()
    gp = ExplainableGP.create(
        symbols=['cartPos', 'cartVel'],  # Input variables from DataFrame
        df_train=df_train,
        rootdir=output_dir,
        operators=operator_dict,         # Custom operators (or use preset='math_simple')
        depth_max=5,
        nodes_max=25,
        pop_max_size=20,
        gen_end=5,
        clip_range=(0.0, 2.0),           # Clip predictions to [0, 2]
        error_metric='rmse',             # Use RMSE (also: 'mse', 'mae')
    )

    print(f"GP initialized. Output: {output_dir}\n")

    # -------------------------------------------------------------------------
    # STEP 5: Create initial population
    # -------------------------------------------------------------------------
    print("Creating initial population...")
    gp.gen_create_initial()
    print(f"  -> {len(gp.pop_genepool)} candidates created")
    print(f"  -> Pareto front: {len(gp.paretofront)} non-dominated solutions\n")

    # -------------------------------------------------------------------------
    # STEP 6: Run evolution for a few generations
    # -------------------------------------------------------------------------
    print("Running evolution...")

    for gen in range(3):
        print(f"\n--- Generation {gp.gen_id} ---")

        # Strategy 1: Reproduction (copy good trees)
        @gp.create_trees(rate=0.2)
        def reproduction():
            """Select a good tree and copy it."""
            return selection_tournament(gp.pop_genepool, n=3)

        # Strategy 2: Mutation (modify existing trees)
        @gp.create_trees(rate=0.4)
        def mutation():
            """Select a tree and mutate a branch."""
            tree = selection_tournament(gp.pop_genepool, n=3)
            return gp.evolve.evolve_mutate_branch_depth(tree, depth_goal=3, p_term=0.3)

        # Strategy 3: New random trees
        @gp.create_trees(rate=0.2)
        def random_new():
            """Create a new random tree."""
            depth = np.random.choice([2, 3, 4])
            return gp.evolve.evolve_new_tree_depth(float, depth, p_term=0.1)

        # Strategy 4: Crossover (combine two trees)
        @gp.create_trees(rate=0.2, crossover=True)
        def crossover():
            """Select two trees and swap branches."""
            tree_a = selection_tournament(gp.pop_genepool, n=3)
            tree_b = selection_tournament(gp.pop_genepool, n=3)
            return gp.evolve.evolve_crossover(tree_a, tree_b)

        # End generation: update Pareto front, prepare for next
        gp.end_generation()

        print(f"  Population: {len(gp.pop_genepool)}, Pareto front: {len(gp.paretofront)}")

    # -------------------------------------------------------------------------
    # STEP 7: Inspect results
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("RESULTS - Pareto Front (Trade-off: Fitness vs Complexity)")
    print("="*60)

    for i, candidate in enumerate(gp.paretofront[:5]):  # Show top 5
        print(f"\n{i+1}. Complexity: {candidate.parsimony:2d} | Fitness: {candidate.fitness:.4f} | Expression: {candidate.tree.get_sympy_expr()}")

    if len(gp.paretofront) > 5:
        print(f"\n... and {len(gp.paretofront) - 5} more solutions")

    # -------------------------------------------------------------------------
    # STEP 8: Save results
    # -------------------------------------------------------------------------
    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print(f"\n" + "="*60)
    print(f"Demo complete! Results saved to: {output_dir}")
    print("="*60 + "\n")

    return gp


# =============================================================================
# EXISTING TEST RUNS (kept for backwards compatibility)
# =============================================================================

def _test_simple(dir_name, chained_on=True):
    """SIMPLE"""

    # Setup logging for this test
    setup_logging(
        log_file=rootdir / dir_name / 'run.log',
        console_level=logging.INFO,
        verbose=False
    )
    log_info(f"Starting test run: {dir_name}")

    evolve = Evolution(symbol_list=sympy.symbols(['cartVel', 'cartPos']), operators=operator_dict, allow_chain=chained_on)
    gp = ExplainableGP(evolve, df_train, rootdir=rootdir / dir_name, pop_max_size=50, gen_end=20, eval_autocast=eval_autocast, allow_chain=chained_on, eval_error_metric=eval_error_metric)

    gp.gen_create_initial()

    for _ in range(1):
        @gp.create_trees(rate=1)
        def rand2():
            d = np.clip(int(random.normalvariate(3.5, 1)), 3, 5)
            return gp.evolve.evolve_new_tree_depth(float, d, p_term=0)

        gp.end_generation()

    for _ in range(1):
        @gp.create_trees(rate=1)
        def rand_huge():
            d = np.clip(int(random.normalvariate(7, 1)), 7, 10)
            return gp.evolve.evolve_new_tree_depth(float, d, p_term=0)
        gp.end_generation()

    for _ in range(2):
        @gp.create_trees(rate=1)
        def rand2_CHAINA():  # noqa
            d = np.clip(int(random.normalvariate(4.5, 1)), 3, gp.evolve.depth_max)
            _tree = gp.evolve.evolve_new_tree_depth(float, d, p_term=0)
            _tree = tree_simplification(_tree, allow_chain=chained_on)
            _tree.repair_depth(depth=0)  # sfeh repairing depth should be part of any evolution
            return _tree

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


def _test_random_pop(dir_name, chained_on=True, simplicate=False, try_load_backup=False):
    """Testrun"""

    # Setup logging for this comprehensive test
    setup_logging(
        log_file=rootdir / dir_name / 'run.log',
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        verbose=False
    )
    log_info(f"Starting comprehensive test run: {dir_name}")
    log_debug(f"Options - chained_on={chained_on}, simplicate={simplicate}, try_load_backup={try_load_backup}")

    operator_dict.update({Ifte: 2, PowRounded: 1, Round: 1})
    operator_dict.update({Sign: 1, And: 1, Or: 1,
                          Xor: 1, Cos: 1, Tan: 0.2,
                          Lt: 1, Le: 1, Eq: 1, Ne: 1,
                          Exp: 1, Acos: 0.1, Asin: 1, Atan: 0.5})

    # operator_presets = {'math_simple':
    #                     {Add: 2, Mul: 2, Div: 1, Square: 0.75, Abs: 0.5, Sign: 0.5, Sqrt: 0.1, Log: 0.1,
    #                      Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}}
    #
    # sym2node = {sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos, sympy.asin: Asin, sympy.atan: Atan,
    #             sympy.tanh: Tanh, sympy.sinh: Sinh, sympy.cosh: Cosh, sympy.Min: Min, sympy.Max: Max, sympy.Add: Add,
    #             sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul, sympy.sqrt: Sqrt,
    #             sympy.exp: Exp, sympy.Xor: Xor, sympy.Not: Not, sympy.Equality: Eq, sympy.Ne: Ne, sympy.And: And,
    #             sympy.Or: Or, sympy.ITE: ITE, sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.Gt: Gt,
    #             sympy.GreaterThan: Ge}

    evolve = Evolution(symbol_list=['cartVel', 'cartPos'], operators=operator_dict, depth_max=9, nodes_max=50)
    gp = ExplainableGP(evolve, df_train, rootdir=rootdir / dir_name, pop_max_size=50, gen_end=20, eval_autocast=eval_autocast, eval_error_metric=eval_error_metric)
    try:
        if try_load_backup:
            gp.backup_load()
        else:
            printpl('i', 'Ignore loading backup!')
    except FileNotFoundError as ex:
        printpl('i', f'No backup file found at {ex}. Starting a new run.')

    if gp.gen_id == 0:
        gp.gen_create_initial()  # sfeh check if last pop is empty? +info/warnung: neue generation?

    gp.time_genstart = time.perf_counter()  # sfeh here?

    while gp.gen_id <= gp.gen_end and not gp.run_custom_exit_condition():

        @gp.create_trees(rate=0.1)
        def repro1():
            _tree = selection_tournament(gp.pop_genepool, n=3)
            return _tree

        if chained_on:

            @gp.create_trees(rate=0.4, simplicate=simplicate)
            def mx_branch_d_CHAIN():
                _tree = selection_tournament(gp.pop_genepool, n=3)
                _tree = gp.evolve.evolve_mutate_branch_depth(_tree, 4, chained_on, p_term=0.5)
                return _tree

            @gp.create_trees(rate=0.3, simplicate=simplicate)
            def rand2_CHAINB():
                tree = gp.evolve.evolve_new_tree_depth(float, np.clip(int(random.normalvariate(3.5, 1)), 3, 5),
                                                       p_term=0)
                # _tree = tree_simplification(_tree, allow_chain=gp.allow_chain)
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
                _tree = selection_tournament(gp.pop_genepool, n=3)
                n = np.random.choice([1, 5, 10, 15, 17, 19, 20, 30, 40, 50, 60])
                _tree = gp.evolve.evolve_mutate_branch_nodes(_tree, n, p_term=0.2)
                return _tree

            @gp.create_trees(rate=0.1)  # was error source?
            def mut_br():
                _tree = selection_tournament(gp.pop_genepool, n=3)
                _tree = gp.evolve.evolve_mutate_branch_nodes(_tree, 4, p_term=0)
                return _tree

            @gp.create_trees(rate=0.1)
            def filter_optimize():
                _tree = selection_tournament(gp.pop_genepool, n=3)
                return gp.evolve.evolve_mutate_filter(_tree)

            @gp.create_trees(rate=0.1)
            def rand2b():
                _tree = gp.evolve.evolve_new_tree_depth(float, np.random.choice([3, 4, 4, 5]), p_term=0)
                return _tree

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
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'demo'  # Default to demo

    if mode == 'demo':
        # Quick demonstration (~30 seconds)
        demo_minimal()

    elif mode == 'test' or mode == 'full':
        # Full test runs (several minutes)
        """All runs: Experiment: Mountain Car dataset setup"""
        col_names = ['cartVel', 'cartPos']
        df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
        DATA_SYMBOLS = sympy.symbols(df[col_names].columns, real=True)
        operator_dict = Evolution.operator_presets['math_simple']

        eval_autocast = lambda x: np.rint(
            np.clip(np.asarray(x, dtype=np.float64), 0.0, 2.0)
        ).astype(np.int64)

        df_train, df_control = train_test_split(df, test_size=0.2, random_state=0)
        eval_error_metric = lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2))

        rootdir = Path.cwd() / '.testruns'

        if mode == 'full':
            _test_simple(dir_name='simple-MTC200_RMSE_scratch', chained_on=False)
            _test_random_pop(dir_name='MTC200_RMSE_scratch', chained_on=False)
            _test_random_pop(dir_name='MTC200_RMSE_scratch_chained', chained_on=True)

    else:
        print(f"""
            Plagih GP - Explainable Genetic Programming
            
            Usage: python plagih_gp.py [mode]
            
            Modes:
              demo   - Quick demonstration (~30 seconds) [default]
              test   - Basic test run with _test_simple
              full   - Complete test runs with all features
            
            Examples:
              python plagih_gp.py demo
              python plagih_gp.py test
              python plagih_gp.py full
            """)
