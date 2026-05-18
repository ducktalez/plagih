"""
Plagih GP - Explainable Genetic Programming Framework

This file demonstrates how to use the plagih GP framework.
Contains:
1. demo_minimal() - A quick, well-documented example (~30s runtime)
2. demo_active_usability_test() - A single non-standard long observation run for active tests
3. _test_simple() - Basic test run with various evolution strategies
4. _test_random_pop() - Complete test run with all features

Run this file to see the GP in action:
    python plagih_gp.py
    python plagih_gp.py fresh      # Like 'full', but with timestamped output dir
"""

import logging
import time

import numpy as np
import pandas as pd
import sympy
from sklearn.model_selection import train_test_split

from plagih.parallel import Strategy
from plagih.trees import *
from plagih.util import *

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
    setup_logging(log_file=Path("./logs/demo_minimal.log"), console_level=logging.INFO, verbose=False)

    log_info("Starting minimal demo")

    print(TEXT_NEWLINE)
    print("PLAGIH GP - Minimal Demo")
    print(TEXT_NEWLINE)

    # -------------------------------------------------------------------------
    # STEP 1: Load data
    # -------------------------------------------------------------------------
    # Data has input features (cartPos, cartVel) and target (action)
    df = pd.read_csv(Path(__file__).parent.absolute() / "benchmarks/mc/gp_files/samples200.csv")
    df = df.astype("float32")
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
        Add: 2,  # Addition (weight 2 = more likely to be selected)
        Mul: 2,  # Multiplication
        Div: 1,  # Division
        Sub: 1,  # Subtraction
        Abs: 1,  # Absolute value
        Square: 1,  # x^2
        Sin: 0.5,  # Sine (weight 0.5 = less likely)
        Cos: 0.5,  # Cosine
        Min: 1,  # Minimum
        Max: 1,  # Maximum
        # Boolean operators for conditions
        Lt: 1,  # Less than (<)
        Le: 1,  # Less or equal (<=)
        And: 1,  # Logical AND
        Or: 1,  # Logical OR
        Not: 1,  # Logical NOT
        Ifte: 1,  # If-then-else
    }

    print(f"Available operators: {len(operator_dict)}")
    print("Input symbols: ['cartPos', 'cartVel']\n")

    # -------------------------------------------------------------------------
    # STEP 3: Create GP system (simplified with factory method)
    # -------------------------------------------------------------------------
    # Create output directory
    output_dir = Path.cwd() / ".results" / "demo_minimal"

    # NEW: Much simpler initialization with ExplainableGP.create()
    gp = ExplainableGP.create(
        symbols=["cartPos", "cartVel"],  # Input variables from DataFrame
        df_train=df_train,
        rootdir=output_dir,
        operators=operator_dict,  # Custom operators (or use preset='math_simple')
        depth_max=5,
        nodes_max=25,
        pop_max_size=20,
        gen_end=5,
        clip_range=(0.0, 2.0),  # Clip predictions to [0, 2]
        error_metric="rmse",  # Use RMSE (also: 'mse', 'mae')
    )

    print(f"GP initialized. Output: {output_dir}\n")

    # -------------------------------------------------------------------------
    # STEP 5: Create initial population
    # -------------------------------------------------------------------------
    print("Creating initial population...")
    gp.gen_create_initial()
    print(f"  -> {len(gp.pop_genepool)} candidates created, {len(gp.paretofront)} non-dominated solutions")

    # -------------------------------------------------------------------------
    # STEP 6: Run evolution for a few generations
    # -------------------------------------------------------------------------
    print("Running evolution...")

    # Define strategies declaratively — each generation uses the same set
    strategies = [
        Strategy("reproduction", rate=0.2, tournament_n=3),
        Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3),
        Strategy("random_new", rate=0.2, depths=[2, 3, 4], p_term=0.1),
        Strategy("crossover", rate=0.2, crossover=True, tournament_n=3),
    ]

    for gen in range(3):
        print(f"--- Generation {gp.gen_id} ---")
        # run_generation handles: create trees, evaluate, update pareto, end generation
        gp.run_generation(strategies)
        log("ggg", f"  Population: {len(gp.pop_genepool)}, Pareto front: {len(gp.paretofront)}")

    # -------------------------------------------------------------------------
    # STEP 7: Inspect results
    # -------------------------------------------------------------------------
    print(f"RESULTS - Pareto Front (Trade-off: Fitness vs Complexity)\n{TEXT_NEWLINE}")

    for i, candidate in enumerate(gp.paretofront[:5]):  # Show top 5
        print(
            f"{i + 1}. Complexity: {candidate.parsimony:2d} | Fitness: {candidate.fitness:.4f} | Expression: {candidate.tree.get_sympy_expr()}"
        )

    if len(gp.paretofront) > 5:
        print(f"\n... and {len(gp.paretofront) - 5} more solutions")

    # -------------------------------------------------------------------------
    # STEP 8: Save results
    # -------------------------------------------------------------------------
    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print(f"{TEXT_NEWLINE}\nDemo complete! Results saved to: {output_dir}\n{TEXT_NEWLINE}")

    return gp


# =============================================================================
# CARTPOLE DEMO - 4 inputs, binary classification
# =============================================================================


def demo_cartpole():
    """
    CartPole demonstration of the plagih GP framework.

    This demo shows GP on the classic CartPole balancing problem:
    - 4 input variables (position, velocity, angle, angular velocity)
    - Binary output (push left or right)
    - Simple known solutions exist (e.g., poleAngle < 0)

    Runtime: ~30 seconds for 5 generations
    """
    setup_logging(log_file=Path("./logs/demo_cartpole.log"), console_level=logging.INFO, verbose=False)

    log_info("Starting CartPole demo")

    print(TEXT_NEWLINE)
    print("PLAGIH GP - CartPole Demo")
    print(TEXT_NEWLINE)

    # -------------------------------------------------------------------------
    # STEP 1: Load CartPole data
    # -------------------------------------------------------------------------
    df = pd.read_csv(Path(__file__).parent.absolute() / "benchmarks/cp/gp_files/samples.csv")
    df = df.astype("float32")

    # Rename columns for clarity
    df = df.rename(columns={"observation2": "poleAngle", "observation3": "poleVel", "action0": "action"})

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Training data: {len(df_train)} samples")
    print(f"Features: {list(df_train.columns[:-1])}")
    print("Target: action (binary: 0=left, 1=right)\n")

    # -------------------------------------------------------------------------
    # STEP 2: Define operators
    # -------------------------------------------------------------------------
    operator_dict = {
        Add: 2,
        Mul: 2,
        Div: 1,
        Sub: 1,
        Abs: 1,
        # Boolean/comparison for decision making
        Lt: 2,  # Less than - important for CartPole
        Le: 1,
        And: 1,
        Or: 1,
        Ifte: 1,  # If-then-else
    }

    print(f"Available operators: {len(operator_dict)}")
    print("Input symbols: ['cartPos', 'cartVel', 'poleAngle', 'poleVel']\n")

    # -------------------------------------------------------------------------
    # STEP 3: Create GP system
    # -------------------------------------------------------------------------
    output_dir = Path.cwd() / ".results" / "demo_cartpole"

    gp = ExplainableGP.create(
        symbols=["cartPos", "cartVel", "poleAngle", "poleVel"],
        df_train=df_train,
        rootdir=output_dir,
        operators=operator_dict,
        depth_max=4,
        nodes_max=20,
        pop_max_size=20,
        gen_end=5,
        clip_range=(0.0, 1.0),  # Binary classification
        error_metric="rmse",
    )

    print(f"GP initialized. Output: {output_dir}\n")

    # -------------------------------------------------------------------------
    # STEP 4: Run evolution
    # -------------------------------------------------------------------------
    print("Creating initial population...")
    gp.gen_create_initial()
    print(f"  -> {len(gp.pop_genepool)} candidates, {len(gp.paretofront)} non-dominated\n")

    print("Running evolution...")

    for gen in range(3):
        print(f"--- Generation {gp.gen_id} ---")

        @gp.create_trees(rate=0.2)
        def reproduction():
            return selection_tournament(gp.pop_genepool, n=3)

        @gp.create_trees(rate=0.4)
        def mutation():
            tree = selection_tournament(gp.pop_genepool, n=3)
            return gp.evolve.evolve_mutate_branch_depth(tree, depth_goal=2, p_term=0.3)

        @gp.create_trees(rate=0.2)
        def random_new():
            depth = np.random.choice([2, 3])
            return gp.evolve.evolve_new_tree_depth(float, depth, p_term=0.1)

        @gp.create_trees(rate=0.2, crossover=True)
        def crossover():
            tree_a = selection_tournament(gp.pop_genepool, n=3)
            tree_b = selection_tournament(gp.pop_genepool, n=3)
            return gp.evolve.evolve_crossover(tree_a, tree_b)

        gp.end_generation()

    # -------------------------------------------------------------------------
    # STEP 5: Show results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARETO FRONT (best trade-offs between fitness and complexity)")
    print("=" * 60)

    for candidate in sorted(gp.paretofront, key=lambda c: c.parsimony):
        print(
            f"  Fitness: {candidate.fitness:.4f}, Complexity: {candidate.parsimony:2d}, "
            f"Expr: {candidate.get_evotree().get_sympy_expr()}"
        )

    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print(f"\n{TEXT_NEWLINE}\nCartPole demo complete! Results saved to: {output_dir}\n{TEXT_NEWLINE}")

    return gp


# =============================================================================
# SYMBOLIC REGRESSION DEMO - Classic GP benchmark
# =============================================================================


def demo_symbolic_regression():
    """
    Symbolic Regression demonstration - classic GP benchmark.

    Goal: Find the formula f(x) = x³ + x² + x

    This is a standard GP benchmark because:
    - There is an exact solution
    - The solution is relatively simple
    - Success is easy to measure

    Runtime: ~20 seconds for 5 generations
    """
    setup_logging(log_file=Path("./logs/demo_symbolic_regression.log"), console_level=logging.INFO, verbose=False)

    log_info("Starting Symbolic Regression demo")

    print(TEXT_NEWLINE)
    print("PLAGIH GP - Symbolic Regression Demo")
    print("Goal: Find f(x) = x³ + x² + x")
    print(TEXT_NEWLINE)

    # -------------------------------------------------------------------------
    # STEP 1: Load polynomial data
    # -------------------------------------------------------------------------
    df = pd.read_csv(Path(__file__).parent.absolute() / "benchmarks/sr/gp_files/polynomial.csv")
    df = df.astype("float32")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Training data: {len(df_train)} samples")
    print("Input: x (range: -2 to 2)")
    print("Target: x³ + x² + x\n")

    # -------------------------------------------------------------------------
    # STEP 2: Define operators - math focused
    # -------------------------------------------------------------------------
    operator_dict = {
        Add: 3,  # Addition - very important
        Mul: 3,  # Multiplication - very important
        Sub: 2,  # Subtraction
        Div: 1,  # Division
        Square: 2,  # x² - helpful
        Lt: 1,  # Comparison (required for bool type)
        # Note: We don't include Cube, so GP must find x*x*x or x*x²
    }

    print(f"Available operators: {list(operator_dict.keys())}")
    print("Note: No 'Cube' operator - GP must discover x³ = x*x*x\n")

    # -------------------------------------------------------------------------
    # STEP 3: Create GP system
    # -------------------------------------------------------------------------
    output_dir = Path.cwd() / ".results" / "demo_symbolic_regression"

    gp = ExplainableGP.create(
        symbols=["x"],
        df_train=df_train,
        rootdir=output_dir,
        operators=operator_dict,
        depth_max=5,
        nodes_max=15,
        pop_max_size=30,
        gen_end=5,
        error_metric="mse",  # Mean squared error for regression
    )

    print(f"GP initialized. Output: {output_dir}\n")

    # -------------------------------------------------------------------------
    # STEP 4: Run evolution
    # -------------------------------------------------------------------------
    print("Creating initial population...")
    gp.gen_create_initial()
    print(f"  -> {len(gp.pop_genepool)} candidates, {len(gp.paretofront)} non-dominated\n")

    print("Running evolution...")

    strategies = [
        Strategy("reproduction", rate=0.15, tournament_n=3),
        Strategy("mutation", rate=0.45, depth_goal=2, p_term=0.2),
        Strategy("random_new", rate=0.2, depths=[2, 3, 4], p_term=0.1),
        Strategy("crossover", rate=0.2, crossover=True, tournament_n=3),
    ]

    for gen in range(5):
        print(f"--- Generation {gp.gen_id} ---")
        gp.run_generation(strategies)

    # -------------------------------------------------------------------------
    # STEP 5: Show results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARETO FRONT")
    print("Target: x³ + x² + x = x*(x² + x + 1) = x*(x+1)² - x")
    print("=" * 60)

    for candidate in sorted(gp.paretofront, key=lambda c: c.fitness):
        expr = candidate.get_evotree().get_sympy_expr()
        try:
            simplified = sympy.simplify(expr)
        except:
            simplified = expr
        print(f"  MSE: {candidate.fitness:.6f}, Complexity: {candidate.parsimony:2d}, Expr: {simplified}")

    # Check if we found the exact solution
    print("\n" + "-" * 60)
    best = min(gp.paretofront, key=lambda c: c.fitness)
    if best.fitness < 0.001:
        print("✅ Found a very good approximation!")
    else:
        print("🔄 More generations might improve the solution.")

    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print(f"\n{TEXT_NEWLINE}\nSymbolic Regression demo complete! Results saved to: {output_dir}\n{TEXT_NEWLINE}")

    return gp


# =============================================================================
# ACTIVE USABILITY TEST - Observe a single non-standard run over longer timescales
# =============================================================================


def _format_duration_compact(seconds):
    """Formats a duration for compact console summaries."""
    total_seconds = max(0, round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _build_longrun_strategies():
    """Returns a balanced strategy mix for longer MountainCar observation runs."""
    return [
        Strategy("reproduction", rate=0.10, tournament_n=3),
        Strategy("simplicate", rate=0.05, tournament_n=3, completely=True),
        Strategy("mutation", rate=0.10, depth_goal=4, p_term=0.4),
        Strategy("mutation_branch_nodes", rate=0.15, tournament_n=3, nodes_goal=10, p_term=0.2),
        Strategy("mutation_branch_nodes", rate=0.10, tournament_n=3, nodes_goal=4, p_term=0.0),
        Strategy("mutation_filter", rate=0.10, tournament_n=3),
        Strategy("random_new", rate=0.15, depths=[3, 4, 5, 6], p_term=0.0),
        Strategy("crossover", rate=0.20, crossover=True, tournament_n=3),
        Strategy("pareto_revive", rate=0.05),
    ]


def _build_active_test_operator_dict():
    """Return a conservative operator set for long interactive usability runs.

    Goal: prefer stable, interpretable trees and avoid overly pathological
    combinations (especially Exp / Round / PowRounded / Piecewise-heavy trees)
    that are more likely to spend a very long time in native NumPy code.
    """
    return dict(Evolution.operator_presets["math_simple"])


def _build_active_test_strategies():
    """Return a debug-friendlier strategy mix for active usability testing."""
    return [
        Strategy("reproduction", rate=0.10, tournament_n=3),
        Strategy("mutation", rate=0.20, depth_goal=4, p_term=0.4),
        Strategy("mutation_branch_nodes", rate=0.20, tournament_n=3, nodes_goal=10, p_term=0.2),
        Strategy("mutation_branch_nodes", rate=0.10, tournament_n=3, nodes_goal=4, p_term=0.0),
        Strategy("mutation_filter", rate=0.10, tournament_n=3),
        Strategy("random_new", rate=0.20, depths=[3, 4, 5, 6], p_term=0.0),
        Strategy("crossover", rate=0.05, crossover=True, tournament_n=3),
        Strategy("pareto_revive", rate=0.05),
    ]


def demo_active_usability_test(
    pop_max_size=5000,
    gen_end=1000,
    parallel=None,
    run_name=None,
    enable_analysis=False,
):
    """Runs a single non-standard long observation run for active usability testing.

    This mode is intentionally NOT part of the normal demo/test defaults. It is
    meant for temporarily observing long-running behaviour with a much larger GP
    configuration than the regular examples use.
    """
    if pop_max_size <= 0:
        raise ValueError("pop_max_size must be > 0")
    if gen_end <= 0:
        raise ValueError("gen_end must be > 0")

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    if run_name is None:
        run_name = f"NONSTANDARD-observe-MTC200-RMSE-pop{pop_max_size}-gen{gen_end}-{timestamp}"

    output_dir = Path.cwd() / ".results" / run_name
    setup_logging(log_file=output_dir / "run.log", console_level=logging.INFO, file_level=logging.DEBUG, verbose=False)
    log_info(f"Starting non-standard active usability test: {run_name}")

    print(TEXT_NEWLINE)
    print("PLAGIH GP - Non-standard Active Usability Test")
    print(TEXT_NEWLINE)

    df = pd.read_csv(Path(__file__).parent.absolute() / "benchmarks/mc/gp_files/samples200.csv").astype("float32")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

    operator_dict = _build_active_test_operator_dict()
    strategies = _build_active_test_strategies()
    parallel = 0 if parallel is None else parallel

    gp = ExplainableGP.create(
        symbols=["cartVel", "cartPos"],
        df_train=df_train,
        rootdir=output_dir,
        operators=operator_dict,
        depth_max=7,
        nodes_max=35,
        pop_max_size=pop_max_size,
        gen_end=gen_end,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=parallel,
        enable_analysis=enable_analysis,
    )

    print(f"Output directory: {output_dir}")
    print(f"Training samples: {len(df_train)} | Control samples: {len(df_test)}")
    print(f"Non-standard configuration: pop_max_size={pop_max_size}, gen_end={gen_end}")
    print("Purpose: active long-timescale usability testing (not a standard demo/test mode)")
    print(f"Analysis during run: {'enabled' if enable_analysis else 'disabled'}")
    print(f"Parallel workers: {parallel}")
    print(f"Strategy mix: {[strategy.name for strategy in strategies]}")
    print(TEXT_NEWLINE)

    gp.gen_create_initial()

    stop_reason = f"generation limit {gp.gen_end} reached"
    while gp.gen_id <= gp.gen_end:
        if gp.run_custom_exit_condition():
            stop_reason = "custom exit condition triggered"
            break

        observed_generation = gp.gen_id
        gp.run_generation(strategies)
        if gp.paretofront:
            best = min(gp.paretofront, key=lambda c: (c.fitness, c.parsimony))
            best_part = f"best_fit={best.fitness:.4f} | best_parsim={best.parsimony}"
        else:
            best_part = "best_fit=n/a"

        log(
            "gg",
            f"observe generation {observed_generation}/{gp.gen_end}: "
            f"population={len(gp.pop_genepool)}/{gp.pop_max_size} | "
            f"pareto={len(gp.paretofront)} | {best_part}",
        )
    else:
        stop_reason = f"generation limit {gp.gen_end} reached"

    completed_generation = max(0, gp.gen_id - 1)
    elapsed_total = time.perf_counter() - gp.time_start

    print("\nTop Pareto solutions after the active usability test:")
    for index, candidate in enumerate(sorted(gp.paretofront, key=lambda c: (c.fitness, c.parsimony))[:5], start=1):
        print(
            f"{index}. Fitness={candidate.fitness:.4f} | Complexity={candidate.parsimony:2d} | Expr={candidate.get_evotree().get_sympy_expr()}"
        )

    log(
        "g",
        f"non-standard usability test finished: {stop_reason}.\n"
        f"Completed generation: {completed_generation}/{gp.gen_end}.\n"
        f"Elapsed: {_format_duration_compact(elapsed_total)}.\n"
        f"Results saved to: {output_dir}",
    )

    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print(f"\n{TEXT_NEWLINE}\nNon-standard usability test complete! Results saved to: {output_dir}\n{TEXT_NEWLINE}")
    return gp


def demo_longrun(*args, **kwargs):
    """Backward-compatible alias for the current active usability test mode."""
    log_warning("'demo_longrun()' is now a compatibility alias for the non-standard active usability test.")
    return demo_active_usability_test(*args, **kwargs)


# =============================================================================
# EXISTING TEST RUNS (kept for backwards compatibility)
# =============================================================================


def _test_simple(dir_name, chained_on=True, enable_analysis=None):
    """SIMPLE"""

    strategy_plan = [
        (1, [Strategy("random_new", rate=1.0, depths=[3, 4, 5], p_term=0.0)]),
        (1, [Strategy("random_new", rate=1.0, depths=[7, 8, 9, 10], p_term=0.0)]),
        (2, [Strategy("random_new", rate=1.0, simplicate=True, depths=[3, 4, 5, 6], p_term=0.0)]),
        (10, [Strategy("mutation_branch_nodes", rate=1.0, tournament_n=3, nodes_goal=16, p_term=0.2)]),
        (10, [Strategy("crossover", rate=1.0, crossover=True, simplicate=True, tournament_n=3)]),
    ]
    planned_generations = sum(repeat_count for repeat_count, _strategies in strategy_plan)

    # Setup logging for this test
    setup_logging(log_file=rootdir / dir_name / "run.log", console_level=logging.INFO, verbose=False)
    log_info(f"Starting test run: {dir_name}")

    evolve = Evolution(
        symbol_list=sympy.symbols(["cartVel", "cartPos"]), operators=operator_dict, allow_chain=chained_on
    )
    gp = ExplainableGP(
        evolve,
        df_train,
        rootdir=rootdir / dir_name,
        pop_max_size=1000,
        gen_end=planned_generations,
        eval_autocast=eval_autocast,
        allow_chain=chained_on,
        eval_error_metric=eval_error_metric,
        enable_analysis=enable_analysis,
    )

    gp.gen_create_initial()

    for repeat_count, strategies in strategy_plan:
        for _ in range(repeat_count):
            gp.run_generation(strategies)

    gp.evoloop_monitoring_plots()
    gp.backup_save()

    print("***Program ending***\n********************\n\n")
    # sys.exit()


def _test_random_pop(dir_name, chained_on=True, simplicate=False, try_load_backup=False, enable_analysis=None):
    """Testrun"""

    # Setup logging for this comprehensive test
    setup_logging(
        log_file=rootdir / dir_name / "run.log", console_level=logging.INFO, file_level=logging.DEBUG, verbose=False
    )
    log_info(f"Starting comprehensive test run: {dir_name}")
    log_debug(f"Options - chained_on={chained_on}, simplicate={simplicate}, try_load_backup={try_load_backup}")

    operator_dict.update({Ifte: 2, PowRounded: 1, Round: 1})
    operator_dict.update(
        {
            Sign: 1,
            And: 1,
            Or: 1,
            Xor: 1,
            Cos: 1,
            Tan: 0.2,
            Lt: 1,
            Le: 1,
            Eq: 1,
            Ne: 1,
            Exp: 1,
            Acos: 0.1,
            Asin: 1,
            Atan: 0.5,
        }
    )

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

    evolve = Evolution(symbol_list=["cartVel", "cartPos"], operators=operator_dict, depth_max=9, nodes_max=50)
    gp = ExplainableGP(
        evolve,
        df_train,
        rootdir=rootdir / dir_name,
        pop_max_size=50,
        gen_end=20,
        eval_autocast=eval_autocast,
        eval_error_metric=eval_error_metric,
        enable_analysis=enable_analysis,
    )
    try:
        if try_load_backup:
            gp.backup_load()
        else:
            log("i", "Ignore loading backup!")
    except FileNotFoundError as ex:
        log("i", f"No backup file found at {ex}. Starting a new run.")

    if gp.gen_id == 0:
        gp.gen_create_initial()  # sfeh check if last pop is empty? +info/warnung: neue generation?

    gp.time_genstart = time.perf_counter()  # sfeh here?

    # Build strategy list based on configuration
    if chained_on:
        strategies = [
            Strategy("reproduction", rate=0.1, tournament_n=3),
            Strategy("mutation", rate=0.4, simplicate=simplicate, depth_goal=4, p_term=0.5),
            Strategy("random_new", rate=0.3, simplicate=simplicate, depths=[3, 4, 5], p_term=0.0),
            Strategy("crossover", rate=0.3, crossover=True, tournament_n=3),
        ]
    else:
        strategies = [
            Strategy("reproduction", rate=0.1, tournament_n=3),
            Strategy("simplicate", rate=0.05, tournament_n=3, completely=True),
            Strategy("mutation", rate=0.10, depth_goal=4, p_term=0.5),
            Strategy("mutation_branch_nodes", rate=0.15, tournament_n=3, nodes_goal=10, p_term=0.2),
            Strategy("mutation_branch_nodes", rate=0.1, tournament_n=3, nodes_goal=4, p_term=0.0),
            Strategy("mutation_filter", rate=0.1, tournament_n=3),
            Strategy("random_new", rate=0.1, depths=[3, 4, 4, 5], p_term=0.0),
            Strategy("crossover", rate=0.3, crossover=True, tournament_n=3),
            Strategy("pareto_revive", rate=0.01),
        ]

    while gp.gen_id <= gp.gen_end and not gp.run_custom_exit_condition():
        gp.run_generation(strategies)

    completed_generation = max(0, gp.gen_id - 1)
    log(
        "g",
        f"done after generation {completed_generation}/{gp.gen_end}.\n"
        f"Time since start: {time.perf_counter() - gp.time_start:4.2f}s",
    )

    gp.backup_save()
    gp.evoloop_monitoring_plots()

    print("***Program ending***\n********************\n\n")
    # sys.exit()


def _make_timestamped_subdir(run_name: str, base: str = ".results") -> Path:
    """Create a timestamped subdir under ``<base>/<run_name>/``.

    Structure: ``<base>/<run_name>/<YYYYMMDD-HHMMSS>/``

    Each run configuration keeps its own top-level folder, with individual
    launches stored in timestamped subdirectories. This makes it easy to
    compare multiple launches of the same configuration.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out = Path.cwd() / base / run_name / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _setup_test_data():
    """Common Mountain Car dataset setup shared by test / full / fresh modes."""
    col_names = ["cartVel", "cartPos"]
    df = pd.read_csv(Path(__file__).parent.absolute() / "benchmarks/mc/gp_files/samples200.csv").astype("float32")
    DATA_SYMBOLS = sympy.symbols(df[col_names].columns, real=True)
    operator_dict = Evolution.operator_presets["math_simple"]
    eval_autocast = lambda x: np.rint(np.clip(np.asarray(x, dtype=np.float64), 0.0, 2.0)).astype(np.int64)
    df_train, df_control = train_test_split(df, test_size=0.2, random_state=0)
    eval_error_metric = lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2))
    return DATA_SYMBOLS, operator_dict, eval_autocast, df_train, df_control, eval_error_metric


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "demo"  # Default to demo

    if mode == "gui":
        # Launch the PySide6 desktop monitoring GUI
        try:
            from plagih.gui.desktop.app import main as _gui_main
        except ImportError as e:
            print(f"GUI not available: {e}")
            print("Install with: pip install 'plagih[gui]'")
            sys.exit(1)
        _gui_main()

    elif mode == "demo":
        # Quick demonstration (~30 seconds)
        demo_minimal()

    elif mode == "active-test" or mode == "usability-test":
        demo_active_usability_test()

    elif mode == "longrun" or mode == "observe":
        log_warning(
            "CLI mode 'longrun/observe' is deprecated; use 'active-test' for the current non-standard usability run."
        )
        demo_active_usability_test()

    elif mode in ("test", "full", "fresh"):
        # test / full  — static dirs (backwards-compatible, can resume from backup)
        # fresh        — timestamped dir, always starts from scratch
        DATA_SYMBOLS, operator_dict, eval_autocast, df_train, df_control, eval_error_metric = _setup_test_data()

        if mode == "fresh":
            # Each sub-run gets its own timestamped dir: .results/<run_name>/<timestamp>/
            rootdir = Path.cwd() / ".results"  # base only; _test_* receive full paths
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            print(f"Fresh run (timestamp {timestamp})")
        else:
            rootdir = Path.cwd() / ".results"

        if mode == "test":
            _test_simple(dir_name="simple-MTC200_RMSE_scratch", chained_on=False)

        elif mode == "full":
            _test_simple(dir_name="simple-MTC200_RMSE_scratch", chained_on=False)
            _test_random_pop(dir_name="MTC200_RMSE_scratch", chained_on=False)
            _test_random_pop(dir_name="MTC200_RMSE_scratch_chained", chained_on=True)

        elif mode == "fresh":
            _test_simple(dir_name=f"simple-MTC200_RMSE_scratch/{timestamp}", chained_on=False, enable_analysis=True)
            _test_random_pop(dir_name=f"MTC200_RMSE_scratch/{timestamp}", chained_on=False, enable_analysis=True)
            _test_random_pop(dir_name=f"MTC200_RMSE_scratch_chained/{timestamp}", chained_on=True, enable_analysis=True)

    else:
        print("""
            Plagih GP - Explainable Genetic Programming

            Usage: python plagih_gp.py [mode]

            Modes:
              gui    - Launch the PySide6 desktop monitoring GUI
              demo   - Quick demonstration (~30 seconds) [default]
              active-test - Non-standard active usability test (pop=5000, gen=1000)
              usability-test - Alias for active-test
              fresh  - Like 'full', but with timestamped output dir (always from scratch)
              test   - Basic test run with _test_simple (static dir, can resume)
              full   - Complete test runs with all features (static dir, can resume)
              longrun - Deprecated alias for active-test
              observe - Deprecated alias for active-test

            Examples:
              python plagih_gp.py gui
              python plagih_gp.py demo
              python plagih_gp.py fresh
              python plagih_gp.py active-test
              python plagih_gp.py test
              python plagih_gp.py full
            """)
