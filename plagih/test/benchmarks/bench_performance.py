"""
Performance benchmarks for plagih GP framework.

Benchmarks verify:
1. Tree creation meets time limits
2. Evaluation is efficient
3. Operations scale reasonably

Run directly:
    python plagih/test/benchmarks/bench_performance.py
"""

import copy
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path (benchmarks/ → test/ → plagih/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import sympy

from plagih.trees import (
    Abs,
    Add,
    And,
    Cos,
    Div,
    Evolution,
    ExplainableGP,
    Le,
    Lt,
    Max,
    Min,
    Mul,
    Not,
    Number,
    Or,
    Sign,
    Sin,
    Sqrt,
    Square,
    Sub,
    Symbol,
    selection_tournament,
    sympy_to_tree,
)

# =============================================================================
# Shared helpers — replace pytest fixtures
# =============================================================================


def _make_float_symbols():
    return [sympy.Symbol("a", real=True), sympy.Symbol("b", real=True)]


def _make_evolution():
    from plagih.trees import Abs, Add, And, Cos, Div, Le, Lt, Max, Min, Mul, Not, Or, Sign, Sin, Sqrt, Square, Sub

    ops = {
        Add: 2,
        Mul: 2,
        Div: 1,
        Sub: 1,
        Abs: 1,
        Sign: 1,
        Square: 1,
        Sqrt: 0.5,
        Sin: 0.5,
        Cos: 0.5,
        Min: 1,
        Max: 1,
        Lt: 1,
        Le: 1,
        And: 1,
        Or: 1,
        Not: 1,
    }
    return Evolution(symbol_list=_make_float_symbols(), operators=ops, depth_max=5, nodes_max=30)


def _make_sample_df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [-1.0, 0.5, 2.0, -0.5, 1.0],
            "c": [True, False, True, False, True],
            "d": [False, True, True, False, False],
            "action": [0.0, 1.0, 2.0, 1.0, 0.0],
        }
    )


def _make_large_df():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.choice([True, False], n),
            "d": np.random.choice([True, False], n),
            "action": np.random.choice([0, 1, 2], n).astype(float),
        }
    )


def _make_gp_instance():
    """Creates a configured ExplainableGP. Returns (gp, temp_dir)."""
    ops = {
        Add: 2,
        Mul: 2,
        Div: 1,
        Sub: 1,
        Abs: 1,
        Sign: 1,
        Square: 1,
        Sqrt: 0.5,
        Sin: 0.5,
        Cos: 0.5,
        Min: 1,
        Max: 1,
        Lt: 1,
        Le: 1,
        And: 1,
        Or: 1,
        Not: 1,
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_bench_"))
    cartpole_df = pd.DataFrame(
        {
            "cartPos": [0.0, 0.5, 1.0, -0.5, -1.0, 0.1, 0.2, -0.3, 0.8, -0.9],
            "cartVel": [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.5, -0.4, 0.1, -0.2],
            "action": [1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        }
    )
    eval_autocast = lambda x: np.clip(np.asarray(x, dtype=np.float64), 0.0, 2.0)
    eval_error_metric = lambda pred, true: np.sqrt(np.mean((pred - true) ** 2))

    evolve = Evolution(symbol_list=["cartPos", "cartVel"], operators=ops, depth_max=5, nodes_max=30)
    gp = ExplainableGP(
        evolve=evolve,
        df_train=cartpole_df,
        rootdir=temp_dir,
        pop_max_size=10,
        gen_end=3,
        eval_autocast=eval_autocast,
        eval_error_metric=eval_error_metric,
        allow_chain=False,
        enable_analysis=False,
    )
    return gp, temp_dir


# =============================================================================
# Tree Creation Performance
# =============================================================================


def bench_create_100_trees():
    """Creating 100 random trees should complete in under 2 seconds."""
    evo = _make_evolution()
    start = time.perf_counter()
    for _ in range(100):
        evo.evolve_create_random(xt_out=float, depth_max_local=5, depth=0)
    elapsed = time.perf_counter() - start
    print(f"  Create 100 trees:           {elapsed:.3f}s {'✓' if elapsed < 2.0 else '✗ SLOW'}")


def bench_create_deep_trees():
    """Creating deep trees should complete in reasonable time."""
    evo = _make_evolution()
    evo.depth_max = 8
    start = time.perf_counter()
    for _ in range(10):
        evo.evolve_new_tree_depth(xt_out=float, depth_goal=7, p_term=0.1)
    elapsed = time.perf_counter() - start
    print(f"  Create 10 deep trees:       {elapsed:.3f}s {'✓' if elapsed < 2.0 else '✗ SLOW'}")


# =============================================================================
# Evaluation Performance
# =============================================================================


def bench_evaluate_1000_rows():
    """Evaluation on 1000 rows should complete quickly."""
    large_df = _make_large_df()
    tree = Sin(Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Number(2.0)))
    start = time.perf_counter()
    for _ in range(10):
        tree.eval_predict_numpy_now(large_df)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / 10
    print(f"  Eval 1000 rows (avg):       {avg_time * 1000:.1f}ms {'✓' if avg_time < 0.1 else '✗ SLOW'}")


def bench_evaluate_complex_tree():
    """Complex tree evaluation should be reasonable."""
    large_df = _make_large_df()
    tree = Add(
        Sin(Mul(Symbol(sympy.Symbol("a")), Number(2.0))),
        Cos(Add(Symbol(sympy.Symbol("b")), Number(1.0))),
    )
    tree = Add(tree, Mul(tree, Number(0.5)))
    start = time.perf_counter()
    tree.eval_predict_numpy_now(large_df)
    elapsed = time.perf_counter() - start
    print(f"  Complex tree eval:          {elapsed * 1000:.1f}ms {'✓' if elapsed < 0.5 else '✗ SLOW'}")


def bench_lambda_evaluation():
    """Lambda evaluation should be comparable to direct evaluation."""
    large_df = _make_large_df()
    tree = Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Number(5.0))

    start = time.perf_counter()
    for _ in range(10):
        result_direct = tree.eval_predict_numpy_now(large_df)
    time_direct = time.perf_counter() - start

    lambda_fn = tree.eval_np_lambdas()
    start = time.perf_counter()
    for _ in range(10):
        result_lambda = lambda_fn(large_df)
    time_lambda = time.perf_counter() - start

    np.testing.assert_array_almost_equal(result_direct, result_lambda)
    ratio = time_lambda / time_direct if time_direct > 0 else float("inf")
    print(f"  Lambda vs direct:           {ratio:.1f}x {'✓' if ratio < 5 else '✗ TOO SLOW'}")


# =============================================================================
# Sympy Conversion Performance
# =============================================================================


def bench_sympy_conversion():
    """Sympy expression generation should be fast."""
    tree = Sin(Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Cos(Symbol(sympy.Symbol("a")))))
    start = time.perf_counter()
    for _ in range(100):
        tree.get_sympy_expr()
    elapsed = time.perf_counter() - start
    print(f"  100 sympy conversions:      {elapsed:.3f}s {'✓' if elapsed < 1.0 else '✗ SLOW'}")


def bench_sympy_to_tree():
    """sympy_to_tree conversion should be fast."""
    a, b = sympy.symbols("a b", real=True)
    expr = sympy.sin(a + b * 2) + sympy.cos(a)
    start = time.perf_counter()
    for _ in range(100):
        sympy_to_tree(expr, allow_chain=False)
    elapsed = time.perf_counter() - start
    print(f"  100 tree conversions:       {elapsed:.3f}s {'✓' if elapsed < 1.0 else '✗ SLOW'}")


# =============================================================================
# Mutation Performance
# =============================================================================


def bench_point_mutation():
    """Point mutation should be fast."""
    evo = _make_evolution()
    tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
    tree.repair_depth()
    start = time.perf_counter()
    for _ in range(100):
        evo.evolve_mutate_point(tree)
    elapsed = time.perf_counter() - start
    print(f"  100 point mutations:        {elapsed:.3f}s {'✓' if elapsed < 1.0 else '✗ SLOW'}")


def bench_branch_mutation():
    """Branch mutation should complete in reasonable time."""
    evo = _make_evolution()
    tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
    tree.repair_depth()
    start = time.perf_counter()
    for _ in range(50):
        evo.evolve_mutate_branch_depth(tree, depth_goal=3)
    elapsed = time.perf_counter() - start
    print(f"  50 branch mutations:        {elapsed:.3f}s {'✓' if elapsed < 2.0 else '✗ SLOW'}")


# =============================================================================
# Crossover Performance
# =============================================================================


def bench_crossover():
    """Crossover should be fast."""
    evo = _make_evolution()
    tree1 = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
    tree2 = Sin(Add(Symbol(sympy.Symbol("a")), Number(3.0)))
    tree1.repair_depth()
    tree2.repair_depth()
    start = time.perf_counter()
    for _ in range(50):
        t1_copy = copy.deepcopy(tree1)
        t2_copy = copy.deepcopy(tree2)
        evo.evolve_crossover(t1_copy, t2_copy)
    elapsed = time.perf_counter() - start
    print(f"  50 crossovers:              {elapsed:.3f}s {'✓' if elapsed < 2.0 else '✗ SLOW'}")


# =============================================================================
# GP Run Performance
# =============================================================================


def bench_initial_population():
    """Initial population creation should be fast."""
    gp, temp_dir = _make_gp_instance()
    try:
        start = time.perf_counter()
        gp.gen_create_initial()
        elapsed = time.perf_counter() - start
        print(f"  Initial population:         {elapsed:.3f}s {'✓' if elapsed < 10.0 else '✗ SLOW'}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bench_generation():
    """A generation should complete in reasonable time."""
    gp, temp_dir = _make_gp_instance()
    try:
        gp.gen_create_initial()

        start = time.perf_counter()

        @gp.create_trees(rate=0.5)
        def reproduce():
            return selection_tournament(gp.pop_genepool, n=2)

        @gp.create_trees(rate=0.5)
        def mutate():
            tree = selection_tournament(gp.pop_genepool, n=2)
            return gp.evolve.evolve_mutate_branch_depth(tree, 3)

        gp.end_generation()
        elapsed = time.perf_counter() - start
        print(f"  Full generation:            {elapsed:.3f}s {'✓' if elapsed < 30.0 else '✗ SLOW'}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Memory Tests (Basic)
# =============================================================================


def bench_tree_size():
    """Tree objects should not be excessively large."""
    tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
    size = sys.getsizeof(tree)
    print(f"  Simple tree size:           {size} bytes {'✓' if size < 1024 else '✗ TOO BIG'}")


def bench_many_trees():
    """Creating many trees should not cause issues."""
    evo = _make_evolution()
    trees = []
    for _ in range(500):
        tree = evo.evolve_create_random(xt_out=float, depth_max_local=4, depth=0)
        trees.append(tree)
    print(f"  Created {len(trees)} trees:          ✓")
    del trees


# =============================================================================
# Scalability Tests
# =============================================================================


def bench_evaluation_scales_linearly():
    """Evaluation time should scale roughly linearly with data size."""
    sample_df = _make_sample_df()
    tree = Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Number(5.0))

    small_df = sample_df.head(5)
    start = time.perf_counter()
    for _ in range(100):
        tree.eval_predict_numpy_now(small_df)
    time_small = time.perf_counter() - start

    large_df = pd.concat([sample_df] * 10, ignore_index=True)
    start = time.perf_counter()
    for _ in range(100):
        tree.eval_predict_numpy_now(large_df)
    time_large = time.perf_counter() - start

    ratio = time_large / time_small if time_small > 0 else float("inf")
    print(f"  Eval scaling (10x data):    {ratio:.1f}x {'✓' if ratio < 20 else '✗ BAD SCALING'}")


def bench_node_count_scales():
    """Operations should scale with tree size."""
    evo = _make_evolution()
    small_tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
    small_tree.repair_depth()
    large_tree = evo.evolve_create_random(xt_out=float, depth_max_local=5, depth=0)
    large_tree.repair_depth()

    small_nodes = small_tree.list_mutable_nodes()
    large_nodes = large_tree.list_mutable_nodes()
    print(f"  Node count (small/large):   {len(small_nodes)}/{len(large_nodes)} ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("plagih GP — Performance Benchmarks")
    print("=" * 60)

    print("\n--- Tree Creation ---")
    bench_create_100_trees()
    bench_create_deep_trees()

    print("\n--- Evaluation ---")
    bench_evaluate_1000_rows()
    bench_evaluate_complex_tree()
    bench_lambda_evaluation()

    print("\n--- SymPy Conversion ---")
    bench_sympy_conversion()
    bench_sympy_to_tree()

    print("\n--- Mutation ---")
    bench_point_mutation()
    bench_branch_mutation()

    print("\n--- Crossover ---")
    bench_crossover()

    print("\n--- GP Run ---")
    bench_initial_population()
    bench_generation()

    print("\n--- Memory ---")
    bench_tree_size()
    bench_many_trees()

    print("\n--- Scalability ---")
    bench_evaluation_scales_linearly()
    bench_node_count_scales()

    print("\n" + "=" * 60)
    print("All benchmarks completed.")
