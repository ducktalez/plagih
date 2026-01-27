"""
Performance tests for plagih GP framework.

Tests verify:
1. Tree creation meets time limits
2. Evaluation is efficient
3. Operations scale reasonably
"""
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import numpy as np
import pandas as pd
import sympy
import time

from plagih.trees import (
    Node, Number, Symbol,
    Add, Mul, Sin, Cos,
    Evolution, ExplainableGP,
    sympy_to_tree, tree_simplification
)


# =============================================================================
# Tree Creation Performance
# =============================================================================

class TestTreeCreationPerformance:
    """Performance tests for tree creation."""

    def test_create_100_trees_under_2_seconds(self, evolution_instance):
        """Tests creating 100 random trees completes in under 2 seconds."""
        start = time.perf_counter()

        for _ in range(100):
            tree = evolution_instance.evolve_create_random(
                xt_out=float,
                depth_max_local=5,
                depth=0
            )

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Creating 100 trees took {elapsed:.2f}s (limit: 2s)"

    def test_create_deep_tree_reasonable_time(self, evolution_instance):
        """Tests creating a deep tree completes in reasonable time."""
        evolution_instance.depth_max = 8

        start = time.perf_counter()

        for _ in range(10):
            tree = evolution_instance.evolve_new_tree_depth(
                xt_out=float,
                depth_goal=7,
                p_term=0.1
            )

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Creating 10 deep trees took {elapsed:.2f}s"


# =============================================================================
# Evaluation Performance
# =============================================================================

class TestEvaluationPerformance:
    """Performance tests for tree evaluation."""

    def test_evaluate_1000_rows_under_100ms(self, large_df):
        """Tests evaluation on 1000 rows completes quickly."""
        tree = Sin(Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Number(2.0)
        ))

        start = time.perf_counter()

        for _ in range(10):
            result = tree.eval_predict_numpy_now(large_df)

        elapsed = time.perf_counter() - start
        avg_time = elapsed / 10

        assert avg_time < 0.1, f"Evaluation avg {avg_time*1000:.1f}ms (limit: 100ms)"

    def test_evaluate_complex_tree_reasonable(self, large_df):
        """Tests complex tree evaluation is reasonable."""
        # Create a more complex tree
        tree = Add(
            Sin(Mul(Symbol(sympy.Symbol('a')), Number(2.0))),
            Cos(Add(Symbol(sympy.Symbol('b')), Number(1.0))),
        )
        tree = Add(tree, Mul(tree, Number(0.5)))  # Make it more complex

        start = time.perf_counter()

        result = tree.eval_predict_numpy_now(large_df)

        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Complex evaluation took {elapsed:.2f}s"

    def test_lambda_evaluation_comparable(self, large_df):
        """Tests lambda evaluation is comparable to direct evaluation."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Number(5.0)
        )

        # Direct evaluation
        start = time.perf_counter()
        for _ in range(10):
            result_direct = tree.eval_predict_numpy_now(large_df)
        time_direct = time.perf_counter() - start

        # Lambda evaluation
        start = time.perf_counter()
        lambda_fn = tree.eval_np_lambdas()
        for _ in range(10):
            result_lambda = lambda_fn(large_df)
        time_lambda = time.perf_counter() - start

        # Results should match
        np.testing.assert_array_almost_equal(result_direct, result_lambda)

        # Lambda should not be drastically slower (within 5x)
        assert time_lambda < time_direct * 5


# =============================================================================
# Sympy Conversion Performance
# =============================================================================

class TestSympyPerformance:
    """Performance tests for SymPy operations."""

    def test_sympy_conversion_reasonable(self):
        """Tests sympy expression generation is fast."""
        tree = Sin(Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Cos(Symbol(sympy.Symbol('a')))
        ))

        start = time.perf_counter()

        for _ in range(100):
            expr = tree.get_sympy_expr()

        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"100 sympy conversions took {elapsed:.2f}s"

    def test_sympy_to_tree_reasonable(self):
        """Tests sympy_to_tree conversion is fast."""
        a, b = sympy.symbols('a b', real=True)
        expr = sympy.sin(a + b * 2) + sympy.cos(a)

        start = time.perf_counter()

        for _ in range(100):
            tree = sympy_to_tree(expr, allow_chain=False)

        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"100 tree conversions took {elapsed:.2f}s"


# =============================================================================
# Mutation Performance
# =============================================================================

class TestMutationPerformance:
    """Performance tests for mutation operations."""

    def test_point_mutation_fast(self, evolution_instance):
        """Tests point mutation is fast."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )
        tree.repair_depth()

        start = time.perf_counter()

        for _ in range(100):
            mutated = evolution_instance.evolve_mutate_point(tree)

        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"100 point mutations took {elapsed:.2f}s"

    def test_branch_mutation_reasonable(self, evolution_instance):
        """Tests branch mutation completes in reasonable time."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )
        tree.repair_depth()

        start = time.perf_counter()

        for _ in range(50):
            mutated = evolution_instance.evolve_mutate_branch_depth(
                tree, depth_goal=3
            )

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"50 branch mutations took {elapsed:.2f}s"


# =============================================================================
# Crossover Performance
# =============================================================================

class TestCrossoverPerformance:
    """Performance tests for crossover operations."""

    def test_crossover_fast(self, evolution_instance):
        """Tests crossover is fast."""
        import copy

        tree1 = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )
        tree2 = Sin(Add(Symbol(sympy.Symbol('a')), Number(3.0)))

        tree1.repair_depth()
        tree2.repair_depth()

        start = time.perf_counter()

        for _ in range(50):
            t1_copy = copy.deepcopy(tree1)
            t2_copy = copy.deepcopy(tree2)
            r1, r2 = evolution_instance.evolve_crossover(t1_copy, t2_copy)

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"50 crossovers took {elapsed:.2f}s"


# =============================================================================
# GP Run Performance
# =============================================================================

class TestGPRunPerformance:
    """Performance tests for complete GP runs."""

    def test_initial_population_under_10s(self, gp_instance):
        """Tests initial population creation is fast."""
        start = time.perf_counter()

        gp_instance.gen_create_initial()

        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"Initial population took {elapsed:.2f}s"

    def test_generation_under_30s(self, gp_instance):
        """Tests a generation completes in reasonable time."""
        from plagih.trees import selection_tournament

        gp_instance.gen_create_initial()

        start = time.perf_counter()

        @gp_instance.create_trees(rate=0.5)
        def reproduce():
            return selection_tournament(gp_instance.pop_genepool, n=2)

        @gp_instance.create_trees(rate=0.5)
        def mutate():
            tree = selection_tournament(gp_instance.pop_genepool, n=2)
            return gp_instance.evolve.evolve_mutate_branch_depth(tree, 3)

        gp_instance.end_generation()

        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, f"Generation took {elapsed:.2f}s"


# =============================================================================
# Memory Tests (Basic)
# =============================================================================

class TestMemoryUsage:
    """Basic memory usage tests."""

    def test_tree_not_excessively_large(self):
        """Tests tree objects are not excessively large."""
        import sys

        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        # Get rough size (not perfect but indicative)
        size = sys.getsizeof(tree)

        # Should be reasonable (under 1KB for simple tree)
        assert size < 1024, f"Simple tree is {size} bytes"

    def test_many_trees_manageable(self, evolution_instance):
        """Tests creating many trees doesn't cause issues."""
        trees = []

        for _ in range(500):
            tree = evolution_instance.evolve_create_random(
                xt_out=float,
                depth_max_local=4,
                depth=0
            )
            trees.append(tree)

        # Should have created all trees
        assert len(trees) == 500

        # Clean up
        del trees


# =============================================================================
# Scalability Tests
# =============================================================================

class TestScalability:
    """Tests for scalability of operations."""

    def test_evaluation_scales_linearly(self, sample_df):
        """Tests evaluation time scales roughly linearly with data size."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Number(5.0)
        )

        # Small dataset
        small_df = sample_df.head(5)
        start = time.perf_counter()
        for _ in range(100):
            tree.eval_predict_numpy_now(small_df)
        time_small = time.perf_counter() - start

        # Larger dataset (10x)
        large_df = pd.concat([sample_df] * 10, ignore_index=True)
        start = time.perf_counter()
        for _ in range(100):
            tree.eval_predict_numpy_now(large_df)
        time_large = time.perf_counter() - start

        # Large should not be more than 20x slower (allowing for overhead)
        assert time_large < time_small * 20

    def test_node_count_scales_tree_operations(self, evolution_instance):
        """Tests operations scale with tree size."""
        # Small tree
        small_tree = Add(Symbol(sympy.Symbol('a')), Number(1.0))
        small_tree.repair_depth()

        # Larger tree
        large_tree = evolution_instance.evolve_create_random(
            xt_out=float,
            depth_max_local=5,
            depth=0
        )
        large_tree.repair_depth()

        # List nodes - should work for both
        small_nodes = small_tree.list_mutable_nodes()
        large_nodes = large_tree.list_mutable_nodes()

        assert len(small_nodes) <= len(large_nodes)
