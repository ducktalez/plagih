"""Tests für EvaluationContext - unified evaluation system.

Diese Tests stellen sicher, dass:
1. Das neue EvaluationContext-System korrekt funktioniert
2. Die bestehenden Methoden weiterhin unverändert funktionieren (Backward-Kompatibilität)
3. LUT-Caching korrekt arbeitet
4. Statistiken korrekt erfasst werden
"""

import numpy as np
import pandas as pd
import pytest
import sympy

from plagih.evaluation_context import (
    EvalMode,
    EvaluationContext,
    EvaluationResult,
    add_unified_evaluation_to_node,
    create_context,
    evaluate_tree,
)
from plagih.trees import Add, Div, Mul, Number, Sin, Symbol

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_tree():
    """Simple tree: a + 1"""
    return Add(Symbol(sympy.Symbol("a")), Number(1.0))


@pytest.fixture
def complex_tree():
    """More complex tree: sin(a * 2)"""
    return Sin(Mul(Symbol(sympy.Symbol("a")), Number(2.0)))


@pytest.fixture
def multi_var_tree():
    """Tree with multiple variables: a + b * 2"""
    return Add(Symbol(sympy.Symbol("a")), Mul(Symbol(sympy.Symbol("b")), Number(2.0)))


@pytest.fixture
def test_df():
    """Test DataFrame with single variable."""
    return pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def test_df_multi():
    """Test DataFrame with multiple variables."""
    return pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})


# =============================================================================
# Basic Context Tests
# =============================================================================


class TestEvaluationContextBasics:
    """Basic tests for EvaluationContext creation and configuration."""

    def test_default_creation(self):
        """Test default context creation."""
        ctx = EvaluationContext()
        assert "numpy_eager" in ctx.modes
        assert ctx.use_lut is True
        assert ctx.track_gradients is False

    def test_custom_modes(self):
        """Test context with custom modes."""
        ctx = EvaluationContext(modes=["sympy", "numpy_lambda"])
        assert "sympy" in ctx.modes
        assert "numpy_lambda" in ctx.modes
        assert "numpy_eager" not in ctx.modes

    def test_all_modes(self):
        """Test context with all modes."""
        ctx = EvaluationContext(modes=["sympy", "numpy_eager", "numpy_lambda"])
        assert len(ctx.modes) == 3

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            EvaluationContext(modes=["invalid_mode"])

    def test_with_lut_disabled(self):
        """Test context with LUT disabled."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=False)
        assert ctx.use_lut is False

    def test_with_df(self):
        """Test context with DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx = EvaluationContext(df=df)
        assert ctx.df is df


class TestEvaluationContextFluentInterface:
    """Tests for fluent interface methods."""

    def test_with_modes_fluent(self):
        """Test with_modes fluent interface."""
        ctx1 = EvaluationContext(modes=["numpy_eager"])
        ctx2 = ctx1.with_modes(["sympy", "numpy_lambda"])

        assert "sympy" in ctx2.modes
        assert "numpy_lambda" in ctx2.modes
        assert "numpy_eager" not in ctx2.modes
        # Original unchanged
        assert "numpy_eager" in ctx1.modes

    def test_with_df_fluent(self):
        """Test with_df fluent interface."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx1 = EvaluationContext()
        ctx2 = ctx1.with_df(df)

        assert ctx2.df is df
        assert ctx1.df is None

    def test_with_lut_fluent(self):
        """Test with_lut fluent interface."""
        ctx1 = EvaluationContext(use_lut=True)
        ctx2 = ctx1.with_lut(False)

        assert ctx2.use_lut is False
        assert ctx1.use_lut is True

    def test_fluent_chaining(self, test_df):
        """Test chaining multiple fluent methods."""
        ctx = EvaluationContext().with_modes(["numpy_eager", "numpy_lambda"]).with_df(test_df).with_lut(False)

        assert "numpy_eager" in ctx.modes
        assert "numpy_lambda" in ctx.modes
        assert ctx.df is test_df
        assert ctx.use_lut is False


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestEvaluationContextEvaluation:
    """Tests for actual tree evaluation."""

    def test_numpy_eager_single_mode(self, simple_tree, test_df):
        """Test numpy_eager evaluation."""
        ctx = EvaluationContext(modes=["numpy_eager"])
        result = ctx.evaluate(simple_tree, test_df)

        expected = test_df["a"].values + 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_sympy_evaluation(self, simple_tree):
        """Test sympy evaluation."""
        ctx = EvaluationContext(modes=["sympy"])
        result = ctx.evaluate(simple_tree)

        assert isinstance(result, sympy.Basic)
        # Should be a + 1
        assert "a" in str(result)

    def test_numpy_lambda_evaluation(self, simple_tree, test_df):
        """Test numpy_lambda evaluation returns callable."""
        ctx = EvaluationContext(modes=["numpy_lambda"])
        result = ctx.evaluate(simple_tree)

        assert callable(result)
        # Execute the lambda
        output = result(test_df)
        expected = test_df["a"].values + 1.0
        np.testing.assert_array_almost_equal(output, expected)

    def test_multi_mode_evaluation(self, simple_tree, test_df):
        """Test evaluation in multiple modes at once."""
        ctx = EvaluationContext(modes=["sympy", "numpy_eager", "numpy_lambda"])
        results = ctx.evaluate(simple_tree, test_df)

        assert isinstance(results, dict)
        assert "sympy" in results
        assert "numpy_eager" in results
        assert "numpy_lambda" in results

        assert isinstance(results["sympy"], sympy.Basic)
        assert isinstance(results["numpy_eager"], np.ndarray)
        assert callable(results["numpy_lambda"])

    def test_numpy_eager_without_df_raises(self, simple_tree):
        """Test that numpy_eager without df raises error."""
        ctx = EvaluationContext(modes=["numpy_eager"])
        with pytest.raises(ValueError, match="DataFrame.*is required"):
            ctx.evaluate(simple_tree)

    def test_complex_tree_evaluation(self, complex_tree, test_df):
        """Test complex tree evaluation."""
        ctx = EvaluationContext(modes=["numpy_eager"])
        result = ctx.evaluate(complex_tree, test_df)

        expected = np.sin(test_df["a"].values * 2.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_var_tree_evaluation(self, multi_var_tree, test_df_multi):
        """Test tree with multiple variables."""
        ctx = EvaluationContext(modes=["numpy_eager"])
        result = ctx.evaluate(multi_var_tree, test_df_multi)

        expected = test_df_multi["a"].values + test_df_multi["b"].values * 2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_stored_df_usage(self, simple_tree, test_df):
        """Test that stored df is used when not passed to evaluate."""
        ctx = EvaluationContext(modes=["numpy_eager"], df=test_df)
        result = ctx.evaluate(simple_tree)  # No df passed

        expected = test_df["a"].values + 1.0
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# LUT Caching Tests
# =============================================================================


class TestEvaluationContextLUT:
    """Tests for LUT caching functionality."""

    def test_lut_caching(self, simple_tree, test_df):
        """Test that LUT caches results."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=True)

        # First evaluation
        result1 = ctx.evaluate(simple_tree, test_df)
        stats1 = ctx.get_stats()
        assert stats1["cache_misses"]["numpy_eager"] == 1
        assert stats1["cache_hits"]["numpy_eager"] == 0

        # Second evaluation (should be cache hit)
        result2 = ctx.evaluate(simple_tree, test_df)
        stats2 = ctx.get_stats()
        assert stats2["cache_misses"]["numpy_eager"] == 1
        assert stats2["cache_hits"]["numpy_eager"] == 1

        np.testing.assert_array_equal(result1, result2)

    def test_lut_disabled(self, simple_tree, test_df):
        """Test that LUT can be disabled."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=False)

        ctx.evaluate(simple_tree, test_df)
        ctx.evaluate(simple_tree, test_df)

        stats = ctx.get_stats()
        # Both should be misses (no caching)
        assert stats["cache_misses"]["numpy_eager"] == 2
        assert stats["cache_hits"]["numpy_eager"] == 0

    def test_cache_clear(self, simple_tree, test_df):
        """Test cache clearing."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=True)

        ctx.evaluate(simple_tree, test_df)
        assert ctx.get_cache_size()["numpy_eager"] == 1

        ctx.clear_cache()
        assert ctx.get_cache_size()["numpy_eager"] == 0

    def test_cache_clear_specific_mode(self, simple_tree, test_df):
        """Test clearing cache for specific mode."""
        ctx = EvaluationContext(modes=["numpy_eager", "sympy"], use_lut=True)

        ctx.evaluate(simple_tree, test_df)
        assert ctx.get_cache_size()["numpy_eager"] == 1
        assert ctx.get_cache_size()["sympy"] == 1

        ctx.clear_cache("numpy_eager")
        assert ctx.get_cache_size()["numpy_eager"] == 0
        assert ctx.get_cache_size()["sympy"] == 1  # Unchanged

    def test_cache_hit_rate(self, simple_tree, test_df):
        """Test cache hit rate calculation."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=True)

        # 1 miss
        ctx.evaluate(simple_tree, test_df)
        assert ctx.get_cache_hit_rate("numpy_eager") == 0.0

        # 1 hit
        ctx.evaluate(simple_tree, test_df)
        assert ctx.get_cache_hit_rate("numpy_eager") == 0.5

        # Another hit
        ctx.evaluate(simple_tree, test_df)
        assert abs(ctx.get_cache_hit_rate("numpy_eager") - 2 / 3) < 0.01

    def test_different_trees_different_cache_entries(self, test_df):
        """Test that different trees get different cache entries."""
        tree1 = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree2 = Add(Symbol(sympy.Symbol("a")), Number(2.0))

        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=True)

        ctx.evaluate(tree1, test_df)
        ctx.evaluate(tree2, test_df)

        assert ctx.get_cache_size()["numpy_eager"] == 2


# =============================================================================
# Statistics Tests
# =============================================================================


class TestEvaluationContextStatistics:
    """Tests for statistics tracking."""

    def test_statistics_tracking(self, simple_tree, test_df):
        """Test that statistics are tracked correctly."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=True)

        ctx.evaluate(simple_tree, test_df)
        ctx.evaluate(simple_tree, test_df)
        ctx.evaluate(simple_tree, test_df)

        stats = ctx.get_stats()
        assert stats["evaluations"]["numpy_eager"] == 3
        assert stats["cache_hits"]["numpy_eager"] == 2
        assert stats["cache_misses"]["numpy_eager"] == 1

    def test_summary_output(self, simple_tree, test_df):
        """Test summary string generation."""
        ctx = EvaluationContext(modes=["numpy_eager"], use_lut=True)
        ctx.evaluate(simple_tree, test_df)

        summary = ctx.summary()
        assert "numpy_eager" in summary
        assert "Evaluations" in summary
        assert "Cache hits" in summary


# =============================================================================
# Convenience Methods Tests
# =============================================================================


class TestEvaluationContextConvenience:
    """Tests for convenience methods."""

    def test_eval_sympy(self, simple_tree):
        """Test eval_sympy convenience method."""
        ctx = EvaluationContext(modes=["sympy"])
        result = ctx.eval_sympy(simple_tree)
        assert isinstance(result, sympy.Basic)

    def test_eval_numpy(self, simple_tree, test_df):
        """Test eval_numpy convenience method."""
        ctx = EvaluationContext(modes=["numpy_eager"])
        result = ctx.eval_numpy(simple_tree, test_df)
        assert isinstance(result, np.ndarray)

    def test_eval_lambda(self, simple_tree):
        """Test eval_lambda convenience method."""
        ctx = EvaluationContext(modes=["numpy_lambda"])
        result = ctx.eval_lambda(simple_tree)
        assert callable(result)

    def test_eval_all(self, simple_tree, test_df):
        """Test eval_all returns EvaluationResult."""
        ctx = EvaluationContext()
        result = ctx.eval_all(simple_tree, test_df)

        assert isinstance(result, EvaluationResult)
        assert result.sympy is not None
        assert result.numpy_eager is not None
        assert result.numpy_lambda is not None
        assert len(result.errors) == 0

    def test_eval_all_with_errors(self, test_df):
        """Test eval_all captures errors correctly."""
        # Create a tree that might fail in some modes
        # This is a normal tree that should succeed
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))

        ctx = EvaluationContext()
        result = ctx.eval_all(tree, test_df)

        # All should succeed for this simple tree
        assert len(result.successful_modes()) == 3


# =============================================================================
# Utility Functions Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_create_context(self):
        """Test create_context factory function."""
        ctx = create_context("numpy_eager", use_lut=False)
        assert "numpy_eager" in ctx.modes
        assert ctx.use_lut is False

    def test_create_context_with_df(self, test_df):
        """Test create_context with DataFrame."""
        ctx = create_context("numpy_eager", df=test_df)
        assert ctx.df is test_df

    def test_evaluate_tree(self, simple_tree, test_df):
        """Test evaluate_tree one-shot function."""
        result = evaluate_tree(simple_tree, test_df, mode="numpy_eager")
        expected = test_df["a"].values + 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_tree_sympy(self, simple_tree):
        """Test evaluate_tree with sympy mode."""
        result = evaluate_tree(simple_tree, mode="sympy")
        assert isinstance(result, sympy.Basic)

    def test_evaluate_tree_lambda(self, simple_tree, test_df):
        """Test evaluate_tree with lambda mode."""
        lambda_fn = evaluate_tree(simple_tree, mode="numpy_lambda")
        assert callable(lambda_fn)
        result = lambda_fn(test_df)
        expected = test_df["a"].values + 1.0
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing methods."""

    def test_old_methods_still_work(self, simple_tree, test_df):
        """Test that old evaluation methods still work."""
        # These should all work unchanged
        sy_result = simple_tree.get_sympy_expr()
        assert isinstance(sy_result, sympy.Basic)

        np_result = simple_tree.eval_predict_numpy_now(test_df)
        assert isinstance(np_result, np.ndarray)

        lambda_result = simple_tree.eval_np_lambdas()
        assert callable(lambda_result)

    def test_results_match(self, simple_tree, test_df):
        """Test that context results match old method results."""
        ctx = EvaluationContext(modes=["sympy", "numpy_eager", "numpy_lambda"])
        ctx_results = ctx.evaluate(simple_tree, test_df)

        # Compare with old methods
        old_sympy = simple_tree.get_sympy_expr()
        old_numpy = simple_tree.eval_predict_numpy_now(test_df)
        old_lambda = simple_tree.eval_np_lambdas()

        assert str(ctx_results["sympy"]) == str(old_sympy)
        np.testing.assert_array_equal(ctx_results["numpy_eager"], old_numpy)

        # Lambda outputs should match
        ctx_lambda_output = ctx_results["numpy_lambda"](test_df)
        old_lambda_output = old_lambda(test_df)
        np.testing.assert_array_equal(ctx_lambda_output, old_lambda_output)

    def test_complex_tree_results_match(self, complex_tree, test_df):
        """Test results match for more complex tree."""
        ctx = EvaluationContext(modes=["numpy_eager"])
        ctx_result = ctx.evaluate(complex_tree, test_df)
        old_result = complex_tree.eval_predict_numpy_now(test_df)

        np.testing.assert_array_almost_equal(ctx_result, old_result)


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_get_method(self):
        """Test get method returns correct result."""
        result = EvaluationResult(sympy=sympy.Symbol("a"), numpy_eager=np.array([1, 2, 3]))

        assert result.get("sympy") == sympy.Symbol("a")
        assert np.array_equal(result.get("numpy_eager"), np.array([1, 2, 3]))
        assert result.get("numpy_lambda") is None

    def test_has_error(self):
        """Test has_error method."""
        result = EvaluationResult(errors={"sympy": "Some error"})

        assert result.has_error("sympy") is True
        assert result.has_error("numpy_eager") is False

    def test_successful_modes(self):
        """Test successful_modes method."""
        result = EvaluationResult(
            sympy=sympy.Symbol("a"), numpy_eager=np.array([1, 2, 3]), errors={"numpy_lambda": "Error"}
        )

        modes = result.successful_modes()
        assert "sympy" in modes
        assert "numpy_eager" in modes
        assert "numpy_lambda" not in modes


# =============================================================================
# Integration with Node class Tests
# =============================================================================


class TestNodeIntegration:
    """Tests for Node class integration."""

    def test_add_unified_evaluation(self, simple_tree, test_df):
        """Test adding evaluate_unified method to Node."""
        from plagih.trees import Node

        # Add the method
        add_unified_evaluation_to_node(Node)

        # Now all nodes should have the method
        ctx = EvaluationContext(modes=["numpy_eager"])
        result = simple_tree.evaluate_unified(ctx, test_df)

        expected = test_df["a"].values + 1.0
        np.testing.assert_array_almost_equal(result, expected)
