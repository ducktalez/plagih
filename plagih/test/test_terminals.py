"""
Unit tests for Terminal nodes: Number, Symbol, Boolean.

Tests verify:
1. Correct construction and value retrieval
2. Proper evaluation on DataFrames
3. SymPy expression generation
4. String representations
"""
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import numpy as np
import sympy
import pandas as pd

from plagih.trees import (
    Terminal, Number, Symbol, Boolean,
    Add, Mul
)


# =============================================================================
# Number Tests
# =============================================================================

class TestNumber:
    """Tests for Number terminal nodes."""

    def test_number_construction(self):
        """Tests Number node construction."""
        n = Number(3.14)
        assert n.get_value() == pytest.approx(3.14, rel=1e-3)

    def test_number_from_int(self):
        """Tests Number construction from integer."""
        n = Number(42)
        assert float(n.get_value()) == pytest.approx(42.0)

    def test_number_from_sympy(self):
        """Tests Number construction from sympy Float."""
        n = Number(sympy.Float(2.5))
        assert float(n.get_value()) == pytest.approx(2.5)

    def test_number_evaluation(self, sample_df):
        """Tests Number evaluation returns constant array."""
        n = Number(5.0)
        result = n.eval_predict_numpy_now(sample_df)

        assert len(result) == len(sample_df)
        np.testing.assert_array_almost_equal(result, np.full(len(sample_df), 5.0))

    def test_number_sympy_expr(self):
        """Tests Number SymPy expression generation."""
        n = Number(3.0)
        expr = n.get_sympy_expr()
        assert float(expr) == pytest.approx(3.0)

    def test_number_string_repr(self):
        """Tests Number string representation."""
        n = Number(2.5)
        s = str(n)
        assert '2.5' in s or '2.50' in s

    def test_number_len(self):
        """Tests Number length is 1."""
        n = Number(1.0)
        assert len(n) == 1

    def test_number_is_terminal(self):
        """Tests Number is recognized as terminal."""
        n = Number(1.0)
        assert n.is_term()
        assert not n.is_operator()

    def test_number_negative(self, sample_df):
        """Tests negative Number evaluation."""
        n = Number(-3.5)
        result = n.eval_predict_numpy_now(sample_df)
        np.testing.assert_array_almost_equal(result, np.full(len(sample_df), -3.5))

    def test_number_zero(self, sample_df):
        """Tests zero Number evaluation."""
        n = Number(0.0)
        result = n.eval_predict_numpy_now(sample_df)
        np.testing.assert_array_almost_equal(result, np.zeros(len(sample_df)))

    def test_number_set_value(self):
        """Tests setting Number value."""
        n = Number(1.0)
        n.set_value(2.0)
        assert float(n.get_value()) == pytest.approx(2.0)


# =============================================================================
# Symbol Tests
# =============================================================================

class TestSymbol:
    """Tests for Symbol terminal nodes."""

    def test_symbol_construction(self):
        """Tests Symbol node construction."""
        s = Symbol(sympy.Symbol('x'))
        assert 'x' in str(s.get_value())

    def test_symbol_from_string_sympy(self):
        """Tests Symbol with sympy Symbol."""
        sym = sympy.Symbol('myvar', real=True)
        s = Symbol(sym)
        assert 'myvar' in str(s.get_value())

    def test_symbol_evaluation(self, sample_df):
        """Tests Symbol evaluation retrieves DataFrame column."""
        s = Symbol(sympy.Symbol('a'))
        result = s.eval_predict_numpy_now(sample_df)

        np.testing.assert_array_almost_equal(result, sample_df['a'].values)

    def test_symbol_evaluation_different_column(self, sample_df):
        """Tests Symbol with different column."""
        s = Symbol(sympy.Symbol('b'))
        result = s.eval_predict_numpy_now(sample_df)

        np.testing.assert_array_almost_equal(result, sample_df['b'].values)

    def test_symbol_sympy_expr(self):
        """Tests Symbol SymPy expression generation."""
        s = Symbol(sympy.Symbol('x'))
        expr = s.get_sympy_expr()
        assert expr.is_Symbol or 'x' in str(expr)

    def test_symbol_string_repr(self):
        """Tests Symbol string representation."""
        s = Symbol(sympy.Symbol('myvar'))
        assert 'myvar' in str(s)

    def test_symbol_len(self):
        """Tests Symbol length is 1."""
        s = Symbol(sympy.Symbol('x'))
        assert len(s) == 1

    def test_symbol_is_terminal(self):
        """Tests Symbol is recognized as terminal."""
        s = Symbol(sympy.Symbol('x'))
        assert s.is_term()
        assert s.is_term_and_symbol()

    def test_symbol_missing_column_raises(self, sample_df):
        """Tests Symbol with missing column raises error."""
        s = Symbol(sympy.Symbol('nonexistent'))
        with pytest.raises(KeyError):
            s.eval_predict_numpy_now(sample_df)


# =============================================================================
# Boolean Tests
# =============================================================================

class TestBoolean:
    """Tests for Boolean terminal nodes."""

    def test_boolean_true(self):
        """Tests Boolean True construction."""
        b = Boolean(True)
        assert b.get_value() == True

    def test_boolean_false(self):
        """Tests Boolean False construction."""
        b = Boolean(False)
        assert b.get_value() == False

    def test_boolean_from_sympy_true(self):
        """Tests Boolean from sympy.true."""
        b = Boolean(sympy.true)
        assert bool(b.get_value()) == True

    def test_boolean_from_sympy_false(self):
        """Tests Boolean from sympy.false."""
        b = Boolean(sympy.false)
        assert bool(b.get_value()) == False

    def test_boolean_evaluation_true(self, sample_df):
        """Tests Boolean True evaluation returns True array."""
        b = Boolean(True)
        result = b.eval_predict_numpy_now(sample_df)

        assert len(result) == len(sample_df)
        assert all(result)

    def test_boolean_evaluation_false(self, sample_df):
        """Tests Boolean False evaluation returns False array."""
        b = Boolean(False)
        result = b.eval_predict_numpy_now(sample_df)

        assert len(result) == len(sample_df)
        assert not any(result)

    def test_boolean_sympy_expr_true(self):
        """Tests Boolean True SymPy expression."""
        b = Boolean(True)
        expr = b.get_sympy_expr()
        # SymPy true is truthy
        assert expr == sympy.true or bool(expr) == True

    def test_boolean_sympy_expr_false(self):
        """Tests Boolean False SymPy expression."""
        b = Boolean(False)
        expr = b.get_sympy_expr()
        # Check it's falsy or equals sympy.false
        assert expr == sympy.false or (not expr) == True or str(expr) == 'False'

    def test_boolean_string_repr(self):
        """Tests Boolean string representation."""
        b_true = Boolean(True)
        b_false = Boolean(False)

        assert 'True' in str(b_true) or 'true' in str(b_true).lower()
        assert 'False' in str(b_false) or 'false' in str(b_false).lower()

    def test_boolean_len(self):
        """Tests Boolean length is 1."""
        b = Boolean(True)
        assert len(b) == 1

    def test_boolean_is_terminal(self):
        """Tests Boolean is recognized as terminal."""
        b = Boolean(True)
        assert b.is_term()
        assert not b.is_operator()


# =============================================================================
# Terminal Integration Tests
# =============================================================================

class TestTerminalIntegration:
    """Integration tests for terminals in trees."""

    def test_number_in_expression(self, sample_df):
        """Tests Number used in expression."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(10.0))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values + 10.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_symbol_in_expression(self, sample_df):
        """Tests Symbol used in expression."""
        tree = Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b')))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values * sample_df['b'].values
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_terminals(self, sample_df):
        """Tests expression with multiple terminal types."""
        # Add(Mul(a, 2), b)
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )
        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values * 2.0 + sample_df['b'].values
        np.testing.assert_array_almost_equal(result, expected)

    def test_xtype_number(self):
        """Tests Number xtype is correct."""
        n = Number(1.0)
        assert n.xtype == ((), float)

    def test_xtype_symbol(self):
        """Tests Symbol xtype is correct."""
        s = Symbol(sympy.Symbol('x'))
        assert s.xtype == ((), float)

    def test_xtype_boolean(self):
        """Tests Boolean xtype is correct."""
        b = Boolean(True)
        assert b.xtype == ((), bool)


# =============================================================================
# Export/Import Tests
# =============================================================================

class TestTerminalExport:
    """Tests for terminal serialization."""

    def test_number_export_tree(self):
        """Tests Number export_tree format."""
        n = Number(2.5)
        export = n.export_tree()
        assert 'Number' in export
        assert '2.5' in export

    def test_symbol_export_tree(self):
        """Tests Symbol export_tree format."""
        s = Symbol(sympy.Symbol('myvar'))
        export = s.export_tree()
        assert 'Symbol' in export
        assert 'myvar' in export

    def test_boolean_export_tree(self):
        """Tests Boolean export_tree format."""
        b = Boolean(True)
        export = b.export_tree()
        assert 'Boolean' in export
        assert 'True' in export
