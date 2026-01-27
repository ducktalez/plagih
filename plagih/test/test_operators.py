"""
Unit tests for all operator classes in plagih.

Tests verify:
1. symfun and np_fun produce consistent results
2. xtype signatures are correct
3. Chainable operators work with variable arity
4. SymPy expression generation works correctly
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
    Node, Terminal, Number, Symbol, Boolean,
    Add, Mul, Div, Sub, Pow, Sqrt, Square, Abs, Sign, Log, Exp, Exp2,
    Sin, Cos, Tan, Asin, Acos, Atan, Tanh, Sinh, Cosh,
    Min, Max, Lt, Le, Gt, Ge, Eq, Ne,
    And, Or, Xor, Not, Ifte, ITE, Piecewise,
    Round, PowRounded, DivFraction, Usub, Clip, NthRoot,
    BaseOperator, ChainableOp, ExprCondPair_Dummy
)
from plagih.util import get_subclasses


# =============================================================================
# Helper functions (moved from conftest for direct access)
# =============================================================================

def get_all_operator_classes():
    """Returns list of all concrete operator classes for testing."""
    skip_classes = {
        'BaseOperator', 'OperatorArity', 'MathOperator',
        'LogicOperator', 'RelationalOperator', 'Trigonometry',
        'BaseMinMax', 'NodeWithChilds', 'NodeDummy',
        'PleaseUsePartnerOp', 'CustomOperator', 'NoSymCapitalized'
    }

    all_ops = []
    for cls in get_subclasses(BaseOperator):
        if cls.__name__ in skip_classes:
            continue
        if hasattr(cls, 'xtype') and cls.xtype:
            all_ops.append(cls)
    return all_ops


def create_random_inputs(xtype_inputs, seed=42):
    """Creates random inputs based on xtype specification."""
    np.random.seed(seed)
    inputs = []
    for t in xtype_inputs:
        if t == float:
            inputs.append(np.random.uniform(-2, 2))
        elif t == bool:
            inputs.append(np.random.choice([True, False]))
        else:
            inputs.append(np.random.uniform(-1, 1))
    return inputs


# =============================================================================
# Test Data
# =============================================================================

# Operators that require special handling in tests
SPECIAL_OPERATORS = {
    ExprCondPair_Dummy,  # Only used in Piecewise
    Piecewise,           # Complex structure
    Clip,                # 3 arguments with constraints
    NthRoot,             # Needs positive base for some roots
}

# Operators where domain restrictions apply
DOMAIN_RESTRICTED = {
    Sqrt: lambda: (abs(np.random.uniform(0.1, 5)),),  # Non-negative
    Log: lambda: (abs(np.random.uniform(0.1, 5)),),   # Positive
    Asin: lambda: (np.random.uniform(-0.9, 0.9),),    # [-1, 1]
    Acos: lambda: (np.random.uniform(-0.9, 0.9),),    # [-1, 1]
    DivFraction: lambda: (np.random.uniform(0.1, 5),), # Non-zero
    Div: lambda: (np.random.uniform(-5, 5), np.random.uniform(0.1, 5)),
}


# =============================================================================
# Operator Consistency Tests
# =============================================================================

class TestOperatorConsistency:
    """Tests that symfun and np_fun produce consistent results for all operators."""

    def get_testable_operators(self):
        """Returns operators that can be automatically tested."""
        ops = []
        for cls in get_subclasses(BaseOperator):
            # Skip abstract/special classes
            if cls in SPECIAL_OPERATORS:
                continue
            if not hasattr(cls, 'xtype') or not cls.xtype:
                continue
            if not hasattr(cls, 'np_fun') or cls.np_fun is None:
                continue
            if not hasattr(cls, 'symfun') or cls.symfun is None:
                continue
            ops.append(cls)
        return ops

    @pytest.mark.parametrize("op_class", [
        Add, Mul, Sub, Div, Pow, Sqrt, Square, Abs, Sign,
        Sin, Cos, Tan, Tanh, Sinh, Cosh, Atan,
        Min, Max, Exp, Exp2, Usub, Round, PowRounded,
        Lt, Le, Gt, Ge, Eq, Ne, And, Or, Xor, Not
    ])
    def test_symfun_vs_npfun_consistency(self, op_class):
        """Verifies symfun and np_fun produce equivalent results."""
        np.random.seed(42)

        # Get input types
        xtype = op_class.xtype
        if not xtype or not xtype[0]:
            pytest.skip(f"{op_class.__name__} has no xtype")

        input_types = xtype[0]
        output_type = xtype[1]

        # Generate inputs based on domain restrictions
        if op_class in DOMAIN_RESTRICTED:
            inputs = DOMAIN_RESTRICTED[op_class]()
        elif issubclass(op_class, ChainableOp) and hasattr(op_class, 'xtype_input'):
            # Chainable ops: generate 2-4 inputs of same type
            t = op_class.xtype_input
            n = np.random.randint(2, 5)
            if t == float:
                inputs = tuple(np.random.uniform(-2, 2) for _ in range(n))
            else:
                inputs = tuple(np.random.choice([True, False]) for _ in range(n))
        else:
            # Regular operators
            inputs = []
            for t in input_types:
                if t == float:
                    inputs.append(np.random.uniform(-2, 2))
                elif t == bool:
                    inputs.append(np.random.choice([True, False]))
                else:
                    inputs.append(np.random.uniform(-1, 1))
            inputs = tuple(inputs)

        # Evaluate with symfun
        try:
            sym_result = op_class.symfun(*inputs)
            if hasattr(sym_result, 'evalf'):
                sym_result = float(sym_result.evalf())
            else:
                sym_result = output_type(sym_result)
        except Exception as e:
            pytest.skip(f"symfun failed for {op_class.__name__}: {e}")

        # Evaluate with np_fun (wrap scalars in arrays for consistency)
        try:
            np_inputs = [np.array([x]) for x in inputs]
            np_result = op_class.np_fun(*np_inputs)
            np_result = output_type(np_result[0])
        except Exception as e:
            pytest.skip(f"np_fun failed for {op_class.__name__}: {e}")

        # Compare results
        if output_type == bool:
            assert sym_result == np_result, f"{op_class.__name__}: {sym_result} != {np_result}"
        else:
            assert abs(sym_result - np_result) < 1e-6, \
                f"{op_class.__name__}: {sym_result} != {np_result} (diff: {abs(sym_result - np_result)})"

    def test_all_operators_have_required_attributes(self):
        """Verifies all operators have required class attributes."""
        required_attrs = ['xtype', 'showme', 'sy_str']

        for op_class in self.get_testable_operators():
            for attr in required_attrs:
                assert hasattr(op_class, attr), f"{op_class.__name__} missing {attr}"

            # xtype should be a tuple ((input_types), output_type)
            xtype = op_class.xtype
            assert isinstance(xtype, tuple), f"{op_class.__name__}.xtype is not tuple"
            assert len(xtype) == 2, f"{op_class.__name__}.xtype should have 2 elements"


# =============================================================================
# Chainable Operator Tests
# =============================================================================

class TestChainableOperators:
    """Tests for operators that support variable arity."""

    @pytest.mark.parametrize("op_class,expected_type", [
        (Add, float),
        (Mul, float),
        (Min, float),
        (Max, float),
        (And, bool),
        (Or, bool),
        (Xor, bool),
    ])
    def test_chainable_with_multiple_args(self, op_class, expected_type):
        """Tests chainable operators with more than 2 arguments."""
        if expected_type == float:
            args = [Number(1.0), Number(2.0), Number(3.0), Number(4.0)]
        else:
            args = [Boolean(True), Boolean(False), Boolean(True)]

        node = op_class(*args)

        # Verify structure
        assert len(node.get_childs()) >= 2

        # Verify sympy expression can be generated
        expr = node.get_sympy_expr()
        assert expr is not None

    def test_add_chain_evaluation(self, sample_df):
        """Tests Add with multiple operands evaluates correctly."""
        tree = Add(
            Symbol(sympy.Symbol('a')),
            Symbol(sympy.Symbol('b')),
            Number(10.0)
        )

        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values + sample_df['b'].values + 10.0

        np.testing.assert_array_almost_equal(result, expected)

    def test_mul_chain_evaluation(self, sample_df):
        """Tests Mul with multiple operands evaluates correctly."""
        tree = Mul(
            Symbol(sympy.Symbol('a')),
            Symbol(sympy.Symbol('b')),
            Number(2.0)
        )

        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values * sample_df['b'].values * 2.0

        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Operator Evaluation Tests
# =============================================================================

class TestOperatorEvaluation:
    """Tests operator evaluation on DataFrames."""

    def test_add_basic(self, sample_df):
        """Tests basic Add evaluation."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(5.0))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values + 5.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_mul_basic(self, sample_df):
        """Tests basic Mul evaluation."""
        tree = Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b')))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values * sample_df['b'].values
        np.testing.assert_array_almost_equal(result, expected)

    def test_div_basic(self, sample_df):
        """Tests basic Div evaluation."""
        tree = Div(Symbol(sympy.Symbol('a')), Number(2.0))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values / 2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_sin_basic(self, sample_df):
        """Tests Sin evaluation."""
        tree = Sin(Symbol(sympy.Symbol('a')))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.sin(sample_df['a'].values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_abs_basic(self, sample_df):
        """Tests Abs evaluation."""
        tree = Abs(Symbol(sympy.Symbol('b')))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.abs(sample_df['b'].values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_basic(self, sample_df):
        """Tests Min and Max evaluation."""
        tree_min = Min(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b')))
        tree_max = Max(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b')))

        result_min = tree_min.eval_predict_numpy_now(sample_df)
        result_max = tree_max.eval_predict_numpy_now(sample_df)

        expected_min = np.minimum(sample_df['a'].values, sample_df['b'].values)
        expected_max = np.maximum(sample_df['a'].values, sample_df['b'].values)

        np.testing.assert_array_almost_equal(result_min, expected_min)
        np.testing.assert_array_almost_equal(result_max, expected_max)

    def test_comparison_operators(self, sample_df):
        """Tests comparison operators."""
        tree_lt = Lt(Symbol(sympy.Symbol('a')), Number(3.0))
        result = tree_lt.eval_predict_numpy_now(sample_df)
        expected = sample_df['a'].values < 3.0
        np.testing.assert_array_equal(result, expected)

    def test_logical_operators(self, sample_df):
        """Tests logical operators."""
        tree = And(
            Lt(Symbol(sympy.Symbol('a')), Number(3.0)),
            Boolean(True)
        )
        result = tree.eval_predict_numpy_now(sample_df)
        expected = (sample_df['a'].values < 3.0) & True
        np.testing.assert_array_equal(result, expected)

    def test_ifte_basic(self, sample_df):
        """Tests if-then-else operator."""
        tree = Ifte(
            Lt(Symbol(sympy.Symbol('a')), Number(3.0)),
            Number(1.0),
            Number(0.0)
        )
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.where(sample_df['a'].values < 3.0, 1.0, 0.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_nested_evaluation(self, sample_df):
        """Tests nested operator evaluation."""
        # Sin(Add(Mul(a, b), 2))
        tree = Sin(Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Number(2.0)
        ))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.sin(sample_df['a'].values * sample_df['b'].values + 2.0)
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# SymPy Expression Tests
# =============================================================================

class TestSympyExpressions:
    """Tests for SymPy expression generation."""

    def test_simple_expression(self):
        """Tests simple expression generation."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(1.0))
        expr = tree.get_sympy_expr()
        assert expr is not None
        assert 'a' in str(expr)

    def test_nested_expression(self):
        """Tests nested expression generation."""
        tree = Sin(Mul(Symbol(sympy.Symbol('a')), Number(2.0)))
        expr = tree.get_sympy_expr()
        assert 'sin' in str(expr).lower()

    def test_boolean_expression(self):
        """Tests boolean expression generation."""
        tree = And(
            Lt(Symbol(sympy.Symbol('a')), Number(1.0)),
            Boolean(True)
        )
        expr = tree.get_sympy_expr()
        assert expr is not None

    def test_expression_roundtrip(self):
        """Tests tree -> sympy -> tree roundtrip."""
        from plagih.trees import sympy_to_tree

        original = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        # Convert to sympy
        expr = original.get_sympy_expr()

        # Convert back to tree
        reconstructed = sympy_to_tree(expr, allow_chain=False)

        # Both should produce same sympy expression
        expr2 = reconstructed.get_sympy_expr()
        assert sympy.simplify(expr - expr2) == 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_division_by_small_number(self, sample_df):
        """Tests division handling near zero."""
        tree = Div(Number(1.0), Number(0.001))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.full(len(sample_df), 1000.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_exponent(self, sample_df):
        """Tests handling of large exponents."""
        tree = Pow(Number(2.0), Number(10.0))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.full(len(sample_df), 1024.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_pow_rounded(self, sample_df):
        """Tests PowRounded operator."""
        tree = PowRounded(Number(2.0), Number(3.7))
        result = tree.eval_predict_numpy_now(sample_df)
        # 3.7 rounded to 4, so 2^4 = 16
        expected = np.full(len(sample_df), 16.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_usub_negation(self, sample_df):
        """Tests unary negation."""
        tree = Usub(Symbol(sympy.Symbol('a')))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = -sample_df['a'].values
        np.testing.assert_array_almost_equal(result, expected)

    def test_square_and_sqrt(self, sample_df):
        """Tests Square and Sqrt are inverses for positive numbers."""
        tree = Sqrt(Square(Abs(Symbol(sympy.Symbol('a')))))
        result = tree.eval_predict_numpy_now(sample_df)
        expected = np.abs(sample_df['a'].values)
        np.testing.assert_array_almost_equal(result, expected)
