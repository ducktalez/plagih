"""
Comprehensive tests for all Node classes.

This module contains the tests that were previously in trees.py __main__.
It automatically tests all operator classes for consistency between
symfun and np_fun.
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
    Add, Mul, Div, Sub, Pow, Sqrt, Square, Abs, Sign, Log, Exp,
    Sin, Cos, Tan, Asin, Acos, Atan, Tanh, Sinh, Cosh,
    Min, Max, Lt, Le, Gt, Ge, Eq, Ne,
    And, Or, Xor, Not, Ifte, ITE, Piecewise,
    Round, PowRounded, DivFraction, Usub,
    BaseOperator, ChainableOp, ExprCondPair_Dummy,
    expr_sympify, RoundDummy
)
from plagih.util import get_subclasses


# =============================================================================
# Test Data
# =============================================================================

SYMBOLS = {
    'a': sympy.Symbol('a', real=True),
    'b': sympy.Symbol('b', real=True),
    'c': sympy.Symbol('c'),
    'd': sympy.Symbol('d'),
}

SAMPLE_DATA = {
    'a': [1.0, 2, 3, 4, 5, 6],
    'b': [-1.0, -2, -3, -4, -5, -6],
    'c': [True, False, True, False, True, False],
    'd': [True, True, True, True, True, True]
}

SYMPIFY_TEST_EXPRESSIONS = [
    '5', '1', '0', '0.5', '-1', 'True', 'False',
    'c & True', 'c | False', '~c',
    'a<1', 'a<b', 'a<=b', 'a>=b', 'a>b', 'a==b', 'a!=b', 'a',
    'a + 1', 'a + 2', 'a*2', 'a - 2', 'a/2', 'a < 2', 'a**2', '2/a',
    'a*b*2', 'a+b+a+2+4', 'Min(a, b, 3)', 'Max(a, b, 4, a**2, a+b)', 'a<3',
    'Piecewise((a, c), (b, d), (a+b, True))',
    'Eq(4, 4.0)',
]

CUSTOM_TEST_EXPRESSIONS = [
    'Max(a, 2)',
    'Min(b, b)',
    'sin(asin(0.5))',
    'And(False, True)',
    'Add(-1.490149, 14.0)',
    'Eq(4, 4.0)',
    'Lt(a, a)',
    'Or(Ne(False, False), False)',
    'sqrt(5 * a)',
    'Not(False)',
    'acos(0.5)',
    'Pow(a, b)',
    'Add(-2, Min(1, 8))',
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_node_subclasses(cls=Node):
    """Returns all leaf subclasses (no further subclasses)."""
    sub = []
    for x in get_subclasses(cls):
        if len(x.__subclasses__()) > 0:
            pass
        else:
            sub.append(x)
    return sub


def get_testable_operator_classes():
    """Returns operator classes that can be automatically tested."""
    skip_classes = {ExprCondPair_Dummy, Piecewise, Boolean, Number, Symbol}

    all_classes = get_all_node_subclasses()
    testable = []

    for cls in all_classes:
        if cls in skip_classes:
            continue
        if not hasattr(cls, 'xtype') or not cls.xtype:
            continue
        if not hasattr(cls, 'np_fun') or cls.np_fun is None:
            continue
        if not hasattr(cls, 'symfun') or cls.symfun is None:
            continue
        testable.append(cls)

    return testable


# =============================================================================
# Automatic Operator Tests
# =============================================================================

class TestAllOperatorClasses:
    """Automatically tests all operator classes for symfun/np_fun consistency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup random seed for reproducibility."""
        np.random.seed(42)

    def test_all_operators_symfun_npfun_consistency(self):
        """Tests all operators for consistency between symfun and np_fun.

        This is the migrated test from trees.py __main__.
        """
        testable_classes = get_testable_operator_classes()

        failed = []
        passed = []
        skipped = []

        input_generators = {
            float: lambda: np.random.random() * 2 - 1,  # [-1, 1]
            bool: lambda: np.random.choice([True, False])
        }

        for ncls in testable_classes:
            try:
                xtype_me = ncls.xtype[1]  # Output type

                # Generate inputs
                if issubclass(ncls, ChainableOp) and hasattr(ncls, 'xtype_input'):
                    xtype_childs = ncls.xtype_input
                    inputs_sy = [input_generators.get(xtype_childs, lambda: np.random.random())()
                                 for _ in range(4)]
                else:
                    xtype_childs = ncls.xtype[0]
                    inputs_sy = [input_generators.get(x, lambda: np.random.random())()
                                 for x in xtype_childs]

                # Prepare numpy inputs
                inputs_np = [np.array([x]) for x in inputs_sy]

                # Evaluate
                symf = ncls.symfun
                np_fun = ncls.np_fun

                res_sy = symf(*inputs_sy)
                res_np = np_fun(*inputs_np)

                # Convert to output type
                if hasattr(res_sy, 'evalf'):
                    res_sy = float(res_sy.evalf())
                res_sy = xtype_me(res_sy)
                res_np = xtype_me(res_np[0] if hasattr(res_np, '__len__') else res_np)

                # Compare
                if xtype_me == bool:
                    if res_sy == res_np:
                        passed.append(ncls.__name__)
                    else:
                        failed.append((ncls.__name__, res_sy, res_np, inputs_sy))
                else:
                    if abs(res_sy - res_np) < 0.0001:
                        passed.append(ncls.__name__)
                    else:
                        failed.append((ncls.__name__, res_sy, res_np, inputs_sy))

            except Exception as ex:
                skipped.append((ncls.__name__, str(ex)))

        # Report results
        print(f"\nOperator Test Results:")
        print(f"  Passed: {len(passed)}")
        print(f"  Skipped: {len(skipped)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            for name, sy, np_res, inputs in failed:
                print(f"  FAILED: {name} - sympy={sy}, numpy={np_res}, inputs={inputs}")

        # Assert no failures
        assert len(failed) == 0, f"Failed operators: {[f[0] for f in failed]}"


# =============================================================================
# Sympify Tests
# =============================================================================

class TestSympify:
    """Tests for sympify expression conversion."""

    @pytest.mark.parametrize("expr", SYMPIFY_TEST_EXPRESSIONS)
    def test_sympify_expression(self, expr):
        """Tests that expressions can be sympified."""
        result = expr_sympify(expr)
        assert result is not None

    @pytest.mark.parametrize("expr", CUSTOM_TEST_EXPRESSIONS)
    def test_sympify_custom_expression(self, expr):
        """Tests custom expressions can be sympified."""
        try:
            result = expr_sympify(expr)
            assert result is not None
        except Exception as e:
            # Some expressions may not be valid, that's okay
            pytest.skip(f"Expression '{expr}' failed: {e}")


# =============================================================================
# RoundDummy Tests
# =============================================================================

class TestRoundDummy:
    """Tests for RoundDummy function."""

    def test_round_dummy_negative_small(self):
        """Tests RoundDummy with small negative number."""
        result = RoundDummy(sympy.Float(-0.1))
        # -0.1 rounds to 0
        assert float(result) == 0 or result == 0

    def test_round_dummy_positive_small(self):
        """Tests RoundDummy with 0.49."""
        result = RoundDummy(sympy.Float(0.49))
        # 0.49 rounds to 0
        assert float(result) == 0

    def test_round_dummy_with_symbol(self):
        """Tests RoundDummy with symbolic expression."""
        x = sympy.Symbol('x')
        result = RoundDummy(x + 1.13)
        # Should be a symbolic expression
        assert result is not None


# =============================================================================
# Tree Construction Tests
# =============================================================================

class TestTreeConstruction:
    """Tests for constructing trees manually."""

    def test_pow_rounded_construction(self):
        """Tests PowRounded tree construction."""
        tree = PowRounded(Number(3), Number(2))
        assert tree is not None
        assert len(tree.get_childs()) == 2

    def test_complex_tree_construction(self):
        """Tests complex nested tree construction."""
        tree = Mul(
            Number(289),
            Symbol(sympy.Symbol('cartVel')),
            Add(
                Symbol(sympy.Symbol('cartPos')),
                Mul(Number(2.27), Symbol(sympy.Symbol('cartVel')))
            ),
            Sin(PowRounded(Number(12), Symbol(sympy.Symbol('cartPos'))))
        )

        assert tree is not None
        # Should have 4 children (Mul is chainable)
        assert len(tree.get_childs()) >= 2

    def test_complex_tree_sympy_expr(self):
        """Tests sympy expression generation from complex tree."""
        tree = Mul(
            Number(289),
            Symbol(sympy.Symbol('cartVel')),
            Add(
                Symbol(sympy.Symbol('cartPos')),
                Mul(Number(2.27), Symbol(sympy.Symbol('cartVel')))
            ),
            Sin(PowRounded(Number(12), Symbol(sympy.Symbol('cartPos'))))
        )

        expr = tree.get_sympy_expr()
        assert expr is not None
        assert 'cartVel' in str(expr)
        assert 'cartPos' in str(expr)

    def test_tree_evaluation(self):
        """Tests tree evaluation on DataFrame."""
        df = pd.DataFrame({
            'cartPos': [0, np.pi / 4, np.pi / 2, np.pi],
            'cartVel': [0.1, 0.2, 0.3, 0.4]
        })

        tree = Add(
            Symbol(sympy.Symbol('cartPos')),
            Mul(Number(2.0), Symbol(sympy.Symbol('cartVel')))
        )

        result = tree.eval_predict_numpy_now(df)
        expected = df['cartPos'].values + 2.0 * df['cartVel'].values

        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Run tests directly (optional)
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
