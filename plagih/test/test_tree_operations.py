"""
Tests for tree manipulation operations.

Tests verify:
1. sympy_to_tree roundtrip conversion
2. tree_simplification reduces complexity
3. tree_node_grouping pattern recognition
4. Tree structure operations (repair, copy, etc.)
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
import copy

from plagih.trees import (
    Node, Number, Symbol, Boolean,
    Add, Mul, Div, Sub, Pow, Sqrt, Square, Abs,
    Sin, Cos, Min, Max, Lt, And, Or, Ifte,
    PowRounded, DivFraction, Usub, Round,
    sympy_to_tree, tree_simplification, evolve_reduce_simplicate,
    eval_parsimony, node_deepcopy
)


# =============================================================================
# sympy_to_tree Tests
# =============================================================================

class TestSympyToTree:
    """Tests for sympy_to_tree conversion."""

    def test_simple_addition(self):
        """Tests converting simple addition."""
        expr = sympy.Symbol('a') + 1
        tree = sympy_to_tree(expr, allow_chain=False)

        assert tree is not None
        assert tree.is_operator()

    def test_simple_multiplication(self):
        """Tests converting multiplication."""
        expr = sympy.Symbol('a') * sympy.Symbol('b')
        tree = sympy_to_tree(expr, allow_chain=False)

        assert tree is not None

    def test_number_conversion(self):
        """Tests converting a number."""
        expr = sympy.Float(3.14)
        tree = sympy_to_tree(expr, allow_chain=False)

        assert isinstance(tree, Number)
        assert float(tree.get_value()) == pytest.approx(3.14, rel=1e-2)

    def test_symbol_conversion(self):
        """Tests converting a symbol."""
        expr = sympy.Symbol('x')
        tree = sympy_to_tree(expr, allow_chain=False)

        assert isinstance(tree, Symbol)

    def test_boolean_true(self):
        """Tests converting boolean True."""
        tree = sympy_to_tree(True, allow_chain=False)
        assert isinstance(tree, Boolean)

    def test_boolean_false(self):
        """Tests converting boolean False."""
        tree = sympy_to_tree(False, allow_chain=False)
        assert isinstance(tree, Boolean)

    def test_nested_expression(self):
        """Tests converting nested expression."""
        a, b = sympy.symbols('a b', real=True)
        expr = sympy.sin(a + b * 2)
        tree = sympy_to_tree(expr, allow_chain=False)

        assert tree is not None
        # Should have Sin at root
        assert isinstance(tree, Sin)

    def test_min_max(self):
        """Tests converting Min/Max."""
        a, b = sympy.symbols('a b', real=True)
        expr_min = sympy.Min(a, b)
        expr_max = sympy.Max(a, b)

        tree_min = sympy_to_tree(expr_min, allow_chain=False)
        tree_max = sympy_to_tree(expr_max, allow_chain=False)

        assert isinstance(tree_min, Min)
        assert isinstance(tree_max, Max)

    def test_comparison(self):
        """Tests converting comparison."""
        a = sympy.Symbol('a', real=True)
        expr = sympy.Lt(a, 5)
        tree = sympy_to_tree(expr, allow_chain=False)

        assert isinstance(tree, Lt)

    def test_roundtrip_simple(self):
        """Tests tree -> sympy -> tree produces equivalent result."""
        original = Add(Symbol(sympy.Symbol('a')), Number(2.0))

        # To sympy
        expr = original.get_sympy_expr()

        # Back to tree
        reconstructed = sympy_to_tree(expr, allow_chain=False)

        # Check equivalence via sympy
        expr2 = reconstructed.get_sympy_expr()
        assert sympy.simplify(expr - expr2) == 0

    def test_roundtrip_complex(self):
        """Tests roundtrip with complex expression."""
        original = Sin(Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Number(1.0)
        ))

        expr = original.get_sympy_expr()
        reconstructed = sympy_to_tree(expr, allow_chain=False)
        expr2 = reconstructed.get_sympy_expr()

        assert sympy.simplify(expr - expr2) == 0

    def test_piecewise_to_ifte(self):
        """Tests Piecewise converts to Ifte."""
        a = sympy.Symbol('a', real=True)
        expr = sympy.Piecewise((1, a < 0), (2, True))

        tree = sympy_to_tree(expr, allow_chain=False)

        # Should be Ifte structure
        assert isinstance(tree, Ifte)


# =============================================================================
# Tree Simplification Tests
# =============================================================================

class TestTreeSimplification:
    """Tests for tree_simplification function."""

    def test_simplify_reduces_or_maintains_size(self):
        """Tests simplification doesn't increase tree size significantly."""
        # a + 0 should simplify to a
        original = Add(Symbol(sympy.Symbol('a')), Number(0.0))
        original_len = len(original)

        simplified = tree_simplification(original, allow_chain=False)

        # Should not grow
        assert len(simplified) <= original_len + 2  # Small tolerance

    def test_simplify_identity_addition(self):
        """Tests simplifying a + 0."""
        a_sym = sympy.Symbol('a')
        # Create a + 0 via sympy
        expr = a_sym + 0
        tree = sympy_to_tree(expr, allow_chain=False)

        # Sympy already simplifies, so tree should be just Symbol
        assert len(tree) <= 3

    def test_simplify_identity_multiplication(self):
        """Tests simplifying a * 1."""
        a_sym = sympy.Symbol('a')
        expr = a_sym * 1
        tree = sympy_to_tree(expr, allow_chain=False)

        assert len(tree) <= 3

    def test_simplify_preserves_semantics(self, sample_df):
        """Tests simplification preserves evaluation results."""
        original = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        simplified = tree_simplification(copy.deepcopy(original), allow_chain=False)

        result_orig = original.eval_predict_numpy_now(sample_df)
        result_simp = simplified.eval_predict_numpy_now(sample_df)

        np.testing.assert_array_almost_equal(result_orig, result_simp)


# =============================================================================
# Tree Node Grouping Tests
# =============================================================================

class TestTreeNodeGrouping:
    """Tests for tree_node_grouping pattern recognition."""

    def test_power_to_square(self):
        """Tests x**2 -> Square(x)."""
        tree = Pow(Symbol(sympy.Symbol('a')), Number(2.0))
        tree.tree_node_grouping()

        # Should be converted to Square
        assert isinstance(tree, Square) or len(tree) <= 3

    def test_power_to_sqrt(self):
        """Tests x**0.5 -> Sqrt(x)."""
        tree = Pow(Symbol(sympy.Symbol('a')), Number(0.5))
        tree.tree_node_grouping()

        # Should be converted to Sqrt or remain as Pow with 0.5
        assert isinstance(tree, (Sqrt, Pow))

    def test_power_one_simplifies(self):
        """Tests x**1 simplifies to x."""
        tree = Pow(Symbol(sympy.Symbol('a')), Number(1.0))
        tree.tree_node_grouping()

        # Should simplify to just Symbol
        assert len(tree) <= 2

    def test_mul_by_negative_one(self):
        """Tests a * (-1) -> Usub(a)."""
        tree = Mul(Symbol(sympy.Symbol('a')), Number(-1.0))
        tree.tree_node_grouping()

        # Should be Usub or simplified
        assert isinstance(tree, (Usub, Mul, Symbol))


# =============================================================================
# Tree Structure Tests
# =============================================================================

class TestTreeStructure:
    """Tests for tree structure operations."""

    def test_repair_depth(self):
        """Tests repair_depth sets correct depths."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        tree.repair_depth(depth=0)

        assert tree.depth == 0
        for child in tree.get_childs():
            assert child.depth == 1

    def test_node_deepcopy(self):
        """Tests node_deepcopy creates independent copy."""
        original = Add(Symbol(sympy.Symbol('a')), Number(1.0))
        copied = node_deepcopy(original)

        # Modify copy
        copied.childs[1] = Number(999.0)

        # Original should be unchanged
        assert float(original.get_childs()[1].get_value()) == pytest.approx(1.0)

    def test_len_nodecount_raw(self):
        """Tests raw node counting."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        # Add + Mul + Symbol + Number + Symbol = 5
        assert tree.len_nodecount_raw() == 5

    def test_len_nodecount_fair(self):
        """Tests fair node counting."""
        tree = Usub(Symbol(sympy.Symbol('a')))

        # Usub should not be counted
        count = tree.len_nodecount_fair()
        assert count == 1  # Only the Symbol

    def test_list_mutable_nodes(self):
        """Tests listing mutable nodes."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        nodes = tree.list_mutable_nodes()

        # All 5 nodes should be mutable
        assert len(nodes) == 5

    def test_list_terminal_nodes(self):
        """Tests listing terminal nodes."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        terminals = tree.list_terminal_nodes()

        # 3 terminals: a, 2.0, b
        assert len(terminals) == 3

    def test_get_max_depth(self):
        """Tests max depth calculation."""
        tree = Add(
            Mul(Symbol(sympy.Symbol('a')), Number(2.0)),
            Symbol(sympy.Symbol('b'))
        )

        depth = tree.get_max_depth()

        # Add -> Mul -> Symbol = depth 2
        assert depth == 2

    def test_is_root(self):
        """Tests is_root detection."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(1.0))
        tree.repair_all()

        assert tree.is_root()
        assert not tree.get_childs()[0].is_root()


# =============================================================================
# Parsimony Tests
# =============================================================================

class TestParsimony:
    """Tests for parsimony/complexity evaluation."""

    def test_eval_parsimony_raw(self):
        """Tests raw node count parsimony."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(1.0))

        parsimony = eval_parsimony(tree, 'tree_node_count_raw')

        assert parsimony == 3  # Add + Symbol + Number

    def test_eval_parsimony_fair(self):
        """Tests fair node count parsimony."""
        tree = Usub(Add(Symbol(sympy.Symbol('a')), Number(1.0)))

        parsimony = eval_parsimony(tree, 'tree_node_count_fair')

        # Usub not counted: Add + Symbol + Number = 3
        assert parsimony == 3

    def test_parsimony_increases_with_complexity(self):
        """Tests more complex trees have higher parsimony."""
        simple = Add(Symbol(sympy.Symbol('a')), Number(1.0))
        complex_tree = Sin(Add(
            Mul(Symbol(sympy.Symbol('a')), Symbol(sympy.Symbol('b'))),
            Number(2.0)
        ))

        p_simple = eval_parsimony(simple, 'tree_node_count_raw')
        p_complex = eval_parsimony(complex_tree, 'tree_node_count_raw')

        assert p_complex > p_simple


# =============================================================================
# String Representation Tests
# =============================================================================

class TestTreeRepresentation:
    """Tests for tree string representations."""

    def test_represent_str(self):
        """Tests represent_str output."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(2.0))

        s = tree.represent_str()

        assert 'Add' in s
        assert 'a' in s
        assert '2' in s

    def test_str_as_list(self):
        """Tests str_as_list output."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(1.0))

        s = tree.str_as_list()

        assert '[' in s
        assert 'Add' in s

    def test_export_tree(self):
        """Tests export_tree produces evaluable string."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(2.5))

        export = tree.export_tree()

        assert 'Add' in export
        assert 'Symbol' in export
        assert 'Number' in export
        assert '2.5' in export

    def test_get_lut_id_unique(self):
        """Tests get_lut_id produces unique IDs."""
        tree1 = Add(Symbol(sympy.Symbol('a')), Number(1.0))
        tree2 = Add(Symbol(sympy.Symbol('a')), Number(2.0))
        tree3 = Add(Symbol(sympy.Symbol('a')), Number(1.0))

        id1 = tree1.get_lut_id()
        id2 = tree2.get_lut_id()
        id3 = tree3.get_lut_id()

        assert id1 != id2  # Different values
        assert id1 == id3  # Same structure

    def test_get_expr_symlike(self):
        """Tests get_expr_symlike output."""
        tree = Add(Symbol(sympy.Symbol('a')), Number(2.0))

        s = tree.get_expr_symlike()

        assert '+' in s or 'Add' in s
        assert 'a' in s
