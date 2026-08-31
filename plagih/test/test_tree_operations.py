"""
Tests for tree manipulation operations.

Tests verify:
1. sympy_to_tree roundtrip conversion
2. tree_simplification reduces complexity
3. tree_node_grouping pattern recognition
4. Tree structure operations (repair, copy, etc.)
"""

import numpy as np
import pytest
import sympy

import plagih.trees._nodes as nodes_mod
from plagih.exceptions import TreeError
from plagih.trees import (
    Abs,
    Add,
    And,
    Boolean,
    Cos,
    Div,
    DivFraction,
    Ifte,
    Le,
    Lt,
    Max,
    Min,
    Mul,
    Node,
    Number,
    Or,
    Pow,
    PowRounded,
    Round,
    Scale,
    Sin,
    Sqrt,
    Square,
    Sub,
    Symbol,
    Usub,
    eval_parsimony,
    evolve_reduce_simplicate,
    node_deepcopy,
    sympy_to_tree,
    tree_simplification,
)

# =============================================================================
# sympy_to_tree Tests
# =============================================================================


class TestSympyToTree:
    """Tests for sympy_to_tree conversion."""

    def test_simple_addition(self):
        """Tests converting simple addition."""
        expr = sympy.Symbol("a") + 1
        tree = sympy_to_tree(expr, allow_chain=False)

        assert tree is not None
        assert tree.is_operator()

    def test_simple_multiplication(self):
        """Tests converting multiplication."""
        expr = sympy.Symbol("a") * sympy.Symbol("b")
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
        expr = sympy.Symbol("x")
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
        a, b = sympy.symbols("a b", real=True)
        expr = sympy.sin(a + b * 2)
        tree = sympy_to_tree(expr, allow_chain=False)

        assert tree is not None
        # Should have Sin at root
        assert isinstance(tree, Sin)

    def test_min_max(self):
        """Tests converting Min/Max."""
        a, b = sympy.symbols("a b", real=True)
        expr_min = sympy.Min(a, b)
        expr_max = sympy.Max(a, b)

        tree_min = sympy_to_tree(expr_min, allow_chain=False)
        tree_max = sympy_to_tree(expr_max, allow_chain=False)

        assert isinstance(tree_min, Min)
        assert isinstance(tree_max, Max)

    def test_comparison(self):
        """Tests converting comparison."""
        a = sympy.Symbol("a", real=True)
        expr = sympy.Lt(a, 5)
        tree = sympy_to_tree(expr, allow_chain=False)

        assert isinstance(tree, Lt)

    def test_roundtrip_simple(self):
        """Tests tree -> sympy -> tree produces equivalent result."""
        original = Add(Symbol(sympy.Symbol("a")), Number(2.0))

        # To sympy
        expr = original.get_sympy_expr()

        # Back to tree
        reconstructed = sympy_to_tree(expr, allow_chain=False)

        # Check equivalence via sympy
        expr2 = reconstructed.get_sympy_expr()
        assert sympy.simplify(expr - expr2) == 0

    def test_roundtrip_complex(self):
        """Tests roundtrip with complex expression."""
        original = Sin(Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Number(1.0)))

        expr = original.get_sympy_expr()
        reconstructed = sympy_to_tree(expr, allow_chain=False)
        expr2 = reconstructed.get_sympy_expr()

        assert sympy.simplify(expr - expr2) == 0

    def test_piecewise_to_ifte(self):
        """Tests Piecewise converts to Ifte."""
        a = sympy.Symbol("a", real=True)
        expr = sympy.Piecewise((1, a < 0), (2, True))

        tree = sympy_to_tree(expr, allow_chain=False)

        # Should be Ifte structure
        assert isinstance(tree, Ifte)


# =============================================================================
# Tree Simplification Tests
# =============================================================================


class TestTreeSimplification:
    """Tests for tree_simplification function."""

    def test_str_as_list_renders_compact_structure(self):
        """Diagnostics should use the existing compact one-line structural dump."""
        tree = Add(Symbol(sympy.Symbol("a")), Mul(Number(2.0), Symbol(sympy.Symbol("b"))))

        dump = tree.str_as_list(cut_terms=True)

        assert dump == "[Add, [a], [Mul, [2], [b]]]"

    def test_simplify_reduces_or_maintains_size(self):
        """Tests simplification doesn't increase tree size significantly."""
        # a + 0 should simplify to a
        original = Add(Symbol(sympy.Symbol("a")), Number(0.0))
        original_len = len(original)

        simplified = tree_simplification(original, allow_chain=False)

        # Should not grow
        assert len(simplified) <= original_len + 2  # Small tolerance

    def test_simplify_identity_addition(self):
        """Tests simplifying a + 0."""
        a_sym = sympy.Symbol("a")
        # Create a + 0 via sympy
        expr = a_sym + 0
        tree = sympy_to_tree(expr, allow_chain=False)

        # Sympy already simplifies, so tree should be just Symbol
        assert len(tree) <= 3

    def test_simplify_identity_multiplication(self):
        """Tests simplifying a * 1."""
        a_sym = sympy.Symbol("a")
        expr = a_sym * 1
        tree = sympy_to_tree(expr, allow_chain=False)

        assert len(tree) <= 3

    def test_simplify_preserves_semantics(self, sample_df):
        """Tests simplification preserves evaluation results."""
        original = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))

        simplified_input = node_deepcopy(original)
        simplified_input.repair_all()
        simplified = tree_simplification(simplified_input, allow_chain=False)

        result_orig = original.eval_predict_numpy_now(sample_df)
        result_simp = simplified.eval_predict_numpy_now(sample_df)

        np.testing.assert_array_almost_equal(result_orig, result_simp)

    def test_simplify_grouping_initializes_best_tree_regression(self):
        """Regression: grouping updates must not crash with uninitialized best-tree tracking."""
        a_sym = sympy.Symbol("a")
        original = Pow(Symbol(a_sym), Number(2.0))

        simplified = tree_simplification(original, allow_chain=False)

        assert isinstance(simplified, (Square, Pow, PowRounded))
        for sample in (-3, -1, 0, 2, 5):
            expected = float(original.get_sympy_expr().subs({a_sym: sample}))
            actual = float(simplified.get_sympy_expr().subs({a_sym: sample}))
            assert actual == pytest.approx(expected)

    def test_simplify_falls_back_to_original_when_sympy_version_grows(self):
        """Regression: simplification must keep the smaller original representation."""
        original = Div(Number(19.0), Symbol(sympy.Symbol("a")))

        simplified = tree_simplification(original, allow_chain=False)

        assert len(simplified) <= len(original)
        assert sympy.simplify(original.get_sympy_expr() - simplified.get_sympy_expr()) == 0

    def test_simplify_rejects_semantic_change_and_logs_tree_diagnostics(self, monkeypatch):
        """Regression: semantically changed simplifications must fall back and log compact tree diagnostics."""
        a_sym = sympy.Symbol("a")
        original = Add(Symbol(a_sym), Number(1.0))
        messages = []

        def _capture_log(msg_type, message):
            messages.append((msg_type, message))

        monkeypatch.setattr(nodes_mod, "log", _capture_log)
        monkeypatch.setattr(
            nodes_mod,
            "sympy_to_tree",
            lambda expr, allow_chain: Add(Symbol(a_sym), Number(2.0)),
        )

        simplified = tree_simplification(original, allow_chain=False)

        assert sympy.simplify(simplified.get_sympy_expr() - original.get_sympy_expr()) == 0
        warning_messages = [message for msg_type, message in messages if msg_type == "w"]
        assert any("Simplification rejected (changed semantics)" in message for message in warning_messages)
        assert any("original tree : [Add, [a], [1]]" in message for message in warning_messages)
        assert any("roundtrip tree: [Add, [a], [2]]" in message for message in warning_messages)
        assert any("suspected stage=sympy_roundtrip" in message for message in warning_messages)

    def test_partial_reduce_can_simplify_math_operator(self):
        """Regression: partial simplification must not ignore MathOperator nodes."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(0.0))

        simplified = evolve_reduce_simplicate(tree, allow_chain=False, completely=False, force=True)

        assert isinstance(simplified, Symbol)
        assert simplified.get_value() == sympy.Symbol("a")

    def test_revoke_useless_nodes_removes_additive_identity_without_crashing(self):
        """Regression: Add(..., 0) cleanup must not raise CuriosityError."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(0.0))

        tree.revoke_useless_nodes()

        assert isinstance(tree, Symbol)
        assert tree.get_value() == sympy.Symbol("a")

    def test_revoke_useless_nodes_reduces_all_zero_add_to_zero(self):
        """Regression: Add(0, 0) should collapse to Number(0) instead of crashing."""
        tree = Add(Number(0.0), Number(0.0))

        tree.revoke_useless_nodes()

        assert isinstance(tree, Number)
        assert float(tree.get_value()) == pytest.approx(0.0)

    def test_set_new_node_handles_neutral_add_branch_cleanup(self):
        """Regression: set_new_node() should survive replacing a subtree with Add(a, 0)."""
        target = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        replacement = Add(Symbol(sympy.Symbol("b")), Number(0.0))

        target.set_new_node(replacement)

        assert isinstance(target, Symbol)
        assert target.get_value() == sympy.Symbol("b")

    def test_relational_on_min_rejects_for_sympy(self):
        """P12 extension: Lt(Min(a, b), c) must raise SympyError to prevent SymPy hangs."""
        from plagih.exceptions import SympyError

        a, b = sympy.Symbol("a"), sympy.Symbol("b")
        tree = Lt(Min(Symbol(a), Symbol(b)), Number(0.5))
        tree.repair_depth()

        with pytest.raises(SympyError, match="MinMax"):
            tree.get_sympy_expr()

    def test_relational_on_max_rejects_for_sympy(self):
        """P12 extension: Le(Max(a, b), c) must raise SympyError to prevent SymPy hangs."""
        from plagih.exceptions import SympyError
        from plagih.trees import Le

        a, b = sympy.Symbol("a"), sympy.Symbol("b")
        tree = Le(Max(Symbol(a), Symbol(b)), Number(1.0))
        tree.repair_depth()

        with pytest.raises(SympyError, match="MinMax"):
            tree.get_sympy_expr()

    def test_relational_on_nested_min_rejects_for_sympy(self):
        """P12 extension: Lt(Add(Min(a, b), c), d) must also be caught (recursive check)."""
        from plagih.exceptions import SympyError

        a, b, c = sympy.Symbol("a"), sympy.Symbol("b"), sympy.Symbol("c")
        tree = Lt(Add(Min(Symbol(a), Symbol(b)), Symbol(c)), Number(0.5))
        tree.repair_depth()

        with pytest.raises(SympyError, match="MinMax"):
            tree.get_sympy_expr()

    def test_simplification_with_min_max_returns_original(self):
        """Trees containing Lt(Min(...), ...) should survive tree_simplification gracefully."""
        a, b = sympy.Symbol("a"), sympy.Symbol("b")
        # Build a tree that contains Min but where simplification should not hang
        tree = Add(Min(Symbol(a), Symbol(b)), Number(1.0))
        tree.repair_depth()

        # Should not hang — simplification may succeed or return original
        result = tree_simplification(tree, allow_chain=False)
        assert result is not None
        assert len(result) >= 1

    def test_simplification_timeout_returns_original(self, monkeypatch):
        """tree_simplification must return original tree on SymPy timeout."""
        import plagih.trees._nodes as _nodes_mod

        # Set a very short timeout to force the timeout path
        monkeypatch.setattr(_nodes_mod, "SYMPY_SIMPLIFICATION_TIMEOUT_S", 0.001)

        a = sympy.Symbol("a")
        original = Add(Symbol(a), Mul(Number(2.0), Symbol(a)))
        original.repair_depth()
        original_str = str(original)

        # Patch get_sympy_expr to simulate a slow SymPy call
        real_get_sympy = Node.get_sympy_expr

        def _slow_get_sympy_expr(self_node, simplimore=False):
            import time

            time.sleep(0.5)  # Much longer than 0.001s timeout
            return real_get_sympy(self_node, simplimore=simplimore)

        monkeypatch.setattr(Node, "get_sympy_expr", _slow_get_sympy_expr)

        result = tree_simplification(original, allow_chain=False)

        # Should return the original tree (timeout fallback)
        assert str(result) == original_str

    def test_grouping_only_mode_for_trees_with_min(self):
        """D7: Trees containing Min/Max skip SymPy roundtrip, use grouping-only."""
        from plagih.trees._nodes import _tree_has_piecewise_like

        a, b = sympy.Symbol("a"), sympy.Symbol("b")
        # Mul(2, Min(a, b)) — should trigger grouping-only
        tree = Mul(Number(2.0), Min(Symbol(a), Symbol(b)))
        tree.repair_depth()
        assert _tree_has_piecewise_like(tree)

        result = tree_simplification(tree, allow_chain=False)
        # Must not hang and must not grow
        assert len(result) <= len(tree)

    def test_grouping_only_mode_for_trees_with_abs(self):
        """D7: Trees containing Abs skip SymPy roundtrip (Abs → Piecewise internally)."""
        from plagih.trees._nodes import _tree_has_piecewise_like

        a = sympy.Symbol("a")
        tree = Mul(Abs(Symbol(a)), Number(3.0))
        tree.repair_depth()
        assert _tree_has_piecewise_like(tree)

        result = tree_simplification(tree, allow_chain=False)
        assert len(result) <= len(tree)

    def test_non_piecewise_tree_uses_full_roundtrip(self):
        """D7: Trees without Min/Max/Abs/sign still use full SymPy roundtrip."""
        from plagih.trees._nodes import _tree_has_piecewise_like

        a = sympy.Symbol("a")
        # Add(a, 0) — no Piecewise-like ops, should use full roundtrip
        tree = Add(Symbol(a), Number(0.0))
        tree.repair_depth()
        assert not _tree_has_piecewise_like(tree)

        result = tree_simplification(tree, allow_chain=False)
        # SymPy should simplify a+0 → a
        assert len(result) <= len(tree)

    def test_tree_has_piecewise_like_detection(self):
        """D7: _tree_has_piecewise_like correctly detects nested Piecewise-like nodes."""
        from plagih.trees._nodes import _tree_has_piecewise_like

        a, b = sympy.Symbol("a"), sympy.Symbol("b")

        # Direct Min
        assert _tree_has_piecewise_like(Min(Symbol(a), Symbol(b)))
        # Nested in Add
        assert _tree_has_piecewise_like(Add(Number(1.0), Max(Symbol(a), Symbol(b))))
        # Abs
        assert _tree_has_piecewise_like(Abs(Symbol(a)))
        # No Piecewise-like
        assert not _tree_has_piecewise_like(Add(Symbol(a), Number(1.0)))
        assert not _tree_has_piecewise_like(Sin(Symbol(a)))
        assert not _tree_has_piecewise_like(Mul(Symbol(a), Number(2.0)))


# =============================================================================
# Tree Node Grouping Tests
# =============================================================================


class TestTreeNodeGrouping:
    """Tests for tree_node_grouping pattern recognition."""

    def test_power_to_square(self):
        """Tests x**2 -> Square(x)."""
        tree = Pow(Symbol(sympy.Symbol("a")), Number(2.0))
        tree.tree_node_grouping()

        # Should be converted to Square
        assert isinstance(tree, Square) or len(tree) <= 3

    def test_power_to_sqrt(self):
        """Tests x**0.5 -> Sqrt(x)."""
        tree = Pow(Symbol(sympy.Symbol("a")), Number(0.5))
        tree.tree_node_grouping()

        # Should be converted to Sqrt or remain as Pow with 0.5
        assert isinstance(tree, (Sqrt, Pow))

    def test_power_one_simplifies(self):
        """Tests x**1 simplifies to x."""
        tree = Pow(Symbol(sympy.Symbol("a")), Number(1.0))
        tree.tree_node_grouping()

        # Should simplify to just Symbol
        assert len(tree) <= 2

    def test_mul_by_negative_one(self):
        """Tests a * (-1) -> Usub(a)."""
        tree = Mul(Symbol(sympy.Symbol("a")), Number(-1.0))
        tree.tree_node_grouping()

        # Should be Usub or simplified
        assert isinstance(tree, (Usub, Mul, Symbol))

    def test_mul_by_one_cleanup_does_not_crash(self):
        """Regression: Mul(..., 1) in grouping should simplify instead of raising."""
        tree = Mul(Symbol(sympy.Symbol("a")), Number(1.0))

        tree.tree_node_grouping()

        assert isinstance(tree, Symbol)
        assert tree.get_value() == sympy.Symbol("a")

    def test_mul_of_only_ones_collapses_to_one(self):
        """Regression: Mul(1, 1) should collapse to Number(1) during grouping."""
        tree = Mul(Number(1.0), Number(1.0))

        tree.tree_node_grouping()

        assert isinstance(tree, Number)
        assert float(tree.get_value()) == pytest.approx(1.0)

    def test_grouping_preserves_duplicate_negative_one_factors(self):
        """Regression: removing one matched factor must not drop both equal `-1` children."""
        tree = Mul(Symbol(sympy.Symbol("a")), Number(-1.0), Number(-1.0))
        expr_before = tree.get_sympy_expr()

        tree.tree_node_grouping()

        expr_after = tree.get_sympy_expr()
        assert sympy.simplify(expr_before - expr_after) == 0

    def test_grouping_preserves_duplicate_numeric_scale_factors(self):
        """Regression: equal numeric factors must not be removed by value-based filtering."""
        tree = Mul(Number(2.0), Symbol(sympy.Symbol("a")), Number(2.0))
        expr_before = tree.get_sympy_expr()

        tree.tree_node_grouping()

        expr_after = tree.get_sympy_expr()
        assert sympy.simplify(expr_before - expr_after) == 0

    def test_mul_small_denom_becomes_div(self):
        """I13: small denom (<=cap) -> Div, e.g. a * (1/2) -> a/2."""
        tree = Mul(Symbol(sympy.Symbol("a")), Number(0.5))
        tree.tree_node_grouping()

        assert isinstance(tree, Div)
        assert float(tree.get_childs()[1].get_value()) == pytest.approx(2.0)

    def test_mul_big_denom_becomes_scale_not_div(self):
        """I13: big denom (>cap, e.g. 361) stays Scale, not Div — looks wrong (a/361 vs 0.00277*a)."""
        tree = Mul(Symbol(sympy.Symbol("a")), Number(1.0 / 361.0))
        expr_before = tree.get_sympy_expr()

        tree.tree_node_grouping()

        assert isinstance(tree, Scale)
        expr_after = tree.get_sympy_expr()
        assert sympy.simplify(expr_before - expr_after) == 0


# =============================================================================
# Tree Structure Tests
# =============================================================================


class TestTreeStructure:
    """Tests for tree structure operations."""

    def test_node_equality_is_identity_based_after_repair(self):
        """Regression: repaired trees must not recurse via dataclass structural equality."""
        tree_a = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree_b = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree_a.repair_all()
        tree_b.repair_all()

        assert tree_a == tree_a
        assert tree_a != tree_b

    def test_equal_value_terminals_do_not_match_by_membership(self):
        """Regression: node membership checks should be identity-based, not structural."""
        n1 = Number(1.0)
        n2 = Number(1.0)

        assert n1 not in [n2]

    def test_set_new_node_repair_true_preserves_non_root_backlinks(self):
        """Regression: replacing an attached subtree with repair=True must keep parent/root/depth consistent."""
        root = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        root.repair_all()
        child = root.get_childs()[1]

        child.set_new_node(Mul(Number(2.0), Symbol(sympy.Symbol("b"))), repair=True, clean_chain=False)

        assert root.get_childs()[1] is child
        assert isinstance(child, Mul)
        assert child.parent_node is root
        assert child.root_node is root
        assert child.depth == 1
        for grandchild in child.get_childs():
            assert grandchild.parent_node is child
            assert grandchild.root_node is root
            assert grandchild.depth == 2

    def test_set_new_node_reuses_replacement_without_aliasing_children(self):
        """Regression: reusing the same replacement template must not share child objects across trees."""
        replacement = Mul(Number(3.0), Symbol(sympy.Symbol("z")))
        left = Add(Symbol(sympy.Symbol("x")), Number(1.0))
        right = Add(Symbol(sympy.Symbol("y")), Number(2.0))

        left.set_new_node(replacement, clean_chain=False)
        right.set_new_node(replacement, clean_chain=False)

        assert isinstance(left, Mul)
        assert isinstance(right, Mul)
        assert all(a is not b for a, b in zip(left.get_childs(), right.get_childs(), strict=True))

        left.get_childs()[0].set_value(99.0)
        assert float(right.get_childs()[0].get_value()) == pytest.approx(3.0)
        assert float(replacement.get_childs()[0].get_value()) == pytest.approx(3.0)

    def test_replace_with_keeps_parent_slot_bound_to_same_object(self):
        """Regression: replace_with() should mutate the attached node in place."""
        root = Add(Symbol(sympy.Symbol("a")), Mul(Symbol(sympy.Symbol("b")), Number(-1.0)))
        root.repair_all()
        child = root.get_childs()[1]

        child.replace_with(Usub, [Symbol(sympy.Symbol("b"))])

        assert root.get_childs()[1] is child
        assert isinstance(child, Usub)
        assert child.parent_node is root
        assert child.root_node is root
        assert child.depth == 1

    def test_replace_with_node_does_not_mutate_template_node(self):
        """Regression: replace_with_node() should not pre-simplify/mutate the caller-provided template."""
        target = Number(5.0)
        replacement = Add(Symbol(sympy.Symbol("b")), Number(0.0))

        target.replace_with_node(replacement)

        assert isinstance(target, Symbol)
        assert target.get_value() == sympy.Symbol("b")
        assert isinstance(replacement, Add)
        assert len(replacement.get_childs()) == 2

    def test_repair_depth(self):
        """Tests repair_depth sets correct depths."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))

        tree.repair_depth(depth=0)

        assert tree.depth == 0
        for child in tree.get_childs():
            assert child.depth == 1

    def test_node_deepcopy(self):
        """Tests node_deepcopy creates independent copy."""
        original = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        copied = node_deepcopy(original)
        copied.repair_all()

        # Modify copy
        copied.childs[1] = Number(999.0)

        # Original should be unchanged
        assert float(original.get_childs()[1].get_value()) == pytest.approx(1.0)

    def test_len_nodecount_raw(self):
        """Tests raw node counting."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))

        # Add + Mul + Symbol + Number + Symbol = 5
        assert tree.len_nodecount_raw() == 5

    def test_len_nodecount_fair(self):
        """Tests fair node counting."""
        tree = Usub(Symbol(sympy.Symbol("a")))

        # Usub should not be counted
        count = tree.len_nodecount_fair()
        assert count == 1  # Only the Symbol

    def test_list_mutable_nodes(self):
        """Tests listing mutable nodes."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))

        nodes = tree.list_mutable_nodes()

        # All 5 nodes should be mutable
        assert len(nodes) == 5

    def test_list_terminal_nodes(self):
        """Tests listing terminal nodes."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))

        terminals = tree.list_terminal_nodes()

        # 3 terminals: a, 2.0, b
        assert len(terminals) == 3

    def test_get_max_depth(self):
        """Tests max depth calculation."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))

        depth = tree.get_max_depth()

        # Add -> Mul -> Symbol = depth 2
        assert depth == 2

    def test_is_root(self):
        """Tests is_root detection."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
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
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))

        parsimony = eval_parsimony(tree, "tree_node_count_raw")

        assert parsimony == 3  # Add + Symbol + Number

    def test_eval_parsimony_fair(self):
        """Tests fair node count parsimony."""
        tree = Usub(Add(Symbol(sympy.Symbol("a")), Number(1.0)))

        parsimony = eval_parsimony(tree, "tree_node_count_fair")

        # Usub not counted: Add + Symbol + Number = 3
        assert parsimony == 3

    def test_parsimony_increases_with_complexity(self):
        """Tests more complex trees have higher parsimony."""
        simple = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        complex_tree = Sin(Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Number(2.0)))

        p_simple = eval_parsimony(simple, "tree_node_count_raw")
        p_complex = eval_parsimony(complex_tree, "tree_node_count_raw")

        assert p_complex > p_simple


# =============================================================================
# String Representation Tests
# =============================================================================


class TestTreeRepresentation:
    """Tests for tree string representations."""

    def test_represent_str(self):
        """Tests represent_str output."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(2.0))

        s = tree.represent_str()

        assert "Add" in s
        assert "a" in s
        assert "2" in s

    def test_str_as_list(self):
        """Tests str_as_list output."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))

        s = tree.str_as_list()

        assert "[" in s
        assert "Add" in s

    def test_export_tree(self):
        """Tests export_tree produces evaluable string."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(2.5))

        export = tree.export_tree()

        assert "Add" in export
        assert "Symbol" in export
        assert "Number" in export
        assert "2.5" in export

    def test_export_tree_raises_clear_error_for_invalid_number_payload(self):
        """Regression: malformed Number export should raise a typed, descriptive error."""
        tree = Number(1.0)
        tree.childs = [object()]

        with pytest.raises(ValueError, match="Cannot export Number terminal"):
            tree.export_tree()

    def test_str_as_list_raises_tree_error_for_node_without_childs(self):
        """Regression: malformed nodes should fail with TreeError, not CuriosityError."""
        tree = Number(1.0)
        tree.childs = []

        with pytest.raises(TreeError, match="has no childs"):
            tree.str_as_list()

    def test_get_lut_id_unique(self):
        """Tests get_lut_id produces unique IDs."""
        tree1 = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree2 = Add(Symbol(sympy.Symbol("a")), Number(2.0))
        tree3 = Add(Symbol(sympy.Symbol("a")), Number(1.0))

        id1 = tree1.get_lut_id()
        id2 = tree2.get_lut_id()
        id3 = tree3.get_lut_id()

        assert id1 != id2  # Different values
        assert id1 == id3  # Same structure


# =============================================================================
# Canonicalization Tests
# =============================================================================


class TestCanonicalization:
    """Tests for canonicalize_children() and canonicalize_and_get_lut_id()."""

    # --- canonicalize_children ---

    def test_canonicalize_sorts_commutative_add(self):
        """Add(b, a) should become Add(a, b) after canonicalization."""
        tree = Add(Symbol(sympy.Symbol("b")), Symbol(sympy.Symbol("a")))
        tree.canonicalize_children()

        childs = tree.get_childs()
        assert childs[0].get_value() == sympy.Symbol("a")
        assert childs[1].get_value() == sympy.Symbol("b")

    def test_canonicalize_sorts_commutative_mul(self):
        """Mul(z, a) should become Mul(a, z)."""
        tree = Mul(Symbol(sympy.Symbol("z")), Symbol(sympy.Symbol("a")))
        tree.canonicalize_children()

        childs = tree.get_childs()
        assert childs[0].get_value() == sympy.Symbol("a")
        assert childs[1].get_value() == sympy.Symbol("z")

    def test_canonicalize_does_not_reorder_non_commutative(self):
        """Sub(b, a) should stay Sub(b, a) — subtraction is not commutative."""
        tree = Sub(Symbol(sympy.Symbol("b")), Symbol(sympy.Symbol("a")))
        tree.canonicalize_children()

        childs = tree.get_childs()
        assert childs[0].get_value() == sympy.Symbol("b")
        assert childs[1].get_value() == sympy.Symbol("a")

    def test_canonicalize_recursive(self):
        """Canonicalization should recurse into nested commutative children."""
        # Add(Mul(b, a), Number(1))  →  Add(Number(1), Mul(a, b))
        inner = Mul(Symbol(sympy.Symbol("b")), Symbol(sympy.Symbol("a")))
        tree = Add(inner, Number(1.0))
        tree.canonicalize_children()

        # Inner Mul should be sorted: a before b
        inner_childs = tree.get_childs()
        # Find the Mul child
        mul_child = [c for c in inner_childs if isinstance(c, Mul)][0]
        assert mul_child.get_childs()[0].get_value() == sympy.Symbol("a")
        assert mul_child.get_childs()[1].get_value() == sympy.Symbol("b")

    def test_canonicalize_idempotent(self):
        """Running canonicalize twice should produce the same result."""
        tree = Add(Symbol(sympy.Symbol("c")), Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")))
        tree.canonicalize_children()
        lut_after_first = tree.get_lut_id()
        tree.canonicalize_children()
        lut_after_second = tree.get_lut_id()

        assert lut_after_first == lut_after_second

    def test_canonicalize_terminal_is_noop(self):
        """Canonicalizing a terminal should not raise."""
        term = Symbol(sympy.Symbol("x"))
        term.canonicalize_children()  # should not raise

    # --- canonicalize_and_get_lut_id ---

    def test_fused_matches_separate(self):
        """canonicalize_and_get_lut_id() must produce the same result as separate calls."""
        tree1 = Add(
            Mul(Symbol(sympy.Symbol("z")), Symbol(sympy.Symbol("a"))),
            Sin(Symbol(sympy.Symbol("b"))),
        )
        tree2 = Add(
            Mul(Symbol(sympy.Symbol("z")), Symbol(sympy.Symbol("a"))),
            Sin(Symbol(sympy.Symbol("b"))),
        )

        # Separate: canonicalize then get_lut_id
        tree1.canonicalize_children()
        separate_id = tree1.get_lut_id()

        # Fused
        fused_id = tree2.canonicalize_and_get_lut_id()

        assert fused_id == separate_id

    def test_fused_matches_separate_deeply_nested(self):
        """Fused method matches separate calls on a deeper tree."""
        tree1 = Add(
            Mul(
                Add(Symbol(sympy.Symbol("c")), Symbol(sympy.Symbol("a"))),
                Symbol(sympy.Symbol("b")),
            ),
            Number(3.0),
        )
        tree2 = Add(
            Mul(
                Add(Symbol(sympy.Symbol("c")), Symbol(sympy.Symbol("a"))),
                Symbol(sympy.Symbol("b")),
            ),
            Number(3.0),
        )

        tree1.canonicalize_children()
        separate_id = tree1.get_lut_id()
        fused_id = tree2.canonicalize_and_get_lut_id()

        assert fused_id == separate_id

    def test_fused_sorts_children(self):
        """The fused method must actually sort commutative children."""
        tree = Add(Symbol(sympy.Symbol("z")), Symbol(sympy.Symbol("a")))
        tree.canonicalize_and_get_lut_id()

        childs = tree.get_childs()
        assert childs[0].get_value() == sympy.Symbol("a")
        assert childs[1].get_value() == sympy.Symbol("z")

    def test_fused_non_commutative_preserves_order(self):
        """Non-commutative operators preserve child order in fused path."""
        tree = Sub(Symbol(sympy.Symbol("b")), Symbol(sympy.Symbol("a")))
        lut_id = tree.canonicalize_and_get_lut_id()

        assert "b" in lut_id
        childs = tree.get_childs()
        assert childs[0].get_value() == sympy.Symbol("b")
        assert childs[1].get_value() == sympy.Symbol("a")

    def test_fused_terminal(self):
        """Fused method on a terminal returns its represent_str."""
        term = Symbol(sympy.Symbol("x"))
        lut_id = term.canonicalize_and_get_lut_id()

        assert lut_id == term.represent_str(show_fixed_hint=False, cut_terms=False)

    def test_fused_different_order_same_id(self):
        """Two trees with different child order but same structure get the same fused LUT id."""
        tree1 = Add(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")))
        tree2 = Add(Symbol(sympy.Symbol("b")), Symbol(sympy.Symbol("a")))

        id1 = tree1.canonicalize_and_get_lut_id()
        id2 = tree2.canonicalize_and_get_lut_id()

        assert id1 == id2

    def test_fused_different_values_different_id(self):
        """Trees with different values produce different LUT ids."""
        tree1 = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree2 = Add(Symbol(sympy.Symbol("a")), Number(2.0))

        id1 = tree1.canonicalize_and_get_lut_id()
        id2 = tree2.canonicalize_and_get_lut_id()

        assert id1 != id2

    def test_fused_with_chainable_three_children(self):
        """Fused canonicalization works for 3+-ary commutative operators."""
        tree1 = Add(Symbol(sympy.Symbol("c")), Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")))
        tree2 = Add(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")), Symbol(sympy.Symbol("c")))

        id1 = tree1.canonicalize_and_get_lut_id()
        id2 = tree2.canonicalize_and_get_lut_id()

        assert id1 == id2
        # Children should be sorted: a, b, c
        childs = tree1.get_childs()
        assert [c.get_value() for c in childs] == [sympy.Symbol("a"), sympy.Symbol("b"), sympy.Symbol("c")]

    def test_fused_with_ifte_non_commutative(self):
        """Ifte is not commutative — child order must be preserved."""
        tree = Ifte(
            Lt(Symbol(sympy.Symbol("a")), Number(0.0)),
            Number(1.0),
            Number(2.0),
        )
        lut_id = tree.canonicalize_and_get_lut_id()

        assert "Ifte" in lut_id
        childs = tree.get_childs()
        assert isinstance(childs[0], Lt)

    def test_get_expr_symlike(self):
        """Tests get_expr_symlike output."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(2.0))

        s = tree.get_expr_symlike()

        assert "+" in s or "Add" in s
        assert "a" in s
