"""Tests for the intrinsic Zhang-Shasha tree edit distance implementation.

Tests verify:
1. Distance correctness for identical, single-node, and multi-node trees
2. Structural vs. full vs. structural_plus_leaf_diff modes
3. Insert / delete / rename operations in the edit mapping
4. pairwise_ted_matrix utility
5. TedConfig validation
6. Backward-compatible eval_parsimony integration
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pytest

from plagih.tree_complexity.tree_edit_distance import (
    TedConfig,
    TedResult,
    zhang_shasha_ted,
)
from plagih.trees import (
    Add,
    Boolean,
    Cos,
    Div,
    Ifte,
    Lt,
    Mul,
    Number,
    Sin,
    Sub,
    Symbol,
    compute_ted,
    eval_parsimony,
    pairwise_ted_matrix,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_tree(root_cls, *children):
    """Build a simple plagih tree from class + children (Node instances)."""
    node = root_cls(*children)
    node.repair_all()
    return node


def _sym(name: str) -> Symbol:
    import sympy

    return Symbol(sympy.Symbol(name, real=True))


def _num(val: float) -> Number:
    return Number(val)


def _bool(val: bool) -> Boolean:
    return Boolean(val)


# =============================================================================
# Basic distance tests
# =============================================================================


class TestBasicDistance:
    """Core distance computation correctness."""

    def test_identical_single_node(self):
        """Same single-node tree → distance 0."""
        t1 = _num(42)
        t2 = _num(42)
        r = compute_ted(t1, t2)
        assert r.distance == 0.0

    def test_identical_tree(self):
        """Structurally and value-identical tree → distance 0."""
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Add, _sym("x"), _num(1))
        r = compute_ted(t1, t2)
        assert r.distance == 0.0

    def test_single_rename(self):
        """Replacing one leaf value → distance 1 (full mode)."""
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Add, _sym("x"), _num(2))
        r = compute_ted(t1, t2)
        assert r.distance == 1.0

    def test_single_insert(self):
        """One tree is a subtree of the other (one extra node).

        TED insertion means adding a node between parent and child.
        Number(5) → Sin(Number(5)):  insert Sin as new parent of Number → cost 1.
        """
        t1 = _num(5)
        t2 = _make_tree(Sin, _num(5))
        r = compute_ted(t1, t2)
        assert r.distance == 1.0

    def test_single_delete(self):
        """Reverse of insert: extra node removed.

        Sin(Number(5)) → Number(5):  delete Sin, its child Number is kept → cost 1.
        """
        t1 = _make_tree(Sin, _num(5))
        t2 = _num(5)
        r = compute_ted(t1, t2)
        assert r.distance == 1.0

    def test_different_operators(self):
        """Same structure, different operator → 1 rename."""
        t1 = _make_tree(Add, _sym("x"), _sym("y"))
        t2 = _make_tree(Mul, _sym("x"), _sym("y"))
        r = compute_ted(t1, t2)
        assert r.distance == 1.0

    def test_completely_different(self):
        """Completely different trees → distance = sum of sizes."""
        t1 = _num(1)
        t2 = _make_tree(Add, _sym("x"), _sym("y"))
        r = compute_ted(t1, t2)
        # t1 has 1 node, t2 has 3 nodes
        # Rename Number→Add (1) + insert Symbol (1) + insert Symbol (1) = 3
        # OR: delete Number (1) + insert Add (1) + insert Sym (1) + insert Sym (1) = 4
        # Optimal: 3 (rename root + 2 inserts)
        assert r.distance == 3.0

    def test_deeper_tree(self):
        """Deeper nested tree distance."""
        # Add(Mul(x, y), z) vs Add(x, z)
        t1 = _make_tree(Add, _make_tree(Mul, _sym("x"), _sym("y")), _sym("z"))
        t2 = _make_tree(Add, _sym("x"), _sym("z"))
        r = compute_ted(t1, t2)
        # Mul → x (rename Mul→Symbol cost 1) + delete y (cost 1) = 2
        assert r.distance == 2.0

    def test_symmetric_distance(self):
        """TED is symmetric: d(a,b) == d(b,a)."""
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Mul, _sym("y"), _make_tree(Sin, _num(3)))
        r1 = compute_ted(t1, t2)
        r2 = compute_ted(t2, t1)
        assert r1.distance == r2.distance


# =============================================================================
# Mode tests
# =============================================================================


class TestModes:
    """Test the three label-comparison modes."""

    def test_structural_ignores_values(self):
        """Structural mode: same structure, different values → distance 0."""
        t1 = _make_tree(Add, _num(1), _num(2))
        t2 = _make_tree(Add, _num(99), _num(100))
        r = compute_ted(t1, t2, TedConfig(mode="structural"))
        assert r.distance == 0.0

    def test_structural_ignores_symbol_names(self):
        """Structural mode: different symbol names → distance 0."""
        t1 = _make_tree(Add, _sym("x"), _sym("y"))
        t2 = _make_tree(Add, _sym("a"), _sym("b"))
        r = compute_ted(t1, t2, TedConfig(mode="structural"))
        assert r.distance == 0.0

    def test_full_detects_value_diff(self):
        """Full mode: same structure, different values → distance > 0."""
        t1 = _make_tree(Add, _num(1), _num(2))
        t2 = _make_tree(Add, _num(99), _num(100))
        r = compute_ted(t1, t2, TedConfig(mode="full"))
        assert r.distance == 2.0  # two leaf renames

    def test_full_detects_symbol_name_diff(self):
        """Full mode: different symbol names → distance > 0."""
        t1 = _make_tree(Add, _sym("x"), _sym("y"))
        t2 = _make_tree(Add, _sym("a"), _sym("b"))
        r = compute_ted(t1, t2, TedConfig(mode="full"))
        assert r.distance == 2.0

    def test_structural_plus_leaf_diff(self):
        """structural_plus_leaf_diff: structural distance 0, leaf_diff > 0."""
        t1 = _make_tree(Add, _num(1), _sym("x"))
        t2 = _make_tree(Add, _num(99), _sym("y"))
        cfg = TedConfig(mode="structural_plus_leaf_diff")
        r = compute_ted(t1, t2, cfg)
        assert r.distance == 0.0  # structurally identical
        assert r.leaf_diff_count == 2  # both leaves differ

    def test_structural_plus_leaf_diff_partial(self):
        """structural_plus_leaf_diff: one leaf same, one different."""
        t1 = _make_tree(Add, _num(5), _sym("x"))
        t2 = _make_tree(Add, _num(5), _sym("y"))
        cfg = TedConfig(mode="structural_plus_leaf_diff")
        r = compute_ted(t1, t2, cfg)
        assert r.distance == 0.0
        assert r.leaf_diff_count == 1  # only Symbol differs

    def test_leaf_diff_none_in_non_structural_mode(self):
        """leaf_diff_count should be None when mode is not structural_plus_leaf_diff."""
        t1 = _make_tree(Add, _num(1), _num(2))
        t2 = _make_tree(Add, _num(1), _num(2))
        r_full = compute_ted(t1, t2, TedConfig(mode="full"))
        r_struct = compute_ted(t1, t2, TedConfig(mode="structural"))
        assert r_full.leaf_diff_count is None
        assert r_struct.leaf_diff_count is None


# =============================================================================
# Mapping tests
# =============================================================================


class TestMapping:
    """Test edit mapping extraction."""

    def test_identical_tree_mapping(self):
        """Identical trees: all nodes mapped 1:1, no None entries."""
        t1 = _make_tree(Add, _num(1), _sym("x"))
        t2 = _make_tree(Add, _num(1), _sym("x"))
        r = compute_ted(t1, t2)
        assert r.distance == 0.0
        # All mapping entries should be (node, node), no None
        assert all(a is not None and b is not None for a, b in r.mapping)
        assert len(r.mapping) == 3  # Add + Number + Symbol

    def test_mapping_has_insertion(self):
        """Mapping includes (None, node) for insertions."""
        t1 = _num(1)
        t2 = _make_tree(Add, _num(1), _num(2))
        r = compute_ted(t1, t2)
        insertions = [(a, b) for a, b in r.mapping if a is None]
        assert len(insertions) > 0

    def test_mapping_has_deletion(self):
        """Mapping includes (node, None) for deletions."""
        t1 = _make_tree(Add, _num(1), _num(2))
        t2 = _num(1)
        r = compute_ted(t1, t2)
        deletions = [(a, b) for a, b in r.mapping if b is None]
        assert len(deletions) > 0

    def test_mapping_cost_equals_distance(self):
        """Sum of mapping costs should equal the reported distance."""
        t1 = _make_tree(Add, _make_tree(Mul, _sym("x"), _num(2)), _sym("y"))
        t2 = _make_tree(Add, _sym("x"), _make_tree(Sin, _num(3)))
        cfg = TedConfig(mode="full")
        r = compute_ted(t1, t2, cfg)

        # Recompute cost from mapping
        cost = 0.0
        for a, b in r.mapping:
            if a is None:
                cost += cfg.insert_cost
            elif b is None:
                cost += cfg.delete_cost
            else:
                label_a = type(a) if not a.is_term() else (type(a), a.get_value())
                label_b = type(b) if not b.is_term() else (type(b), b.get_value())
                if label_a != label_b:
                    cost += 1.0
        assert cost == pytest.approx(r.distance)


# =============================================================================
# TedConfig validation
# =============================================================================


class TestTedConfig:
    """Test configuration validation and custom costs."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="TedConfig.mode"):
            TedConfig(mode="invalid")

    def test_custom_costs(self):
        """Custom insert/delete costs are respected."""
        t1 = _num(1)
        t2 = _make_tree(Add, _num(1), _num(2))
        cfg = TedConfig(mode="full", insert_cost=10.0, delete_cost=5.0)
        r = compute_ted(t1, t2, cfg)
        # Should use the custom costs
        assert r.distance > 0


# =============================================================================
# Pairwise matrix
# =============================================================================


class TestPairwiseMatrix:
    """Test pairwise_ted_matrix utility."""

    def test_diagonal_zero(self):
        trees = [
            _make_tree(Add, _num(1), _num(2)),
            _make_tree(Mul, _num(3), _num(4)),
            _num(5),
        ]
        mat = pairwise_ted_matrix(trees)
        assert mat.shape == (3, 3)
        for i in range(3):
            assert mat[i, i] == 0.0

    def test_symmetric(self):
        trees = [
            _make_tree(Add, _num(1), _num(2)),
            _make_tree(Mul, _num(3), _num(4)),
            _make_tree(Sin, _sym("x")),
        ]
        mat = pairwise_ted_matrix(trees)
        np.testing.assert_array_equal(mat, mat.T)

    def test_single_tree(self):
        mat = pairwise_ted_matrix([_num(1)])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    def test_empty_list(self):
        mat = pairwise_ted_matrix([])
        assert mat.shape == (0, 0)


# =============================================================================
# Integration: eval_parsimony
# =============================================================================


class TestEvalParsimonyIntegration:
    """Test that eval_parsimony still works with the new TED backend."""

    def test_edit_distance_identical(self):
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Add, _sym("x"), _num(1))
        d = eval_parsimony(t1, "tree_edit_distance", origin_tree=t2)
        assert d == 0

    def test_edit_distance_different(self):
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Mul, _sym("y"), _num(2))
        d = eval_parsimony(t1, "tree_edit_distance", origin_tree=t2)
        # eval_parsimony uses structural mode, so only operator change counts
        assert d >= 1

    def test_node_count_still_works(self):
        """Verify other complexity measures are unaffected."""
        t = _make_tree(Add, _sym("x"), _num(1))
        raw = eval_parsimony(t, "tree_node_count_raw")
        fair = eval_parsimony(t, "tree_node_count_fair")
        assert raw == 3
        assert fair == 3


# =============================================================================
# Edge cases & regression
# =============================================================================


class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_single_leaf_vs_single_leaf_same_type_diff_value(self):
        """Two Number terminals with different values (full mode)."""
        t1 = _num(1)
        t2 = _num(2)
        r = compute_ted(t1, t2)
        assert r.distance == 1.0

    def test_single_leaf_vs_single_leaf_diff_type(self):
        """Number vs Symbol."""
        t1 = _num(1)
        t2 = _sym("x")
        r = compute_ted(t1, t2)
        assert r.distance == 1.0  # rename

    def test_boolean_terminal(self):
        """Boolean terminals."""
        t1 = _bool(True)
        t2 = _bool(False)
        r = compute_ted(t1, t2)
        assert r.distance == 1.0

    def test_chain_operator(self):
        """Chained Add with 3 children."""
        t1 = Add(_num(1), _num(2), _num(3))
        t1.repair_all()
        t2 = Add(_num(1), _num(2), _num(4))
        t2.repair_all()
        r = compute_ted(t1, t2)
        assert r.distance == 1.0  # only Number(3) → Number(4)

    def test_structural_chain_operator(self):
        """Chained Add: structurally identical regardless of values."""
        t1 = Add(_num(1), _num(2), _num(3))
        t1.repair_all()
        t2 = Add(_num(10), _num(20), _num(30))
        t2.repair_all()
        r = compute_ted(t1, t2, TedConfig(mode="structural"))
        assert r.distance == 0.0

    def test_known_example_abc(self):
        """Known example: structural insert.

        Tree1: Add(Number(1), Number(2))       — 3 nodes
        Tree2: Add(Sin(Number(3)), Number(2))  — 4 nodes

        Structural mode: Number types match freely regardless of value.
        Sin is inserted between Add and its left child → cost 1.
        """
        t1 = _make_tree(Add, _num(1), _num(2))
        t2 = _make_tree(Add, _make_tree(Sin, _num(3)), _num(2))
        r_struct = compute_ted(t1, t2, TedConfig(mode="structural"))
        assert r_struct.distance == 1.0

        # Full mode: additionally Number(1) → Number(3) is a rename (cost 1)
        r_full = compute_ted(t1, t2, TedConfig(mode="full"))
        assert r_full.distance == 2.0


# =============================================================================
# Node.compute_ted method (delegation test)
# =============================================================================


class TestNodeMethod:
    """Verify that Node.compute_ted() delegates correctly."""

    def test_method_matches_free_function(self):
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Mul, _sym("y"), _num(2))
        r_method = t1.compute_ted(t2)
        r_free = compute_ted(t1, t2)
        assert r_method.distance == r_free.distance

    def test_method_accepts_config(self):
        t1 = _make_tree(Add, _num(1), _num(2))
        t2 = _make_tree(Add, _num(99), _num(100))
        r = t1.compute_ted(t2, TedConfig(mode="structural"))
        assert r.distance == 0.0

    def test_method_returns_node_references(self):
        """Mapping entries should reference the actual Node objects."""
        t1 = _make_tree(Add, _sym("x"), _num(1))
        t2 = _make_tree(Add, _sym("x"), _num(1))
        r = t1.compute_ted(t2)
        for a, b in r.mapping:
            if a is not None:
                assert isinstance(a, (Add, Number, Symbol))
            if b is not None:
                assert isinstance(b, (Add, Number, Symbol))


# =============================================================================
# Traversal order methods
# =============================================================================


class TestTraversalOrders:
    """Test to_postorder(), to_preorder(), str_as_list(order=...)."""

    def test_postorder_leaf(self):
        """Single leaf: postorder = [leaf]."""
        t = _num(5)
        po = t.to_postorder()
        assert len(po) == 1
        assert po[0] is t

    def test_postorder_simple_tree(self):
        """Add(x, y) → postorder = [x, y, Add]."""
        x, y = _sym("x"), _sym("y")
        t = _make_tree(Add, x, y)
        po = t.to_postorder()
        assert len(po) == 3
        # Children before parent
        assert isinstance(po[0], Symbol)
        assert isinstance(po[1], Symbol)
        assert isinstance(po[2], Add)
        # Root is last
        assert po[-1] is t

    def test_postorder_nested(self):
        """Add(Mul(x, y), z) → postorder = [x, y, Mul, z, Add]."""
        x, y, z = _sym("x"), _sym("y"), _sym("z")
        mul = _make_tree(Mul, x, y)
        t = _make_tree(Add, mul, z)
        po = t.to_postorder()
        assert len(po) == 5
        # Deepest leaves first
        assert po[0].is_term()  # x
        assert po[1].is_term()  # y
        assert isinstance(po[2], Mul)
        assert po[3].is_term()  # z
        assert isinstance(po[4], Add)

    def test_preorder_simple_tree(self):
        """Add(x, y) → preorder = [Add, x, y]."""
        x, y = _sym("x"), _sym("y")
        t = _make_tree(Add, x, y)
        pre = t.to_preorder()
        assert len(pre) == 3
        assert isinstance(pre[0], Add)
        assert pre[0] is t
        assert isinstance(pre[1], Symbol)
        assert isinstance(pre[2], Symbol)

    def test_preorder_nested(self):
        """Add(Mul(x, y), z) → preorder = [Add, Mul, x, y, z]."""
        x, y, z = _sym("x"), _sym("y"), _sym("z")
        mul = _make_tree(Mul, x, y)
        t = _make_tree(Add, mul, z)
        pre = t.to_preorder()
        assert len(pre) == 5
        assert isinstance(pre[0], Add)
        assert isinstance(pre[1], Mul)

    def test_postorder_count_matches_nodecount(self):
        t = _make_tree(Add, _make_tree(Sin, _num(3)), _sym("x"))
        assert len(t.to_postorder()) == t.len_nodecount_raw()
        assert len(t.to_preorder()) == t.len_nodecount_raw()

    def test_str_as_list_default_is_preorder(self):
        """Default str_as_list is preorder: [Add, [x], [1]]."""
        t = _make_tree(Add, _sym("x"), _num(1))
        s = t.str_as_list()
        assert s.startswith("[Add")

    def test_str_as_list_postorder(self):
        """Postorder str_as_list: [[x], [1], Add]."""
        t = _make_tree(Add, _sym("x"), _num(1))
        s = t.str_as_list(order="post")
        assert s.endswith("Add]")
        assert not s.startswith("[Add")

    def test_str_as_list_preorder_explicit(self):
        """Explicit pre order matches default."""
        t = _make_tree(Add, _sym("x"), _num(1))
        assert t.str_as_list(order="pre") == t.str_as_list()
