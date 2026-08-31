"""
Tests for Population Merge Module

Tests the "One-evaluation-tree" strategy for merging multiple trees
from a population into a unified evaluation graph.
"""

import pytest
import sympy

from plagih.population_merge import (
    MergedEvaluationGraph,
    MergedNode,
    TrunkInfo,
    analyze_population_sharing,
    build_one_evaluation_tree,
    find_trunks,
    get_expressions_by_depth,
    suggest_origin_templates,
    suggest_origin_trees,
    trunk_to_origin_tree,
)
from plagih.trees import Abs, Add, Mul, Number, Symbol

# =============================================================================
# Helper Functions
# =============================================================================


def make_symbol(name: str) -> Symbol:
    """Create a Symbol terminal node."""
    return Symbol(sympy.Symbol(name))


def make_number(val: float) -> Number:
    """Create a Number terminal node."""
    return Number(val)


# =============================================================================
# Test MergedNode and MergedEvaluationGraph Data Structures
# =============================================================================


class TestMergedNode:
    """Tests for MergedNode dataclass."""

    def test_create_merged_node(self):
        """Test basic MergedNode creation."""
        expr = sympy.Symbol("x")
        node = MergedNode(node_id="n0", sympy_expr=expr, depth=0, node_type="terminal")
        assert node.node_id == "n0"
        assert node.sympy_expr == expr
        assert node.depth == 0
        assert node.node_type == "terminal"
        assert not node.is_root
        assert node.original_nodes == []

    def test_merged_node_repr(self):
        """Test string representation of MergedNode."""
        expr = sympy.Symbol("x")
        node = MergedNode(node_id="n0", sympy_expr=expr)
        assert "n0" in repr(node)
        assert "x" in repr(node)


class TestMergedEvaluationGraph:
    """Tests for MergedEvaluationGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = MergedEvaluationGraph()
        assert len(graph.nodes) == 0
        assert len(graph.root_ids) == 0
        assert graph.tree_count == 0

    def test_get_or_create_node_new(self):
        """Test creating a new node."""
        graph = MergedEvaluationGraph()
        expr = sympy.Symbol("x")

        node_id = graph.get_or_create_node(sympy_expr=expr, depth=0, node_type="terminal")

        assert node_id == "n0"
        assert expr in graph.expr_to_id
        assert node_id in graph.nodes

    def test_get_or_create_node_existing(self):
        """Test that duplicate expressions return existing node."""
        graph = MergedEvaluationGraph()
        expr = sympy.Symbol("x")

        node_id1 = graph.get_or_create_node(sympy_expr=expr, depth=0, node_type="terminal")
        node_id2 = graph.get_or_create_node(sympy_expr=expr, depth=0, node_type="terminal")

        assert node_id1 == node_id2
        assert len(graph.nodes) == 1

    def test_mark_as_root(self):
        """Test marking a node as root."""
        graph = MergedEvaluationGraph()
        expr = sympy.Symbol("x")

        node_id = graph.get_or_create_node(sympy_expr=expr, depth=0, node_type="terminal")
        graph.mark_as_root(node_id, tree_index=0)

        assert graph.nodes[node_id].is_root
        assert 0 in graph.nodes[node_id].root_of_trees
        assert node_id in graph.root_ids

    def test_statistics(self):
        """Test statistics calculation."""
        graph = MergedEvaluationGraph()
        graph.tree_count = 2

        # Add some nodes
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")

        n1 = graph.get_or_create_node(x, 0, "terminal")
        n2 = graph.get_or_create_node(y, 0, "terminal")
        n3 = graph.get_or_create_node(x + y, 1, "operator", child_ids=[n1, n2])

        # Simulate both trees using these nodes
        graph.add_original_node_mapping(n1, 0, None)
        graph.add_original_node_mapping(n1, 1, None)
        graph.add_original_node_mapping(n2, 0, None)
        graph.add_original_node_mapping(n3, 0, None)

        stats = graph.get_statistics()

        assert stats["total_nodes"] == 3
        assert stats["terminal_nodes"] == 2
        assert stats["operator_nodes"] == 1
        assert stats["tree_count"] == 2
        assert stats["shared_nodes"] == 1  # x is shared

    def test_to_string(self):
        """Test string representation."""
        graph = MergedEvaluationGraph()
        graph.tree_count = 1

        x = sympy.Symbol("x")
        node_id = graph.get_or_create_node(x, 0, "terminal")
        graph.mark_as_root(node_id, 0)

        output = graph.to_string()

        assert "MERGED EVALUATION GRAPH" in output
        assert "Trees merged: 1" in output
        assert "x" in output

    def test_to_evaluation_order(self):
        """Test evaluation order (children before parents)."""
        graph = MergedEvaluationGraph()

        x = sympy.Symbol("x")
        y = sympy.Symbol("y")

        n1 = graph.get_or_create_node(x, 0, "terminal")
        n2 = graph.get_or_create_node(y, 0, "terminal")
        n3 = graph.get_or_create_node(x + y, 1, "operator", child_ids=[n1, n2])

        order = graph.to_evaluation_order()

        # Children should come before parent
        assert order.index(n1) < order.index(n3)
        assert order.index(n2) < order.index(n3)

    def test_to_graphviz_dot(self):
        """Test Graphviz DOT output."""
        graph = MergedEvaluationGraph()

        x = sympy.Symbol("x")
        node_id = graph.get_or_create_node(x, 0, "terminal")

        dot = graph.to_graphviz_dot()

        assert "digraph" in dot
        assert "n0" in dot


# =============================================================================
# Test Expression Collection by Depth
# =============================================================================


class TestGetExpressionsByDepth:
    """Tests for get_expressions_by_depth function."""

    def test_single_terminal(self):
        """Test with a single terminal node."""
        tree = make_symbol("x")
        layers = get_expressions_by_depth(tree)

        assert len(layers) == 1
        assert len(layers[0]) == 1
        assert layers[0][0][0] == sympy.Symbol("x")

    def test_simple_binary_op(self):
        """Test with a simple binary operation."""
        # a + b
        tree = Add(make_symbol("a"), make_symbol("b"))
        layers = get_expressions_by_depth(tree)

        assert len(layers) == 2
        # Depth 0: terminals
        assert len(layers[0]) == 2
        # Depth 1: the Add
        assert len(layers[1]) == 1

    def test_nested_operations(self):
        """Test with nested operations: (a + b) * c"""
        tree = Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c"))
        layers = get_expressions_by_depth(tree)

        # Depth 0: a, b, c (3 terminals)
        # Depth 1: a + b
        # Depth 2: (a + b) * c
        assert len(layers) == 3
        assert len(layers[0]) == 3  # a, b, c


# =============================================================================
# Test Building One-Evaluation-Tree
# =============================================================================


class TestBuildOneEvaluationTree:
    """Tests for the main build_one_evaluation_tree function."""

    def test_single_tree(self):
        """Test merging a single tree."""
        tree = Add(make_symbol("a"), make_number(1))

        graph = build_one_evaluation_tree([tree])

        assert graph.tree_count == 1
        assert len(graph.root_ids) == 1
        stats = graph.get_statistics()
        assert stats["total_nodes"] >= 3  # a, 1, a+1

    def test_identical_trees(self):
        """Test that identical trees share all nodes."""
        tree1 = Add(make_symbol("a"), make_number(1))
        tree2 = Add(make_symbol("a"), make_number(1))

        graph = build_one_evaluation_tree([tree1, tree2])

        # Should have same nodes despite two trees
        stats = graph.get_statistics()
        assert graph.tree_count == 2
        # Both trees should map to the same root
        assert len(graph.root_ids) == 1

    def test_shared_subexpression(self):
        """Test that shared sub-expressions are deduplicated."""
        # Tree 1: a + b
        tree1 = Add(make_symbol("a"), make_symbol("b"))

        # Tree 2: (a + b) * c - shares (a + b)
        tree2 = Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c"))

        graph = build_one_evaluation_tree([tree1, tree2])

        # The (a + b) expression should be shared
        stats = graph.get_statistics()
        assert stats["shared_nodes"] >= 1  # At least a+b is shared

        # Savings should be positive
        assert stats["savings_percent"] > 0

    def test_different_trees_share_terminals(self):
        """Test that different trees share terminal nodes."""
        # Tree 1: a + 1
        tree1 = Add(make_symbol("a"), make_number(1))

        # Tree 2: a * 2
        tree2 = Mul(make_symbol("a"), make_number(2))

        graph = build_one_evaluation_tree([tree1, tree2])

        # 'a' should be shared
        stats = graph.get_statistics()
        assert stats["shared_nodes"] >= 1

    def test_normalize_strategy_sympify(self):
        """Test that normalization works (a+b == b+a)."""
        # These should be recognized as the same expression
        tree1 = Add(make_symbol("a"), make_symbol("b"))
        tree2 = Add(make_symbol("b"), make_symbol("a"))

        graph = build_one_evaluation_tree([tree1, tree2], normalize_strategy="sympify")

        # Both should map to the same root (a+b is commutative)
        assert len(graph.root_ids) == 1

    def test_empty_population(self):
        """Test with empty population."""
        graph = build_one_evaluation_tree([])

        assert graph.tree_count == 0
        assert len(graph.nodes) == 0


# =============================================================================
# Test Population Sharing Analysis
# =============================================================================


class TestAnalyzePopulationSharing:
    """Tests for analyze_population_sharing function."""

    def test_analyze_sharing(self):
        """Test the sharing analysis function."""
        tree1 = Add(make_symbol("a"), make_symbol("b"))
        tree2 = Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c"))

        stats = analyze_population_sharing([tree1, tree2])

        assert "total_nodes" in stats
        assert "shared_nodes" in stats
        assert "savings_percent" in stats
        assert "most_shared_expressions" in stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with more complex scenarios."""

    def test_complex_population(self):
        """Test with a more realistic population."""
        # Create several related trees
        population = [
            # Tree 0: a + b
            Add(make_symbol("a"), make_symbol("b")),
            # Tree 1: (a + b) * c
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),
            # Tree 2: (a + b) + 1
            Add(Add(make_symbol("a"), make_symbol("b")), make_number(1)),
            # Tree 3: a * 2
            Mul(make_symbol("a"), make_number(2)),
            # Tree 4: |a + b|
            Abs(Add(make_symbol("a"), make_symbol("b"))),
        ]

        graph = build_one_evaluation_tree(population)

        # Basic sanity checks
        assert graph.tree_count == 5
        assert len(graph.root_ids) >= 1  # Some trees might have identical roots

        # Should have significant sharing
        stats = graph.get_statistics()
        assert stats["shared_nodes"] >= 2  # a, b, and (a+b) should be shared

        # String output should work
        output = graph.to_string()
        assert "MERGED EVALUATION GRAPH" in output

        # DOT output should work
        dot = graph.to_graphviz_dot()
        assert "digraph" in dot

    def test_evaluation_order_is_valid(self):
        """Test that evaluation order respects dependencies."""
        tree = Mul(Add(make_symbol("a"), make_number(1)), Add(make_symbol("b"), make_number(2)))

        graph = build_one_evaluation_tree([tree])
        order = graph.to_evaluation_order()

        # For each node, all children should come before it
        for node_id in order:
            node = graph.nodes[node_id]
            node_idx = order.index(node_id)

            for child_id in node.child_ids:
                child_idx = order.index(child_id)
                assert child_idx < node_idx, f"Child {child_id} should come before parent {node_id}"


# =============================================================================
# Trunk Analysis (Targeted Optimization §3.5)
# =============================================================================


class TestFindTrunks:
    """Tests for find_trunks()."""

    def _make_population(self):
        """3 trees sharing the (a + b) trunk, 1 unrelated."""
        return [
            Add(make_symbol("a"), make_symbol("b")),  # a + b
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),  # (a+b)*c
            Abs(Add(make_symbol("a"), make_symbol("b"))),  # |a+b|
            Mul(make_symbol("a"), make_number(2)),  # a*2 — no trunk
        ]

    def test_finds_shared_trunk(self):
        graph = build_one_evaluation_tree(self._make_population())
        trunks = find_trunks(graph)

        assert trunks, "Expected at least one trunk"
        best = trunks[0]
        assert best.n_trees == 3
        assert set(best.tree_indices) == {0, 1, 2}
        assert best.subtree_size == 3  # Add + a + b
        assert "a" in best.expr and "b" in best.expr

    def test_min_trees_filter(self):
        graph = build_one_evaluation_tree(self._make_population())
        trunks = find_trunks(graph, min_trees=4)
        assert trunks == []

    def test_min_size_filter(self):
        graph = build_one_evaluation_tree(self._make_population())
        trunks = find_trunks(graph, min_size=10)
        assert trunks == []

    def test_no_sharing_no_trunks(self):
        population = [
            Add(make_symbol("a"), make_number(1)),
            Mul(make_symbol("b"), make_number(2)),
        ]
        graph = build_one_evaluation_tree(population)
        assert find_trunks(graph) == []

    def test_nested_trunk_excluded(self):
        """Inner trunk with same coverage must be dropped when nested."""
        # Both trees share the whole (a+b)*c — inner (a+b) has same coverage
        population = [
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),
        ]
        graph = build_one_evaluation_tree(population)

        trunks = find_trunks(graph)
        assert len(trunks) == 1  # only the outer trunk survives

        all_trunks = find_trunks(graph, exclude_nested=False)
        assert len(all_trunks) == 2  # outer + inner

    def test_ranked_by_score(self):
        graph = build_one_evaluation_tree(self._make_population())
        trunks = find_trunks(graph, exclude_nested=False)
        scores = [t.score for t in trunks]
        assert scores == sorted(scores, reverse=True)


class TestSuggestOriginTrees:
    """Tests for suggest_origin_trees()."""

    def test_returns_tree_copies(self):
        population = [
            Add(make_symbol("a"), make_symbol("b")),
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),
        ]
        suggestions = suggest_origin_trees(population)

        assert suggestions
        tree, info = suggestions[0]
        assert isinstance(info, TrunkInfo)
        # Copy — not one of the original node objects
        original_ids = set()
        for t in population:
            original_ids.add(id(t))
            for child in t.get_childs():
                original_ids.add(id(child))
        assert id(tree) not in original_ids

    def test_inputs_not_modified(self):
        population = [
            Add(make_symbol("a"), make_symbol("b")),
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),
        ]
        before = [str(t) for t in population]
        suggest_origin_trees(population)
        assert [str(t) for t in population] == before

    def test_top_n_limit(self):
        population = [
            Mul(Add(make_symbol("a"), make_symbol("b")), Add(make_symbol("c"), make_number(1))),
            Mul(Add(make_symbol("a"), make_symbol("b")), Add(make_symbol("c"), make_number(1))),
        ]
        suggestions = suggest_origin_trees(population, top_n=1, min_trees=2)
        assert len(suggestions) <= 1

    def test_empty_population_like_input(self):
        population = [
            Add(make_symbol("a"), make_number(1)),
            Mul(make_symbol("b"), make_number(2)),
        ]
        assert suggest_origin_trees(population) == []


class TestTrunkToOriginTree:
    """Tests for trunk_to_origin_tree() / suggest_origin_templates()."""

    def test_operators_frozen_terminals_mutable(self):
        trunk = Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c"))
        template = trunk_to_origin_tree(trunk)

        assert template.is_fix
        add_node = template.get_childs()[0]
        assert add_node.is_fix
        # Terminals stay mutable
        for term in [*add_node.get_childs(), template.get_childs()[1]]:
            assert not term.is_fix

    def test_fully_frozen(self):
        trunk = Add(make_symbol("a"), make_symbol("b"))
        template = trunk_to_origin_tree(trunk, mutable_terminals=False)

        assert template.is_fix
        assert all(cc.is_fix for cc in template.get_childs())

    def test_input_not_modified(self):
        trunk = Add(make_symbol("a"), make_symbol("b"))
        trunk.repair_all()
        template = trunk_to_origin_tree(trunk)

        assert id(template) != id(trunk)
        assert not trunk.is_fix

    def test_suggest_origin_templates(self):
        population = [
            Add(make_symbol("a"), make_symbol("b")),
            Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c")),
        ]
        templates = suggest_origin_templates(population)

        assert templates
        template, info = templates[0]
        assert isinstance(info, TrunkInfo)
        assert template.is_fix

    def test_evolution_preserves_skeleton(self):
        """Evolution(origin_tree=template) must keep the frozen skeleton."""
        from plagih.trees import Sub
        from plagih.trees._evolution import Evolution
        from plagih.trees._nodes import Ifte, Le

        trunk = Mul(Add(make_symbol("a"), make_symbol("b")), make_symbol("c"))
        template = trunk_to_origin_tree(trunk)

        ev = Evolution(
            symbol_list=[sympy.Symbol("a"), sympy.Symbol("b"), sympy.Symbol("c")],
            operators={Add: 1, Mul: 1, Sub: 1, Ifte: 1, Le: 1},
            nodes_max=30,
            depth_max=6,
            origin_tree=template,
        )

        for _ in range(5):
            tree = ev.evolve_new_tree_depth(float, depth_goal=2, p_term=0.3)
            # Skeleton: root Mul with fixed Add child survives
            assert type(tree) is Mul
            assert tree.is_fix
            assert type(tree.get_childs()[0]) is Add
            assert tree.get_childs()[0].is_fix


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
