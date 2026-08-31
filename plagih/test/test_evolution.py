"""
Tests for the Evolution class.

Tests verify:
1. Random tree creation
2. Mutation operations (point, branch, filter)
3. Crossover operations
4. Tree pruning
5. Constraint enforcement (depth, nodes)
"""

import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import copy

import numpy as np
import pytest
import sympy

from plagih.trees import (
    Abs,
    Add,
    And,
    Boolean,
    Candidate,
    Cos,
    Div,
    Evolution,
    Ifte,
    Lt,
    Mul,
    Node,
    NodeSelect,
    Number,
    Scale,
    Sin,
    Symbol,
    Terminal,
    fast_tree_copy,
    is_terminal,
    selection_tournament,
)

# =============================================================================
# Evolution Initialization Tests
# =============================================================================


class TestEvolutionInit:
    """Tests for Evolution class initialization."""

    def test_basic_initialization(self, float_symbols, basic_operator_dict):
        """Tests basic Evolution initialization."""
        evo = Evolution(symbol_list=float_symbols, operators=basic_operator_dict, depth_max=5, nodes_max=30)

        assert evo.depth_max == 5
        assert evo.nodes_max == 30
        assert evo.node_selector is not None

    def test_initialization_with_string_symbols(self, basic_operator_dict):
        """Tests Evolution with string symbol names."""
        evo = Evolution(symbol_list=["x", "y", "z"], operators=basic_operator_dict)

        assert len(evo.symbol_list) == 3
        assert all(isinstance(s, sympy.Symbol) for s in evo.symbol_list)

    def test_initialization_with_preset(self, float_symbols):
        """Tests Evolution with operator preset."""
        evo = Evolution(symbol_list=float_symbols, operators="math_simple")

        assert evo.node_selector is not None

    def test_default_operators(self, float_symbols):
        """Tests Evolution with default operators."""
        evo = Evolution(symbol_list=float_symbols)

        assert evo.node_selector is not None


# =============================================================================
# Tree Creation Tests
# =============================================================================


class TestTreeCreation:
    """Tests for random tree creation."""

    def test_create_random_simple(self, evolution_instance):
        """Tests creating a random tree."""
        tree = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=3, depth=0)

        assert tree is not None
        assert isinstance(tree, Node)

    def test_create_random_respects_depth(self, evolution_instance):
        """Tests tree respects depth limit."""
        tree = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=2, depth=0)

        max_depth = tree.get_max_depth()
        assert max_depth <= 2

    def test_create_random_float_output(self, evolution_instance):
        """Tests tree produces float output type."""
        tree = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=3, depth=0)

        # Root should produce float
        assert tree.get_xtype_self() == float

    def test_create_random_bool_output(self, evolution_instance):
        """Tests tree produces bool output type."""
        tree = evolution_instance.evolve_create_random(xt_out=bool, depth_max_local=3, depth=0)

        assert tree.get_xtype_self() == bool

    def test_evolve_new_tree_depth(self, evolution_instance):
        """Tests evolve_new_tree_depth creates valid tree."""
        tree = evolution_instance.evolve_new_tree_depth(xt_out=float, depth_goal=4, p_term=0.1)

        assert tree is not None
        assert isinstance(tree, Node)

    def test_create_multiple_trees(self, evolution_instance):
        """Tests creating multiple unique trees."""
        trees = []
        for _ in range(10):
            tree = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=4, depth=0)
            trees.append(str(tree))

        # Should have some variety
        unique_trees = set(trees)
        assert len(unique_trees) >= 3  # At least some variety

    def test_p_term_affects_depth(self, evolution_instance):
        """Tests p_term parameter affects tree termination."""
        depths_low_pterm = []
        depths_high_pterm = []

        for _ in range(20):
            tree_low = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=5, depth=0, p_term=0.0)
            tree_high = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=5, depth=0, p_term=0.5)
            depths_low_pterm.append(tree_low.get_max_depth())
            depths_high_pterm.append(tree_high.get_max_depth())

        # High p_term should produce shallower trees on average
        assert np.mean(depths_high_pterm) <= np.mean(depths_low_pterm) + 1


# =============================================================================
# Mutation Tests
# =============================================================================


class TestMutation:
    """Tests for mutation operations."""

    def test_mutate_point_preserves_type(self, evolution_instance):
        """Tests point mutation preserves output type."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree.repair_depth()

        mutated = evolution_instance.evolve_mutate_point(tree)

        # Output type should still be float
        assert mutated.get_xtype_self() == float

    def test_mutate_point_changes_tree(self, evolution_instance):
        """Tests point mutation can change tree."""
        original = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        original.repair_depth()

        changed = False
        for _ in range(10):
            mutated = evolution_instance.evolve_mutate_point(original)
            if str(mutated) != str(original):
                changed = True
                break

        # Should change at least once in 10 tries
        assert changed

    def test_mutate_branch_depth(self, evolution_instance):
        """Tests branch mutation with depth goal."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
        tree.repair_depth()

        mutated = evolution_instance.evolve_mutate_branch_depth(tree, depth_goal=2, p_term=0.3)

        assert mutated is not None

    def test_mutate_branch_nodes(self, evolution_instance):
        """Tests branch mutation with node goal."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
        tree.repair_depth()

        mutated = evolution_instance.evolve_mutate_branch_nodes(tree, nodes_goal=5, p_term=0.2)

        assert mutated is not None

    def test_mutate_filter_gauss(self, evolution_instance, sample_df):
        """Tests Gaussian filter mutation changes numbers."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree.repair_depth()

        original_value = float(tree.get_childs()[1].get_value())

        # Apply filter mutation multiple times
        changed = False
        for _ in range(10):
            test_tree = copy.deepcopy(tree)
            evolution_instance.evolve_mutate_filter(test_tree)
            new_value = float(test_tree.get_childs()[1].get_value())
            if abs(new_value - original_value) > 0.001:
                changed = True
                break

        assert changed


# =============================================================================
# Terminal-Only Mutation Tests (I7)
# =============================================================================


class TestMutateTerminals:
    """Tests for terminal-only mutation (evolve_mutate_terminals)."""

    def test_preserves_operator_structure(self, evolution_instance):
        """Operator topology must remain identical after terminal mutation."""
        tree = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Sin(Symbol(sympy.Symbol("b"))))
        tree.repair_depth()
        original_ops = [type(n).__name__ for n in tree.list_mutable_nodes() if not is_terminal(n)]

        mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree))
        mutated_ops = [type(n).__name__ for n in mutated.list_mutable_nodes() if not is_terminal(n)]

        assert original_ops == mutated_ops

    def test_changes_at_least_one_terminal(self, evolution_instance):
        """With enough tries, at least one terminal should change."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree.repair_depth()
        original_str = str(tree)

        changed = False
        for _ in range(20):
            mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree))
            if str(mutated) != original_str:
                changed = True
                break
        assert changed

    def test_xtype_preserved(self, evolution_instance):
        """Replacement terminals must match the xtype of the original."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(3.0))
        tree.repair_depth()

        for _ in range(10):
            mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree))
            terminals = [n for n in mutated.list_mutable_nodes() if is_terminal(n)]
            for t in terminals:
                assert t.get_xtype_self() == float

    def test_fixed_nodes_untouched(self, evolution_instance):
        """Terminals with is_fix=True must never be replaced."""
        fixed_sym = Symbol(sympy.Symbol("a"), is_fix=True)
        tree = Add(fixed_sym, Number(5.0))
        tree.repair_depth()

        for _ in range(10):
            t = fast_tree_copy(tree)
            evolution_instance.evolve_mutate_terminals(t, n_terminals=5)
            # The fixed node should still be a Symbol named "a"
            childs = t.get_childs()
            fixed_child = [c for c in childs if c.is_fix]
            assert len(fixed_child) == 1
            assert isinstance(fixed_child[0], Symbol)
            assert str(fixed_child[0].get_value()) == "a"

    def test_n_terminals_clamp(self, evolution_instance):
        """n_terminals > available terminals should not raise."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree.repair_depth()

        # 2 mutable terminals, requesting 10 — should work without error
        mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree), n_terminals=10)
        assert mutated is not None

    def test_multiple_terminals_changed(self, evolution_instance):
        """With n_terminals=3, up to 3 terminals should differ."""
        tree = Add(
            Mul(Number(1.0), Number(2.0)),
            Add(Number(3.0), Number(4.0)),
        )
        tree.repair_depth()

        changed_count = 0
        for _ in range(20):
            mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree), n_terminals=3)
            orig_terms = sorted(str(n) for n in tree.list_mutable_nodes() if is_terminal(n))
            mut_terms = sorted(str(n) for n in mutated.list_mutable_nodes() if is_terminal(n))
            if orig_terms != mut_terms:
                changed_count += 1
        # Should change in most attempts
        assert changed_count >= 10

    def test_empty_tree_returns_unchanged(self, evolution_instance):
        """A terminal-only tree (single Number) should still work."""
        tree = Number(42.0)
        tree.repair_depth()
        mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree))
        assert mutated is not None

    def test_bool_xtype_terminals(self, evolution_instance):
        """Bool terminals should be replaced with bool terminals."""
        tree = And(Boolean(True), Lt(Symbol(sympy.Symbol("a")), Number(0.0)))
        tree.repair_depth()

        for _ in range(10):
            mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree))
            bool_terms = [n for n in mutated.list_mutable_nodes() if is_terminal(n) and n.get_xtype_self() == bool]
            for bt in bool_terms:
                assert isinstance(bt, Boolean)

    def test_p_symbol_high_produces_symbols(self, evolution_instance):
        """With p_symbol≈1, replaced terminals should mostly be Symbols."""
        tree = Add(Number(1.0), Number(2.0))
        tree.repair_depth()

        symbol_count = 0
        total = 0
        for _ in range(30):
            mutated = evolution_instance.evolve_mutate_terminals(fast_tree_copy(tree), n_terminals=2, p_symbol=0.95)
            for n in mutated.list_mutable_nodes():
                if is_terminal(n):
                    total += 1
                    if isinstance(n, Symbol):
                        symbol_count += 1

        # Most should be Symbols
        assert symbol_count > total * 0.5


# =============================================================================
# Crossover Tests
# =============================================================================


class TestCrossover:
    """Tests for crossover operations."""

    def test_crossover_produces_two_trees(self, evolution_instance):
        """Tests crossover produces two trees."""
        tree1 = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
        tree2 = Sin(Add(Symbol(sympy.Symbol("a")), Number(3.0)))

        tree1.repair_depth()
        tree2.repair_depth()

        result1, result2 = evolution_instance.evolve_crossover(copy.deepcopy(tree1), copy.deepcopy(tree2))

        assert result1 is not None
        assert result2 is not None

    def test_crossover_changes_trees(self, evolution_instance):
        """Tests crossover can change both trees."""
        tree1 = Add(Mul(Symbol(sympy.Symbol("a")), Number(2.0)), Symbol(sympy.Symbol("b")))
        tree2 = Sin(Add(Symbol(sympy.Symbol("a")), Number(3.0)))

        tree1.repair_depth()
        tree2.repair_depth()

        str1_orig = str(tree1)
        str2_orig = str(tree2)

        # Try multiple times
        changed = False
        for _ in range(10):
            t1_copy = copy.deepcopy(tree1)
            t2_copy = copy.deepcopy(tree2)
            r1, r2 = evolution_instance.evolve_crossover(t1_copy, t2_copy)

            if str(r1) != str1_orig or str(r2) != str2_orig:
                changed = True
                break

        assert changed

    def test_crossover_maintains_validity(self, evolution_instance, sample_df):
        """Tests crossover produces evaluable trees."""
        tree1 = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree2 = Mul(Symbol(sympy.Symbol("a")), Number(2.0))

        tree1.repair_depth()
        tree2.repair_depth()

        r1, r2 = evolution_instance.evolve_crossover(copy.deepcopy(tree1), copy.deepcopy(tree2))

        # Both should be evaluable
        result1 = r1.eval_predict_numpy_now(sample_df)
        result2 = r2.eval_predict_numpy_now(sample_df)

        assert len(result1) == len(sample_df)
        assert len(result2) == len(sample_df)


# =============================================================================
# Pruning Tests
# =============================================================================


class TestPruning:
    """Tests for tree pruning."""

    def test_prune_respects_depth(self, evolution_instance):
        """Tests pruning respects depth limit."""
        # Create a potentially deep tree
        tree = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=10, depth=0)

        pruned = evolution_instance.evolve_prune_tree(tree)

        assert pruned.get_max_depth() <= evolution_instance.depth_max

    def test_prune_respects_nodes(self, evolution_instance):
        """Tests pruning respects node limit."""
        evolution_instance.nodes_max = 15

        tree = evolution_instance.evolve_create_random(xt_out=float, depth_max_local=6, depth=0)

        pruned = evolution_instance.evolve_prune_tree(tree)

        assert len(pruned) <= evolution_instance.nodes_max

    def test_prune_returns_root_not_subtree(self, evolution_instance):
        """Regression: pruning must return the root object, not a random subtree."""
        import sympy

        from plagih.trees import Add, Number, Symbol

        evolution_instance.nodes_max = 5
        tree = Symbol(sympy.Symbol("a"))
        for _ in range(8):
            tree = Add(tree, Number(1.0))
        tree.repair_all()
        assert len(tree) > evolution_instance.nodes_max

        pruned = evolution_instance.evolve_prune_tree(tree)

        assert pruned is tree, "must return the same root object"
        assert len(pruned) <= evolution_instance.nodes_max

    def test_prune_oversized_tree_reaches_limit(self, evolution_instance):
        """Regression: loop must measure the root, not the pruned branch."""
        import sympy

        from plagih.trees import Add, Number, Symbol

        evolution_instance.nodes_max = 7
        for _ in range(10):
            tree = Symbol(sympy.Symbol("a"))
            for _ in range(12):
                tree = Add(tree, Number(1.0))
            tree.repair_all()

            pruned = evolution_instance.evolve_prune_tree(tree)
            assert len(pruned) <= evolution_instance.nodes_max

    def test_prune_fully_fixed_tree_does_not_crash(self, evolution_instance):
        """Regression: frozen origin_tree skeletons have no prunable branch."""
        import sympy

        from plagih.trees import Add, Number, Symbol

        evolution_instance.nodes_max = 3
        tree = Symbol(sympy.Symbol("a"))
        for _ in range(5):
            tree = Add(tree, Number(1.0))
        tree.repair_all()

        # Freeze everything
        for node in tree.to_traversal("pre"):
            node.is_fix = True

        pruned = evolution_instance.evolve_prune_tree(tree)  # must not raise
        assert pruned is tree

    def test_prune_mostly_fixed_tree_does_not_crash(self, evolution_instance):
        """Regression: IndexError when no mutable branch reaches prune_now."""
        import sympy

        from plagih.trees import Add, Number, Symbol

        evolution_instance.nodes_max = 4
        tree = Symbol(sympy.Symbol("a"))
        for _ in range(10):
            tree = Add(tree, Number(1.0))
        tree.repair_all()

        # Freeze all operators; only terminals stay mutable (len 1 each)
        for node in tree.to_traversal("pre"):
            if node.has_childs():
                node.is_fix = True

        pruned = evolution_instance.evolve_prune_tree(tree)  # must not raise
        assert pruned is tree


# =============================================================================
# Node Selector Tests
# =============================================================================


class TestNodeSelector:
    """Tests for NodeSelect class."""

    def test_choose_operator_class(self, evolution_instance):
        """Tests choosing operator class."""
        op = evolution_instance.node_selector.choose_operator_class(float)

        assert op is not None
        assert hasattr(op, "xtype")
        assert op.xtype[1] == float

    def test_choose_terminal_node(self, evolution_instance):
        """Tests choosing terminal node."""
        term = evolution_instance.node_selector.choose_terminal_node(float)

        assert term is not None
        assert term.is_term()

    def test_choose_symbol_node(self, evolution_instance):
        """Tests choosing symbol node."""
        sym = evolution_instance.node_selector.choose_symbol_node(float)

        assert sym is not None
        assert isinstance(sym, Symbol)

    def test_choose_constant_node(self, evolution_instance):
        """Tests choosing constant node."""
        const = evolution_instance.node_selector.choose_constant_node(float)

        assert const is not None
        assert isinstance(const, Number)

    def test_choose_boolean_constant(self, evolution_instance):
        """Tests choosing boolean constant."""
        const = evolution_instance.node_selector.choose_constant_node(bool)

        assert const is not None
        assert isinstance(const, Boolean)


# =============================================================================
# Tournament Selection Tests
# =============================================================================


class TestTournamentSelection:
    """Tests for tournament selection."""

    def test_selection_returns_tree(self):
        """Tests tournament selection returns a tree."""
        # Create fake population
        pop = []
        for i in range(10):
            tree = Add(Symbol(sympy.Symbol("a")), Number(float(i)))
            cand = Candidate(tree, fitness=float(i), parsimony=3, tag="test")
            pop.append(cand)

        selected = selection_tournament(pop, n=3)

        assert selected is not None
        assert isinstance(selected, Node)

    def test_selection_favors_fitter(self):
        """Tests tournament tends to select fitter individuals."""
        # Create population with varying fitness
        pop = []
        for i in range(20):
            tree = Add(Symbol(sympy.Symbol("a")), Number(float(i)))
            cand = Candidate(tree, fitness=float(i), parsimony=3, tag="test")
            pop.append(cand)

        # Run many selections
        selections = []
        for _ in range(100):
            selected = selection_tournament(pop, n=5)
            # Find which candidate this was
            for cand in pop:
                if str(cand.tree) == str(selected):
                    selections.append(cand.fitness)
                    break

        # Average selection should be below median fitness
        avg_selected = np.mean(selections)
        median_fitness = 9.5  # 0-19, median is 9.5

        assert avg_selected < median_fitness
