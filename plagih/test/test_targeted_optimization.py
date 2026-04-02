"""Tests for plagih.targeted_optimization module.

Covers Phase 1 (analysis infrastructure), Phase 2 (Ifte scoring),
and Phase 2b (targeted_ifte strategy integration).
"""

import copy

import numpy as np
import pandas as pd
import pytest
import sympy

from plagih.targeted_optimization import (
    BestPerDatapointResult,
    IfteAnalysisResult,
    best_per_datapoint,
    eval_node_intermediates,
    ifte_component_scores,
    piecewise_component_scores,
    soft_optimum_error,
)
from plagih.trees import (
    Add,
    Boolean,
    Candidate,
    ExprCondPair_Dummy,
    Ifte,
    Le,
    Max,
    Min,
    Mul,
    Number,
    Piecewise,
    Square,
    Sub,
    Symbol,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_df():
    """Small DataFrame for targeted optimization tests."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 1.0, 0.0, -1.0, -2.0],
            "action": [3.0, 3.0, 3.0, 3.0, 3.0],
        }
    )


@pytest.fixture
def target_flat():
    """Target values: all 3.0."""
    return np.array([3.0, 3.0, 3.0, 3.0, 3.0])


@pytest.fixture
def tree_add_ab():
    """Tree: Add(a, b) = a + b."""
    return Add(Symbol(sympy.Symbol("a", real=True)), Symbol(sympy.Symbol("b", real=True)))


# Need sympy for Symbol creation
import sympy


@pytest.fixture
def sym_a():
    return sympy.Symbol("a", real=True)


@pytest.fixture
def sym_b():
    return sympy.Symbol("b", real=True)


# =============================================================================
# Phase 1: eval_node_intermediates
# =============================================================================


class TestEvalNodeIntermediates:
    """Tests for per-node intermediate value evaluation."""

    def test_terminal_values(self, simple_df, sym_a, sym_b):
        """Terminal nodes should return column/constant values."""
        sym_node = Symbol(sym_a)
        intermediates = eval_node_intermediates(sym_node, simple_df)

        assert id(sym_node) in intermediates
        np.testing.assert_array_equal(intermediates[id(sym_node)], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_operator_and_children(self, simple_df, sym_a, sym_b):
        """Operator and its children should all appear in intermediates."""
        a_node = Symbol(sym_a)
        b_node = Symbol(sym_b)
        add_node = Add(a_node, b_node)

        intermediates = eval_node_intermediates(add_node, simple_df)

        # All three nodes should be present
        assert id(a_node) in intermediates
        assert id(b_node) in intermediates
        assert id(add_node) in intermediates

        # Check values
        np.testing.assert_array_equal(intermediates[id(a_node)], [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(intermediates[id(b_node)], [2.0, 1.0, 0.0, -1.0, -2.0])
        np.testing.assert_array_equal(intermediates[id(add_node)], [3.0, 3.0, 3.0, 3.0, 3.0])

    def test_nested_tree(self, simple_df, sym_a, sym_b):
        """Nested tree: Mul(Add(a, b), a) — all nodes should have intermediates."""
        a1 = Symbol(sym_a)
        b1 = Symbol(sym_b)
        a2 = Symbol(sym_a)
        add_node = Add(a1, b1)
        mul_node = Mul(add_node, a2)

        intermediates = eval_node_intermediates(mul_node, simple_df)

        assert len(intermediates) == 5  # a1, b1, a2, add, mul
        # Mul(Add(a,b), a) = (a+b)*a
        expected = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
        np.testing.assert_array_almost_equal(intermediates[id(mul_node)], expected)

    def test_number_terminal(self, simple_df):
        """Number terminals should return constant arrays."""
        num = Number(sympy.Float(42.0))
        intermediates = eval_node_intermediates(num, simple_df)

        np.testing.assert_array_equal(intermediates[id(num)], np.full(5, 42.0))


# =============================================================================
# Phase 1: best_per_datapoint & soft_optimum_error
# =============================================================================


class TestBestPerDatapoint:
    """Tests for Oracle Selector / best-per-datapoint analysis."""

    def test_single_candidate(self, simple_df, target_flat, sym_a, sym_b):
        """Single candidate: it wins every row."""
        tree = Add(Symbol(sym_a), Symbol(sym_b))  # a + b = [3, 3, 3, 3, 3]
        cand = Candidate(tree, fitness=0.0, parsimony=3, tag="test")

        result = best_per_datapoint([cand], simple_df, target_flat)

        assert isinstance(result, BestPerDatapointResult)
        np.testing.assert_array_equal(result.winner_indices, [0, 0, 0, 0, 0])
        assert result.soft_optimum_error == 0.0
        assert result.candidate_contributions[0] == 5
        assert len(result.minimum_set_indices) == 1

    def test_complementary_candidates(self, sym_a, sym_b):
        """Two candidates that complement each other."""
        df = pd.DataFrame({"a": [1.0, 10.0], "b": [0.0, 0.0]})
        target = np.array([1.0, 1.0])

        # Candidate 0: a → [1, 10] — perfect on row 0
        tree0 = Symbol(sym_a)
        cand0 = Candidate(tree0, fitness=4.5, parsimony=1, tag="t0")

        # Candidate 1: constant 1 → [1, 1] — perfect on both, but same as target
        tree1 = Number(sympy.Float(1.0))
        cand1 = Candidate(tree1, fitness=0.0, parsimony=1, tag="t1")

        result = best_per_datapoint([cand0, cand1], df, target)

        # Both candidates hit row 0 perfectly; cand1 also hits row 1
        # Row 0: both error=0, argmin picks 0 (first)
        # Row 1: cand0 error=9, cand1 error=0 → winner=1
        assert result.winner_indices[1] == 1
        assert result.soft_optimum_error == 0.0

    def test_soft_optimum_error_convenience(self, simple_df, target_flat, sym_a, sym_b):
        """soft_optimum_error() should return same as full analysis."""
        tree = Add(Symbol(sym_a), Symbol(sym_b))
        cand = Candidate(tree, fitness=0.0, parsimony=3, tag="test")

        so = soft_optimum_error([cand], simple_df, target_flat)
        assert so == 0.0

    def test_empty_population_raises(self, simple_df, target_flat):
        """Empty population should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            best_per_datapoint([], simple_df, target_flat)

    def test_minimum_set_covers_all_rows(self, sym_a, sym_b):
        """Minimum set should cover all rows."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
        target = np.array([1.0, 2.0, 3.0])

        # Perfect candidate
        tree = Symbol(sym_a)
        cand = Candidate(tree, fitness=0.0, parsimony=1, tag="t")
        result = best_per_datapoint([cand], df, target)

        assert len(result.minimum_set_indices) >= 1

    def test_squared_metric(self, sym_a):
        """Squared error metric should work."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        target = np.array([0.0, 0.0, 0.0])
        tree = Symbol(sym_a)
        cand = Candidate(tree, fitness=0.0, parsimony=1, tag="t")

        result = best_per_datapoint([cand], df, target, error_metric="squared")
        # errors should be [1, 4, 9]
        np.testing.assert_array_almost_equal(result.winner_errors, [1.0, 4.0, 9.0])


# =============================================================================
# Phase 2: ifte_component_scores
# =============================================================================


class TestIfteComponentScores:
    """Tests for Ifte pseudo-backpropagation scoring."""

    def test_no_ifte_returns_empty(self, simple_df, target_flat, sym_a, sym_b):
        """Tree without Ifte should return empty list."""
        tree = Add(Symbol(sym_a), Symbol(sym_b))
        scores = ifte_component_scores(tree, simple_df, target_flat)
        assert scores == []

    def test_identical_branches(self, sym_a, sym_b):
        """When both branches return the same value, error_sum should be 0."""
        # Ifte(a < 3, 3.0, 3.0) → both branches return 3, condition is irrelevant
        df = pd.DataFrame({"a": [1.0, 2.0, 4.0, 5.0], "b": [0.0, 0.0, 0.0, 0.0]})
        target = np.array([3.0, 3.0, 3.0, 3.0])

        tree = Ifte(
            Le(Symbol(sym_a), Number(sympy.Float(3.0))),
            Number(sympy.Float(3.0)),
            Number(sympy.Float(3.0)),
        )

        results = ifte_component_scores(tree, df, target)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, IfteAnalysisResult)
        # Both branches are identical → no error regardless of condition
        assert result.scores["condition"].error_sum == 0.0
        assert result.scores["then"].error_sum == 0.0
        assert result.scores["else"].error_sum == 0.0

    def test_bad_condition(self, sym_a):
        """Condition that always picks the worse branch."""
        # Ifte(a < 3, 100.0, 0.0) with target=0 and a=[1,2]
        # → condition is True (picks 100) but target is 0, else=0 is better
        df = pd.DataFrame({"a": [1.0, 2.0]})
        target = np.array([0.0, 0.0])

        tree = Ifte(
            Le(Symbol(sym_a), Number(sympy.Float(3.0))),
            Number(sympy.Float(100.0)),  # then — bad
            Number(sympy.Float(0.0)),  # else — good
        )

        results = ifte_component_scores(tree, df, target)
        assert len(results) == 1
        # Condition always True, but else (0.0) is always closer to target (0.0)
        # → condition accuracy = 0.0
        assert results[0].condition_accuracy == 0.0
        assert results[0].weakest == "condition"

    def test_identifies_weak_then_branch(self, sym_a):
        """When then-branch is far from target, it should be weakest."""
        # a=[1, 2, 10, 20], target=[0, 0, 0, 0]
        # Ifte(a < 5, a, 0)
        # Rows 0,1: cond=True, then=a=[1,2], else=0 → else is closer to 0
        # Rows 2,3: cond=False, else=0 → perfect
        df = pd.DataFrame({"a": [1.0, 2.0, 10.0, 20.0]})
        target = np.array([0.0, 0.0, 0.0, 0.0])

        tree = Ifte(
            Le(Symbol(sym_a), Number(sympy.Float(5.0))),
            Symbol(sym_a),  # then — returns [1, 2] on active rows, worse than 0
            Number(sympy.Float(0.0)),  # else — perfect
        )

        results = ifte_component_scores(tree, df, target)
        assert len(results) == 1
        r = results[0]

        # Then-branch on its active rows [1, 2] vs target [0, 0]: count_score
        # then_err=[1,2], else_err=[0,0] → then is never better → count_score=0
        assert r.scores["then"].count_score == 0.0
        assert r.scores["else"].count_score == 1.0

    def test_nested_ifte(self, sym_a):
        """Nested Ifte nodes should all be found."""
        df = pd.DataFrame({"a": [1.0, 5.0]})
        target = np.array([0.0, 0.0])

        inner = Ifte(
            Le(Symbol(sym_a), Number(sympy.Float(3.0))),
            Number(sympy.Float(1.0)),
            Number(sympy.Float(2.0)),
        )
        outer = Ifte(
            Le(Symbol(sym_a), Number(sympy.Float(10.0))),
            inner,
            Number(sympy.Float(0.0)),
        )

        results = ifte_component_scores(outer, df, target)
        assert len(results) == 2  # Both outer and inner Ifte


# =============================================================================
# Phase 2b — targeted_ifte strategy integration
# =============================================================================


class TestTargetedIfteStrategy:
    """Tests for the _strategy_targeted_ifte builtin strategy."""

    @pytest.fixture
    def evolve_and_df(self):
        """Create a minimal Evolution instance and DataFrame for strategy tests."""
        from plagih.trees._evolution import Evolution

        operators = {Add: 1, Mul: 1, Sub: 1, Min: 1, Max: 1, Ifte: 1, Le: 1, Square: 1}
        terminals_float = [sympy.Symbol("a"), sympy.Symbol("b")]

        ev = Evolution(
            symbol_list=terminals_float,
            operators=operators,
            nodes_max=30,
            depth_max=6,
        )

        df = pd.DataFrame(
            {
                "a": np.linspace(-5, 5, 20),
                "b": np.linspace(0, 10, 20),
                "action": np.linspace(1, 3, 20),
            }
        )
        target = df["action"].values
        return ev, df, target

    def test_strategy_without_ifte_falls_back(self, evolve_and_df):
        """Strategy should fall back to branch mutation when tree has no Ifte."""
        from plagih.parallel import _strategy_targeted_ifte

        ev, df, target = evolve_and_df
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        tree.repair_all()

        result = _strategy_targeted_ifte(
            ev,
            [],
            [],
            False,
            _pre_selected=[copy.deepcopy(tree)],
            _df_train=df,
            _target=target,
        )
        # Should return a valid tree (fallback to mutation)
        assert result is not None
        assert hasattr(result, "get_childs") or hasattr(result, "get_value")

    def test_strategy_with_ifte_mutates_weakest(self, evolve_and_df):
        """Strategy should identify and mutate the weakest Ifte component."""
        from plagih.parallel import _strategy_targeted_ifte

        ev, df, target = evolve_and_df

        # Build: Ifte(a > 0, 100, b) — condition is good, then-branch is terrible
        ifte_tree = Ifte(
            Le(Number(0.0), Symbol(sympy.Symbol("a"))),  # a >= 0
            Number(100.0),  # then: always 100 (bad for target ~1-3)
            Symbol(sympy.Symbol("b")),  # else: b
        )
        ifte_tree.repair_all()

        result = _strategy_targeted_ifte(
            ev,
            [],
            [],
            False,
            _pre_selected=[copy.deepcopy(ifte_tree)],
            _df_train=df,
            _target=target,
        )
        assert result is not None

    def test_strategy_without_df_falls_back(self, evolve_and_df):
        """Without df_train/target, strategy should fall back to mutation."""
        from plagih.parallel import _strategy_targeted_ifte

        ev, df, target = evolve_and_df
        ifte_tree = Ifte(
            Le(Number(0.0), Symbol(sympy.Symbol("a"))),
            Number(1.0),
            Number(2.0),
        )
        ifte_tree.repair_all()

        result = _strategy_targeted_ifte(
            ev,
            [],
            [],
            False,
            _pre_selected=[copy.deepcopy(ifte_tree)],
            # No _df_train or _target
        )
        assert result is not None

    def test_strategy_registered_in_builtins(self):
        """targeted_ifte must be in BUILTIN_STRATEGIES."""
        from plagih.parallel import BUILTIN_STRATEGIES

        assert "targeted_ifte" in BUILTIN_STRATEGIES


class TestTreeHasIfte:
    """Tests for the _tree_has_ifte helper."""

    def test_simple_tree_no_ifte(self):
        from plagih.parallel import _tree_has_ifte

        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        assert _tree_has_ifte(tree) is False

    def test_tree_with_ifte(self):
        from plagih.parallel import _tree_has_ifte

        tree = Ifte(Boolean(True), Number(1.0), Number(2.0))
        assert _tree_has_ifte(tree) is True

    def test_nested_ifte(self):
        from plagih.parallel import _tree_has_ifte

        tree = Add(
            Symbol(sympy.Symbol("a")),
            Ifte(Boolean(True), Number(1.0), Number(2.0)),
        )
        assert _tree_has_ifte(tree) is True

    def test_terminal_no_ifte(self):
        from plagih.parallel import _tree_has_ifte

        tree = Number(1.0)
        assert _tree_has_ifte(tree) is False
