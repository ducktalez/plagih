"""
Smoke tests for plagih.demo_helpers.

Ensures that all pre-built trees, DataFrames, Evolution instances, and
display utilities survive refactors.  All rendering is done with the
matplotlib Agg backend (headless) — no windows are opened.
"""

import copy

import matplotlib
import pytest

matplotlib.use("Agg")  # headless — must be set before any pyplot import

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Tree factories — ensure they create valid, non-empty trees
# ---------------------------------------------------------------------------


class TestTreeFactories:
    """Each make_tree_* helper must return a repaired tree with depth ≥ 0."""

    _FACTORIES = [
        "make_tree_simple",
        "make_tree_trig",
        "make_tree_ifte",
        "make_tree_boolean",
        "make_tree_simplifiable",
        "make_tree_crossover_parent_a",
        "make_tree_crossover_parent_b",
        "make_tree_cartpole",
        "make_tree_ifte_cartpole",
        "make_tree_redundant",
    ]

    @pytest.mark.parametrize("factory_name", _FACTORIES)
    def test_factory_returns_valid_tree(self, factory_name):
        from plagih import demo_helpers

        factory = getattr(demo_helpers, factory_name)
        tree = factory()

        # Basic structural checks
        assert tree is not None
        assert len(tree) >= 1, f"{factory_name} returned an empty tree"
        assert tree.depth >= 0
        assert tree.represent_str() != ""

    @pytest.mark.parametrize("factory_name", _FACTORIES)
    def test_factory_trees_are_deepcopyable(self, factory_name):
        from plagih import demo_helpers

        factory = getattr(demo_helpers, factory_name)
        tree = factory()
        tree_copy = copy.deepcopy(tree)
        assert tree_copy.represent_str() == tree.represent_str()


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------


class TestDataFrameFactories:
    def test_make_sample_df(self):
        from plagih.demo_helpers import make_sample_df

        df = make_sample_df()
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 5

    def test_make_cartpole_df(self):
        from plagih.demo_helpers import make_cartpole_df

        df = make_cartpole_df()
        assert "cartPos" in df.columns
        assert "cartVel" in df.columns
        assert "action" in df.columns
        assert len(df) == 10


# ---------------------------------------------------------------------------
# Evolution factories
# ---------------------------------------------------------------------------


class TestEvolutionFactories:
    def test_make_evolution(self):
        from plagih.demo_helpers import make_evolution

        evo = make_evolution()
        assert evo is not None
        assert len(evo.symbol_list) == 2  # a, b

    def test_make_cartpole_evolution(self):
        from plagih.demo_helpers import make_cartpole_evolution

        evo = make_cartpole_evolution()
        assert evo is not None
        assert len(evo.symbol_list) == 2  # cartPos, cartVel

    def test_evolution_can_create_random_tree(self):
        from plagih.demo_helpers import make_evolution

        evo = make_evolution()
        tree = evo.evolve_create_random(xt_out=float, depth_max_local=2, p_term=0.3)
        assert tree is not None
        assert len(tree) >= 1


# ---------------------------------------------------------------------------
# Crossover helper
# ---------------------------------------------------------------------------


class TestCrossover:
    def test_do_crossover_returns_two_trees(self):
        from plagih.demo_helpers import (
            do_crossover,
            make_tree_crossover_parent_a,
            make_tree_crossover_parent_b,
        )

        parent_a = make_tree_crossover_parent_a()
        parent_b = make_tree_crossover_parent_b()
        child_a, child_b = do_crossover(parent_a, parent_b)
        assert child_a is not None
        assert child_b is not None
        assert len(child_a) >= 1
        assert len(child_b) >= 1

    def test_crossover_does_not_mutate_originals(self):
        from plagih.demo_helpers import (
            do_crossover,
            make_tree_crossover_parent_a,
            make_tree_crossover_parent_b,
        )

        parent_a = make_tree_crossover_parent_a()
        parent_b = make_tree_crossover_parent_b()
        repr_a = parent_a.represent_str()
        repr_b = parent_b.represent_str()
        do_crossover(parent_a, parent_b)
        assert parent_a.represent_str() == repr_a
        assert parent_b.represent_str() == repr_b


# ---------------------------------------------------------------------------
# Display utilities (headless rendering)
# ---------------------------------------------------------------------------


class TestDisplayUtilities:
    """Verify that show_* functions run without error in headless mode."""

    def test_show_tree(self):
        from plagih.demo_helpers import make_tree_simple, show_tree

        plt.close("all")
        # show_tree calls plt.show() which is a no-op with Agg backend
        show_tree(make_tree_simple(), title="smoke test")
        plt.close("all")

    def test_show_trees(self):
        from plagih.demo_helpers import (
            make_tree_crossover_parent_a,
            make_tree_crossover_parent_b,
            show_trees,
        )

        plt.close("all")
        trees = [
            (make_tree_crossover_parent_a(), "A"),
            (make_tree_crossover_parent_b(), "B"),
        ]
        show_trees(trees, suptitle="smoke test")
        plt.close("all")

    def test_show_tree_with_scores(self):
        from plagih.demo_helpers import make_tree_ifte, show_tree_with_scores

        plt.close("all")
        tree = make_tree_ifte()

        # Build a simple score dict — score the root and one child
        scores = {id(tree): 0.5}
        for child in tree.get_childs():
            scores[id(child)] = 0.8

        show_tree_with_scores(tree, node_scores=scores, title="score test")
        plt.close("all")

    def test_show_tree_with_scores_empty_dict(self):
        from plagih.demo_helpers import make_tree_simple, show_tree_with_scores

        plt.close("all")
        show_tree_with_scores(make_tree_simple(), node_scores={}, title="empty scores")
        plt.close("all")


# ---------------------------------------------------------------------------
# make_ifte_node_scores
# ---------------------------------------------------------------------------


class TestMakeIfteNodeScores:
    def test_empty_input(self):
        from plagih.demo_helpers import make_ifte_node_scores

        result = make_ifte_node_scores([])
        assert result == {}
