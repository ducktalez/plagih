"""
Demo helpers for the plagih Feature Showcase Notebook (docs/demo.ipynb).

Provides pre-built example trees, sample DataFrames, and inline display
utilities so notebook cells stay short and focused on the concept being
demonstrated rather than on setup boilerplate.

Usage::

    from plagih.demo_helpers import (
        make_tree_simple,
        make_tree_ifte,
        make_tree_crossover_parent_a,
        make_cartpole_df,
        make_evolution,
        show_tree,
        show_trees,
        show_tree_with_scores,
    )
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import sympy

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sym(name: str):
    """Return a Symbol node for the given variable name."""
    from plagih.trees import Symbol

    return Symbol(sympy.Symbol(name, real=True))


def _num(v: float):
    """Return a Number node for the given constant."""
    from plagih.trees import Number

    return Number(sympy.Float(v, 6))


def _build(tree):
    """Call repair_all on a freshly constructed tree and return it."""
    tree.repair_all(depth=0)
    return tree


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------


def make_sample_df() -> pd.DataFrame:
    """Return a small 5-row DataFrame with float variables *a*, *b*.

    Suitable for most arithmetic tree evaluations.
    """
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [-1.0, 0.5, 2.0, -0.5, 1.0],
        }
    )


def make_cartpole_df() -> pd.DataFrame:
    """Return a 10-row DataFrame mimicking the cartpole benchmark.

    Columns: *cartPos*, *cartVel*, *action*.
    """
    return pd.DataFrame(
        {
            "cartPos": [0.0, 0.5, 1.0, -0.5, -1.0, 0.1, 0.2, -0.3, 0.8, -0.9],
            "cartVel": [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.5, -0.4, 0.1, -0.2],
            "action": [1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# Pre-built example trees
# ---------------------------------------------------------------------------


def make_tree_simple():
    """``Add(a, 1)`` — the simplest possible non-trivial tree."""
    from plagih.trees import Add

    return _build(Add(_sym("a"), _num(1.0)))


def make_tree_trig():
    """``sin(a * b + 2)`` — a slightly deeper arithmetic tree."""
    from plagih.trees import Add, Mul, Sin

    return _build(Sin(Add(Mul(_sym("a"), _sym("b")), _num(2.0))))


def make_tree_ifte():
    """``Ifte(a < 0, −1, 1)`` — if-then-else with a numeric result."""
    from plagih.trees import Ifte, Lt

    return _build(Ifte(Lt(_sym("a"), _num(0.0)), _num(-1.0), _num(1.0)))


def make_tree_boolean():
    """``And(a < 1, NOT (b > 0))`` — a pure boolean tree."""
    from plagih.trees import And, Gt, Lt, Not

    return _build(And(Lt(_sym("a"), _num(1.0)), Not(Gt(_sym("b"), _num(0.0)))))


def make_tree_simplifiable():
    """``1*a + a^2`` — a tree with obvious simplification opportunities.

    After ``tree_simplification``, SymPy reduces it to ``a*(a + 1)``
    and ``tree_node_grouping`` rewrites ``Pow(a, 2)`` to ``Square(a)``.
    """
    from plagih.trees import Add, Mul, Pow

    return _build(Add(Mul(_num(1.0), _sym("a")), Pow(_sym("a"), _num(2.0))))


def make_tree_crossover_parent_a():
    """``sin(a) + b`` — parent A for the crossover demo."""
    from plagih.trees import Add, Sin

    return _build(Add(Sin(_sym("a")), _sym("b")))


def make_tree_crossover_parent_b():
    """``a * |b|`` — parent B for the crossover demo."""
    from plagih.trees import Abs, Mul

    return _build(Mul(_sym("a"), Abs(_sym("b"))))


def make_tree_cartpole():
    """``cartPos + 2 * cartVel`` — simple linear cartpole policy."""
    from plagih.trees import Add, Mul

    return _build(Add(_sym("cartPos"), Mul(_num(2.0), _sym("cartVel"))))


def make_tree_ifte_cartpole():
    """``Ifte(cartPos < 0.5, cartVel, -1 * cartPos)`` — branched cartpole policy."""
    from plagih.trees import Ifte, Lt, Mul

    return _build(Ifte(Lt(_sym("cartPos"), _num(0.5)), _sym("cartVel"), Mul(_num(-1.0), _sym("cartPos"))))


def make_tree_redundant():
    """``(a + 0) * (b * 1) - 0`` — tree with trivially removable operations.

    Demonstrates SymPy's algebraic simplification in the roundtrip.
    """
    from plagih.trees import Add, Mul, Sub

    return _build(Sub(Mul(Add(_sym("a"), _num(0.0)), Mul(_sym("b"), _num(1.0))), _num(0.0)))


# ---------------------------------------------------------------------------
# Evolution instance for crossover / mutation demos
# ---------------------------------------------------------------------------


def make_evolution():
    """Return a minimal :class:`Evolution` instance for demo operations.

    Operator set covers both float and bool types for type-closure
    (float→float, bool→bool, float→bool, bool→float via ``Ifte``).
    Variables: *a*, *b*.
    """
    from plagih.trees import Abs, Add, And, Cos, Evolution, Ifte, Lt, Mul, Not, Or, Sin, Sub

    return Evolution(
        symbol_list=["a", "b"],
        operators={
            Add: 2,
            Mul: 2,
            Sub: 1,
            Abs: 1,
            Sin: 0.5,
            Cos: 0.5,
            Lt: 1,
            And: 1,
            Or: 1,
            Not: 1,
            Ifte: 0.5,  # bool→float: required for type-closure
        },
        depth_max=5,
        nodes_max=30,
        allow_chain=False,
    )


def make_cartpole_evolution():
    """Return an :class:`Evolution` instance configured for cartpole variables."""
    from plagih.trees import Abs, Add, And, Cos, Evolution, Le, Lt, Max, Min, Mul, Not, Or, Sign, Sin, Sqrt, Square, Sub

    return Evolution(
        symbol_list=["cartPos", "cartVel"],
        operators={
            Add: 2,
            Mul: 2,
            Sub: 1,
            Abs: 1,
            Sign: 1,
            Square: 1,
            Sqrt: 0.5,
            Sin: 0.5,
            Cos: 0.5,
            Min: 1,
            Max: 1,
            Lt: 1,
            Le: 1,
            And: 1,
            Or: 1,
            Not: 1,
        },
        depth_max=5,
        nodes_max=30,
        allow_chain=False,
    )


# ---------------------------------------------------------------------------
# Crossover helper
# ---------------------------------------------------------------------------


def do_crossover(parent_a, parent_b, evo=None):
    """Perform crossover on deep copies of *parent_a* and *parent_b*.

    Returns ``(child_a, child_b)`` — the modified copies.  The originals
    are not altered.

    Args:
        parent_a: First parent tree.
        parent_b: Second parent tree.
        evo: Optional :class:`Evolution` instance.  Defaults to
            :func:`make_evolution`.

    Returns:
        Tuple ``(child_a, child_b)`` with swapped subtrees, both repaired.
    """
    if evo is None:
        evo = make_evolution()
    child_a = copy.deepcopy(parent_a)
    child_b = copy.deepcopy(parent_b)
    child_a, child_b = evo.evolve_crossover(child_a, child_b)
    child_a.repair_all(depth=0)
    child_b.repair_all(depth=0)
    return child_a, child_b


# ---------------------------------------------------------------------------
# Inline display utilities (Jupyter / matplotlib)
# ---------------------------------------------------------------------------


def show_tree(tree, title: str = "", figsize: tuple = (5, 4)) -> None:
    """Render *tree* inline as a Jupyter matplotlib figure.

    Args:
        tree: Root node of the tree to display.
        title: Optional subplot title.
        figsize: Figure size ``(width, height)`` in inches.
    """
    from plagih.visualization.tree_renderer import _render_tree_on_axes

    fig, ax = plt.subplots(figsize=figsize)
    _render_tree_on_axes(ax, tree)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_trees(trees_titles: List[Tuple], suptitle: str = "", figsize_per: tuple = (4, 3)) -> None:
    """Render multiple trees side-by-side in a single figure.

    Args:
        trees_titles: List of ``(tree, title)`` tuples.
        suptitle: Optional super-title for the entire figure.
        figsize_per: Per-panel ``(width, height)`` in inches.
    """
    from plagih.visualization.tree_renderer import _render_tree_on_axes

    n = len(trees_titles)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]
    for ax, (tree, title) in zip(axes, trees_titles):
        _render_tree_on_axes(ax, tree)
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold")
    if suptitle:
        plt.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()


def show_tree_with_scores(tree, node_scores: Dict[int, float], title: str = "", figsize: tuple = (5, 4)) -> None:
    """Render *tree* with per-node score tinting.

    Nodes are tinted from their base colour (score=0.0, best) to red
    (score=1.0, worst).  Use this with :func:`plagih.targeted_optimization.ifte_component_scores`.

    Args:
        tree: Root node of the tree.
        node_scores: Mapping ``id(node) → float`` in ``[0, 1]``.
        title: Optional subplot title.
        figsize: Figure size in inches.
    """
    from plagih.visualization.tree_renderer import _render_tree_on_axes

    fig, ax = plt.subplots(figsize=figsize)
    _render_tree_on_axes(ax, tree, node_scores=node_scores)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    # Colour legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#ECEFF1", edgecolor="#607D8B", label="score = 0.0 (best)"),
        Patch(facecolor="#FF5252", edgecolor="#B71C1C", label="score = 1.0 (worst)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def make_ifte_node_scores(analysis_results) -> Dict[int, float]:
    """Convert :func:`ifte_component_scores` output to a ``node_scores`` dict.

    Maps each Ifte node's *condition*, *then*, and *else* child nodes to a
    normalised weakness score (``1 − count_score``) for use with
    :func:`show_tree_with_scores`.

    Args:
        analysis_results: List of :class:`IfteAnalysisResult` objects.

    Returns:
        Dict ``{id(node): float}`` ready for :func:`show_tree_with_scores`.
    """
    scores: Dict[int, float] = {}
    for result in analysis_results:
        for comp_score in result.scores.values():
            # weakness = 1 - count_score, already in [0, 1]
            weakness = 1.0 - comp_score.count_score
            scores[result.node_id] = weakness
    return scores
