"""
Targeted Optimization Module for plagih GP Framework.

Provides analysis tools for per-tree and per-population optimization
beyond random evolution. This is the implementation of Phase 1 & 2
from docs/TARGETED_OPTIMIZATION.md.

Key features:
- Per-node intermediate-value evaluation
- Best-per-datapoint (Oracle Selector) analysis
- SoftOptimum error bound computation
- Ifte/Piecewise pseudo-backpropagation scoring

Usage::

    from plagih.targeted_optimization import (
        eval_node_intermediates,
        best_per_datapoint,
        soft_optimum_error,
        ifte_component_scores,
    )

    # Phase 1: Analysis
    intermediates = eval_node_intermediates(tree, df)
    bpd = best_per_datapoint(population, df, target)
    so_error = soft_optimum_error(population, df, target)

    # Phase 2: Ifte scoring
    scores = ifte_component_scores(tree, df, target)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from plagih.trees._evolution import Candidate
    from plagih.trees._nodes import Node


# ---------------------------------------------------------------------------
# Phase 1 — Analysis Infrastructure
# ---------------------------------------------------------------------------


def eval_node_intermediates(
    tree: Node,
    df: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    """Evaluate a tree and return intermediate values at every node.

    Walks the tree bottom-up.  Each node is evaluated via
    ``eval_predict_numpy_now`` and the result is stored keyed by the
    node's ``id()``.

    Args:
        tree: Root node of the computation tree.
        df: Input DataFrame with feature columns.

    Returns:
        Dictionary mapping ``id(node)`` → ``np.ndarray`` of shape
        ``(n_rows,)`` with the intermediate output at that node.
    """
    intermediates: Dict[int, np.ndarray] = {}
    _eval_recursive(tree, df, intermediates)
    return intermediates


def _eval_recursive(
    node: Node,
    df: pd.DataFrame,
    out: Dict[int, np.ndarray],
) -> np.ndarray:
    """Recursively evaluate *node* and populate *out* with intermediate values."""
    from plagih.trees._nodes import Terminal

    if isinstance(node, Terminal):
        result = node.eval_predict_numpy_now(df)
        out[id(node)] = result
        return result

    # Evaluate children first (bottom-up)
    child_results = []
    for child in node.get_childs():
        child_results.append(_eval_recursive(child, df, out))

    # Evaluate this node — use its own eval_predict_numpy_now which
    # will re-evaluate children, but the overhead is acceptable for
    # an analysis tool.  A future optimisation could pass pre-computed
    # children directly, but that would require changing the Node API.
    result = node.eval_predict_numpy_now(df)
    out[id(node)] = result
    return result


# ---------------------------------------------------------------------------
# Best-per-datapoint & SoftOptimum
# ---------------------------------------------------------------------------


@dataclass
class BestPerDatapointResult:
    """Result of a best-per-datapoint analysis.

    Attributes:
        winner_indices: ``(n_rows,)`` int array — index into *population*
            of the candidate closest to the target on each row.
        winner_errors: ``(n_rows,)`` float array — absolute error of the
            winning candidate on each row.
        soft_optimum_error: Scalar — mean of *winner_errors* (the Oracle
            Selector Bound).
        candidate_contributions: ``(n_candidates,)`` int array — how many
            rows each candidate "wins".
        minimum_set_indices: List of candidate indices that form an
            approximate minimum set (greedy set cover).
    """

    winner_indices: np.ndarray
    winner_errors: np.ndarray
    soft_optimum_error: float
    candidate_contributions: np.ndarray
    minimum_set_indices: List[int]


def best_per_datapoint(
    population: List[Candidate],
    df: pd.DataFrame,
    target: np.ndarray,
    error_metric: str = "absolute",
) -> BestPerDatapointResult:
    """Determine which candidate predicts each training row best.

    For every row in *df*, finds the candidate whose prediction is
    closest to *target*.  Also computes the SoftOptimum (Oracle
    Selector) error and an approximate Minimum Set via greedy set cover.

    Args:
        population: List of Candidate objects (must have evaluated trees).
        df: Training DataFrame.
        target: ``(n_rows,)`` array of target values.
        error_metric: ``"absolute"`` (default) or ``"squared"``.

    Returns:
        :class:`BestPerDatapointResult` with all analysis outputs.

    Raises:
        ValueError: If population is empty.
    """
    if not population:
        raise ValueError("Cannot analyse empty population")

    n_rows = len(target)
    n_cands = len(population)

    # Build prediction matrix (n_candidates * n_rows)
    pred_matrix = np.empty((n_cands, n_rows), dtype=np.float64)
    for i, cand in enumerate(population):
        try:
            pred_matrix[i] = cand.get_evotree().eval_predict_numpy_now(df)
        except Exception:
            pred_matrix[i] = np.nan

    # Compute per-row errors
    if error_metric == "squared":
        error_matrix = (pred_matrix - target[np.newaxis, :]) ** 2
    else:  # absolute
        error_matrix = np.abs(pred_matrix - target[np.newaxis, :])

    # Replace NaN with inf so they never win
    error_matrix = np.where(np.isfinite(error_matrix), error_matrix, np.inf)

    # Winner per row
    winner_indices = np.argmin(error_matrix, axis=0)  # (n_rows,)
    winner_errors = error_matrix[winner_indices, np.arange(n_rows)]  # (n_rows,)

    # SoftOptimum error = mean of best-per-row errors
    finite_mask = np.isfinite(winner_errors)
    so_error = float(np.mean(winner_errors[finite_mask])) if finite_mask.any() else float("inf")

    # Candidate contributions
    contributions = np.bincount(winner_indices, minlength=n_cands)

    # Greedy Minimum Set (approximate set cover)
    min_set = _greedy_minimum_set(error_matrix)

    return BestPerDatapointResult(
        winner_indices=winner_indices,
        winner_errors=winner_errors,
        soft_optimum_error=so_error,
        candidate_contributions=contributions,
        minimum_set_indices=min_set,
    )


def soft_optimum_error(
    population: List[Candidate],
    df: pd.DataFrame,
    target: np.ndarray,
) -> float:
    """Compute the SoftOptimum (Oracle Selector Bound) error.

    This is the mean absolute error achievable if an oracle could pick
    the best candidate per training row.  It is the theoretical lower
    bound for any per-row selector over the current population.

    Args:
        population: List of Candidate objects.
        df: Training DataFrame.
        target: ``(n_rows,)`` target values.

    Returns:
        SoftOptimum error (float).  Lower is better.
    """
    return best_per_datapoint(population, df, target).soft_optimum_error


def _greedy_minimum_set(error_matrix: np.ndarray) -> List[int]:
    """Greedy set-cover approximation for the Minimum Set.

    Repeatedly picks the candidate that covers (= is best for) the most
    uncovered rows, until all rows are covered.

    Args:
        error_matrix: ``(n_candidates, n_rows)`` error matrix.

    Returns:
        List of candidate indices forming the approximate minimum set.
    """
    n_cands, n_rows = error_matrix.shape
    uncovered = np.ones(n_rows, dtype=bool)
    selected: List[int] = []

    while uncovered.any():
        # For each candidate, count how many uncovered rows it wins
        sub_errors = error_matrix[:, uncovered]
        # Which candidate is best per uncovered row?
        best_per_row = np.argmin(sub_errors, axis=0)
        # Count wins per candidate
        wins = np.bincount(best_per_row, minlength=n_cands)

        if wins.max() == 0:
            break  # all remaining rows are inf — no candidate can help

        best_cand = int(np.argmax(wins))
        selected.append(best_cand)

        # Mark rows where this candidate is best as covered
        all_row_best = np.argmin(error_matrix[:, uncovered], axis=0)
        covered_mask = all_row_best == best_cand
        uncovered_indices = np.where(uncovered)[0]
        uncovered[uncovered_indices[covered_mask]] = False

    return selected


# ---------------------------------------------------------------------------
# Phase 2 — Ifte / Piecewise Pseudo-Backpropagation
# ---------------------------------------------------------------------------


@dataclass
class IfteComponentScore:
    """Performance score for one component of an Ifte node.

    Attributes:
        component: ``"condition"``, ``"then"``, or ``"else"``.
        count_score: Fraction of rows where this component performs
            correctly / better (0.0 = always wrong, 1.0 = always right).
        error_sum: Total error contribution of this component.
        n_rows: Number of rows this component is responsible for
            (all rows for condition, cond=True rows for then, etc.).
    """

    component: str
    count_score: float
    error_sum: float
    n_rows: int


@dataclass
class IfteAnalysisResult:
    """Full pseudo-backpropagation analysis for one Ifte node.

    Attributes:
        node_id: ``id()`` of the analysed Ifte node.
        scores: Dict mapping component name to its score.
        weakest: Name of the weakest component (highest priority
            for targeted mutation).
        condition_accuracy: Fraction of rows where the condition
            selects the better branch.
    """

    node_id: int
    scores: Dict[str, IfteComponentScore]
    weakest: str
    condition_accuracy: float


def ifte_component_scores(
    tree: Node,
    df: pd.DataFrame,
    target: np.ndarray,
) -> List[IfteAnalysisResult]:
    """Compute pseudo-backpropagation scores for all Ifte nodes in a tree.

    For each ``Ifte(cond, then, else)`` node found in *tree*:

    1. Evaluate ``cond``, ``then``, and ``else`` independently on all rows.
    2. Score each component (see :class:`IfteComponentScore`).
    3. Identify the weakest component for targeted mutation.

    Args:
        tree: Root node of the tree to analyse.
        df: Training DataFrame.
        target: ``(n_rows,)`` target values.

    Returns:
        List of :class:`IfteAnalysisResult`, one per Ifte node found.
        Empty list if the tree contains no Ifte nodes.

    Examples::

        import numpy as np
        from plagih.demo_helpers import (
            make_cartpole_df,
            make_ifte_node_scores,
            make_tree_ifte_cartpole,
            show_tree_with_scores,
        )
        from plagih.targeted_optimization import ifte_component_scores

        df = make_cartpole_df()
        tree = make_tree_ifte_cartpole()  # Ifte(cartPos < 0.5, cartVel, -1 * cartPos)
        target = df["action"].to_numpy()

        results = ifte_component_scores(tree, df, target)
        for r in results:
            print(f"Weakest component: {r.weakest}")
            for name, score in r.scores.items():
                print(f"  {name}: count_score={score.count_score:.2f}, error_sum={score.error_sum:.3f}")

        # Visualise: colour Ifte branches by weakness
        node_scores = make_ifte_node_scores(results)
        show_tree_with_scores(tree, node_scores, title="Ifte weakness map")

        # See docs/demo.ipynb §7 for the full interactive example.
    """
    from plagih.trees._nodes import Ifte

    results: List[IfteAnalysisResult] = []
    _find_and_score_ifte(tree, df, target, results, Ifte)
    return results


def _find_and_score_ifte(
    node: Node,
    df: pd.DataFrame,
    target: np.ndarray,
    results: List[IfteAnalysisResult],
    ifte_cls: type,
) -> None:
    """Recursively find Ifte nodes and compute their component scores."""
    from plagih.trees._nodes import Terminal

    if isinstance(node, Terminal):
        return

    if isinstance(node, ifte_cls):
        analysis = _score_single_ifte(node, df, target)
        if analysis is not None:
            results.append(analysis)

    # Recurse into children
    for child in node.get_childs():
        _find_and_score_ifte(child, df, target, results, ifte_cls)


def _score_single_ifte(
    node: Node,
    df: pd.DataFrame,
    target: np.ndarray,
) -> Optional[IfteAnalysisResult]:
    """Score a single Ifte(cond, then, else) node.

    Returns None if evaluation fails (e.g. malformed tree).
    """
    childs = node.get_childs()
    if len(childs) != 3:
        return None

    cond_node, then_node, else_node = childs

    try:
        cond_vals = cond_node.eval_predict_numpy_now(df)
        then_vals = then_node.eval_predict_numpy_now(df).astype(np.float64)
        else_vals = else_node.eval_predict_numpy_now(df).astype(np.float64)
    except Exception:
        return None

    n_rows = len(target)

    # Ensure cond_vals is boolean
    cond_mask = np.asarray(cond_vals, dtype=bool)

    # --- Score CONDITION ---
    # "correct" if cond selects the branch closer to target
    then_err = np.abs(then_vals - target)
    else_err = np.abs(else_vals - target)
    better_is_then = then_err <= else_err  # True where then is at least as good

    # Condition is "correct" when: (cond=True AND then is better) OR (cond=False AND else is better)
    cond_correct = (cond_mask & better_is_then) | (~cond_mask & ~better_is_then)
    cond_count_score = float(np.mean(cond_correct))

    # Error sum for condition: when it picks the worse branch, sum |better - worse|
    cond_picked_worse = ~cond_correct
    # Difference between the two branches
    branch_diff = np.abs(then_vals - else_vals)
    cond_error_sum = float(np.sum(branch_diff[cond_picked_worse]))

    cond_score = IfteComponentScore(
        component="condition",
        count_score=cond_count_score,
        error_sum=cond_error_sum,
        n_rows=n_rows,
    )

    # --- Score THEN branch ---
    then_active = cond_mask
    n_then = int(np.sum(then_active))
    if n_then > 0:
        then_count = float(np.mean(then_err[then_active] <= else_err[then_active]))
        then_error = float(np.sum(then_err[then_active]))
    else:
        then_count = 1.0  # vacuously good
        then_error = 0.0

    then_score = IfteComponentScore(
        component="then",
        count_score=then_count,
        error_sum=then_error,
        n_rows=n_then,
    )

    # --- Score ELSE branch ---
    else_active = ~cond_mask
    n_else = int(np.sum(else_active))
    if n_else > 0:
        else_count = float(np.mean(else_err[else_active] <= then_err[else_active]))
        else_error = float(np.sum(else_err[else_active]))
    else:
        else_count = 1.0
        else_error = 0.0

    else_score = IfteComponentScore(
        component="else",
        count_score=else_count,
        error_sum=else_error,
        n_rows=n_else,
    )

    # --- Determine weakest component ---
    scores = {"condition": cond_score, "then": then_score, "else": else_score}

    # Weakest = lowest count_score (worst performer)
    weakest = min(scores, key=lambda k: scores[k].count_score)

    return IfteAnalysisResult(
        node_id=id(node),
        scores=scores,
        weakest=weakest,
        condition_accuracy=cond_count_score,
    )


# ---------------------------------------------------------------------------
# Piecewise extension (Phase 2b)
# ---------------------------------------------------------------------------


@dataclass
class PiecewiseComponentScore:
    """Score for one branch of a Piecewise node.

    Attributes:
        branch_index: Index of the ExprCondPair in the Piecewise.
        condition_accuracy: How often this condition fires when its
            expression is best.
        expression_error: Sum of |expr - target| on rows where this
            branch is active.
        n_active_rows: Number of rows where this branch is selected.
        is_default: Whether this is the default (last, True) branch.
    """

    branch_index: int
    condition_accuracy: float
    expression_error: float
    n_active_rows: int
    is_default: bool


def piecewise_component_scores(
    tree: Node,
    df: pd.DataFrame,
    target: np.ndarray,
) -> List[Dict[str, any]]:
    """Analyse all Piecewise nodes in a tree.

    For each ``Piecewise`` node, evaluates every branch independently
    and scores condition accuracy and expression quality.

    Args:
        tree: Root node.
        df: Training DataFrame.
        target: ``(n_rows,)`` target values.

    Returns:
        List of dicts, one per Piecewise node.  Each dict has:
        - ``"node_id"``: id() of the Piecewise node
        - ``"branches"``: list of :class:`PiecewiseComponentScore`
        - ``"weakest_branch"``: index of the worst-performing branch
    """
    from plagih.trees._nodes import Piecewise, Terminal

    results = []
    _find_piecewise(tree, df, target, results, Piecewise, Terminal)
    return results


def _find_piecewise(node, df, target, results, pw_cls, term_cls):
    """Recursively find Piecewise nodes and score them."""
    if isinstance(node, term_cls):
        return

    if isinstance(node, pw_cls):
        analysis = _score_single_piecewise(node, df, target)
        if analysis is not None:
            results.append(analysis)

    for child in node.get_childs():
        _find_piecewise(child, df, target, results, pw_cls, term_cls)


def _score_single_piecewise(node, df, target):
    """Score a single Piecewise node."""
    pairs = node.get_childs()  # List of ExprCondPair_Dummy nodes
    if not pairs:
        return None

    n_rows = len(target)
    branch_scores = []

    # Evaluate all expressions and conditions
    expressions = []
    conditions = []
    for pair in pairs:
        try:
            expr_val = pair.childs[0].eval_predict_numpy_now(df).astype(np.float64)
            cond_val = pair.childs[1].eval_predict_numpy_now(df)
            expressions.append(expr_val)
            conditions.append(np.asarray(cond_val, dtype=bool))
        except Exception:
            return None

    # Determine which branch is actually selected per row
    # (first True condition wins, like numpy Piecewise evaluation)
    active_branch = np.full(n_rows, len(pairs) - 1, dtype=int)  # default = last
    for i in range(len(pairs) - 2, -1, -1):
        active_branch[conditions[i]] = i

    # Find which expression is best per row (regardless of condition)
    expr_errors = np.array([np.abs(e - target) for e in expressions])  # (n_branches, n_rows)
    best_branch = np.argmin(expr_errors, axis=0)  # (n_rows,)

    for i in range(len(pairs)):
        is_default = i == len(pairs) - 1
        is_active = active_branch == i
        n_active = int(np.sum(is_active))

        # Condition accuracy: of the rows where this branch's expression
        # is best, how often does the condition actually select it?
        best_for_this = best_branch == i
        n_best = int(np.sum(best_for_this))
        if n_best > 0:
            cond_acc = float(np.sum(best_for_this & is_active)) / n_best
        else:
            cond_acc = 1.0  # vacuously correct

        # Expression error on active rows
        if n_active > 0:
            expr_error = float(np.sum(np.abs(expressions[i][is_active] - target[is_active])))
        else:
            expr_error = 0.0

        branch_scores.append(
            PiecewiseComponentScore(
                branch_index=i,
                condition_accuracy=cond_acc,
                expression_error=expr_error,
                n_active_rows=n_active,
                is_default=is_default,
            )
        )

    # Weakest branch = lowest condition accuracy (excluding default which has no condition)
    non_default = [s for s in branch_scores if not s.is_default]
    if non_default:
        weakest = min(non_default, key=lambda s: s.condition_accuracy).branch_index
    else:
        weakest = 0

    return {
        "node_id": id(node),
        "branches": branch_scores,
        "weakest_branch": weakest,
    }
