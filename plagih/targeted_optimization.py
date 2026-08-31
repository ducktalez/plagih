"""
Targeted Optimization Module for plagih GP Framework.

Provides analysis tools for per-tree and per-population optimization
beyond random evolution. This is the implementation of Phase 1, 2 & 3
from docs/TARGETED_OPTIMIZATION.md.

Key features:
- Per-node intermediate-value evaluation
- Best-per-datapoint (Oracle Selector) analysis
- SoftOptimum error bound computation
- Ifte/Piecewise pseudo-backpropagation scoring
- Node-level optimization gaps for invertible operators

Usage::

    from plagih.targeted_optimization import (
        eval_node_intermediates,
        best_per_datapoint,
        soft_optimum_error,
        ifte_component_scores,
        node_optimization_gaps,
    )

    # Phase 1: Analysis
    intermediates = eval_node_intermediates(tree, df)
    bpd = best_per_datapoint(population, df, target)
    so_error = soft_optimum_error(population, df, target)

    # Phase 2: Ifte scoring
    scores = ifte_component_scores(tree, df, target)

    # Phase 3: Node-level optimization gaps
    gaps = node_optimization_gaps(tree, df, target)
    worst = largest_gap_node(gaps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
) -> List[Dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Phase 3 — General node-level optimization (§3.2)
# ---------------------------------------------------------------------------

# Operators whose output can be inverted exactly for one child, given the
# node's ideal output and the actual values of the sibling children.
# Non-invertible operators (Abs, Sign, Square, trigonometry, Ifte, ...) stop
# the backward propagation — their children get no ideal value.
INVERTIBLE_OPERATORS = frozenset(
    {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Scale",
        "Usub",
        "DivFraction",
    }
)


@dataclass
class NodeGap:
    """Optimization gap of one node (§3.2 in docs/TARGETED_OPTIMIZATION.md).

    The *ideal value* is what this node would have to output — with the rest
    of the tree unchanged — so the tree output equals the target.  The gap is
    the distance between actual and ideal output.

    Attributes:
        node_id: ``id()`` of the node.  Valid only for the analysed tree object.
        operator: Class name of the node (e.g. ``"Add"``, ``"Symbol"``).
        depth: Distance from the root (root = 0).
        gap_mean: Mean ``|actual - ideal|`` over rows with finite values.
        gap_sum: Sum of ``|actual - ideal|`` over those rows.
        n_finite: Number of rows where the ideal value is finite.
        is_root: Whether this node is the analysed tree's root.
    """

    node_id: int
    operator: str
    depth: int
    gap_mean: float
    gap_sum: float
    n_finite: int
    is_root: bool


def node_optimization_gaps(
    tree: Node,
    df: pd.DataFrame,
    target: np.ndarray,
) -> List[NodeGap]:
    """Compute per-node optimization gaps by inverse (backward) propagation.

    Starting at the root — whose ideal output *is* the target — the ideal
    value is pushed down through invertible operators
    (:data:`INVERTIBLE_OPERATORS`).  For each reached node the gap between
    its actual and ideal output is recorded.

    Example: for ``Add(a, b, c)`` the ideal value of ``c`` on row ``i`` is
    ``target_i - a_i - b_i``; the gap of ``c`` is ``|c_i - ideal_c_i|``.

    Non-invertible operators (``Abs``, ``Sign``, ``Square``, trigonometry,
    ``Ifte``, ...) terminate the propagation: they still get their own gap,
    but their children do not.  Use :func:`ifte_component_scores` for the
    Ifte/Piecewise-specific analysis instead.

    Args:
        tree: Root node of the tree to analyse (read-only, never modified).
        df: Training DataFrame.
        target: ``(n_rows,)`` target values.

    Returns:
        List of :class:`NodeGap`, ordered depth-first from the root.
        Empty list if the tree cannot be evaluated.

    Examples::

        from plagih.targeted_optimization import largest_gap_node, node_optimization_gaps

        gaps = node_optimization_gaps(tree, df, target)
        for g in gaps:
            print(f"{g.operator:12s} depth={g.depth} gap_mean={g.gap_mean:.3f}")

        worst = largest_gap_node(gaps)  # best candidate for targeted mutation
    """
    try:
        intermediates = eval_node_intermediates(tree, df)
    except Exception:
        return []

    target_arr = np.asarray(target, dtype=np.float64)
    gaps: List[NodeGap] = []
    _propagate_ideal(tree, target_arr, intermediates, gaps, depth=0, is_root=True)
    return gaps


def largest_gap_node(
    gaps: List[NodeGap],
    exclude_root: bool = True,
) -> Optional[NodeGap]:
    """Return the node with the largest mean optimization gap.

    This is the "weakest link" — the preferred target for gap-guided
    mutation (Phase 3 of docs/TARGETED_OPTIMIZATION.md).

    Args:
        gaps: Output of :func:`node_optimization_gaps`.
        exclude_root: Skip the root node.  Mutating the root replaces the
            whole tree, which is what plain branch mutation already does.

    Returns:
        The :class:`NodeGap` with the highest ``gap_mean``, or ``None`` if
        no node has a usable (finite, non-zero-row) gap.
    """
    usable = [g for g in gaps if g.n_finite > 0 and np.isfinite(g.gap_mean)]
    if exclude_root:
        usable = [g for g in usable if not g.is_root]
    if not usable:
        return None
    return max(usable, key=lambda g: g.gap_mean)


def _propagate_ideal(
    node: Node,
    ideal: np.ndarray,
    intermediates: Dict[int, np.ndarray],
    out: List[NodeGap],
    depth: int,
    is_root: bool = False,
) -> None:
    """Record the gap of *node* and push ideal values down to its children."""
    actual = intermediates.get(id(node))
    if actual is None:
        return

    out.append(_make_node_gap(node, actual, ideal, depth, is_root))

    if not node.has_childs():
        return

    childs = node.get_childs()
    if type(node).__name__ not in INVERTIBLE_OPERATORS:
        return  # non-invertible — stop propagation here

    child_vals = [intermediates.get(id(c)) for c in childs]
    if any(v is None for v in child_vals):
        return

    child_ideals = _invert_operator(node, ideal, child_vals)
    if child_ideals is None:
        return

    for child, child_ideal in zip(childs, child_ideals):
        if child_ideal is not None:
            _propagate_ideal(child, child_ideal, intermediates, out, depth + 1)


def _make_node_gap(
    node: Node,
    actual: np.ndarray,
    ideal: np.ndarray,
    depth: int,
    is_root: bool,
) -> NodeGap:
    """Build a :class:`NodeGap` from actual vs. ideal output vectors."""
    actual_f = np.asarray(actual, dtype=np.float64)
    diff = np.abs(actual_f - ideal)
    finite = np.isfinite(diff)
    n_finite = int(np.sum(finite))

    if n_finite > 0:
        gap_sum = float(np.sum(diff[finite]))
        gap_mean = gap_sum / n_finite
    else:
        gap_sum = float("inf")
        gap_mean = float("inf")

    return NodeGap(
        node_id=id(node),
        operator=type(node).__name__,
        depth=depth,
        gap_mean=gap_mean,
        gap_sum=gap_sum,
        n_finite=n_finite,
        is_root=is_root,
    )


def _invert_operator(
    node: Node,
    ideal: np.ndarray,
    child_vals: List[np.ndarray],
) -> Optional[List[Optional[np.ndarray]]]:
    """Compute the ideal value of each child, given the node's ideal output.

    Args:
        node: The (invertible) operator node.
        ideal: ``(n_rows,)`` ideal output of *node*.
        child_vals: Actual output of each child.

    Returns:
        List with one entry per child — an ideal-value array, or ``None``
        when that child cannot be inverted.  Returns ``None`` entirely if
        the operator arity does not match expectations.
    """
    op = type(node).__name__
    vals = [np.asarray(v, dtype=np.float64) for v in child_vals]

    # Division by zero produces inf/nan; those rows are masked out later.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "Add":
            total = np.sum(vals, axis=0)
            # ideal_ci = ideal - sum(siblings) = ideal - (total - ci)
            return [ideal - (total - v) for v in vals]

        if op == "Sub":
            if len(vals) != 2:
                return None
            a, b = vals
            return [ideal + b, a - ideal]

        if op == "Mul":
            product = np.prod(vals, axis=0)
            result: List[Optional[np.ndarray]] = []
            for v in vals:
                siblings = np.divide(product, v, out=np.full_like(product, np.nan), where=v != 0)
                result.append(np.divide(ideal, siblings, out=np.full_like(ideal, np.nan), where=siblings != 0))
            return result

        if op == "Div":
            if len(vals) != 2:
                return None
            a, b = vals
            ideal_a = ideal * b
            ideal_b = np.divide(a, ideal, out=np.full_like(a, np.nan), where=ideal != 0)
            return [ideal_a, ideal_b]

        if op == "Scale":
            # Scale(c, expr) == c * expr, c is a Number terminal
            if len(vals) != 2:
                return None
            c, expr = vals
            ideal_c = np.divide(ideal, expr, out=np.full_like(ideal, np.nan), where=expr != 0)
            ideal_expr = np.divide(ideal, c, out=np.full_like(ideal, np.nan), where=c != 0)
            return [ideal_c, ideal_expr]

        if op == "Usub":
            if len(vals) != 1:
                return None
            return [-ideal]

        if op == "DivFraction":
            # out == 1 / a  =>  ideal_a == 1 / ideal
            if len(vals) != 1:
                return None
            return [np.divide(1.0, ideal, out=np.full_like(ideal, np.nan), where=ideal != 0)]

    return None
