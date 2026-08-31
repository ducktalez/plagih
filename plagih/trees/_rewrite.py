"""
Local rewrite-rule engine for plagih trees (D11 direction 2).

SymPy-free structural simplification: exact, deterministic, O(n) per pass,
idempotent via fixpoint iteration.  Complements ``tree_node_grouping()``
(which handles SymPy round-trip artefacts) with classic algebraic cleanup
that needs no symbolic engine at all.

Rules (all semantics-exact in float arithmetic):

===============================  ==========================================
Rule                             Example
===============================  ==========================================
constant folding                 ``Add(2, 3) -> 5``; ``Abs(-4) -> 4``
additive neutral                 ``Add(x, 0) -> x``  (chain-aware)
multiplicative neutral           ``Mul(x, 1) -> x``  (chain-aware)
multiplicative absorber          ``Mul(x, 0) -> 0``
subtractive neutral              ``Sub(x, 0) -> x``
division neutral                 ``Div(x, 1) -> x``
double negation                  ``Usub(Usub(x)) -> x``
double logical negation          ``Not(Not(x)) -> x``
===============================  ==========================================

Guarantees:
- ``is_fix`` nodes are never replaced or dropped (origin_tree skeletons).
- Only finite fold results are accepted (``Div(1, 0)`` stays untouched).
- Never grows the tree.

Usage::

    from plagih.trees._rewrite import rewrite_fixpoint

    tree = rewrite_fixpoint(tree)  # in-place, returns the same root
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from plagih.trees._nodes import Node

# Operators safe for scalar constant folding (exact float semantics,
# no display-relevant symbolic forms like sin(1) — see D6 question 2).
_FOLDABLE = ("Add", "Sub", "Mul", "Div", "Usub", "Square", "Abs", "Min", "Max")

MAX_PASSES = 10


def rewrite_fixpoint(tree: Node, max_passes: int = MAX_PASSES) -> Node:
    """Apply local rewrite rules until nothing changes.

    In-place: node identity of the root is preserved (``set_new_node``
    swaps class/dict).  Back-references are repaired once at the end.

    Args:
        tree: Root of the tree to simplify.
        max_passes: Safety bound for the fixpoint loop.

    Returns:
        The same root object, simplified.
    """
    changed_any = False
    for _ in range(max_passes):
        if not _rewrite_pass(tree):
            break
        changed_any = True
    if changed_any:
        tree.repair_all()
    return tree


def _rewrite_pass(node: Node) -> bool:
    """One bottom-up pass. Returns True when something was rewritten."""
    from plagih.trees._nodes import Terminal

    if isinstance(node, Terminal):
        return False

    changed = False
    for child in list(node.get_childs()):
        changed |= _rewrite_pass(child)

    if node.is_fix:
        return changed  # frozen skeleton — structure untouchable

    changed |= _try_rules(node)
    return changed


def _num_value(node: Node) -> Optional[float]:
    """Float value when *node* is a mutable Number, else None."""
    from plagih.trees._nodes import Number

    if isinstance(node, Number) and not node.is_fix:
        try:
            return float(node.get_value())
        except (TypeError, ValueError):
            return None
    return None


def _make_number(value: float) -> Node:
    from plagih.trees._nodes import Number

    return Number(value)


def _replace(node: Node, replacement: Node) -> None:
    """Swap *node* for *replacement* in place (root-safe)."""
    node.set_new_node(replacement)


def _try_rules(node: Node) -> bool:
    """Apply the first matching rule to *node*. Returns True on rewrite."""
    from plagih.trees._nodes import Not, Usub

    op = type(node).__name__
    childs = node.get_childs()

    # --- constant folding -------------------------------------------------
    if op in _FOLDABLE:
        vals = [_num_value(c) for c in childs]
        if all(v is not None for v in vals):
            folded = _fold(op, vals)  # type: ignore[arg-type]
            if folded is not None and math.isfinite(folded):
                _replace(node, _make_number(folded))
                return True

    # --- neutral / absorbing elements ------------------------------------
    if op == "Add":
        return _drop_neutral_children(node, neutral=0.0)

    if op == "Mul":
        # Absorber first: any literal 0 kills the product
        if any(v == 0.0 for v in (_num_value(c) for c in childs) if v is not None):
            if not any(c.is_fix for c in childs):
                _replace(node, _make_number(0.0))
                return True
        return _drop_neutral_children(node, neutral=1.0)

    if op == "Sub" and len(childs) == 2 and _num_value(childs[1]) == 0.0:
        _replace(node, childs[0])
        return True

    if op == "Div" and len(childs) == 2 and _num_value(childs[1]) == 1.0:
        _replace(node, childs[0])
        return True

    if isinstance(node, Usub) and isinstance(childs[0], Usub) and not childs[0].is_fix:
        _replace(node, childs[0].get_childs()[0])
        return True

    if isinstance(node, Not) and isinstance(childs[0], Not) and not childs[0].is_fix:
        _replace(node, childs[0].get_childs()[0])
        return True

    return False


def _fold(op: str, vals: list) -> Optional[float]:
    """Scalar fold for _FOLDABLE operators."""
    if op == "Add":
        return float(sum(vals))
    if op == "Sub":
        return float(vals[0] - vals[1]) if len(vals) == 2 else None
    if op == "Mul":
        out = 1.0
        for v in vals:
            out *= v
        return float(out)
    if op == "Div":
        if len(vals) != 2 or vals[1] == 0.0:
            return None
        return float(vals[0] / vals[1])
    if op == "Usub":
        return float(-vals[0])
    if op == "Square":
        return float(vals[0] * vals[0])
    if op == "Abs":
        return float(abs(vals[0]))
    if op == "Min":
        return float(min(vals))
    if op == "Max":
        return float(max(vals))
    return None


def _drop_neutral_children(node: Node, neutral: float) -> bool:
    """Remove literal *neutral* operands from a chainable node.

    Keeps arity >= 1; a single survivor replaces the node itself.
    Fixed children are never dropped.
    """
    childs = node.get_childs()
    keep = [c for c in childs if c.is_fix or _num_value(c) != neutral]

    if len(keep) == len(childs) or not keep:
        return False  # nothing to drop, or everything was neutral (leave to folding)

    if len(keep) == 1:
        _replace(node, keep[0])
    else:
        node.set_childs(keep)
    return True
