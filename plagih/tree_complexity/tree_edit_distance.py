"""Tree Edit Distance for the plagih GP Framework.

Implements the Zhang-Shasha algorithm [1] directly on plagih Node objects.
No string conversion, no intermediate data structure, no external library.

The algorithm traverses the Node tree via ``node.has_childs()`` /
``node.get_childs()`` and compares labels via ``type(node)`` /
``node.get_value()``.  The only auxiliary data are two plain Python lists
per tree:

- ``po``  (``List[Node]``) – node references in postorder (no copies)
- ``lld`` (``List[int]``)  – leftmost-leaf-descendant postorder index

Three label-comparison modes are supported via :class:`TedConfig`:

* **structural** – compares only node type (class).
* **full** – type for operators; type + value for terminals.
* **structural_plus_leaf_diff** – structural TED + leaf-value mismatch count.

Architecture Note – Upgrade Path to APTED
==========================================
Zhang-Shasha is the *LEFT-only* special case of the full APTED algorithm.
APTED extends ZS by computing an optimal decomposition strategy that
chooses the cheapest path type (LEFT / RIGHT / INNER) per sub-problem.

To upgrade later:
  (a) Add a ``NodeIndexer`` with pre-/post-order indices, LLD/RLD, sizes.
  (b) Implement ``compute_opt_strategy()`` for LEFT/RIGHT/INNER selection.
  (c) Replace the fixed left-path decomposition with ``gted()`` driven
      by the optimal strategy.

.. todo::
   Explore LR-path decomposition for:
   - Path-based crossover-point selection (crossover at the mapping break).
   - Tree similarity along dominant paths for diversity clustering.

.. todo::
   Edit-Mapping can be used for *targeted* mutation: mutate where the tree
   diverges most from the origin / target.  Also useful for intelligent
   crossover (crossover point at mapping break).
   See ``Node.compute_ted(...).mapping``.

References
----------
[1] K. Zhang and D. Shasha.  SIAM J. Comput. 18(6), 1989.
[2] M. Pawlik and N. Augsten.  ACM TODS 40(1), 2015.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from plagih.trees import Node


# ---------------------------------------------------------------------------
# Configuration & Result types
# ---------------------------------------------------------------------------


@dataclass
class TedConfig:
    """Configuration for tree edit distance computation.

    Attributes:
        mode: Label comparison mode.
            ``"structural"``  – compare only node type (class).
            ``"full"``        – type for operators; type+value for terminals.
            ``"structural_plus_leaf_diff"`` – structural TED + leaf-diff count.
        insert_cost: Cost of inserting a node (default 1).
        delete_cost: Cost of deleting a node (default 1).
        rename_cost: Fixed rename cost.  ``None`` (default) → auto:
            0 when labels are identical, 1 otherwise.
    """

    mode: str = "full"
    insert_cost: float = 1.0
    delete_cost: float = 1.0
    rename_cost: Optional[float] = None  # None → auto (0 same, 1 different)

    def __post_init__(self):
        valid_modes = ("structural", "full", "structural_plus_leaf_diff")
        if self.mode not in valid_modes:
            raise ValueError(f"TedConfig.mode must be one of {valid_modes}, got {self.mode!r}")


@dataclass
class TedResult:
    """Result of a tree edit distance computation.

    Attributes:
        distance: The tree edit distance (total cost of edit operations).
        mapping: List of ``(node_from_tree1, node_from_tree2)`` pairs.
            ``None`` on either side indicates an insertion or deletion.
        leaf_diff_count: Count of mapped terminal pairs whose *values*
            differ (only populated for mode ``"structural_plus_leaf_diff"``).
    """

    distance: float
    mapping: List[Tuple[Optional[Any], Optional[Any]]] = field(default_factory=list)
    leaf_diff_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Postorder traversal + leftmost-leaf-descendant (LLD)
# ---------------------------------------------------------------------------


def _postorder_with_lld(root: Node) -> Tuple[List[Node], List[int]]:
    """Collect *root*'s nodes in postorder and compute LLD indices.

    Traverses the tree directly via ``node.has_childs()`` /
    ``node.get_childs()``.  No intermediate data structure is created —
    the returned list contains references to the original Node objects.

    Returns:
        po:  Node references in postorder.  ``po[i]`` has 1-based
             postorder id ``i + 1``.
        lld: 1-based int array.  ``lld[k]`` is the postorder id of the
             leftmost leaf descendant of the node with postorder id ``k``.
             ``lld[0]`` is unused padding.
    """
    po: List[Node] = []
    lld: List[int] = [0]  # index 0 unused; 1-based

    def _visit(node: Node) -> int:
        first_child_lld: Optional[int] = None
        if node.has_childs():
            for child in node.get_childs():
                child_lld = _visit(child)
                if first_child_lld is None:
                    first_child_lld = child_lld

        po.append(node)
        my_lld = first_child_lld if first_child_lld is not None else len(po)
        lld.append(my_lld)
        return my_lld

    _visit(root)
    return po, lld


# ---------------------------------------------------------------------------
# Key roots
# ---------------------------------------------------------------------------


def _keyroots(lld: List[int], n: int) -> List[int]:
    """Compute Zhang-Shasha key roots from the LLD array.

    Returns sorted postorder indices (1-based) where each unique LLD value
    is represented by the node with the highest postorder index.
    """
    lld_to_max: Dict[int, int] = {}
    for i in range(1, n + 1):
        v = lld[i]
        if v not in lld_to_max or i > lld_to_max[v]:
            lld_to_max[v] = i
    return sorted(lld_to_max.values())


# ---------------------------------------------------------------------------
# Rename cost (reads type / value directly from Node objects)
# ---------------------------------------------------------------------------


def _rename_cost(a: Node, b: Node, config: TedConfig) -> float:
    """Compute the rename cost between two Node objects."""
    if config.rename_cost is not None:
        if type(a) is type(b):
            if not a.is_term():
                return 0.0
            if a.get_value() == b.get_value():
                return 0.0
        return config.rename_cost

    mode = config.mode
    if mode in ("structural", "structural_plus_leaf_diff"):
        return 0.0 if type(a) is type(b) else 1.0
    # mode == "full"
    if type(a) is not type(b):
        return 1.0
    if a.is_term() and a.get_value() != b.get_value():
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Zhang-Shasha: distance computation
# ---------------------------------------------------------------------------


def _compute_distances(
    po1: List[Node],
    lld1: List[int],
    po2: List[Node],
    lld2: List[int],
    config: TedConfig,
) -> List[List[float]]:
    """Compute the tree-distance matrix ``td[i][j]`` (1-based).

    Traverses the Node objects directly via the postorder lists.
    """
    n, m = len(po1), len(po2)
    kr1 = _keyroots(lld1, n)
    kr2 = _keyroots(lld2, m)

    del_c = config.delete_cost
    ins_c = config.insert_cost

    td: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]

    for x in kr1:
        for y in kr2:
            lx, ly = lld1[x], lld2[y]

            fd: Dict[Tuple[int, int], float] = {(lx - 1, ly - 1): 0.0}
            for i in range(lx, x + 1):
                fd[(i, ly - 1)] = fd[(i - 1, ly - 1)] + del_c
            for j in range(ly, y + 1):
                fd[(lx - 1, j)] = fd[(lx - 1, j - 1)] + ins_c

            for i in range(lx, x + 1):
                node_i = po1[i - 1]
                for j in range(ly, y + 1):
                    node_j = po2[j - 1]
                    rc = _rename_cost(node_i, node_j, config)

                    if lld1[i] == lx and lld2[j] == ly:
                        fd[(i, j)] = min(
                            fd[(i - 1, j)] + del_c,
                            fd[(i, j - 1)] + ins_c,
                            fd[(i - 1, j - 1)] + rc,
                        )
                        td[i][j] = fd[(i, j)]
                    else:
                        fd[(i, j)] = min(
                            fd[(i - 1, j)] + del_c,
                            fd[(i, j - 1)] + ins_c,
                            fd[(lld1[i] - 1, lld2[j] - 1)] + td[i][j],
                        )
    return td


# ---------------------------------------------------------------------------
# Zhang-Shasha: mapping extraction via backtracking
# ---------------------------------------------------------------------------


def _extract_mapping(
    po1: List[Node],
    lld1: List[int],
    po2: List[Node],
    lld2: List[int],
    td: List[List[float]],
    config: TedConfig,
) -> List[Tuple[Optional[Node], Optional[Node]]]:
    """Extract the edit mapping by backtracking through forest distances."""
    n, m = len(po1), len(po2)
    del_c = config.delete_cost
    ins_c = config.insert_cost

    mapping: List[Tuple[Optional[Node], Optional[Node]]] = []
    worklist: List[Tuple[int, int]] = [(n, m)]

    while worklist:
        ri, rj = worklist.pop()
        li, lj = lld1[ri], lld2[rj]

        fd: Dict[Tuple[int, int], float] = {(li - 1, lj - 1): 0.0}
        for i in range(li, ri + 1):
            fd[(i, lj - 1)] = fd[(i - 1, lj - 1)] + del_c
        for j in range(lj, rj + 1):
            fd[(li - 1, j)] = fd[(li - 1, j - 1)] + ins_c

        for i in range(li, ri + 1):
            for j in range(lj, rj + 1):
                rc = _rename_cost(po1[i - 1], po2[j - 1], config)
                if lld1[i] == li and lld2[j] == lj:
                    fd[(i, j)] = min(
                        fd[(i - 1, j)] + del_c,
                        fd[(i, j - 1)] + ins_c,
                        fd[(i - 1, j - 1)] + rc,
                    )
                else:
                    fd[(i, j)] = min(
                        fd[(i - 1, j)] + del_c,
                        fd[(i, j - 1)] + ins_c,
                        fd[(lld1[i] - 1, lld2[j] - 1)] + td[i][j],
                    )

        # Backtrack
        i, j = ri, rj
        while i >= li and j >= lj:
            rc = _rename_cost(po1[i - 1], po2[j - 1], config)

            if lld1[i] == li and lld2[j] == lj:
                if fd[(i, j)] == fd[(i - 1, j)] + del_c:
                    mapping.append((po1[i - 1], None))
                    i -= 1
                elif fd[(i, j)] == fd[(i, j - 1)] + ins_c:
                    mapping.append((None, po2[j - 1]))
                    j -= 1
                else:
                    mapping.append((po1[i - 1], po2[j - 1]))
                    i -= 1
                    j -= 1
            else:
                if fd[(i, j)] == fd[(i - 1, j)] + del_c:
                    mapping.append((po1[i - 1], None))
                    i -= 1
                elif fd[(i, j)] == fd[(i, j - 1)] + ins_c:
                    mapping.append((None, po2[j - 1]))
                    j -= 1
                else:
                    worklist.append((i, j))
                    i = lld1[i] - 1
                    j = lld2[j] - 1

        while i >= li:
            mapping.append((po1[i - 1], None))
            i -= 1
        while j >= lj:
            mapping.append((None, po2[j - 1]))
            j -= 1

    return mapping


# ---------------------------------------------------------------------------
# Public entry point (called by Node.compute_ted)
# ---------------------------------------------------------------------------


def zhang_shasha_ted(
    tree1: Node,
    tree2: Node,
    config: Optional[TedConfig] = None,
) -> TedResult:
    """Compute tree edit distance between two plagih Node trees.

    Uses the Zhang-Shasha algorithm [1] operating directly on Node objects.
    No string conversion or external library required.

    Args:
        tree1: Source tree root node.
        tree2: Destination tree root node.
        config: TED configuration (label mode, costs).
            Defaults to ``TedConfig(mode="full")``.

    Returns:
        :class:`TedResult` with distance, edit mapping, and optional
        ``leaf_diff_count``.

    Complexity:
        Time  O(n² · m²),  Space O(n · m)
    """
    if config is None:
        config = TedConfig()

    po1, lld1 = _postorder_with_lld(tree1)
    po2, lld2 = _postorder_with_lld(tree2)

    n, m = len(po1), len(po2)

    if n == 0 and m == 0:
        return TedResult(distance=0.0)
    if n == 0:
        return TedResult(distance=m * config.insert_cost, mapping=[(None, nd) for nd in po2])
    if m == 0:
        return TedResult(distance=n * config.delete_cost, mapping=[(nd, None) for nd in po1])

    td = _compute_distances(po1, lld1, po2, lld2, config)
    distance = td[n][m]

    mapping = _extract_mapping(po1, lld1, po2, lld2, td, config)

    leaf_diff_count: Optional[int] = None
    if config.mode == "structural_plus_leaf_diff":
        leaf_diff_count = sum(
            1
            for a, b in mapping
            if a is not None and b is not None and a.is_term() and b.is_term() and a.get_value() != b.get_value()
        )

    return TedResult(distance=distance, mapping=mapping, leaf_diff_count=leaf_diff_count)


# ---------------------------------------------------------------------------
# Deprecated legacy wrapper
# ---------------------------------------------------------------------------


def apted_distance(expr1: str, expr2: str):
    """**Deprecated.** Was used with the external ``apted`` library.

    Use :meth:`Node.compute_ted` instead.
    """
    warnings.warn(
        "apted_distance() is deprecated.  Use Node.compute_ted() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from apted import APTED  # type: ignore[import-untyped]
        from apted.helpers import Tree as aptree  # type: ignore[import-untyped]

        tree1 = aptree.from_text(expr1)
        tree2 = aptree.from_text(expr2)
        ap = APTED(tree1, tree2)
        ted = ap.compute_edit_distance()
        mapping = ap.compute_edit_mapping()
        return ted, mapping
    except ImportError:
        raise ImportError(
            "The 'apted' package is no longer a dependency.  Use Node.compute_ted(other_node) instead."
        ) from None
