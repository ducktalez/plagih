"""
Population Merge Module for plagih GP Framework

Provides strategies for merging multiple trees from a population into unified
structures for efficient evaluation, visualization, and analysis.

=============================================================================
IMPLEMENTED STRATEGIES:
=============================================================================

### "One-evaluation-tree" (IMPLEMENTED)
- Collects all unique expressions per depth level from the population.
- Builds a DAG (Directed Acyclic Graph) where no expression is computed twice.
- Suitable for feed-forward batch evaluation (e.g., with TensorFlow).

=============================================================================
PLANNED STRATEGIES (TODO):
=============================================================================

### "Expand-random-tree"
- A random tree is selected as the starting point.
- For each additional tree, only the necessary nodes/branches are added.
- Results in a large tree containing all trees, but only required nodes.
- Non-deterministic, heavily dependent on the starting tree.

### "Clustered-tree"
- All trees are grouped into clusters based on similarity (e.g., using APTED).
- For each cluster, a "median tree" is created with common structures.
- Individual trees are represented as deviations from the median tree.
- Results in a hierarchical structure highlighting similarities/differences.

=============================================================================
SUB-STRATEGIES (can be combined with main strategies):
=============================================================================

### "Sympified-merged-tree"
- All trees are converted to sympy expressions.
- Different sympy simplification strategies can be applied:
  - factorized, expanded, trigsimp, etc.

### "Operator-focused-tree"
- Number nodes can optionally be included without values.
- Symbols could also be ignored, creating a merge-tree with operators only.
- Useful for structural analysis independent of constants.

### "Chainable-merged-tree"
- Chainable nodes (e.g., Add/Mul/Max as Floor/Ceiling functions) are merged.
- This is partially done by sympy already.
- Could be extended for custom chain handling.

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union, TYPE_CHECKING

import sympy

from plagih.util import TEXT_NEWLINE

if TYPE_CHECKING:
    from plagih.trees import Node, Candidate


# =============================================================================
# Data Structures for Merged Evaluation Graph
# =============================================================================

@dataclass
class MergedNode:
    """A node in the merged evaluation graph.

    Represents a unique (sub-)expression that may be shared across multiple
    trees in the population.

    Attributes:
        node_id: Unique identifier for this node in the merged graph.
        sympy_expr: The normalized sympy expression this node represents.
        original_nodes: List of (tree_index, original_node) tuples that map to this.
        child_ids: List of node_ids of child nodes (for operators).
        parent_ids: Set of node_ids that use this node as input.
        depth: The depth level in the expression tree (0 = terminals).
        node_type: String indicating the type ('terminal', 'operator', 'root').
        operator_name: For operators, the name (e.g., 'Add', 'Mul').
        is_root: Whether this node is a root of one or more original trees.
        root_of_trees: List of tree indices where this node is the root.
    """
    node_id: str
    sympy_expr: sympy.Basic
    original_nodes: List[Tuple[int, 'Node']] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    parent_ids: Set[str] = field(default_factory=set)
    depth: int = 0
    node_type: str = "unknown"
    operator_name: str = ""
    is_root: bool = False
    root_of_trees: List[int] = field(default_factory=list)

    def __repr__(self):
        root_marker = " [ROOT]" if self.is_root else ""
        return f"MergedNode({self.node_id}: {self.sympy_expr}{root_marker})"


@dataclass
class ExpressionLayer:
    """Represents all expressions at a specific depth level.

    Used during the bottom-up construction of the merged graph.

    Attributes:
        depth: The depth level (0 = terminals, increasing towards root).
        expressions: Dict mapping sympy_expr -> MergedNode for this level.
    """
    depth: int
    expressions: Dict[sympy.Basic, MergedNode] = field(default_factory=dict)


class MergedEvaluationGraph:
    """A merged DAG representing multiple trees from a population.

    This structure enables efficient batch evaluation where each unique
    sub-expression is computed only once, regardless of how many trees
    use it.

    Attributes:
        nodes: Dict mapping node_id -> MergedNode.
        expr_to_id: Dict mapping sympy expression -> node_id for deduplication.
        root_ids: List of node_ids that are roots of original trees.
        tree_count: Number of trees merged into this graph.
        layers: List of ExpressionLayer for bottom-up construction.
    """

    def __init__(self):
        self.nodes: Dict[str, MergedNode] = {}
        self.expr_to_id: Dict[sympy.Basic, str] = {}
        self.root_ids: List[str] = []
        self.tree_count: int = 0
        self.layers: List[ExpressionLayer] = []
        self._node_counter: int = 0

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        node_id = f"n{self._node_counter}"
        self._node_counter += 1
        return node_id

    def get_or_create_node(
        self,
        sympy_expr: sympy.Basic,
        depth: int,
        node_type: str,
        operator_name: str = "",
        child_ids: List[str] = None
    ) -> str:
        """Get existing node for expression or create a new one.

        Args:
            sympy_expr: The normalized sympy expression.
            depth: Depth level in the tree.
            node_type: 'terminal' or 'operator'.
            operator_name: Name of operator (for operators).
            child_ids: List of child node IDs (for operators).

        Returns:
            The node_id of the (existing or new) node.
        """
        # Check if expression already exists
        if sympy_expr in self.expr_to_id:
            return self.expr_to_id[sympy_expr]

        # Create new node
        node_id = self._generate_node_id()
        node = MergedNode(
            node_id=node_id,
            sympy_expr=sympy_expr,
            depth=depth,
            node_type=node_type,
            operator_name=operator_name,
            child_ids=child_ids or []
        )

        self.nodes[node_id] = node
        self.expr_to_id[sympy_expr] = node_id

        # Update parent references for children
        for child_id in node.child_ids:
            if child_id in self.nodes:
                self.nodes[child_id].parent_ids.add(node_id)

        return node_id

    def mark_as_root(self, node_id: str, tree_index: int) -> None:
        """Mark a node as a root of an original tree."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.is_root = True
            node.root_of_trees.append(tree_index)
            if node_id not in self.root_ids:
                self.root_ids.append(node_id)

    def add_original_node_mapping(
        self,
        node_id: str,
        tree_index: int,
        original_node: 'Node'
    ) -> None:
        """Track which original nodes map to a merged node."""
        if node_id in self.nodes:
            self.nodes[node_id].original_nodes.append((tree_index, original_node))

    def get_statistics(self) -> Dict:
        """Get statistics about the merged graph."""
        total_nodes = len(self.nodes)
        terminal_nodes = sum(1 for n in self.nodes.values() if n.node_type == 'terminal')
        operator_nodes = sum(1 for n in self.nodes.values() if n.node_type == 'operator')
        root_nodes = len(self.root_ids)

        # Count how many nodes are shared (used by multiple trees)
        shared_nodes = sum(
            1 for n in self.nodes.values()
            if len(n.original_nodes) > 1
        )

        # Calculate potential savings
        total_original_nodes = sum(
            len(n.original_nodes) for n in self.nodes.values()
        )
        savings_percent = (
            (1 - total_nodes / total_original_nodes) * 100
            if total_original_nodes > 0 else 0
        )

        return {
            'total_nodes': total_nodes,
            'terminal_nodes': terminal_nodes,
            'operator_nodes': operator_nodes,
            'root_nodes': root_nodes,
            'shared_nodes': shared_nodes,
            'tree_count': self.tree_count,
            'total_original_nodes': total_original_nodes,
            'savings_percent': savings_percent
        }

    def to_string(self, show_details: bool = True) -> str:
        """Generate a string representation of the merged graph.

        Args:
            show_details: If True, show which trees share each node.

        Returns:
            Formatted string representation.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MERGED EVALUATION GRAPH")
        lines.append("=" * 60)

        # Statistics
        stats = self.get_statistics()
        lines.append(f"\nStatistics:")
        lines.append(f"  Trees merged: {stats['tree_count']}")
        lines.append(f"  Total nodes in graph: {stats['total_nodes']}")
        lines.append(f"  - Terminal nodes: {stats['terminal_nodes']}")
        lines.append(f"  - Operator nodes: {stats['operator_nodes']}")
        lines.append(f"  Root nodes: {stats['root_nodes']}")
        lines.append(f"  Shared nodes: {stats['shared_nodes']}")
        lines.append(f"  Original node count: {stats['total_original_nodes']}")
        lines.append(f"  Computation savings: {stats['savings_percent']:.1f}%")

        # Group nodes by depth
        max_depth = max((n.depth for n in self.nodes.values()), default=0)

        lines.append(f"\n{'=' * 60}")
        lines.append("GRAPH STRUCTURE (bottom-up by depth)")
        lines.append("=" * 60)

        for depth in range(max_depth + 1):
            depth_nodes = [n for n in self.nodes.values() if n.depth == depth]
            if not depth_nodes:
                continue

            lines.append(f"\n--- Depth {depth} ({len(depth_nodes)} nodes) ---")

            for node in sorted(depth_nodes, key=lambda x: x.node_id):
                # Format expression
                expr_str = str(node.sympy_expr)
                if len(expr_str) > 40:
                    expr_str = expr_str[:37] + "..."

                # Root marker
                root_marker = ""
                if node.is_root:
                    trees_str = ", ".join(f"T{i}" for i in node.root_of_trees)
                    root_marker = f" [ROOT of {trees_str}]"

                # Usage count
                usage_count = len(node.original_nodes)
                usage_str = f" (used {usage_count}x)" if usage_count > 1 else ""

                # Children
                children_str = ""
                if node.child_ids:
                    children_str = f" <- [{', '.join(node.child_ids)}]"

                lines.append(
                    f"  {node.node_id}: {expr_str}{root_marker}{usage_str}{children_str}"
                )

                if show_details and len(node.original_nodes) > 1:
                    trees = set(t for t, _ in node.original_nodes)
                    lines.append(f"       Shared by trees: {sorted(trees)}")

        # Show roots summary
        lines.append(f"\n{'=' * 60}")
        lines.append("ROOT EXPRESSIONS (one per original tree)")
        lines.append("=" * 60)

        for root_id in self.root_ids:
            node = self.nodes[root_id]
            for tree_idx in node.root_of_trees:
                lines.append(f"  Tree {tree_idx}: {node.sympy_expr}")

        return "\n".join(lines)

    def to_evaluation_order(self) -> List[str]:
        """Return node IDs in valid evaluation order (children before parents).

        Returns:
            List of node_ids sorted so dependencies come before dependents.
        """
        # Topological sort by depth (terminals first)
        return sorted(
            self.nodes.keys(),
            key=lambda nid: (self.nodes[nid].depth, nid)
        )

    def to_graphviz_dot(self) -> str:
        """Generate Graphviz DOT format for visualization.

        Returns:
            DOT format string that can be rendered with graphviz.
        """
        lines = ["digraph MergedEvaluationGraph {"]
        lines.append("  rankdir=BT;")  # Bottom to top
        lines.append("  node [fontname=Arial, fontsize=10];")
        lines.append("  edge [color=gray];")

        # Group by depth using subgraphs
        max_depth = max((n.depth for n in self.nodes.values()), default=0)

        for depth in range(max_depth + 1):
            lines.append(f"  subgraph cluster_depth_{depth} {{")
            lines.append(f'    label="Depth {depth}";')
            lines.append("    style=dashed;")
            lines.append("    color=lightgray;")

            depth_nodes = [n for n in self.nodes.values() if n.depth == depth]
            for node in depth_nodes:
                # Style based on type
                if node.node_type == 'terminal':
                    shape = "ellipse"
                    color = "lightgreen" if 'Number' in str(type(node.sympy_expr)) else "lightblue"
                else:
                    shape = "box"
                    color = "lightyellow"

                if node.is_root:
                    color = "orange"

                # Escape special characters
                label = str(node.sympy_expr).replace('"', '\\"')
                if len(label) > 25:
                    label = label[:22] + "..."

                usage = len(node.original_nodes)
                if usage > 1:
                    label += f"\\n({usage}x)"

                lines.append(
                    f'    {node.node_id} [label="{label}", shape={shape}, '
                    f'style=filled, fillcolor={color}];'
                )

            lines.append("  }")

        # Add edges
        for node in self.nodes.values():
            for child_id in node.child_ids:
                lines.append(f"  {child_id} -> {node.node_id};")

        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# Core Functions for Building Merged Graph
# =============================================================================

def get_expressions_by_depth(
    tree: 'Node',
    normalize: bool = True
) -> List[List[Tuple[sympy.Basic, 'Node']]]:
    """Extract all expressions from a tree, organized by depth (bottom-up).

    Traverses the tree and collects sympy expressions at each depth level,
    starting from terminals (depth 0) up to the root.

    Args:
        tree: The root node of the tree to analyze.
        normalize: If True, use sympy to normalize expressions (e.g., a+b == b+a).

    Returns:
        List of lists, where index i contains (sympy_expr, node) tuples at depth i.
        Index 0 contains terminals, higher indices contain expressions built from them.

    Example:
        For tree (a + b*c):
        - Depth 0: [(a, node_a), (b, node_b), (c, node_c)]
        - Depth 1: [(b*c, node_mul)]
        - Depth 2: [(a + b*c, node_add)]
    """
    # First pass: compute actual depth for each node (max distance to any leaf)
    def compute_depth(node: 'Node') -> int:
        """Compute depth as max distance from any leaf (terminals = 0)."""
        if node.is_term():
            return 0
        else:
            child_depths = [compute_depth(c) for c in node.get_childs()]
            return max(child_depths) + 1

    # Second pass: collect expressions by depth
    max_depth = compute_depth(tree)
    layers: List[List[Tuple[sympy.Basic, 'Node']]] = [[] for _ in range(max_depth + 1)]

    def collect_expressions(node: 'Node', depth_from_leaf: int = None):
        """Recursively collect expressions."""
        if depth_from_leaf is None:
            depth_from_leaf = compute_depth(node)

        try:
            sympy_expr = node.get_sympy_expr()
            if normalize:
                # Basic normalization - sympify ensures canonical form
                sympy_expr = sympy.sympify(sympy_expr)
            layers[depth_from_leaf].append((sympy_expr, node))
        except Exception as e:
            # If sympy conversion fails, use string representation
            print(f"Warning: Could not get sympy expr for node: {e}")
            pass

        if not node.is_term():
            for child in node.get_childs():
                collect_expressions(child)

    collect_expressions(tree)
    return layers


def build_one_evaluation_tree(
    population: List[Union['Node', 'Candidate']],
    normalize_strategy: str = 'sympify'
) -> MergedEvaluationGraph:
    """Build a merged evaluation graph from a population of trees.

    This is the main entry point for the "One-evaluation-tree" strategy.
    Merges all trees into a single DAG where each unique sub-expression
    exists only once.

    Args:
        population: List of Node trees or Candidate objects.
        normalize_strategy: How to normalize expressions for deduplication:
            - 'sympify': Basic normalization via sympy.sympify (default)
            - 'simplify': Full simplification via sympy.simplify (slower)
            - 'expand': Expand all expressions before comparing
            - 'factor': Factor all expressions before comparing
            - 'none': No normalization (exact structural match only)

    Returns:
        MergedEvaluationGraph containing the unified DAG.

    Example:
        >>> import sympy
        >>> from plagih.trees import Add, Mul, Number, Symbol
        >>> tree1 = Add(Symbol(sympy.Symbol('a')), Number(1))
        >>> tree2 = Add(Symbol(sympy.Symbol('a')), Number(1))  # Same as tree1
        >>> tree3 = Mul(Symbol(sympy.Symbol('a')), Number(2))
        >>> graph = build_one_evaluation_tree([tree1, tree2, tree3])
        >>> graph.tree_count
        3
        >>> len(graph.root_ids)  # tree1 and tree2 share the same root
        2
    """
    from plagih.trees import Candidate

    graph = MergedEvaluationGraph()

    # Extract trees from Candidates if needed
    trees: List['Node'] = []
    for item in population:
        if isinstance(item, Candidate):
            trees.append(item.get_evotree())
        else:
            trees.append(item)

    graph.tree_count = len(trees)

    # Process each tree
    for tree_idx, tree in enumerate(trees):
        _process_tree_into_graph(
            graph=graph,
            tree=tree,
            tree_index=tree_idx,
            normalize_strategy=normalize_strategy
        )

    return graph


def _normalize_expression(
    expr: sympy.Basic,
    strategy: str
) -> sympy.Basic:
    """Normalize a sympy expression according to the chosen strategy.

    Args:
        expr: The sympy expression to normalize.
        strategy: Normalization strategy name.

    Returns:
        Normalized sympy expression.
    """
    if strategy == 'none':
        return expr
    elif strategy == 'sympify':
        return sympy.sympify(expr)
    elif strategy == 'simplify':
        return sympy.simplify(expr)
    elif strategy == 'expand':
        return sympy.expand(expr)
    elif strategy == 'factor':
        return sympy.factor(expr)
    else:
        return sympy.sympify(expr)


def _process_tree_into_graph(
    graph: MergedEvaluationGraph,
    tree: 'Node',
    tree_index: int,
    normalize_strategy: str
) -> str:
    """Process a single tree and add its nodes to the merged graph.

    Args:
        graph: The MergedEvaluationGraph to add to.
        tree: The root node of the tree to process.
        tree_index: Index of this tree in the population.
        normalize_strategy: Expression normalization strategy.

    Returns:
        The node_id of the root node in the merged graph.
    """
    def process_node(node: 'Node', depth_from_leaf: int) -> str:
        """Recursively process a node and its children."""

        # Get sympy expression
        try:
            sympy_expr = node.get_sympy_expr()
            sympy_expr = _normalize_expression(sympy_expr, normalize_strategy)
        except Exception as e:
            # Fallback: use string representation
            sympy_expr = sympy.Symbol(f"_err_{id(node)}")

        if node.is_term():
            # Terminal node
            node_id = graph.get_or_create_node(
                sympy_expr=sympy_expr,
                depth=0,
                node_type='terminal',
                operator_name=type(node).__name__
            )
        else:
            # Operator node - first process children
            child_ids = []
            child_depths = []

            for child in node.get_childs():
                # Compute child's depth from leaf
                child_depth = _compute_node_depth(child)
                child_id = process_node(child, child_depth)
                child_ids.append(child_id)
                child_depths.append(child_depth)

            # This node's depth is max(child_depths) + 1
            node_depth = max(child_depths) + 1 if child_depths else 1

            node_id = graph.get_or_create_node(
                sympy_expr=sympy_expr,
                depth=node_depth,
                node_type='operator',
                operator_name=type(node).__name__,
                child_ids=child_ids
            )

        # Track mapping from original node
        graph.add_original_node_mapping(node_id, tree_index, node)

        return node_id

    # Process the tree starting from root
    root_depth = _compute_node_depth(tree)
    root_id = process_node(tree, root_depth)

    # Mark as root
    graph.mark_as_root(root_id, tree_index)

    return root_id


def _compute_node_depth(node: 'Node') -> int:
    """Compute depth of a node (terminals = 0, increasing towards root)."""
    if node.is_term():
        return 0
    else:
        child_depths = [_compute_node_depth(c) for c in node.get_childs()]
        return max(child_depths) + 1 if child_depths else 1


# =============================================================================
# Utility Functions
# =============================================================================

def analyze_population_sharing(
    population: List[Union['Node', 'Candidate']]
) -> Dict:
    """Analyze how much computation can be shared across a population.

    Args:
        population: List of trees or candidates.

    Returns:
        Dictionary with sharing statistics.
    """
    graph = build_one_evaluation_tree(population)
    stats = graph.get_statistics()

    # Add additional analysis
    stats['most_shared_expressions'] = []

    # Find most shared expressions
    shared_nodes = [
        (len(n.original_nodes), n.sympy_expr, n.node_id)
        for n in graph.nodes.values()
        if len(n.original_nodes) > 1
    ]
    # Sort by count (descending), then node_id (for determinism)
    # Avoid comparing sympy expressions directly as they can't be compared
    shared_nodes.sort(key=lambda x: (-x[0], x[2]))

    stats['most_shared_expressions'] = [
        {'count': count, 'expr': str(expr), 'node_id': nid}
        for count, expr, nid in shared_nodes[:10]
    ]

    return stats


def visualize_merged_graph(
    graph: MergedEvaluationGraph,
    output_path: str = None,
    show: bool = True
) -> Optional[str]:
    """Visualize the merged graph using graphviz if available.

    Args:
        graph: The MergedEvaluationGraph to visualize.
        output_path: Path to save the image (without extension).
        show: Whether to display the graph.

    Returns:
        Path to the generated image, or None if graphviz unavailable.
    """
    try:
        from graphviz import Source

        dot_source = graph.to_graphviz_dot()
        src = Source(dot_source)

        if output_path:
            src.render(output_path, format='png', cleanup=True)
            return f"{output_path}.png"

        if show:
            src.view()

        return None

    except ImportError:
        print("graphviz not available. Install with: pip install graphviz")
        print("DOT source:\n", graph.to_graphviz_dot())
        return None


# =============================================================================
# Example Usage and Testing
# =============================================================================

def _demo():
    """Demonstrate the population merge functionality."""
    from plagih.trees import Add, Mul, Number, Symbol

    # Create some example trees
    a = Symbol(sympy.Symbol('a'))
    b = Symbol(sympy.Symbol('b'))
    c = Symbol(sympy.Symbol('c'))

    # Tree 1: a + b
    tree1 = Add(
        Symbol(sympy.Symbol('a')),
        Symbol(sympy.Symbol('b'))
    )

    # Tree 2: (a + b) * c  -- shares (a + b) with tree1
    tree2 = Mul(
        Add(
            Symbol(sympy.Symbol('a')),
            Symbol(sympy.Symbol('b'))
        ),
        Symbol(sympy.Symbol('c'))
    )

    # Tree 3: a + b + 1  -- shares a, b with others
    tree3 = Add(
        Add(
            Symbol(sympy.Symbol('a')),
            Symbol(sympy.Symbol('b'))
        ),
        Number(1)
    )

    # Tree 4: a * 2  -- shares only 'a'
    tree4 = Mul(
        Symbol(sympy.Symbol('a')),
        Number(2)
    )

    population = [tree1, tree2, tree3, tree4]

    print(f"Creating merged evaluation graph from 4 trees...")

    # Build merged graph
    graph = build_one_evaluation_tree(population)

    # Print string representation
    print(graph.to_string())

    # Print DOT format for graphviz
    print(f"GRAPHVIZ DOT FORMAT: {graph.to_graphviz_dot()}")


    return graph


if __name__ == "__main__":
    _demo()
