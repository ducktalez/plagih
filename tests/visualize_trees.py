"""
Tree Visualization for plagih GP Framework

Uses graphviz for clean hierarchical tree layouts (if available),
otherwise falls back to matplotlib + networkx.
Generates PNG images for quick inspection of evolved trees.
"""

import os
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Try graphviz
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

# Try networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Import tree structures
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from plagih.trees import (
    Node, Terminal, Number, Symbol, Boolean,
    BaseOperator, MathOperator, LogicOperator,
    Add, Mul
)


def get_node_label(node: Node) -> str:
    """
    Get a human-readable label for a node.
    - Terminals: show value (number, symbol name, bool)
    - Operators: show operator name
    """
    if node.is_term():
        val = node.get_value()
        if isinstance(node, Number):
            # Format numbers nicely
            fval = float(val) if not isinstance(val, float) else val
            if fval == int(fval):
                return str(int(fval))
            return f"{fval:.4g}"
        elif isinstance(node, Symbol):
            return str(val)
        elif isinstance(node, Boolean):
            return str(bool(val))
        else:
            return str(val)
    else:
        # Operator node: use showme or class name
        return node.showme or type(node).__name__


def get_node_color(node: Node) -> tuple:
    """Get colors (fill, border) for different node types."""
    if node.is_term():
        if isinstance(node, Number):
            return '#E8F5E9', '#4CAF50'  # green
        elif isinstance(node, Symbol):
            return '#E3F2FD', '#2196F3'  # blue
        elif isinstance(node, Boolean):
            return '#FFF3E0', '#FF9800'  # orange
        else:
            return '#F5F5F5', '#9E9E9E'  # gray
    else:
        if isinstance(node, MathOperator):
            return '#FCE4EC', '#E91E63'  # pink
        elif isinstance(node, LogicOperator):
            return '#F3E5F5', '#9C27B0'  # purple
        else:
            return '#ECEFF1', '#607D8B'  # blue-gray


def get_node_style(node: Node) -> dict:
    """
    Get graphviz style attributes for different node types.
    Returns dict with shape, color, style, fontcolor etc.
    """
    fill, border = get_node_color(node)

    if node.is_term():
        return {
            'shape': 'ellipse',
            'style': 'filled',
            'fillcolor': fill,
            'color': border,
            'fontcolor': '#1B5E20' if isinstance(node, Number) else '#0D47A1',
        }
    else:
        if isinstance(node, LogicOperator):
            return {
                'shape': 'diamond',
                'style': 'filled',
                'fillcolor': fill,
                'color': border,
                'fontcolor': '#4A148C',
            }
        else:
            return {
                'shape': 'box',
                'style': 'filled,rounded',
                'fillcolor': fill,
                'color': border,
                'fontcolor': '#880E4F' if isinstance(node, MathOperator) else '#263238',
            }


# ============================================================================
# Graphviz-based visualization
# ============================================================================

def tree_to_graphviz(
    root: Node,
    name: str = "gp_tree",
    rankdir: str = "TB",
) -> 'Digraph':
    """Convert a plagih tree to a graphviz Digraph object."""
    if not HAS_GRAPHVIZ:
        raise ImportError("graphviz is required for tree visualization")

    dot = Digraph(name=name, format='png')
    dot.attr(rankdir=rankdir)
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', color='#666666', arrowsize='0.7')

    node_counter = [0]

    def add_node_recursive(node: Node, parent_id: Optional[str] = None):
        node_id = f"n{node_counter[0]}"
        node_counter[0] += 1

        label = get_node_label(node)
        style = get_node_style(node)

        dot.node(node_id, label=label, **style)

        if parent_id is not None:
            dot.edge(parent_id, node_id)

        if not node.is_term():
            for child in node.get_childs():
                add_node_recursive(child, node_id)

        return node_id

    add_node_recursive(root)
    return dot


# ============================================================================
# Matplotlib/NetworkX-based visualization (Fallback)
# ============================================================================

def _compute_tree_layout(root: Node) -> tuple:
    """
    Compute hierarchical tree layout using Buchheim algorithm variant.
    Returns (positions dict, labels dict, colors dict, edges list)
    """
    positions = {}
    labels = {}
    colors = {}
    edges = []

    node_counter = [0]

    def get_subtree_width(node: Node) -> int:
        """Calculate width needed for subtree."""
        if node.is_term():
            return 1
        children = node.get_childs()
        if not children:
            return 1
        return sum(get_subtree_width(c) for c in children)

    def layout_node(node: Node, depth: int, left_bound: float, parent_id: Optional[int] = None):
        """Recursively assign positions."""
        node_id = node_counter[0]
        node_counter[0] += 1

        labels[node_id] = get_node_label(node)
        fill, border = get_node_color(node)
        colors[node_id] = fill

        if parent_id is not None:
            edges.append((parent_id, node_id))

        if node.is_term():
            # Leaf node
            x = left_bound + 0.5
            positions[node_id] = (x, -depth)
            return node_id, 1

        children = node.get_childs()
        if not children:
            x = left_bound + 0.5
            positions[node_id] = (x, -depth)
            return node_id, 1

        # Layout children first
        total_width = 0
        child_positions = []
        current_left = left_bound

        for child in children:
            child_id, child_width = layout_node(child, depth + 1, current_left, node_id)
            child_positions.append(positions[child_id][0])
            current_left += child_width
            total_width += child_width

        # Center parent above children
        x = (child_positions[0] + child_positions[-1]) / 2
        positions[node_id] = (x, -depth)

        return node_id, total_width

    layout_node(root, 0, 0)

    return positions, labels, colors, edges


def visualize_tree_matplotlib(
    root: Node,
    filename: str = "gp_tree",
    output_dir: Optional[str] = None,
    figsize: tuple = (12, 8),
    dpi: int = 150,
) -> str:
    """
    Visualize tree using matplotlib (no external dependencies).
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tree_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute layout
    positions, labels, colors, edges = _compute_tree_layout(root)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges first (behind nodes)
    for parent_id, child_id in edges:
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        ax.plot([x1, x2], [y1, y2], '-', linewidth=1.5, zorder=1, color='#666666')

    # Draw nodes
    node_size = 0.4
    for node_id, (x, y) in positions.items():
        color = colors[node_id]
        label = labels[node_id]

        # Draw circle/ellipse
        circle = mpatches.FancyBboxPatch(
            (x - node_size/2, y - node_size/4),
            node_size, node_size/2,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02),
            facecolor=color,
            edgecolor='#333333',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(circle)

        # Draw label
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                fontweight='bold', zorder=3)

    # Adjust view
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin = 0.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # Save
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Tree saved to: {output_path}")
    return str(output_path)


# ============================================================================
# Main visualization function (auto-selects backend)
# ============================================================================

def visualize_tree(
    root: Node,
    filename: str = "gp_tree",
    output_dir: Optional[str] = None,
    view: bool = False,
    format: str = "png",
    rankdir: str = "TB",
    cleanup: bool = True,
    backend: str = "auto",  # "auto", "graphviz", "matplotlib"
) -> str:
    """
    Visualize a tree and save as image file.

    Args:
        root: Root node of the tree to visualize
        filename: Output filename (without extension)
        output_dir: Directory for output (default: tree_output/)
        view: If True, open the image after creation (graphviz only)
        format: Output format (png, svg, pdf - graphviz only)
        rankdir: Layout direction TB=top-down, LR=left-right (graphviz only)
        cleanup: Remove intermediate .dot file (graphviz only)
        backend: "auto", "graphviz", or "matplotlib"

    Returns:
        Path to the generated image file
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tree_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select backend
    use_graphviz = False
    if backend == "graphviz":
        use_graphviz = True
    elif backend == "matplotlib":
        use_graphviz = False
    else:  # auto
        # Try graphviz first, fall back to matplotlib
        if HAS_GRAPHVIZ:
            use_graphviz = True

    if use_graphviz:
        try:
            dot = tree_to_graphviz(root, name=filename, rankdir=rankdir)
            dot.format = format
            output_path = output_dir / filename
            dot.render(str(output_path), view=view, cleanup=cleanup)
            result_path = f"{output_path}.{format}"
            print(f"Tree saved to: {result_path}")
            return result_path
        except Exception as e:
            print(f"Graphviz failed ({e}), falling back to matplotlib...")

    # Matplotlib fallback
    return visualize_tree_matplotlib(root, filename, output_dir)


def visualize_multiple_trees(
    trees: list,
    prefix: str = "tree",
    output_dir: Optional[str] = None,
    **kwargs
) -> list:
    """
    Visualize multiple trees with numbered filenames.
    """
    paths = []
    for i, tree in enumerate(trees):
        filename = f"{prefix}_{i+1:03d}"
        path = visualize_tree(tree, filename=filename, output_dir=output_dir, **kwargs)
        paths.append(path)
    return paths


# ============================================================================
# Test / Demo
# ============================================================================

def _demo():
    """Demo visualization with sample trees."""
    from plagih.trees import Add, Mul, Sin, Cos, Number, Symbol

    # Example 1: Simple expression: (x + 2) * sin(y)
    tree1 = Mul(
        Add(Symbol('x'), Number(2)),
        Sin(Symbol('y'))
    )
    visualize_tree(tree1, filename="example_simple", backend="matplotlib")

    # Example 2: More complex expression
    tree2 = Add(
        Mul(Number(3), Symbol('x')),
        Cos(Add(Symbol('y'), Number(1))),
        Number(-5)
    )
    visualize_tree(tree2, filename="example_complex", backend="matplotlib")

    # Example 3: Deeper tree
    tree3 = Mul(
        Add(
            Sin(Symbol('x')),
            Cos(Symbol('y'))
        ),
        Add(
            Number(2),
            Mul(Symbol('z'), Number(3))
        )
    )
    visualize_tree(tree3, filename="example_deep", backend="matplotlib")

    print("\nDemo complete! Check tree_output/ folder for images.")


if __name__ == "__main__":
    _demo()
