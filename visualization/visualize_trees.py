"""
Tree Visualization for plagih GP Framework

This module provides backward-compatible visualization functions.
The actual rendering is now handled by the unified tree_renderer module.

For new code, prefer using:
    from visualization.tree_renderer import render_tree, render_merged_tree

Legacy functions (visualize_tree, visualize_tree_matplotlib, etc.) are
maintained for backward compatibility.
"""

from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Try graphviz (still used for some legacy functions)
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    Digraph = None  # type: ignore

# Import tree structures
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from plagih.util import *
from plagih.trees import (
    Node, Number, Symbol, Boolean,
    MathOperator, LogicOperator,
)

# Import new unified renderer
from visualization.tree_renderer import (
    render_tree as _render_tree,
    TreeRendererConfig,
    Orientation,
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
    backend: str = "auto",  # "auto", "graphviz", "matplotlib", "new"
) -> str:
    """
    Visualize a tree and save as image file.

    This function now uses the unified tree_renderer by default.
    Set backend="graphviz" to use the legacy graphviz backend.

    Args:
        root: Root node of the tree to visualize
        filename: Output filename (without extension)
        output_dir: Directory for output (default: tree_output/)
        view: If True, open the image after creation (graphviz only)
        format: Output format (png, svg, pdf - graphviz only)
        rankdir: Layout direction TB=top-down, LR=left-right, BT=bottom-up, RL=right-left
        cleanup: Remove intermediate .dot file (graphviz only)
        backend: "auto"/"new" uses new renderer, "graphviz"/"matplotlib" use legacy

    Returns:
        Path to the generated image file
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tree_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use new renderer by default
    if backend in ("auto", "new", "matplotlib"):
        return _render_tree(
            tree=root,
            filename=filename,
            output_dir=output_dir,
            orientation=rankdir,
            show=view
        )

    # Legacy graphviz backend
    if backend == "graphviz" and HAS_GRAPHVIZ:
        try:
            dot = tree_to_graphviz(root, name=filename, rankdir=rankdir)
            dot.format = format
            output_path = output_dir / filename
            dot.render(str(output_path), view=view, cleanup=cleanup)
            result_path = f"{output_path}.{format}"
            print(f"Tree saved to: {result_path}")
            return result_path
        except Exception as e:
            print(f"Graphviz failed ({e}), falling back to new renderer...")

    # Fallback to new renderer
    return _render_tree(
        tree=root,
        filename=filename,
        output_dir=output_dir,
        orientation=rankdir,
        show=view
    )


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


def visualize_paretofront(
    paretofront: list,
    filename: str = "paretofront_trees",
    output_dir: Optional[Path] = None,
    figsize_per_tree: tuple = (5, 4),
    dpi: int = 150,
    max_cols: int = 4,
    save_individual: bool = True,
) -> str:
    """
    Visualize all Paretofront candidates in a single combined image.

    Args:
        paretofront: List of Candidate objects from the Paretofront
        filename: Output filename (without extension)
        output_dir: Directory for output (default: tree_output/)
        figsize_per_tree: Size (width, height) for each individual tree subplot
        dpi: Resolution of the output image
        max_cols: Maximum number of columns in the grid
        save_individual: If True, also save each tree as individual image in paretoCandidates/

    Returns:
        Path to the generated image file
    """
    if not paretofront:
        print("Paretofront is empty, nothing to visualize.")
        return ""

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tree_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort paretofront by parsimony for consistent display
    sorted_front = sorted(paretofront, key=lambda c: (c.get_parsim(), c.get_fitness()))

    # Save individual images if requested
    if save_individual:
        individual_dir = output_dir / "paretoCandidates"
        individual_dir.mkdir(parents=True, exist_ok=True)

        for idx, candidate in enumerate(sorted_front):
            tree = candidate.get_evotree()
            parsim = candidate.get_parsim()
            fitness = candidate.get_fitness()
            individual_filename = f"pareto_{idx+1:03d}_P{parsim:.0f}_F{fitness:.4g}"
            _visualize_single_tree_to_file(
                tree, individual_filename, individual_dir,
                title=f"P={parsim:.0f}, F={fitness:.4g}",
                dpi=dpi
            )
        printez('ff', f"Individual Paretofront trees saved to: {individual_dir}")

    # Calculate grid layout
    n_trees = len(paretofront)
    n_cols = min(n_trees, max_cols)
    n_rows = (n_trees + n_cols - 1) // n_cols  # Ceiling division

    # Calculate dynamic figure size based on tree complexity
    max_depth = max(_get_tree_depth(c.get_evotree()) for c in sorted_front)
    max_width = max(_get_tree_width(c.get_evotree()) for c in sorted_front)

    # Scale subplot size based on tree complexity
    width_scale = max(1.0, max_width / 4)
    height_scale = max(1.0, max_depth / 3)

    fig_width = figsize_per_tree[0] * n_cols * width_scale
    fig_height = figsize_per_tree[1] * n_rows * height_scale

    # Ensure minimum size
    fig_width = max(fig_width, 8)
    fig_height = max(fig_height, 6)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Ensure axes is always 2D array
    if n_trees == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, candidate in enumerate(sorted_front):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get tree from candidate
        tree = candidate.get_evotree()

        # Compute layout for this tree
        positions, labels, colors, edges = _compute_tree_layout(tree)

        # Calculate node size based on tree complexity
        tree_width = _get_tree_width(tree)
        node_size = max(0.3, min(0.5, 2.0 / max(tree_width, 1)))

        # Draw edges
        for parent_id, child_id in edges:
            x1, y1 = positions[parent_id]
            x2, y2 = positions[child_id]
            ax.plot([x1, x2], [y1, y2], '-', linewidth=1.2, zorder=1, color='#666666')

        # Draw nodes
        for node_id, (x, y) in positions.items():
            color = colors[node_id]
            label = labels[node_id]

            # Adjust node size based on label length
            label_width = max(node_size, len(label) * 0.08)

            circle = mpatches.FancyBboxPatch(
                (x - label_width/2, y - node_size/4),
                label_width, node_size/2,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                facecolor=color,
                edgecolor='#333333',
                linewidth=1.0,
                zorder=2
            )
            ax.add_patch(circle)

            # Adjust font size based on label length
            fontsize = max(6, min(9, 72 / max(len(label), 1)))
            ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', zorder=3)

        # Adjust view with proper margins
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            margin_x = max(0.5, (max(xs) - min(xs)) * 0.1)
            margin_y = max(0.5, (max(ys) - min(ys)) * 0.1)
            ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
            ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

        ax.set_aspect('equal')
        ax.axis('off')

        # Add title with parsimony and fitness
        parsim = candidate.get_parsim()
        fitness = candidate.get_fitness()
        ax.set_title(f"P={parsim:.0f}, F={fitness:.4g}", fontsize=10, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_trees, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle(f"Paretofront ({n_trees} candidates)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Paretofront visualization saved to: {output_path}")
    return str(output_path)


def _get_tree_depth(node: Node) -> int:
    """Calculate the depth of a tree."""
    if node.is_term():
        return 1
    children = node.get_childs()
    if not children:
        return 1
    return 1 + max(_get_tree_depth(c) for c in children)


def _get_tree_width(node: Node) -> int:
    """Calculate the width (number of leaves) of a tree."""
    if node.is_term():
        return 1
    children = node.get_childs()
    if not children:
        return 1
    return sum(_get_tree_width(c) for c in children)


def _visualize_single_tree_to_file(
    tree: Node,
    filename: str,
    output_dir: Path,
    title: str = "",
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> str:
    """
    Visualize a single tree and save to file with dynamic sizing.
    """
    # Calculate dynamic figure size based on tree complexity
    depth = _get_tree_depth(tree)
    width = _get_tree_width(tree)

    fig_width = max(figsize[0], width * 1.2)
    fig_height = max(figsize[1], depth * 1.5)

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Compute layout
    positions, labels, colors, edges = _compute_tree_layout(tree)

    # Calculate node size based on tree complexity
    node_size = max(0.3, min(0.5, 3.0 / max(width, 1)))

    # Draw edges
    for parent_id, child_id in edges:
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        ax.plot([x1, x2], [y1, y2], '-', linewidth=1.5, zorder=1, color='#666666')

    # Draw nodes
    for node_id, (x, y) in positions.items():
        color = colors[node_id]
        label = labels[node_id]

        # Adjust node size based on label length
        label_width = max(node_size, len(label) * 0.1)

        circle = mpatches.FancyBboxPatch(
            (x - label_width/2, y - node_size/4),
            label_width, node_size/2,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02),
            facecolor=color,
            edgecolor='#333333',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(circle)

        # Adjust font size based on label length
        fontsize = max(7, min(11, 88 / max(len(label), 1)))
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=3)

    # Adjust view
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin_x = max(0.5, (max(xs) - min(xs)) * 0.15)
        margin_y = max(0.5, (max(ys) - min(ys)) * 0.15)
        ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
        ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return str(output_path)


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
