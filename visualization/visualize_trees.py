"""
Tree Visualization for plagih GP Framework

Provides `visualize_paretofront` for combined Pareto-front images.

For single-tree rendering, use the unified tree_renderer module:
    from visualization.tree_renderer import render_tree, render_merged_tree
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from plagih.trees import (
    Boolean,
    LogicOperator,
    MathOperator,
    Node,
    Number,
    Symbol,
)
from plagih.util import *

# ============================================================================
# Node helpers (used by layout / paretofront rendering)
# ============================================================================


def get_node_label(node: Node) -> str:
    """
    Get a human-readable label for a node.
    - Terminals: show value (number, symbol name, bool)
    - Operators: show operator name
    """
    if node.is_term():
        val = node.get_value()
        if isinstance(node, Number):
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
        return node.showme or type(node).__name__


def get_node_color(node: Node) -> tuple:
    """Get colors (fill, border) for different node types."""
    if node.is_term():
        if isinstance(node, Number):
            return "#E8F5E9", "#4CAF50"  # green
        elif isinstance(node, Symbol):
            return "#E3F2FD", "#2196F3"  # blue
        elif isinstance(node, Boolean):
            return "#FFF3E0", "#FF9800"  # orange
        else:
            return "#F5F5F5", "#9E9E9E"  # gray
    else:
        if isinstance(node, MathOperator):
            return "#FCE4EC", "#E91E63"  # pink
        elif isinstance(node, LogicOperator):
            return "#F3E5F5", "#9C27B0"  # purple
        else:
            return "#ECEFF1", "#607D8B"  # blue-gray


# ============================================================================
# Tree layout (Buchheim-style hierarchical layout for matplotlib)
# ============================================================================


def _compute_tree_layout(root: Node) -> tuple:
    """
    Compute hierarchical tree layout.
    Returns (positions dict, labels dict, colors dict, edges list)
    """
    positions = {}
    labels = {}
    colors = {}
    edges = []

    node_counter = [0]

    def get_subtree_width(node: Node) -> int:
        if node.is_term():
            return 1
        children = node.get_childs()
        if not children:
            return 1
        return sum(get_subtree_width(c) for c in children)

    def layout_node(node: Node, depth: int, left_bound: float, parent_id: Optional[int] = None):
        node_id = node_counter[0]
        node_counter[0] += 1

        labels[node_id] = get_node_label(node)
        fill, border = get_node_color(node)
        colors[node_id] = fill

        if parent_id is not None:
            edges.append((parent_id, node_id))

        if node.is_term():
            x = left_bound + 0.5
            positions[node_id] = (x, -depth)
            return node_id, 1

        children = node.get_childs()
        if not children:
            x = left_bound + 0.5
            positions[node_id] = (x, -depth)
            return node_id, 1

        total_width = 0
        child_positions = []
        current_left = left_bound

        for child in children:
            child_id, child_width = layout_node(child, depth + 1, current_left, node_id)
            child_positions.append(positions[child_id][0])
            current_left += child_width
            total_width += child_width

        x = (child_positions[0] + child_positions[-1]) / 2
        positions[node_id] = (x, -depth)

        return node_id, total_width

    layout_node(root, 0, 0)

    return positions, labels, colors, edges


# ============================================================================
# Tree metrics
# ============================================================================


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


# ============================================================================
# Single-tree helper (used by visualize_paretofront)
# ============================================================================


def _visualize_single_tree_to_file(
    tree: Node,
    filename: str,
    output_dir: Path,
    title: str = "",
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> str:
    """Render a single tree to a PNG file with dynamic sizing."""
    depth = _get_tree_depth(tree)
    width = _get_tree_width(tree)

    fig_width = max(figsize[0], width * 1.2)
    fig_height = max(figsize[1], depth * 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    positions, labels, colors, edges = _compute_tree_layout(tree)

    node_size = max(0.3, min(0.5, 3.0 / max(width, 1)))

    for parent_id, child_id in edges:
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        ax.plot([x1, x2], [y1, y2], "-", linewidth=1.5, zorder=1, color="#666666")

    for node_id, (x, y) in positions.items():
        color = colors[node_id]
        label = labels[node_id]
        label_width = max(node_size, len(label) * 0.1)

        circle = mpatches.FancyBboxPatch(
            (x - label_width / 2, y - node_size / 4),
            label_width,
            node_size / 2,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02),
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(circle)

        fontsize = max(7, min(11, 88 / max(len(label), 1)))
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=3)

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin_x = max(0.5, (max(xs) - min(xs)) * 0.15)
        margin_y = max(0.5, (max(ys) - min(ys)) * 0.15)
        ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
        ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(output_path)


# ============================================================================
# Paretofront visualization (actively used by ExplainableGP)
# ============================================================================


def visualize_paretofront(
    paretofront: list,
    filename: str = "paretofront_trees",
    output_dir: Optional[Path] = Path.cwd(),
    figsize_per_tree: tuple = (5, 4),
    dpi: int = 150,
    max_cols: int = 4,
    save_individual: bool = True,
):
    """
    Visualize all Paretofront candidates in a single combined image.

    Args:
        paretofront: List of Candidate objects from the Paretofront
        filename: Output filename (without extension)
        output_dir: Directory for output (default: cwd)
        figsize_per_tree: Size (width, height) for each individual tree subplot
        dpi: Resolution of the output image
        max_cols: Maximum number of columns in the grid
        save_individual: If True, also save each tree as individual image in paretoCandidates/

    Returns:
        Path to the generated image file
    """
    if not paretofront:
        print_warning("w", "Paretofront is empty, nothing to visualize.")
        return

    output_dir = path_make_dir(output_dir / "tree_output")

    sorted_front = sorted(paretofront, key=lambda c: (c.get_parsim(), c.get_fitness()))

    # Save individual images if requested
    if save_individual:
        individual_dir = output_dir / "paretoCandidates"
        individual_dir.mkdir(parents=True, exist_ok=True)

        for idx, candidate in enumerate(sorted_front):
            tree = candidate.get_evotree()
            parsim = candidate.get_parsim()
            fitness = candidate.get_fitness()
            individual_filename = f"pareto_{idx + 1:03d}_P{parsim:.0f}_F{fitness:.4g}"
            _visualize_single_tree_to_file(
                tree, individual_filename, individual_dir, title=f"P={parsim:.0f}, F={fitness:.4g}", dpi=dpi
            )
        printez("ff", f"Individual Paretofront trees saved to: {individual_dir}")

    # Calculate grid layout
    n_trees = len(paretofront)
    n_cols = min(n_trees, max_cols)
    n_rows = (n_trees + n_cols - 1) // n_cols

    # Calculate dynamic figure size based on tree complexity
    max_depth = max(_get_tree_depth(c.get_evotree()) for c in sorted_front)
    max_width = max(_get_tree_width(c.get_evotree()) for c in sorted_front)

    width_scale = max(1.0, max_width / 4)
    height_scale = max(1.0, max_depth / 3)

    fig_width = figsize_per_tree[0] * n_cols * width_scale
    fig_height = figsize_per_tree[1] * n_rows * height_scale

    fig_width = max(fig_width, 8)
    fig_height = max(fig_height, 6)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

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

        tree = candidate.get_evotree()
        positions, labels, colors, edges = _compute_tree_layout(tree)

        tree_width = _get_tree_width(tree)
        node_size = max(0.3, min(0.5, 2.0 / max(tree_width, 1)))

        for parent_id, child_id in edges:
            x1, y1 = positions[parent_id]
            x2, y2 = positions[child_id]
            ax.plot([x1, x2], [y1, y2], "-", linewidth=1.2, zorder=1, color="#666666")

        for node_id, (x, y) in positions.items():
            color = colors[node_id]
            label = labels[node_id]
            label_width = max(node_size, len(label) * 0.08)

            circle = mpatches.FancyBboxPatch(
                (x - label_width / 2, y - node_size / 4),
                label_width,
                node_size / 2,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(circle)

            fontsize = max(6, min(9, 72 / max(len(label), 1)))
            ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=3)

        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            margin_x = max(0.5, (max(xs) - min(xs)) * 0.1)
            margin_y = max(0.5, (max(ys) - min(ys)) * 0.1)
            ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
            ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

        ax.set_aspect("equal")
        ax.axis("off")

        parsim = candidate.get_parsim()
        fitness = candidate.get_fitness()
        ax.set_title(f"P={parsim:.0f}, F={fitness:.4g}", fontsize=10, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_trees, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle(f"Paretofront ({n_trees} candidates)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Paretofront visualization saved to: {output_path}")
    return str(output_path)
