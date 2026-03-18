"""
Unified Tree Visualization Module for plagih GP Framework

Provides a centralized, configurable tree rendering system supporting:
- Normal GP trees (Node-based)
- Merged evaluation trees (MergedEvaluationGraph)
- Multiple layout orientations (TB, BT, LR, RL)
- Dynamic node sizing based on content
- Two display modes for merged trees: label-only and full-expression

Layout Algorithm: Modified Reingold-Tilford with variable node sizes.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from plagih.logging_utils import log
from plagih.util import path_make_dir

if TYPE_CHECKING:
    from plagih.population_merge import MergedEvaluationGraph, MergedNode
    from plagih.trees import Node


# =============================================================================
# Enums and Configuration
# =============================================================================


class Orientation(Enum):
    """Tree layout orientation."""

    TOP_DOWN = "TB"  # Root at top, leaves at bottom
    BOTTOM_UP = "BT"  # Root at bottom, leaves at top
    LEFT_RIGHT = "LR"  # Root at left, leaves at right
    RIGHT_LEFT = "RL"  # Root at right, leaves at left


class MergedDisplayMode(Enum):
    """Display mode for merged tree nodes."""

    LABEL_ONLY = auto()  # Show only operator/terminal label
    FULL_EXPRESSION = auto()  # Show complete child expression


class NodeShape(Enum):
    """Shape styles for nodes."""

    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded"
    DIAMOND = "diamond"


@dataclass
class NodeStyle:
    """Style configuration for a node."""

    fill_color: str = "#FFFFFF"
    border_color: str = "#333333"
    text_color: str = "#000000"
    shape: NodeShape = NodeShape.ROUNDED_RECTANGLE
    border_width: float = 1.5
    font_size: float = 9.0
    font_family: str = "sans-serif"


@dataclass
class TreeRendererConfig:
    """Configuration for the tree renderer.

    Attributes:
        orientation: Layout direction (TB=top-down, BT=bottom-up, LR=left-right, RL=right-left)
        min_level_gap: Minimum gap between tree levels (depth layers)
        min_sibling_gap: Minimum gap between sibling nodes
        min_subtree_gap: Minimum gap between subtrees
        node_padding_x: Horizontal padding inside node boxes
        node_padding_y: Vertical padding inside node boxes
        min_node_width: Minimum node width
        min_node_height: Minimum node height
        max_label_width: Maximum characters per line before text wrapping
        merged_display_mode: How to display merged tree nodes
        show_usage_count: Show how many times a merged node is used
        show_root_marker: Show [R] marker for root nodes in merged trees
        max_expression_length: Truncate expressions longer than this
        edge_color: Color for tree edges
        edge_width: Width of tree edges
        figure_dpi: DPI for saved figures
        background_color: Background color for figures
    """

    # Layout
    orientation: Orientation = Orientation.TOP_DOWN
    min_level_gap: float = 0.5  # Minimum vertical gap between levels
    min_sibling_gap: float = 0.15  # Minimum horizontal gap between siblings
    min_subtree_gap: float = 0.3  # Minimum gap between subtrees

    # Node sizing - compact defaults
    node_padding_x: float = 0.08  # Horizontal padding inside nodes
    node_padding_y: float = 0.05  # Vertical padding inside nodes
    min_node_width: float = 0.3  # Smaller minimum width
    min_node_height: float = 0.25  # Smaller minimum height
    max_label_width: int = 20  # Characters before wrapping

    # Merged tree specific
    merged_display_mode: MergedDisplayMode = MergedDisplayMode.LABEL_ONLY
    show_usage_count: bool = True
    show_root_marker: bool = True
    max_expression_length: int = 30  # Truncate expressions earlier for compactness

    # Visual
    edge_color: str = "#666666"
    edge_width: float = 1.5
    figure_dpi: int = 150
    background_color: str = "white"


@dataclass
class LayoutNode:
    """Internal representation of a node for layout calculation.

    Used by the layout engine to compute positions before rendering.
    """

    node_id: str
    label: str
    width: float = 0.0
    height: float = 0.0
    x: float = 0.0
    y: float = 0.0
    depth: int = 0
    children: List[LayoutNode] = field(default_factory=list)
    parent: Optional[LayoutNode] = None
    style: NodeStyle = field(default_factory=NodeStyle)

    # For Reingold-Tilford algorithm
    mod: float = 0.0
    prelim: float = 0.0
    change: float = 0.0
    shift: float = 0.0
    thread: Optional[LayoutNode] = None
    ancestor: Optional[LayoutNode] = None
    number: int = 0  # Position among siblings

    def __post_init__(self):
        if self.ancestor is None:
            self.ancestor = self

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leftmost_sibling(self) -> Optional[LayoutNode]:
        if self.parent is None or self.number == 0:
            return None
        return self.parent.children[0]

    def left_sibling(self) -> Optional[LayoutNode]:
        if self.parent is None or self.number == 0:
            return None
        return self.parent.children[self.number - 1]


# =============================================================================
# Node Measurement
# =============================================================================


class NodeMeasurer:
    """Measures node dimensions based on label content using matplotlib."""

    def __init__(self, config: TreeRendererConfig):
        self.config = config
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._renderer = None
        self._dpi_scale: float = 1.0

    def _ensure_renderer(self):
        """Create a temporary figure for text measurement."""
        if self._fig is None:
            # Use a figure with known dimensions for accurate measurement
            self._fig, self._ax = plt.subplots(figsize=(10, 10))
            self._ax.set_xlim(0, 10)
            self._ax.set_ylim(0, 10)
            self._fig.canvas.draw()
            self._renderer = self._fig.canvas.get_renderer()
            # Calculate the scale factor: how many data units per inch
            self._dpi_scale = 10.0 / self._fig.get_figwidth()  # data units per inch

    def measure_label(self, label: str, font_size: float = None) -> Tuple[float, float]:
        """Measure the dimensions needed for a label.

        Args:
            label: The text label to measure
            font_size: Font size (uses 9.0 as default)

        Returns:
            Tuple of (width, height) in data coordinates
        """
        self._ensure_renderer()

        if font_size is None:
            font_size = 9.0

        # Wrap long labels
        if len(label) > self.config.max_label_width:
            wrapped = textwrap.wrap(label, width=self.config.max_label_width)
            label = "\n".join(wrapped)

        # Count lines for height estimation
        num_lines = label.count("\n") + 1

        # Measure text using matplotlib
        txt = self._ax.text(0, 0, label, fontsize=font_size, fontfamily="sans-serif")
        bbox = txt.get_window_extent(renderer=self._renderer)
        txt.remove()

        # Convert pixels to data coordinates
        # bbox is in display pixels, convert to inches then to data units
        dpi = self._fig.dpi
        width_inches = bbox.width / dpi
        height_inches = bbox.height / dpi

        # Convert to data coordinates using the scale
        width = width_inches * self._dpi_scale + 2 * self.config.node_padding_x
        height = height_inches * self._dpi_scale + 2 * self.config.node_padding_y

        # Apply minimums
        width = max(width, self.config.min_node_width)
        height = max(height, self.config.min_node_height)

        return width, height

    def cleanup(self):
        """Close the temporary figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._renderer = None


# =============================================================================
# Layout Engine - Modified Reingold-Tilford Algorithm
# =============================================================================


class TreeLayoutEngine:
    """
    Computes tree layouts using a modified Reingold-Tilford algorithm.

    Supports variable node sizes and multiple orientations.
    Reference: Reingold & Tilford (1981), improved by Buchheim et al. (2002)
    """

    def __init__(self, config: TreeRendererConfig):
        self.config = config
        self.measurer = NodeMeasurer(config)

    def compute_layout(self, root: LayoutNode) -> Dict[str, Tuple[float, float]]:
        """
        Compute positions for all nodes in the tree.

        Args:
            root: The root LayoutNode

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        # Initialize tree structure
        self._initialize_tree(root, depth=0)

        # First pass: compute preliminary x-coordinates (bottom-up)
        self._first_walk(root)

        # Second pass: compute final x-coordinates (top-down)
        self._second_walk(root, mod=0)

        # Compute y-positions based on depth and adjust for orientation
        positions = self._compute_final_positions(root)

        self.measurer.cleanup()
        return positions

    def _initialize_tree(self, node: LayoutNode, depth: int, parent: LayoutNode = None):
        """Initialize tree structure with depth and parent references."""
        node.depth = depth
        node.parent = parent
        for i, child in enumerate(node.children):
            child.number = i
            self._initialize_tree(child, depth + 1, node)

    def _first_walk(self, node: LayoutNode):
        """First pass: compute preliminary x-coordinates bottom-up."""
        if node.is_leaf():
            # Leaf node: position relative to left sibling
            left = node.left_sibling()
            if left:
                node.prelim = left.prelim + self._spacing(left, node)
            else:
                node.prelim = 0
        else:
            # Internal node: center above children
            default_ancestor = node.children[0]

            for child in node.children:
                self._first_walk(child)
                default_ancestor = self._apportion(child, default_ancestor)

            self._execute_shifts(node)

            # Center node above its children
            first_child = node.children[0]
            last_child = node.children[-1]
            midpoint = (first_child.prelim + last_child.prelim) / 2

            left = node.left_sibling()
            if left:
                node.prelim = left.prelim + self._spacing(left, node)
                node.mod = node.prelim - midpoint
            else:
                node.prelim = midpoint

    def _spacing(self, left: LayoutNode, right: LayoutNode) -> float:
        """Calculate minimum spacing between two nodes."""
        return (left.width + right.width) / 2 + self.config.min_sibling_gap

    def _apportion(self, node: LayoutNode, default_ancestor: LayoutNode) -> LayoutNode:
        """Ensure subtrees don't overlap."""
        left_sibling = node.left_sibling()
        if left_sibling is None:
            return default_ancestor

        # Traverse contours
        vip = node  # Inner right
        vop = node  # Outer right
        vim = left_sibling  # Inner left
        vom = node.leftmost_sibling()  # Outer left

        sip = vip.mod
        sop = vop.mod
        sim = vim.mod
        som = vom.mod if vom else 0

        while self._next_right(vim) and self._next_left(vip):
            vim = self._next_right(vim)
            vip = self._next_left(vip)
            vom = self._next_left(vom) if vom else None
            vop = self._next_right(vop)

            if vop:
                vop.ancestor = node

            shift = (vim.prelim + sim) - (vip.prelim + sip) + self._spacing(vim, vip)

            if shift > 0:
                ancestor = self._get_ancestor(vim, node, default_ancestor)
                self._move_subtree(ancestor, node, shift)
                sip += shift
                sop += shift

            sim += vim.mod
            sip += vip.mod
            if vom:
                som += vom.mod
            if vop:
                sop += vop.mod

        # Set threads for efficient traversal
        if self._next_right(vim) and not self._next_right(vop):
            vop.thread = self._next_right(vim)
            vop.mod += sim - sop

        if self._next_left(vip) and not self._next_left(vom):
            if vom:
                vom.thread = self._next_left(vip)
                vom.mod += sip - som
            default_ancestor = node

        return default_ancestor

    def _next_left(self, node: LayoutNode) -> Optional[LayoutNode]:
        """Get next node on left contour."""
        if node.children:
            return node.children[0]
        return node.thread

    def _next_right(self, node: LayoutNode) -> Optional[LayoutNode]:
        """Get next node on right contour."""
        if node.children:
            return node.children[-1]
        return node.thread

    def _get_ancestor(self, vim: LayoutNode, node: LayoutNode, default: LayoutNode) -> LayoutNode:
        """Get appropriate ancestor for subtree move."""
        if vim.ancestor and vim.ancestor.parent == node.parent:
            return vim.ancestor
        return default

    def _move_subtree(self, wm: LayoutNode, wp: LayoutNode, shift: float):
        """Move subtree to avoid overlap."""
        subtrees = wp.number - wm.number
        if subtrees > 0:
            wp.change -= shift / subtrees
            wp.shift += shift
            wm.change += shift / subtrees
            wp.prelim += shift
            wp.mod += shift

    def _execute_shifts(self, node: LayoutNode):
        """Execute accumulated shifts for children."""
        shift = 0
        change = 0
        for child in reversed(node.children):
            child.prelim += shift
            child.mod += shift
            change += child.change
            shift += child.shift + change

    def _second_walk(self, node: LayoutNode, mod: float):
        """Second pass: compute final x-coordinates top-down."""
        node.x = node.prelim + mod
        for child in node.children:
            self._second_walk(child, mod + node.mod)

    def _compute_final_positions(self, root: LayoutNode) -> Dict[str, Tuple[float, float]]:
        """Compute final (x, y) positions based on orientation."""
        positions = {}
        all_nodes = self._collect_nodes(root)

        # Compute y-positions based on depth with consistent spacing
        depth_heights = self._compute_depth_positions(all_nodes)

        for node in all_nodes:
            base_y = depth_heights[node.depth]
            x = node.x

            # Transform based on orientation
            if self.config.orientation == Orientation.TOP_DOWN:
                positions[node.node_id] = (x, -base_y)
            elif self.config.orientation == Orientation.BOTTOM_UP:
                positions[node.node_id] = (x, base_y)
            elif self.config.orientation == Orientation.LEFT_RIGHT:
                positions[node.node_id] = (base_y, x)
            elif self.config.orientation == Orientation.RIGHT_LEFT:
                positions[node.node_id] = (-base_y, x)

        return positions

    def _collect_nodes(self, root: LayoutNode) -> List[LayoutNode]:
        """Collect all nodes in breadth-first order."""
        nodes = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def _compute_depth_positions(self, nodes: List[LayoutNode]) -> Dict[int, float]:
        """Compute y-position for each depth level."""
        # Group nodes by depth
        by_depth: Dict[int, List[LayoutNode]] = {}
        for node in nodes:
            if node.depth not in by_depth:
                by_depth[node.depth] = []
            by_depth[node.depth].append(node)

        # Compute max height per depth
        max_heights = {}
        for depth, depth_nodes in by_depth.items():
            max_heights[depth] = max(n.height for n in depth_nodes)

        # Compute cumulative y-positions
        depths = sorted(by_depth.keys())
        y_positions = {}
        current_y = 0

        for depth in depths:
            y_positions[depth] = current_y + max_heights[depth] / 2
            current_y += max_heights[depth] + self.config.min_level_gap

        return y_positions


# =============================================================================
# Node Style and Label Helpers
# =============================================================================


def get_node_label_simple(node: Node) -> str:
    """Get a simple label for a regular tree node."""
    if node.is_term():
        val = node.get_value()
        # Check if it's a Number node
        from plagih.trees import Number

        if isinstance(node, Number):
            fval = float(val) if not isinstance(val, float) else val
            if fval == int(fval):
                return str(int(fval))
            return f"{fval:.4g}"
        return str(val)
    return node.showme or type(node).__name__


_SHAPE_MAP = {"ellipse": NodeShape.ELLIPSE, "diamond": NodeShape.DIAMOND, "rounded": NodeShape.ROUNDED_RECTANGLE}


def get_node_style_for_type(node: Node) -> NodeStyle:
    """Get style from node's _viz_* class attributes (set on base classes in trees.py)."""
    return NodeStyle(
        fill_color=node._viz_color,
        border_color=node._viz_border,
        text_color=node._viz_text,
        shape=_SHAPE_MAP.get(getattr(node, "_viz_shape", "rounded"), NodeShape.ROUNDED_RECTANGLE),
    )


def get_merged_node_style(merged_node: MergedNode) -> NodeStyle:
    """Get style for a merged tree node."""
    if merged_node.is_root:
        return NodeStyle(
            fill_color="#FFB74D",
            border_color="#E65100",
            text_color="#000000",
            shape=NodeShape.ROUNDED_RECTANGLE,
            border_width=2.0,
        )
    elif merged_node.node_type == "terminal":
        return NodeStyle(fill_color="#A5D6A7", border_color="#2E7D32", text_color="#1B5E20", shape=NodeShape.ELLIPSE)
    else:
        return NodeStyle(
            fill_color="#90CAF9", border_color="#1565C0", text_color="#0D47A1", shape=NodeShape.ROUNDED_RECTANGLE
        )


# =============================================================================
# Tree Builders - Convert different tree types to LayoutNode
# =============================================================================


def build_layout_tree_from_node(root: Node, config: TreeRendererConfig, measurer: NodeMeasurer) -> LayoutNode:
    """Convert a plagih Node tree to a LayoutNode tree."""
    counter = [0]

    def build_recursive(node: Node) -> LayoutNode:
        node_id = f"n{counter[0]}"
        counter[0] += 1

        label = get_node_label_simple(node)
        style = get_node_style_for_type(node)
        width, height = measurer.measure_label(label, style.font_size)

        layout_node = LayoutNode(node_id=node_id, label=label, width=width, height=height, style=style)

        if not node.is_term():
            for child in node.get_childs():
                child_layout = build_recursive(child)
                layout_node.children.append(child_layout)

        return layout_node

    return build_recursive(root)


def build_layout_tree_from_merged(
    graph: MergedEvaluationGraph, config: TreeRendererConfig, measurer: NodeMeasurer
) -> Tuple[LayoutNode, Dict[str, LayoutNode]]:
    """
    Convert a MergedEvaluationGraph to a LayoutNode structure.

    For merged graphs (DAGs), we need special handling since nodes can have
    multiple parents. We create a tree view for layout purposes.

    Returns:
        Tuple of (root_layout_node, dict mapping original node_ids to LayoutNodes)
    """
    layout_nodes: Dict[str, LayoutNode] = {}

    # Create LayoutNodes for all merged nodes
    for node_id, merged_node in graph.nodes.items():
        # Generate label based on display mode
        if config.merged_display_mode == MergedDisplayMode.LABEL_ONLY:
            label = merged_node.operator_name or str(merged_node.sympy_expr)
            if len(label) > 15:
                label = label[:12] + "..."
        else:  # FULL_EXPRESSION
            label = str(merged_node.sympy_expr)
            if len(label) > config.max_expression_length:
                label = label[: config.max_expression_length - 3] + "..."

        # Add usage count
        if config.show_usage_count:
            usage = len(merged_node.original_nodes)
            if usage > 1:
                label += f"\n({usage}x)"

        # Add root marker
        if config.show_root_marker and merged_node.is_root:
            label = "[R] " + label

        # Wrap long labels
        if "\n" not in label and len(label) > config.max_label_width:
            wrapped = textwrap.wrap(label, width=config.max_label_width)
            label = "\n".join(wrapped)

        style = get_merged_node_style(merged_node)
        width, height = measurer.measure_label(label, style.font_size)

        layout_nodes[node_id] = LayoutNode(
            node_id=node_id, label=label, width=width, height=height, depth=merged_node.depth, style=style
        )

    # Build tree structure - connect children
    # Note: In merged graphs, edges go from children to parents (bottom-up)
    for node_id, merged_node in graph.nodes.items():
        layout_node = layout_nodes[node_id]
        for child_id in merged_node.child_ids:
            if child_id in layout_nodes:
                layout_node.children.append(layout_nodes[child_id])

    # Find root node (highest depth or explicitly marked)
    if graph.root_ids:
        main_root_id = graph.root_ids[0]
    else:
        # Fallback: find node with no parents (max depth)
        main_root_id = max(layout_nodes.keys(), key=lambda nid: graph.nodes[nid].depth)

    return layout_nodes[main_root_id], layout_nodes


# =============================================================================
# Tree Renderer - Main rendering class
# =============================================================================


class TreeRenderer:
    """Renders trees to matplotlib figures."""

    def __init__(self, config: TreeRendererConfig = None):
        self.config = config or TreeRendererConfig()
        self.layout_engine = TreeLayoutEngine(self.config)
        self.measurer = NodeMeasurer(self.config)

    def render_tree(
        self, root: Union[Node, LayoutNode], title: str = "", figsize: Tuple[float, float] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render a tree to a matplotlib figure.

        Args:
            root: Either a plagih Node or a pre-built LayoutNode
            title: Optional title for the figure
            figsize: Optional figure size (auto-computed if None)

        Returns:
            Tuple of (figure, axes)
        """
        # Convert to LayoutNode if needed
        if not isinstance(root, LayoutNode):
            layout_root = build_layout_tree_from_node(root, self.config, self.measurer)
        else:
            layout_root = root

        # Compute layout
        positions = self.layout_engine.compute_layout(layout_root)
        all_nodes = self._collect_layout_nodes(layout_root)

        # Create node lookup
        node_lookup = {n.node_id: n for n in all_nodes}

        # Compute figure size if not provided
        if figsize is None:
            figsize = self._compute_figure_size(positions, node_lookup)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw edges first (behind nodes)
        self._draw_edges(ax, layout_root, positions, node_lookup)

        # Draw nodes
        self._draw_nodes(ax, all_nodes, positions)

        # Configure axes
        self._configure_axes(ax, positions, node_lookup, title)

        return fig, ax

    def render_merged_graph(
        self,
        graph: MergedEvaluationGraph,
        title: str = "",
        figsize: Tuple[float, float] = None,
        show_statistics: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render a merged evaluation graph.

        Merged graphs are DAGs, so we use a layer-based layout that
        respects the depth of each node.

        Args:
            graph: The MergedEvaluationGraph to render
            title: Optional title (statistics added if show_statistics=True)
            figsize: Optional figure size
            show_statistics: Whether to add statistics to title

        Returns:
            Tuple of (figure, axes)
        """
        # Build layout nodes
        layout_root, all_layout_nodes = build_layout_tree_from_merged(graph, self.config, self.measurer)

        # Use layer-based layout for DAGs (more appropriate than tree layout)
        positions = self._compute_layer_layout(graph, all_layout_nodes)

        # Add statistics to title
        if show_statistics:
            stats = graph.get_statistics()
            stat_str = (
                f"Trees: {stats['tree_count']} | "
                f"Nodes: {stats['total_nodes']} | "
                f"Shared: {stats['shared_nodes']} | "
                f"Savings: {stats['savings_percent']:.1f}%"
            )
            if title:
                title = f"{title}\n{stat_str}"
            else:
                title = f"Merged Evaluation Graph\n{stat_str}"

        # Compute figure size
        if figsize is None:
            figsize = self._compute_figure_size(positions, all_layout_nodes)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw edges (all edges from the DAG)
        self._draw_merged_edges(ax, graph, positions, all_layout_nodes)

        # Draw nodes
        nodes_list = list(all_layout_nodes.values())
        self._draw_nodes(ax, nodes_list, positions)

        # Configure axes
        self._configure_axes(ax, positions, all_layout_nodes, title)

        # Add legend
        self._add_merged_legend(ax)

        return fig, ax

    def _compute_layer_layout(
        self, graph: MergedEvaluationGraph, layout_nodes: Dict[str, LayoutNode]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute layer-based layout for merged DAG.

        Uses dynamic spacing based on number of nodes to prevent overlap
        in large graphs.
        """
        positions = {}

        # Group nodes by depth
        max_depth = max((n.depth for n in graph.nodes.values()), default=0)
        layers: Dict[int, List[str]] = {d: [] for d in range(max_depth + 1)}
        for node_id, merged_node in graph.nodes.items():
            layers[merged_node.depth].append(node_id)

        # Calculate total nodes and max layer size for dynamic spacing
        total_nodes = len(graph.nodes)
        max_layer_size = max(len(layer) for layer in layers.values()) if layers else 1

        # Dynamic spacing based on graph complexity
        # Base gaps are now smaller, so scale factors adjusted
        if total_nodes > 80 or max_layer_size > 15:
            # Very large graph: moderate spacing
            h_gap = max(self.config.min_sibling_gap * 1.5, 0.25)
            v_gap = max(self.config.min_level_gap * 1.3, 0.7)
        elif total_nodes > 40 or max_layer_size > 8:
            # Large graph
            h_gap = max(self.config.min_sibling_gap * 1.3, 0.2)
            v_gap = max(self.config.min_level_gap * 1.2, 0.6)
        else:
            # Normal graph - use config values
            h_gap = self.config.min_sibling_gap
            v_gap = self.config.min_level_gap

        # Compute max height per layer
        layer_heights = {}
        for depth, node_ids in layers.items():
            if node_ids:
                layer_heights[depth] = max(layout_nodes[nid].height for nid in node_ids)
            else:
                layer_heights[depth] = self.config.min_node_height

        # Compute y-positions (cumulative)
        y_positions = {}
        current_y = 0
        for depth in range(max_depth + 1):
            y_positions[depth] = current_y + layer_heights.get(depth, 0.5) / 2
            current_y += layer_heights.get(depth, 0.5) + v_gap

        # Compute x-positions per layer (centered)
        for depth, node_ids in layers.items():
            if not node_ids:
                continue

            # Sort nodes for consistent layout
            node_ids = sorted(node_ids)

            # Calculate total width with dynamic gap
            widths = [layout_nodes[nid].width for nid in node_ids]
            total_width = sum(widths) + h_gap * (len(node_ids) - 1)

            # Position nodes centered around 0
            current_x = -total_width / 2
            for nid in node_ids:
                w = layout_nodes[nid].width
                x = current_x + w / 2
                y = y_positions[depth]

                # Apply orientation
                if self.config.orientation == Orientation.TOP_DOWN:
                    positions[nid] = (x, -y)
                elif self.config.orientation == Orientation.BOTTOM_UP:
                    positions[nid] = (x, y)
                elif self.config.orientation == Orientation.LEFT_RIGHT:
                    positions[nid] = (y, x)
                elif self.config.orientation == Orientation.RIGHT_LEFT:
                    positions[nid] = (-y, x)

                current_x += w + h_gap

        return positions

    def _collect_layout_nodes(self, root: LayoutNode) -> List[LayoutNode]:
        """Collect all layout nodes in breadth-first order."""
        nodes = []
        queue = [root]
        seen = set()
        while queue:
            node = queue.pop(0)
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def _compute_figure_size(
        self, positions: Dict[str, Tuple[float, float]], node_lookup: Union[Dict[str, LayoutNode], List[LayoutNode]]
    ) -> Tuple[float, float]:
        """Compute appropriate figure size based on tree extent.

        For large trees, the figure is scaled up to prevent node overlap.
        """
        if not positions:
            return (10, 8)

        # Handle both dict and list inputs
        if isinstance(node_lookup, list):
            node_lookup = {n.node_id: n for n in node_lookup}

        # Compute bounds including node sizes
        x_coords = []
        y_coords = []
        for nid, (x, y) in positions.items():
            if nid in node_lookup:
                node = node_lookup[nid]
                x_coords.extend([x - node.width / 2, x + node.width / 2])
                y_coords.extend([y - node.height / 2, y + node.height / 2])

        if not x_coords:
            return (10, 8)

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # Count nodes for scaling factor
        num_nodes = len(positions)

        # Base scaling: ensure enough space for the content
        # For large graphs, we need more generous sizing
        if num_nodes > 50:
            # Large trees: scale more aggressively
            scale_factor = 1.3
            min_width, min_height = 20, 15
            # No hard max - let it grow as needed
            max_width, max_height = 100, 80
        elif num_nodes > 20:
            # Medium trees
            scale_factor = 1.2
            min_width, min_height = 12, 10
            max_width, max_height = 60, 50
        else:
            # Small trees
            scale_factor = 1.1
            min_width, min_height = 8, 6
            max_width, max_height = 30, 25

        fig_width = max(min_width, min(max_width, (x_range + 2) * scale_factor))
        fig_height = max(min_height, min(max_height, (y_range + 2) * scale_factor))

        return (fig_width, fig_height)

    def _draw_edges(
        self,
        ax: plt.Axes,
        root: LayoutNode,
        positions: Dict[str, Tuple[float, float]],
        node_lookup: Dict[str, LayoutNode],
    ):
        """Draw edges between nodes."""
        drawn = set()

        def draw_recursive(node: LayoutNode):
            if node.node_id not in positions:
                return

            px, py = positions[node.node_id]

            for child in node.children:
                edge_key = (node.node_id, child.node_id)
                if edge_key in drawn:
                    continue
                drawn.add(edge_key)

                if child.node_id not in positions:
                    continue

                cx, cy = positions[child.node_id]

                # Compute connection points based on orientation
                if self.config.orientation in (Orientation.TOP_DOWN, Orientation.BOTTOM_UP):
                    if py > cy:  # Parent above child
                        y1 = py - node.height / 2
                        y2 = cy + child.height / 2
                    else:
                        y1 = py + node.height / 2
                        y2 = cy - child.height / 2
                    ax.plot(
                        [px, cx],
                        [y1, y2],
                        color=self.config.edge_color,
                        linewidth=self.config.edge_width,
                        zorder=1,
                        solid_capstyle="round",
                    )
                else:  # LR or RL
                    if px > cx:
                        x1 = px - node.width / 2
                        x2 = cx + child.width / 2
                    else:
                        x1 = px + node.width / 2
                        x2 = cx - child.width / 2
                    ax.plot(
                        [x1, x2],
                        [py, cy],
                        color=self.config.edge_color,
                        linewidth=self.config.edge_width,
                        zorder=1,
                        solid_capstyle="round",
                    )

                draw_recursive(child)

        draw_recursive(root)

    def _draw_merged_edges(
        self,
        ax: plt.Axes,
        graph: MergedEvaluationGraph,
        positions: Dict[str, Tuple[float, float]],
        layout_nodes: Dict[str, LayoutNode],
    ):
        """Draw edges for merged DAG."""
        for node_id, merged_node in graph.nodes.items():
            if node_id not in positions:
                continue

            px, py = positions[node_id]
            parent_node = layout_nodes[node_id]

            for child_id in merged_node.child_ids:
                if child_id not in positions:
                    continue

                cx, cy = positions[child_id]
                child_node = layout_nodes[child_id]

                # Compute connection points
                if self.config.orientation in (Orientation.TOP_DOWN, Orientation.BOTTOM_UP):
                    if py > cy:
                        y1 = py - parent_node.height / 2
                        y2 = cy + child_node.height / 2
                    else:
                        y1 = py + parent_node.height / 2
                        y2 = cy - child_node.height / 2
                    ax.plot(
                        [px, cx],
                        [y1, y2],
                        color=self.config.edge_color,
                        linewidth=self.config.edge_width,
                        zorder=1,
                        solid_capstyle="round",
                    )
                else:
                    if px > cx:
                        x1 = px - parent_node.width / 2
                        x2 = cx + child_node.width / 2
                    else:
                        x1 = px + parent_node.width / 2
                        x2 = cx - child_node.width / 2
                    ax.plot(
                        [x1, x2],
                        [py, cy],
                        color=self.config.edge_color,
                        linewidth=self.config.edge_width,
                        zorder=1,
                        solid_capstyle="round",
                    )

    def _draw_nodes(self, ax: plt.Axes, nodes: List[LayoutNode], positions: Dict[str, Tuple[float, float]]):
        """Draw all nodes."""
        for node in nodes:
            if node.node_id not in positions:
                continue

            x, y = positions[node.node_id]
            w, h = node.width, node.height
            style = node.style

            # Draw shape based on node style
            if style.shape == NodeShape.ELLIPSE:
                patch = mpatches.Ellipse(
                    (x, y),
                    w,
                    h,
                    facecolor=style.fill_color,
                    edgecolor=style.border_color,
                    linewidth=style.border_width,
                    zorder=2,
                )
            elif style.shape == NodeShape.DIAMOND:
                # Diamond vertices
                verts = [
                    (x, y + h / 2),  # top
                    (x + w / 2, y),  # right
                    (x, y - h / 2),  # bottom
                    (x - w / 2, y),  # left
                ]
                patch = mpatches.Polygon(
                    verts,
                    facecolor=style.fill_color,
                    edgecolor=style.border_color,
                    linewidth=style.border_width,
                    zorder=2,
                )
            elif style.shape == NodeShape.RECTANGLE:
                patch = FancyBboxPatch(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    boxstyle="square,pad=0",
                    facecolor=style.fill_color,
                    edgecolor=style.border_color,
                    linewidth=style.border_width,
                    zorder=2,
                )
            else:  # ROUNDED_RECTANGLE (default)
                patch = FancyBboxPatch(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=style.fill_color,
                    edgecolor=style.border_color,
                    linewidth=style.border_width,
                    zorder=2,
                )

            ax.add_patch(patch)

            # Draw label
            ax.text(
                x,
                y,
                node.label,
                ha="center",
                va="center",
                fontsize=style.font_size,
                color=style.text_color,
                fontfamily=style.font_family,
                fontweight="bold",
                linespacing=1.1,
                zorder=3,
            )

    def _configure_axes(
        self,
        ax: plt.Axes,
        positions: Dict[str, Tuple[float, float]],
        node_lookup: Union[Dict[str, LayoutNode], List[LayoutNode]],
        title: str,
    ):
        """Configure axes appearance."""
        if isinstance(node_lookup, list):
            node_lookup = {n.node_id: n for n in node_lookup}

        if positions and node_lookup:
            x_coords = []
            y_coords = []
            for nid, (x, y) in positions.items():
                if nid in node_lookup:
                    node = node_lookup[nid]
                    x_coords.extend([x - node.width / 2, x + node.width / 2])
                    y_coords.extend([y - node.height / 2, y + node.height / 2])

            if x_coords and y_coords:
                x_margin = max(0.5, (max(x_coords) - min(x_coords)) * 0.08)
                y_margin = max(0.5, (max(y_coords) - min(y_coords)) * 0.08)

                ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
                ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

        ax.set_aspect("equal")
        ax.axis("off")

        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=15)

    def _add_merged_legend(self, ax: plt.Axes):
        """Add legend for merged tree visualization."""
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#FFB74D",
                markeredgecolor="#E65100",
                markersize=12,
                label="Root (output)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#90CAF9",
                markeredgecolor="#1565C0",
                markersize=12,
                label="Operator",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#A5D6A7",
                markeredgecolor="#2E7D32",
                markersize=10,
                label="Terminal",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)


# =============================================================================
# High-Level API Functions
# =============================================================================


def render_tree(
    tree: Node,
    filename: str = "tree",
    output_dir: Union[str, Path] = None,
    orientation: str = "TB",
    config: TreeRendererConfig = None,
    title: str = "",
    show: bool = False,
    **kwargs,
) -> str:
    """
    Render a plagih tree and save to file.

    Args:
        tree: Root node of the tree
        filename: Output filename (without extension)
        output_dir: Directory for output (default: tree_output/)
        orientation: Layout orientation ("TB", "BT", "LR", "RL")
        config: Optional TreeRendererConfig
        title: Optional figure title
        show: If True, display the figure
        **kwargs: Additional config overrides

    Returns:
        Path to saved image
    """
    # Setup config
    if config is None:
        config = TreeRendererConfig()

    config.orientation = Orientation(orientation)

    # Apply kwargs to config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tree_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render
    renderer = TreeRenderer(config)
    fig, ax = renderer.render_tree(tree, title=title)

    # Save
    output_path = output_dir / f"{filename}.png"
    plt.savefig(
        output_path, dpi=config.figure_dpi, bbox_inches="tight", facecolor=config.background_color, edgecolor="none"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Tree saved to: {output_path}")
    # TODO(sfeh): should use log("f", ...) but render_tree is also used standalone
    return str(output_path)


def render_merged_tree(
    graph: MergedEvaluationGraph,
    filename: str = "merged_tree",
    output_dir: Union[str, Path] = None,
    orientation: str = "BT",
    display_mode: str = "label",
    config: TreeRendererConfig = None,
    title: str = "",
    show: bool = False,
    **kwargs,
) -> str:
    """
    Render a merged evaluation graph and save to file.

    Args:
        graph: MergedEvaluationGraph to visualize
        filename: Output filename (without extension)
        output_dir: Directory for output (default: tree_output/)
        orientation: Layout orientation ("TB", "BT", "LR", "RL")
        display_mode: "label" for label-only, "expression" for full expressions
        config: Optional TreeRendererConfig
        title: Optional figure title
        show: If True, display the figure
        **kwargs: Additional options (show_statistics, etc.)

    Returns:
        Path to saved image
    """
    # Setup config
    if config is None:
        config = TreeRendererConfig()

    config.orientation = Orientation(orientation)
    config.merged_display_mode = (
        MergedDisplayMode.FULL_EXPRESSION if display_mode == "expression" else MergedDisplayMode.LABEL_ONLY
    )

    # Apply kwargs to config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tree_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render
    renderer = TreeRenderer(config)
    fig, ax = renderer.render_merged_graph(graph, title=title, show_statistics=kwargs.get("show_statistics", True))

    # Save
    output_path = output_dir / f"{filename}.png"
    plt.savefig(
        output_path, dpi=config.figure_dpi, bbox_inches="tight", facecolor=config.background_color, edgecolor="none"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    log("f", f"Merged tree saved to: {output_path}")
    return str(output_path)


# =============================================================================
# Paretofront Visualization (moved from visualize_trees.py)
# =============================================================================


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


def _compute_simple_tree_layout(root: Node) -> tuple:
    """Compute a simple hierarchical tree layout for subplot rendering.

    Returns (positions, labels, colors, edges) dicts keyed by integer node id.
    Uses get_node_label_simple / get_node_style_for_type for consistency.
    """
    positions, labels, colors, edges = {}, {}, {}, []
    counter = [0]

    def get_subtree_width(node: Node) -> int:
        if node.is_term():
            return 1
        children = node.get_childs()
        return sum(get_subtree_width(c) for c in children) if children else 1

    def layout_node(node: Node, depth: int, left_bound: float, parent_id=None):
        nid = counter[0]
        counter[0] += 1
        labels[nid] = get_node_label_simple(node)
        colors[nid] = get_node_style_for_type(node).fill_color
        if parent_id is not None:
            edges.append((parent_id, nid))
        children = node.get_childs() if not node.is_term() else []
        if not children:
            positions[nid] = (left_bound + 0.5, -depth)
            return nid, 1
        total_width, child_xs, cur = 0, [], left_bound
        for child in children:
            cid, cw = layout_node(child, depth + 1, cur, nid)
            child_xs.append(positions[cid][0])
            cur += cw
            total_width += cw
        positions[nid] = ((child_xs[0] + child_xs[-1]) / 2, -depth)
        return nid, total_width

    layout_node(root, 0, 0)
    return positions, labels, colors, edges


def _render_tree_on_axes(ax, tree: Node, node_size_scale: float = 1.0):
    """Render a single tree onto a matplotlib Axes (for subplot grids)."""
    positions, labels, colors, edges = _compute_simple_tree_layout(tree)
    tree_width = _get_tree_width(tree)
    node_size = max(0.3, min(0.5, node_size_scale / max(tree_width, 1)))

    for pid, cid in edges:
        x1, y1 = positions[pid]
        x2, y2 = positions[cid]
        ax.plot([x1, x2], [y1, y2], "-", linewidth=1.2, zorder=1, color="#666666")

    for nid, (x, y) in positions.items():
        label = labels[nid]
        lw = max(node_size, len(label) * 0.08)
        patch = mpatches.FancyBboxPatch(
            (x - lw / 2, y - node_size / 4),
            lw,
            node_size / 2,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02),
            facecolor=colors[nid],
            edgecolor="#333333",
            linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(patch)
        fs = max(6, min(9, 72 / max(len(label), 1)))
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=3)

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        mx = max(0.5, (max(xs) - min(xs)) * 0.1)
        my = max(0.5, (max(ys) - min(ys)) * 0.1)
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal")
    ax.axis("off")


def _visualize_single_tree_to_file(
    tree: Node,
    filename: str,
    output_dir: Path,
    title: str = "",
    figsize=(8, 6),
    dpi: int = 150,
) -> str:
    """Render a single tree to a PNG file with dynamic sizing."""
    depth = _get_tree_depth(tree)
    width = _get_tree_width(tree)
    fig_w = max(figsize[0], width * 1.2)
    fig_h = max(figsize[1], depth * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _render_tree_on_axes(ax, tree, node_size_scale=3.0)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def visualize_paretofront(
    paretofront: list,
    filename: str = "paretofront_trees",
    output_dir: Optional[Path] = Path.cwd(),
    figsize_per_tree: tuple = (5, 4),
    dpi: int = 150,
    max_cols: int = 4,
    save_individual: bool = True,
):
    """Visualize all Paretofront candidates in a single combined image.

    Args:
        paretofront: List of Candidate objects from the Paretofront.
        filename: Output filename (without extension).
        output_dir: Directory for output (default: cwd).
        figsize_per_tree: Size (width, height) for each individual tree subplot.
        dpi: Resolution of the output image.
        max_cols: Maximum number of columns in the grid.
        save_individual: If True, also save each tree as individual image.

    Returns:
        Path to the generated image file.
    """
    if not paretofront:
        log("w", "Paretofront is empty, nothing to visualize.")
        return

    output_dir = path_make_dir(output_dir / "tree_output")
    sorted_front = sorted(paretofront, key=lambda c: (c.get_parsim(), c.get_fitness()))

    # Save individual images
    if save_individual:
        individual_dir = output_dir / "paretoCandidates"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for idx, cand in enumerate(sorted_front):
            tree = cand.get_evotree()
            p, f = cand.get_parsim(), cand.get_fitness()
            _visualize_single_tree_to_file(
                tree,
                f"pareto_{idx + 1:03d}_P{p:.0f}_F{f:.4g}",
                individual_dir,
                title=f"P={p:.0f}, F={f:.4g}",
                dpi=dpi,
            )
        log("ff", f"Individual Paretofront trees saved to: {individual_dir}")

    # Grid layout
    n_trees = len(paretofront)
    n_cols = min(n_trees, max_cols)
    n_rows = (n_trees + n_cols - 1) // n_cols

    max_depth = max(_get_tree_depth(c.get_evotree()) for c in sorted_front)
    max_width = max(_get_tree_width(c.get_evotree()) for c in sorted_front)
    w_scale = max(1.0, max_width / 4)
    h_scale = max(1.0, max_depth / 3)
    fig_w = max(8, figsize_per_tree[0] * n_cols * w_scale)
    fig_h = max(6, figsize_per_tree[1] * n_rows * h_scale)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_trees == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, cand in enumerate(sorted_front):
        ax = axes[idx // n_cols, idx % n_cols]
        _render_tree_on_axes(ax, cand.get_evotree(), node_size_scale=2.0)
        p, f = cand.get_parsim(), cand.get_fitness()
        ax.set_title(f"P={p:.0f}, F={f:.4g}", fontsize=10, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_trees, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.suptitle(f"Paretofront ({n_trees} candidates)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log("ff", f"Paretofront visualization saved to: {output_path}")
    return str(output_path)
