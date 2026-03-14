"""
Visualization module for plagih GP Framework.

This module provides unified tree visualization functionality for:
- Normal GP trees (Node-based)
- Merged evaluation graphs (DAGs)
- Multiple orientations and display modes
"""


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import of visualization components."""
    if name in _TREE_RENDERER_EXPORTS:
        from visualization import tree_renderer

        return getattr(tree_renderer, name)

    if name in _VISUALIZE_TREES_EXPORTS:
        from visualization import visualize_trees

        return getattr(visualize_trees, name)

    raise AttributeError(f"module 'visualization' has no attribute {name!r}")


_TREE_RENDERER_EXPORTS = {
    # Core classes
    "TreeRenderer",
    "TreeRendererConfig",
    "TreeLayoutEngine",
    "NodeMeasurer",
    "LayoutNode",
    # Enums
    "Orientation",
    "MergedDisplayMode",
    "NodeShape",
    "NodeStyle",
    # High-level functions
    "render_tree",
    "render_merged_tree",
    "visualize_tree",
    "visualize_merged_graph",
}

_VISUALIZE_TREES_EXPORTS = {
    "visualize_multiple_trees",
    "visualize_paretofront",
}

__all__ = list(_TREE_RENDERER_EXPORTS | _VISUALIZE_TREES_EXPORTS)
