"""
Visualization module for plagih GP Framework.

Provides unified tree visualization functionality for:
- Normal GP trees (Node-based)
- Merged evaluation graphs (DAGs)
- Paretofront candidate grids
- LaTeX tree export (latex_renderer)
"""


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import of visualization components."""
    if name in _EXPORTS:
        from visualization import tree_renderer

        return getattr(tree_renderer, name)

    raise AttributeError(f"module 'visualization' has no attribute {name!r}")


_EXPORTS = {
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
    "visualize_paretofront",
}

__all__ = list(_EXPORTS)
