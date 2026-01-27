"""
Sometimes, just sometimes, you gotta hate python.
This file needs to be here, empty. Lel!

Exports for convenient access to plagih functionality.
"""

# Population merge functionality
from plagih.population_merge import (
    build_one_evaluation_tree,
    analyze_population_sharing,
    MergedEvaluationGraph,
    MergedNode,
    get_expressions_by_depth,
    visualize_merged_graph,
)

# Unified evaluation context (optional, backward-compatible)
from plagih.evaluation_context import (
    EvaluationContext,
    EvaluationResult,
    EvalMode,
    create_context,
    evaluate_tree,
    add_unified_evaluation_to_node,
)

# Visualization (lazy import to avoid circular dependencies)
def __getattr__(name):
    """Lazy import for visualization components."""
    _VIZ_EXPORTS = {
        'render_tree', 'render_merged_tree',
        'TreeRenderer', 'TreeRendererConfig',
        'Orientation', 'MergedDisplayMode',
    }
    if name in _VIZ_EXPORTS:
        from visualization import tree_renderer
        return getattr(tree_renderer, name)
    raise AttributeError(f"module 'plagih' has no attribute {name!r}")

