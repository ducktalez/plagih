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

