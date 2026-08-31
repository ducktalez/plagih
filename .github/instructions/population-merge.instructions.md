---
applyTo: "plagih/population_merge.py"
---

# Population Merge – Copilot Instructions

- **Implemented**: "One-evaluation-tree" — builds a DAG where no expression is
  computed twice. Uses `MergedEvaluationGraph` / `MergedNode`.
- **Implemented**: Trunk analysis (§3.5 in `docs/TARGETED_OPTIMIZATION.md`) —
  `find_trunks()` ranks shared subtrees by `n_trees * subtree_size`;
  `suggest_origin_trees()` returns **copies** via `fast_tree_copy` +
  `repair_all()` — never hand out original node references.
- **DAG invariant**: Every `MergedNode` appears at most once in the graph.
  Shared sub-expressions point to the same node instance.
- **Read-only analysis**: All analysis functions must not modify input trees
  or the population.
- **Planned strategies** (see `docs/IMPLEMENTATION_PLAN.md` M3):
  "Expand-random-tree" and "Clustered-tree" are documented in the module
  docstring but not yet implemented.

