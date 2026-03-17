---
applyTo: "plagih/population_merge.py"
---

# Population Merge – Copilot Instructions

- **Implemented**: "One-evaluation-tree" — builds a DAG where no expression is
  computed twice. Uses `MergedEvaluationGraph` / `MergedNode`.
- **DAG invariant**: Every `MergedNode` appears at most once in the graph.
  Shared sub-expressions point to the same node instance.
- **Planned strategies** (see `docs/IMPLEMENTATION_PLAN.md` M3):
  "Expand-random-tree" and "Clustered-tree" are documented in the module
  docstring but not yet implemented.

