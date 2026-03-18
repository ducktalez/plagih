---
applyTo: "plagih/targeted_optimization.py"
---

# Targeted Optimization – Copilot Instructions

## Purpose

Analysis and scoring tools for per-tree targeted optimization beyond
random evolution. See `docs/TARGETED_OPTIMIZATION.md` for the full
design document.

## Key functions

| Function | Phase | Purpose |
|---|---|---|
| `eval_node_intermediates()` | 1 | Per-node intermediate values during evaluation |
| `best_per_datapoint()` | 1 | Oracle Selector — which candidate wins each row |
| `soft_optimum_error()` | 1 | Population-level lower bound (Oracle Bound) |
| `ifte_component_scores()` | 2 | Pseudo-backprop scoring for Ifte nodes |
| `piecewise_component_scores()` | 2 | Pseudo-backprop scoring for Piecewise nodes |

## Rules

1. **No side effects on trees**: Analysis functions must not modify the
   input tree or population. They are read-only diagnostic tools.
2. **Lazy imports**: Import `plagih.trees._nodes` classes inside function
   bodies to avoid circular imports at module level.
3. **NaN handling**: Always treat NaN predictions as `inf` error so they
   never "win" a row in best-per-datapoint analysis.
4. **Performance awareness**: `eval_node_intermediates` re-evaluates
   children via `eval_predict_numpy_now`. This is acceptable for analysis
   but should not be called in hot-path evolution loops.
5. **Target column**: The caller provides `target` as a numpy array. The
   functions do not assume a column name like `"action"`.

