---
applyTo: "plagih/monitoring.py"
---

# Monitoring – Copilot Instructions

- **No `get_sympy_expr()` in metrics** (P10): Use `str(c.tree)` for unique-expression
  counting. SymPy conversion adds ~3–5s per generation with pop=1000.
- **Callback exceptions are swallowed**: `on_generation`, `on_improvement`,
  `on_pareto_update` callbacks catch all exceptions silently. Be careful with
  debugging — add explicit try/except in callback code if needed.
- **DataFrame column mapping**: `to_dataframe()` renames internal metric keys
  for backward compatibility (e.g. `pop_size` → `pop_len`, `gen_time` → `time`).
  When adding new metrics, add the rename mapping too.

