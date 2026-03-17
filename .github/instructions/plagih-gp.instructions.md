---
applyTo: "plagih_gp.py"
---

# plagih_gp – Copilot Instructions

- **Entry point & demos**: Contains `demo_minimal()`, `demo_cartpole()`,
  `demo_symbolic_regression()` and test functions. Not framework internals.
- **`ExplainableGP.create()`**: Factory method — see `docs/ARCHITECTURE.md` §4
  for lifecycle. Key params: `symbols`, `df_train`, `rootdir`, `preset`,
  `pop_max_size`, `gen_end`, `parallel`, `error_metric`.
- **Adding a demo**: Follow existing `demo_*` pattern. Use `enable_analysis=False`
  for timing benchmarks (P9).

