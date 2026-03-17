---
applyTo: "benchmarks/**"
---

# Benchmarks – Copilot Instructions

- **Timing** (P9): Always use `enable_analysis=False` for performance benchmarking.
- **Samples format**: CSV with typed headers (`input1:float,target:float`).
- **Adding a benchmark**: Create `benchmarks/NAME/gp_files/samples.csv`,
  add `demo_NAME()` in `plagih_gp.py`, document in `docs/BENCHMARKS.md`.

Full details: `docs/BENCHMARKS.md`
