---
applyTo: "plagih/parallel.py"
---

# Parallel – Copilot Instructions

## Key constraints

- **Windows pickle** (P4): Strategies and error metrics must be top-level functions
  (no lambdas, closures). `_ClipAutocast` is a picklable class for this reason.
- **`repair_all()` after IPC** (P1): Trees received from workers have stripped
  back-references. Already handled in `_update_worker_state()` and
  `run_generation_parallel()`.
- **Worker count** (P11): Use `cpu_count_physical()` — logical threads hurt
  CPU-bound work. Current sweet spot: 8 workers on 8-core machine.
- **Batch size**: Sweet spot **32..128 tasks per batch** (see diagnosis report).
- **Relational-on-Piecewise** (P12): Workers use small runtime batches + timeout
  to surface pathological SymPy hangs.

Full diagnosis: `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md`
