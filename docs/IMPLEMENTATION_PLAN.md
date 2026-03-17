# Implementation Plan

> Central collection of open tasks, design discussions, and future ideas.
> Add new items here instead of scattering TODOs through source code.
> Reference relevant files/pitfalls with each item.

---

## High Priority

### H1 – Parallelize `gen_create_initial()`
- **Where:** `ExplainableGP.gen_create_initial()` in `trees.py`
- **Problem:** Initial population creation is always sequential (P8). For
  `pop=10000` this costs ~44–47s — a fixed sequential block.
- **Approach:** Reuse existing `run_generation_parallel()` infrastructure.

### H2 – Optimize pre-selection overhead
- **Where:** `pre_select_for_tasks()` in `parallel.py`
- **Problem:** Pre-selection in the main process is currently the main IPC
  cost contributor. See `PARALLEL_BENCHMARK_DIAGNOSIS.md` §4.

### H3 – Reduce worker RAM overhead
- **Where:** `parallel.py` worker pool
- **Problem:** 8 workers consume ~1.1–1.2 GB child RSS (mostly fixed
  interpreter/import overhead). See diagnosis report §5.

---

## Medium Priority

### M1 – `canonicalize_children()` performance benchmark
- **Where:** `Node.canonicalize_children()` in `trees.py` (P14)
- **Current:** Uses `represent_str()` as sort key — recursive string generation.
- **Alternative:** Sort by subtree size (`len(child)`) first, string as tiebreaker.
- **Action:** Benchmark both approaches on large trees before deciding.

### M2 – xtype system extension
- **Where:** `xtype` on `Node` subclasses, `evolve_create_random()` in `trees.py`
- **Idea:** Support node-class constraints (e.g. `(Number, BaseOperator)`)
  beyond current `float`/`bool` primitives.
- **Pro:** Declarative constraints for operators like `Scale`.
- **Contra:** Significant changes to `operatorpool_to_picks`, `choose_operator_class`.
- **Current workaround:** Special-case logic in `evolve_create_random()`.

### M3 – Expand-random-tree and Clustered-tree merge strategies
- **Where:** `population_merge.py`
- **Status:** Documented as PLANNED in module docstring.
- **Depends on:** TED infrastructure in `tree_complexity/`.

### M4 – Grouping rules as class attributes
- **Where:** `tree_node_grouping()` in `trees.py`
- **Idea:** Declare grouping rules as class attributes or decorators on the
  node class, rather than centrally in `tree_node_grouping()`.

### ~~M5 – Migrate legacy logging calls to `log()`~~ ✅
- **Where:** `trees.py`, `parallel.py`, `paretofront.py`, `visualization/tree_renderer.py`
- **What:** All `printpl`/`printez`/`print_warning`/`print_caution` calls replaced
  with `log()`. Legacy aliases kept in `logging_utils.py` for backward compatibility.
- **Status:** Complete (452 tests pass). Verified no legacy calls remain.

### ~~M6 – Split `trees.py` into sub-modules~~ ✅
- **Where:** `plagih/trees/` package (was 4966-line monolith)
- **Layout:** `_nodes.py` (2812 lines — full node hierarchy + sympy bridge),
  `_evolution.py` (899 lines — Candidate, NodeSelect, Evolution),
  `_gp_engine.py` (1288 lines — ExplainableGP), `__init__.py` (re-export).
- **Status:** Complete (455 tests pass). All `from plagih.trees import X`
  statements continue to work via `__init__.py` re-exports.

---

## Low Priority

### L1 – Backend-specific complexity measures
- **Where:** `tree_complexity/`
- **Ideas:** Numba/LLVM IR complexity, ASM instruction count, parallel
  critical-path complexity, branch-sensitive complexity for `Ifte`/`Piecewise`.
- **Status:** Proof-of-concept exists for CPython bytecode (P16).

### L2 – Gradient tracking placeholder
- **Where:** `evaluation_context.py`
- **Status:** `track_gradients` parameter exists, emits `FutureWarning`.
- **Idea:** JAX/PyTorch integration for gradient-based optimization.

### L3 – `analyze_pareto` duplicate cleanup
- **Where:** `paretofront.py`
- **Problem:** Two similar analysis functions exist (code smell).
- **Action:** Consolidate into one.

### L4 – Background analysis process
- **Where:** `analyze_generation()` in `trees.py` (P9)
- **Idea:** Run visualization/backup IO in a separate background process
  so the main evolution loop is never blocked.

---

## Completed

- ✅ Shared memory for `df_train` — 1.68× faster worker startup
- ✅ Pre-selection replaces legacy full-population IPC — 4.8× less IPC cost
- ✅ `get_sympy_expr()` removed from monitoring hot path (P10)
- ✅ Relational-on-Piecewise guard in `get_sympy_expr()` (P12)
- ✅ Physical core detection via `cpu_count_physical()` (P11)
- ✅ Chunked batching with progress diagnostics in parallel (P12)
- ✅ **M5** — Legacy `printpl`/`printez`/`print_warning`/`print_caution` migrated to `log()`/`log_error()`





