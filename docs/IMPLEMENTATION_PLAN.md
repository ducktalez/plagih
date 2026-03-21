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
- **Benchmark note (2026-03-20):** The new tree-creation benchmark shows raw
  tree building is cheap (`~1.5–2.1 ms/tree`), while `gen_create_initial()` is
  dominated by **evaluation** inside `tree_to_candidate()` (`max_evaluate_ms`
  observed up to `~55.9 ms`). Parallelizing only the raw generator will not be
  sufficient; the evaluation hot path must be considered too.

### H4 – Reduce evaluation hot-path cost during tree production
- **Where:** `ExplainableGP.tree_to_candidate()` in `trees/_gp_engine.py` and
  `evaluate_tree_standalone()` in `parallel.py`
- **Problem:** New tree-creation benchmark (`bench_tree_creation.py`) indicates
  that both initial population creation and later generation production are
  dominated by **evaluation**, not raw tree generation.
- **Current evidence:** `initial_population` and `active_generation` both show
  evaluation as the dominant phase for most successful trees, with evaluation
  failures significantly outnumbering creation/simplification failures.
- **Ideas:**
  1. Add a cheap pre-filter before full evaluation for obviously invalid trees.
  2. Investigate `canonicalize_children()` and LUT interactions in the hot path.
  3. Consider lighter-weight or staged evaluation for active-test/debug modes.

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

## Design Discussions

> Open architectural questions that need a decision before implementation.
> Add structural trade-offs, API design choices, and "should we?" questions
> here so they don't get lost. Reference the relevant pitfall or module.

### D1 – `canonicalize_children()` sort-key trade-offs (→ P14)
- **Where:** `Node.canonicalize_children()` in `trees/_nodes.py`
- **Questions:**
  1. **Performance cost**: `represent_str()` is recursive per child. Benchmark
     vs. subtree size (`len(child)`) as primary key, string as tiebreaker.
  2. **Sort key quality**: SymPy's `default_sort_key` produces more canonical
     forms but is significantly more expensive. Is a middle-ground worth it?
  3. **Re-sorting timing**: Currently one-shot in `tree_to_candidate()`. If
     canonical form is ever expected *between* mutation steps, propagation
     becomes a problem.
- **Also tracked as:** M1 (benchmark).

### D2 – Bytecode complexity backends and extensions (→ P16)
- **Where:** `plagih/tree_complexity/python_bytecode_complexity.py`
- **Questions:**
  1. Parallel critical-path complexity — how to define / measure?
  2. Branch-sensitive complexity for `Ifte` / `Piecewise` — weight both
     branches equally or only the taken branch?
  3. Optional Numba / LLVM / ASM backends — worth the dependency?

### D3 – `Number` display: cleaner representation for fixed children
- **Where:** `Number.represent_str()` in `trees/_nodes.py` (line ~1263)
- **Question:** `s += ":fix"` suffix for fixed nodes feels ad-hoc.
  Is there a more natural way to visually distinguish fixed vs. mutable
  terminals without string manipulation?

### D4 – Allow rational inputs for `Number` terminals
- **Where:** `NodeSelect.get_terminals()` in `trees/_evolution.py` (line ~313)
- **Question:** Currently `sympy.Float(_v, precision)` is used. Should
  rational literals like `1/3`, `3/4` be supported as terminal values?
  Pro: more compact expressions. Contra: larger search space, interaction
  with rounding (P17).

### D5 – Targeted Evolutionary Optimization (→ `docs/TARGETED_OPTIMIZATION.md`)
- **Scope:** Per-tree pseudo-backpropagation, node-level optimization gaps,
  SoftOptimum population bound, chained-operator mutation, merged-tree
  trunk analysis.
- **Status:** Phase 1 (analysis infrastructure) + Phase 2 (Ifte/Piecewise
  scoring) implemented in `plagih/targeted_optimization.py` (17 tests pass).
- **Primary focus:** Ifte/Piecewise pseudo-backpropagation (§3.1).
- **Next:** Integration into `run_generation()` as optional strategy (Phase 2b).

### D6 – Idempotent simplification pipeline (→ P19)
- **Where:** `tree_simplification()` → `sympy_to_tree()` →
  `tree_node_grouping()` in `trees/_nodes.py`
- **Problem:** The round-trip `tree → sympy → tree → grouping` is not
  idempotent.  SymPy's canonical form disagrees with the grouped form,
  causing the tree to **grow** during simplification.  In some cases SymPy
  also numerically evaluates constant sub-expressions (e.g.
  `sin(1)**2 → 0.708`), changing the expression semantically.
- **Implemented mitigations:**
  1. **Size guard** — if simplified tree is larger than original, return
     original.
  2. **Min-size tracking** — grouping loop tracks the smallest tree across
     all iterations.
  3. **Oscillation detection** — if a string representation repeats, the
     loop exits early (prevents infinite cycling A→B→A).
  4. **`CuriosityError` debug bomb removed** — replaced with structured
     `log("w", …)` warning on non-convergence.
- **Remaining questions:**
  1. Should `tree_node_grouping` produce a form that round-trips cleanly
     through SymPy?  Or should we stop converting back to SymPy after
     grouping?
  2. Should constant-folding by SymPy be accepted (smaller tree, but
     different expression) or suppressed (keep `sin(1)` unevaluated)?
  3. Could a "grouping-only" simplification mode (skip SymPy round-trip)
     be useful for cases where SymPy expansion is counterproductive?
- **Status:** Core mitigations implemented, pipeline is safe but not
  fully idempotent.

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

### L5 – Crossover time grows across generations
- **Where:** `run_generation()` / crossover strategy
- **Observed:** In test run, crossover avg goes from `10ms → 29ms` over
  20 generations (population=50). Likely caused by trees growing in
  complexity across generations. Investigate whether tree-size limits or
  early-rejection can keep crossover time stable.

### L6 – Generation count exceeds `gen_end`
- **Where:** `run_generation()` loop in `_gp_engine.py`
- **Observed:** Log shows "generation 21/20", "22/20", "23/20", "24/20" —
  the loop continues past `gen_end=20`. Verify the termination condition
  and off-by-one in the generation counter.

### L7 – Population shrinks below `pop_max_size`
- **Where:** Various strategy functions in `parallel.py`
- **Observed:** `genepool=44` to `genepool=48` when `pop_max=50`. Some
  candidates are rejected (fail counts), but the population is not
  back-filled. Consider retry logic or fallback random trees to maintain
  target population size.

---

## Completed

- ✅ Shared memory for `df_train` — 1.68× faster worker startup
- ✅ Pre-selection replaces legacy full-population IPC — 4.8× less IPC cost
- ✅ `get_sympy_expr()` removed from monitoring hot path (P10)
- ✅ Relational-on-Piecewise guard in `get_sympy_expr()` (P12)
- ✅ Physical core detection via `cpu_count_physical()` (P11)
- ✅ Chunked batching with progress diagnostics in parallel (P12)
- ✅ **M5** — Legacy `printpl`/`printez`/`print_warning`/`print_caution` migrated to `log()`/`log_error()`
- ✅ **P18** — `MemoryError` guard in `get_sympy_expr()` for `sympy.exp()` on huge arguments





