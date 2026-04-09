# Implementation Plan

> Central collection of open tasks, design discussions, and future ideas.
> Add new items here instead of scattering TODOs through source code.
> Reference relevant files/pitfalls with each item.

---

## High Priority

### ~~H1 – Parallelize `gen_create_initial()`~~ ✅
- **Where:** `ExplainableGP.gen_create_initial()` in `trees.py`
- **Historical problem:** Initial population creation used to be always
  sequential (P8). For `pop=10000` this cost ~44–47s — a fixed sequential
  block.
- **Approach:** Reuse existing `run_generation_parallel()` infrastructure.
- **Implemented groundwork (2026-03-23):** Generation 0 now uses a generic
  weighted population-fill mechanism instead of hard-wired half/half rate
  decorators.  This is closer to common GP practice (generic fill loop +
  init-specific samplers) and preserves exact target size even when an
  `origin_tree` already occupies one slot.
- **Additional groundwork (2026-03-23):** Generation 0 now also has an
  explicit **declarative strategy plan** with exact per-sampler counts
  (`Strategy(count=...)`) and normal-depth sampling parameters.  This makes the
  init path structurally closer to `build_task_list()` / task execution, even
  though execution is still local to `gen_create_initial()`.
- **Completed (2026-03-24):** Generation 0 now builds `TaskSpec`s via
  `build_task_list()` and executes through the same sequential/parallel task
  runner as regular generations, including shared progress reporting and
  per-tree timing capture.
- **Caveat:** Parallel generation-0 runs currently backfill only cheap
  **exact-tree LUT metadata** (`tree_id -> parsimony/fitness`) into the main
  process. Worker-local SymPy-expression LUTs are still not synchronized back.
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
  1. ~~Add a cheap pre-filter before full evaluation for obviously invalid trees.~~ ✅
     Already covered by existing 7-stage pre-filter chain:
     `force_input_node` → `evolve_prune_tree` → `canonicalize_and_get_lut_id`
     → tree-LUT check → parsimony guard → `get_sympy_expr` (catches imaginary,
     zoo/oo/nan, MemoryError, RecursionError, relational-on-Piecewise) →
     symex-LUT check.  The only remaining failures at NumPy level are
     **data-dependent** NaN/Inf (e.g. `log(x)` with `x ≤ 0` in training data),
     which cannot be detected without actual evaluation.
  2. ~~Investigate `canonicalize_children()` and LUT interactions in the hot path.~~ ✅
  3. Consider lighter-weight or staged evaluation for active-test/debug modes.
- **Implemented optimizations (2026-04-02):**
  1. **Fused `canonicalize_and_get_lut_id()`** — eliminates redundant
     `represent_str()` traversal.  The old path called `canonicalize_children()`
     (recursive `represent_str()` for sorting) and then `get_lut_id()` (full
     `represent_str()` again).  The new `_canonicalize_and_repr()` method
     does both in a single bottom-up pass.
     **Benchmark (30-node trees, 500 samples):** 0.107 ms/tree vs 0.247 ms/tree
     → **2.3× speedup** on the canonicalize+LUT-id step.
  2. **Cached `true_values`** — `df_train[target_column].to_numpy()` was called
     for every LUT-miss evaluation.  Now pre-computed once:
     - In `ExplainableGP.__init__()` as `self._true_values`
     - In worker globals as `_worker_true_values` (set in `_init_worker()`)
     - In `run_task_sequential()` before the evaluation loop
  3. **Fixed NaN check** — `np_fitness == np.nan` was **always False** (NaN ≠ NaN).
     The old code in `_gp_engine.py` relied on `"nan" in str(np_fitness)` as
     fallback (wasteful string conversion).  Replaced with `np.isfinite()` in
     both `tree_to_candidate()` and `evaluate_tree_standalone()`.
- **Remaining ideas:** Staged evaluation for active-test/debug modes (idea 3).

### ~~H2 – Optimize pre-selection overhead~~ ✅
- **Where:** `pre_select_for_tasks()` in `parallel.py`
- **Problem:** Pre-selection in the main process is currently the main IPC
  cost contributor (~198ms/gen at pop=1000). See `PARALLEL_BENCHMARK_DIAGNOSIS.md` §4.
- **Implemented optimizations (2026-04-09):**
  1. **Batch tournament selection with NumPy** — Replaced the per-task Python
     loop (900× `random.choices()` + `min()` with lambda + `selection_tournament()`)
     with a vectorized batch approach: `_batch_tournament_select()` pre-computes
     a fitness array once, generates all tournament indices via
     `np.random.randint(shape=(n, k))`, and finds winners via
     `np.argmin(axis=1)` in a single pass. Tasks are grouped by `tournament_n`
     for efficient batching.
  2. **`pareto_revive` uses `fast_tree_copy`** — Pre-selection of Pareto
     candidates now uses `fast_tree_copy()` (~4.6× faster) instead of
     `copy.deepcopy()`.
  3. **Eliminated `selection_tournament` import** — `pre_select_for_tasks()`
     no longer calls the per-item `selection_tournament()` function from
     `_evolution.py`, avoiding the per-call Python function overhead and
     redundant `random.choices` → `min` → `fast_tree_copy` chain.
- **Index-deferral analysis (2026-04-09):** Considered sending only winner
  indices to workers instead of pre-copied trees, deferring `fast_tree_copy`
  to worker processes. **Rejected:** This would require re-introducing
  `_update_worker_state` IPC to send `pop_genepool` to all workers (~950ms/gen
  at pop=1000 historically — the original IPC bottleneck). The current
  `fast_tree_copy` cost (~40–100ms for 900 trees) is **10× cheaper** than the
  IPC cost it would trade for. Net result would be a significant regression.
- **Status:** Complete. No further optimization of pre-selection is cost-effective
  without a fundamentally different IPC architecture (e.g. shared-memory tree pool).

### H3 – Reduce worker RAM overhead
- **Where:** `parallel.py` worker pool
- **Problem:** 8 workers consume ~1.1–1.2 GB child RSS (mostly fixed
  interpreter/import overhead). See diagnosis report §5.

---

## Medium Priority

### D8 – Feature Demo Notebook (`docs/demo.ipynb`)
- **Where:** `docs/demo.ipynb`, `plagih/demo_helpers.py`
- **Purpose:** Showcase every major feature with hand-crafted, deterministic examples.
  Doubles as a visual sanity-check after refactors — run the notebook, inspect inline
  tree renders, verify nothing looks broken.
- **Layout:**
  - `docs/demo.ipynb` — Jupyter notebook (7 parts, stripped outputs via `nbstripout`)
  - `plagih/demo_helpers.py` — pre-built example trees, DataFrames, display utilities
    (`show_tree`, `show_trees`, `show_tree_with_scores`)
- **Sections:** (1) Building Blocks, (2) Genetic Operations (crossover, mutation),
  (3) Simplification & SymPy Bridge, (4) Evaluation & Complexity,
  (5) Population & Pareto, (6) Monitoring, (7) Targeted Optimization / `targeted_ifte`
- **Key design:** All demo examples are *hand-crafted* (no full GP runs). Complex
  concepts (e.g. merge-tree, Pareto front) use prepared `Candidate` objects so the
  output is reproducible and editable.
- **Renderer extension:** `_render_tree_on_axes` gains an optional `node_scores`
  parameter (`dict[id(node) → float 0–1]`) for score-coloured tree renders in Part 7.
- **Status:** Initial implementation added 2026-04-02.  Smoke test suite
  (`test_demo_helpers.py`, 32 tests) added 2026-04-02 covering all tree
  factories, DataFrame factories, Evolution factories, crossover helper,
  display utilities (headless Agg), and `make_ifte_node_scores`.

### M1 – `canonicalize_children()` performance benchmark
- **Where:** `Node.canonicalize_children()` in `trees/_nodes.py` (P14)
- **Current:** Uses `represent_str()` as sort key — recursive string generation.
- **Alternative:** Sort by subtree size (`len(child)`) first, string as tiebreaker.
- **Benchmark script:** `plagih/test/benchmarks/bench_canonicalize.py`
- **Benchmark results (2026-04-02):**

  | Variant | Small (10 nodes) | Medium (30 nodes) | Large (96 nodes) |
  |---|---:|---:|---:|
  | **(A) `represent_str`** (status quo) | 6.2 ms / 50 trees | 27.7 ms / 50 trees | 131 ms / 30 trees |
  | **(B) `len` + str tiebreaker** | 6.0 ms / 50 trees | 54.1 ms / 50 trees | 151 ms / 30 trees |
  | **(C) `len` only** | 0.8 ms / 50 trees | 6.2 ms / 50 trees | 17.2 ms / 30 trees |

- **Key findings:**
  1. **(B) is slower than (A)** on medium/large trees (~2× on medium).
     The composite `(len, str)` tuple key adds overhead without benefit because
     `len()` is itself O(n) recursive and the tiebreaker still requires full
     string generation.
  2. **(C) is 7–8× faster** than (A), but produces different canonical forms for
     **most** trees (24/50 to 29/30 differ). This would invalidate existing
     LUT keys and potentially reduce LUT hit rates after the switch.
  3. The cost of `canonicalize_children()` scales super-linearly with tree size
     (from ~123 µs/tree at 10 nodes to ~4.4 ms/tree at 96 nodes).
- **Conclusion:** Switching to (B) is **not recommended**. Switching to (C)
  gives a large speedup but changes canonical forms — acceptable only if LUT
  keys are regenerated. The current (A) approach remains the best balance.
  A potential optimization: cache `represent_str()` on nodes to avoid
  redundant recursive string generation during sorts.
- **Action:** → see D1 for further design discussion.

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
  scoring) + Phase 2b (strategy integration) implemented.
- **Primary focus:** Ifte/Piecewise pseudo-backpropagation (§3.1).
- **Phase 2b completed (2026-03-26):** New `targeted_ifte` strategy registered
  in `BUILTIN_STRATEGIES` in `parallel.py`. The strategy selects trees with
  Ifte/Piecewise nodes, uses `ifte_component_scores()` to identify the weakest
  component, and applies focused mutation only to that subtree. Falls back to
  standard branch mutation if no Ifte nodes are found or df_train/target are
  unavailable.  Runtime context (`_df_train`, `_target`) is injected
  automatically by `run_task_sequential()`.
- **Next:** Phase 3 — General node-level optimization (§3.2) with invertible
  operators.

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
  5. **Semantic guard** — if the simplified tree is not SymPy-equivalent to
     the original expression, reject it and keep the original tree.
  6. **Compact structure diagnostics** — rejection logs now include
     `original/roundtrip/grouped` both as expression and as compact structural
     dump via `str_as_list()`, so single-tree failures can be inspected
     without huge multiline output.
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

### D7 – Root-cause analysis for rejected simplifications
- **Where:** `tree_simplification()` / `tree_node_grouping()` in
  `trees/_nodes.py`
- **Problem:** The simplification pipeline is now **safe** (bad rewrites are
  rejected), but real rejected cases still occur in practice and can be slow.
  The current diagnostics show that failures often cluster around
  `Scale`/`DivFraction`/`PowRounded` rewrites, SymPy float canonicalization,
  and grouped forms that are structurally different despite apparently similar
  end expressions.
- **Questions:**
  1. Which rewrite families are responsible for most rejections in long runs?
  2. Are the issues introduced mainly by `sympy_to_tree()` or by
     `tree_node_grouping()` post-processing?
  3. Should some grouping rules be made conditional (e.g. avoid rewrites when
     they are only cosmetic but expand the tree or destabilize round-tripping)?
- **Next step:** Build a small frequency analysis of rejected simplification
  patterns from the new compact diagnostics before changing grouping rules.

### D9 – RuntimeWarning suppression and data-dependent NaN quantification (→ H4)
- **Where:** `ExplainableGP.tree_to_candidate()` in `trees/_gp_engine.py`
  (line ~1238), `evaluate_tree_standalone()` in `parallel.py`
- **Context:** The H4 pre-filter analysis showed that the only trees escaping
  the 7-stage pre-filter chain are those with **data-dependent** NaN/Inf
  (e.g. `log(x)` when `x ≤ 0` in training data). These are caught by the
  `np.isfinite()` check after NumPy evaluation.
- **Questions:**
  1. **Quantification:** How many trees actually fail at the `np.isfinite()`
     check? The existing `_generation_tree_timings` (with `failed_stage =
     "evaluate"`) could answer this. If it's a significant fraction, a
     lightweight domain check (e.g. Log/Sqrt/Div with known-negative inputs)
     might be worthwhile. If it's rare → no further action needed.
  2. **RuntimeWarning blanket suppression:** `warnings.simplefilter("ignore",
     RuntimeWarning)` (`# sfeh:discuss`) hides all RuntimeWarnings during
     NumPy eval (division-by-zero, log-of-negative, etc.). This is "by design"
     (followed by `isfinite()` guard), but could mask real bugs in custom
     `eval_error_metric` implementations. Should the suppression be narrowed
     to specific warning categories, or is the blanket approach acceptable?

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

### ~~L3 – `analyze_pareto` duplicate cleanup~~ ✅
- **Where:** `paretofront.py`
- **Problem:** Two identical `analyze_pareto` definitions existed (second
  shadowed the first), plus ~160 lines of commented-out benchmark-specific
  legacy code.
- **Fix:** Consolidated into a single clean stub with proper docstring.
- **Status:** Complete.

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
- **Root cause (analysed 2026-04-02):** All crossover operations are O(n) in
  tree size.  `selection_tournament()` calls `copy.deepcopy()` (2× per
  crossover), `evolve_crossover()` calls `list_mutable_nodes()` (up to 3×) and
  `copy.deepcopy(subtree)` (1×), and `evolve_prune_tree()` is called 2×.
  Because GP trees grow in size across generations (selection pressure favours
  lower fitness, not smaller trees), all these O(n) operations become
  proportionally more expensive.  The pruning in `evolve_prune_tree()` already
  caps trees at `nodes_max`, but trees at the cap are still larger than initial
  trees (typically depth 3–5).
- **Possible mitigations (not yet implemented):**
  1. Size-aware tournament selection for crossover (prefer smaller parents).
  2. Crossover rejection when both parents are at `nodes_max` (product tree
     would be pruned anyway).
  3. ~~Lighter-weight deepcopy (e.g. copy-on-write or structural sharing).~~ ✅
     **Implemented (2026-04-09):** `fast_tree_copy()` rollout replaced all
     `copy.deepcopy()` on Node trees with ~4.6× faster structural copy.
- **Impact:** Low.  At `pop=50` and `crossover_rate=0.2`, only ~10 crossovers
  per generation.  The 19ms increase is ~190ms total — dwarfed by evaluation.

### ~~L6 – Generation count exceeds `gen_end`~~ ✅
- **Where:** `_test_simple()` in `plagih_gp.py` (not `_gp_engine.py`)
- **Cause found:** `gen_end` was fixed to `20`, but `_test_simple()` executed
  a hard-coded plan of **24** generations (`1 + 1 + 2 + 10 + 10`).
  The misleading logs like `21/20` were therefore caused by a demo/test-plan
  mismatch, not by the core generation counter.
- **Fix:** `_test_simple()` now derives `gen_end` from the declared strategy
  plan, so the log output ends cleanly at `24/24`.
- **Status:** Complete.

### ~~L7 – Population shrinks below `pop_max_size`~~ ✅
- **Where:** `run_generation()` in `trees/_gp_engine.py`
- **Observed:** `genepool=44` to `genepool=48` when `pop_max=50`. Some
  candidates are rejected (fail counts), but the population is not
  back-filled.
- **Implemented (2026-03-26):** After main generation execution, if failures
  reduced the population below the expected task count, remaining slots are
  back-filled with `random_new` trees. Only triggers on failure-induced
  shortfall (not when strategy rates intentionally produce fewer candidates).
- **Status:** Complete (526 tests pass).

### ~~L8 – Logging handlers can outlive their stdout/stderr streams~~ ✅
- **Where:** `logging_utils.py` / benchmark harnesses / tests
- **Observed:** During repeated pytest benchmark runs, the global logger can
  keep a console handler whose underlying stream has already been closed,
  leading to `ValueError: I/O operation on closed file` on later `log()` calls.
- **Fix:** `_handler_has_closed_stream()`, `_prune_closed_handlers()`, and
  `_ensure_live_console_handler()` were added to `logging_utils.py`.  Every
  `log()` / `log_*()` call now auto-prunes stale handlers and reattaches a
  fresh console handler when needed.  `setup_logging()` also removes all
  existing handlers before adding new ones.
- **Status:** Complete.

---

## Ideas Backlog

> Ideas migrated from the old README and other sources. Not yet prioritised
> or scoped — promote to L*/M*/H* when ready to act on them.

### I1 – NaN-escape operator
- **Idea:** New node type that returns a default value when evaluation produces
  NaN or a complex number. Two inputs: `(expression, fallback)`.
- **Distinction:** Separate between NaN from SymPy (imaginary/zoo) and NaN from
  NumPy evaluation (data-dependent). The former is already caught by
  `get_sympy_expr()`, the latter is a runtime concern.
- **Related:** D9 (RuntimeWarning suppression), P20 (NaN check fix).

### I2 – Best-overlapping candidates / Partnering
- **Idea:** Identify candidates that perform well in regions where others fail.
  Use per-datapoint residuals to find complementary trees for `Piecewise`
  composition or targeted crossover.
- **Approach:** For each datapoint, track which candidate has the smallest
  residual. Candidates with good coverage of "uncovered" regions are promoted
  as partners. This could inform an entropy-like metric for merge selection.
- **Related:** D5 (targeted optimisation), M3 (merge strategies).

### I3 – Population mining / adaptive strategy
- **Idea:** Analyse population characteristics and adjust strategies dynamically:
  - Trees too large → reduce `nodes_max` or prefer smaller parents
  - Population too homogeneous → increase `random_new` rate
  - Stagnation → ban dominant sub-structures or change operator pool
  - If constant-filter improved a tree → retry with slightly different adaptation
- **Related:** L5 (crossover time), D5 (targeted optimisation).

### I4 – GPU-accelerated evaluation
- **Idea:** Evaluate trees on GPU for large populations / large datasets.
  Requires batched evaluation (multiple trees simultaneously).
- **Options:** TensorFlow (graph mode), PyTorch, JAX, or CuPy.
- **Challenge:** Current per-tree NumPy evaluation produces 1D arrays;
  batched GPU eval would need 2D (trees × datapoints), requiring a different
  dimensionality model.
- **Related:** L2 (gradient tracking / JAX integration).

### I5 – Sub-populations / races
- **Idea:** Evolve separate sub-populations that occasionally exchange
  individuals. Variant: mine common trunk structure from good candidates and
  seed a new sub-population with just that trunk (or explicitly without it).
- **Related:** I2 (partnering), M3 (merge strategies).

### I6 – `nsimplify` for terminals and expressions
- **Idea:** Use `sympy.nsimplify(expr, tolerance=..., rational=True)` to
  clean up evolved constants. E.g. `3.333*x → (10/3)*x`, or
  `x**1.999 → x**2`. Especially useful for power exponents.
- **Related:** D4 (rational Number terminals), D6 (simplification pipeline).

### I7 – Terminal-only mutation
- **Idea:** Mutation variant that only changes terminal values (Numbers,
  Symbols) without altering tree structure. Preserves proven operator
  structure while fine-tuning constants.
- **Related:** D5 Phase 3 (node-level optimisation).

### I8 – Special constants (π, e) as terminals
- **Idea:** Allow `sympy.pi`, `sympy.E` etc. as terminal values in the
  operator pool / terminal set.

### I9 – Adaptive tournament size
- **Idea:** Adjust `tournament_size` based on population fitness distribution
  (skewness). High skew → smaller tournaments (more exploration). Low skew →
  larger tournaments (more exploitation).
- **Related:** I3 (population mining).

### I10 – GP/NN co-evolution (EM-style)
- **Idea:** Iterative process: (1) Train a GP to approximate an NN.
  (2) Use Pareto-optimal GP trees as features for a smaller NN.
  (3) GP focuses on the residual. Repeat.
- **Variant:** NN identifies regions where GP fails → GP evolves `Ifte`
  structures for those regions.
- **Related:** D5 (targeted optimisation), I2 (partnering).

### I11 – Merged-tree visualisation improvements
- **Idea:** Variant of merge tree without terminal nodes (structure only).
  Use existing evaluation combination for colour coding.
- **Related:** `population_merge.py`, `visualization/`.

### I12 – Tree "styles" / representation modes
- **Idea:** One expression can be rendered in different styles:
  raw (as generated), isolated inputs, factorised, simplified, "better
  mutable" (optimised for further evolution). Currently only raw + simplified
  exist.

### I13 – DivFraction factor limits
- **Observed:** Grouping rewrites `Mul → DivFraction` with large factors
  (e.g. `361/x²`), where the original intent was small factors (`a/2`).
  Consider limiting DivFraction rewrite to small denominators.
- **Related:** D6 (idempotent simplification), D7 (rejected simplifications).

### I14 – Showcase markdown document
- **Idea:** A second demo format (`.md` file) focused on **rendering and
  display** capabilities rather than code usage. A visual "gallery" of tree
  renders, Pareto plots, merge trees, etc.
- **Related:** D8 (demo notebook).

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
- ✅ **L6** — `_test_simple()` now maps its fixed strategy plan to the correct `gen_end`, eliminating misleading `21/20`-style logs
- ✅ **D5 Phase 2b** — `targeted_ifte` strategy integrated into `parallel.py` with automatic df_train/target injection
- ✅ **L7** — Population back-fill after failure-induced shortfall in `run_generation()`
- ✅ **CuriosityError cleanup** — All remaining `raise CuriosityError` in production code replaced with proper exceptions: `SympyError` (RoundDummy), `ValueError` (export_tree), log warnings (revoke_useless_nodes), or handled gracefully (tree_node_grouping Mul×1)
- ✅ **Node.__eq__/__hash__** — Identity-based equality prevents infinite recursion from circular parent_node refs in dataclass-generated `__eq__`
- ✅ **set_new_node deepcopy fix** — Back-references (parent_node, root_node, depth) are now saved before deepcopy and passed to `repair_all()` correctly
- ✅ **mychlds_remove identity fix** — `tree_node_grouping` Mul-factor removal now uses identity (`is`) instead of value equality, fixing double-removal of equal Number nodes
- ✅ **Object-dtype coercion** — `eval_predict_numpy_now` retries with float64 coercion when numpy ufuncs fail on object-dtype child arrays
- ✅ **L3** — Duplicate `analyze_pareto` consolidated into a single clean stub; ~160 lines of commented-out legacy code removed
- ✅ **L8** — Stale logging handler pruning already implemented (`_ensure_live_console_handler`); marked as complete
- ✅ **H4 (partial)** — Fused `canonicalize_and_get_lut_id()` (2.3× speedup), cached `true_values`, fixed NaN check (`np_fitness == np.nan` → `np.isfinite`)
- ✅ **README cleanup** — Restructured from chaotic ~500-line dump to professional ~200-line document. ~100+ scattered TODOs migrated to Ideas Backlog (I1–I14). Removed: TensorFlow references, Python-3.9/Anaconda setup, LaTeX/tikzplotlib deps, biography, debug dumps, `====Everything below here is garbage====` section
- ✅ **H2** — Batch tournament selection with NumPy (`_batch_tournament_select`), `pareto_revive` → `fast_tree_copy`, eliminated per-task `selection_tournament()`. Index-deferral to workers rejected (IPC cost ~10× higher than `fast_tree_copy` savings)
- ✅ **P21 fix** — Mul grouping early-return bug: `0 < mul1 < 1` branch without `else: continue` skipped DivFraction handler; degenerate `Mul(single_child)` in DivFraction handler
- ✅ **`fast_tree_copy` rollout** — All `copy.deepcopy()` on Node trees replaced with `fast_tree_copy()` (~4.6× faster): `set_new_node`, `tree_simplification` (4×), `evolve_reduce_simplicate`, `evolve_new_tree_depth`, `evolve_mutate_node`, `evolve_crossover`. `import copy` removed from `_nodes.py` and `_evolution.py`





