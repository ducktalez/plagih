# Implementation Plan

> Central collection of open tasks, design discussions, and future ideas.
> Add new items here instead of scattering TODOs through source code.
> Reference relevant files/pitfalls with each item.

---

## High Priority

### H4 – Reduce evaluation hot-path cost during tree production ✅ (diagnosed — no action needed)
- **Where:** `ExplainableGP.tree_to_candidate()` in `trees/_gp_engine.py` and
  `evaluate_tree_standalone()` in `parallel.py`
- **Problem:** New tree-creation benchmark (`bench_tree_creation.py`) indicates
  that both initial population creation and later generation production are
  dominated by **evaluation**, not raw tree generation.
- **Current evidence:** `initial_population` and `active_generation` both show
  evaluation as the dominant phase for most successful trees, with evaluation
  failures significantly outnumbering creation/simplification failures.
- **Analysis (2026-04-13):** Evaluation **is** the core work of GP —
  trees exist to be evaluated. The current pipeline already minimises wasted
  evaluation cycles through:
  1. **tree_id LUT** — duplicate trees return cached results instantly.
  2. **symex→fitness LUT** — structurally different trees with identical
     SymPy expressions share fitness.
  3. **Early rejection** — `TreeSizeError` and `SympyError` abort before
     the expensive NumPy pass.
  4. **Vectorised NumPy evaluation** — already the fastest Python-level
     approach (µs–low ms per tree on typical training data).
  A "staged evaluation" or lightweight debug mode would add complexity
  for marginal runtime savings. The only remaining cost is the irreducible
  per-tree NumPy evaluation, which cannot be shortened without changing
  the fitness semantics.
- **Verdict:** No action needed. The evaluation hot-path is already well
  optimised. Revisit only if population sizes grow to 5000+ or training
  datasets exceed 10k rows, where batched/GPU evaluation (→ I4) could help.

### H3 – Reduce worker RAM overhead
- **Where:** `parallel.py` worker pool
- **Problem:** 8 workers consume ~1.1–1.2 GB child RSS (mostly fixed
  interpreter/import overhead). See diagnosis report §5.
- **Profiling results (2026-04-13):**
  Benchmark: `bench_h3_worker_ram.py` + clean verification run.

  **Important:** The original diagnosis report §5 measured ~1.1–1.2 GB at
  8 workers. The H3 benchmark initially showed 298 MB/worker — but this was
  an **artifact** of the benchmark importing TensorFlow (+510 MB) in the main
  process before forking workers. TF is **not** imported during normal GP runs.

  **Corrected measurements (no TF, fork mode):**

  | Metric | Value |
  |---|---|
  | Main process RSS | 196.7 MB |
  | Worker RSS (each) | **147.0 MB** |
  | Worker modules loaded | 2063 |
  | 4-worker total RSS | 588.1 MB |
  | 8-worker total (estimated) | ~1176 MB |

  Import cost breakdown (per-process, no TF):

  | Import group | Delta RSS (MB) |
  |---|---:|
  | numpy + pandas | +56.1 |
  | plagih.trees | +30.7 |
  | sympy | +30.2 |
  | matplotlib | +0.5 |

- **Analysis:** At 147 MB/worker, the overhead is dominated by fixed
  interpreter + numpy/sympy/pandas imports (~117 MB above baseline 12 MB).
  With `fork` on Linux, workers share CoW pages with the parent — actual
  unique RSS per worker is likely lower than reported (shared library pages
  are double-counted in per-process RSS).
- **Verdict:** No easy wins remain. The ~147 MB/worker is the irreducible
  cost of numpy + sympy + pandas in a forked Python process. Switching to
  `spawn` would not help (workers would re-import everything). The only
  significant improvement would require a fundamentally different architecture
  (e.g. shared-memory tree pool, or C/Rust worker processes).
- **Status:** Diagnosed. No action needed unless 8+ worker configs on
  memory-constrained machines become a priority.

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
- **Primary focus:** Ifte/Piecewise pseudo-backpropagation (§3.1).
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
- **Remaining questions:**
  1. Should `tree_node_grouping` produce a form that round-trips cleanly
     through SymPy?  Or should we stop converting back to SymPy after
     grouping?
  2. Should constant-folding by SymPy be accepted (smaller tree, but
     different expression) or suppressed (keep `sin(1)` unevaluated)?
  3. Could a "grouping-only" simplification mode (skip SymPy round-trip)
     be useful for cases where SymPy expansion is counterproductive?

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
- **Frequency analysis results (2026-04-13):**
  Script: `scripts/analyze_simplification_rejections.py`, parsed 2 log files
  with 92 logged rejections (302 including suppressed similar cases).

  **Key finding: 100% of rejections are "changed semantics", 0% are "grew".**
  The size guard works perfectly — it's the semantic equivalence check that
  rejects simplifications.

  | Suspected stage | Count | % |
  |---|---:|---:|
  | sympy_roundtrip+grouping | 117 | 60% |
  | unknown | 48 | 25% |
  | sympy_roundtrip | 30 | 15% |

  **Dominant operators in rejected expressions:**

  | Operator | Occurrences | Notes |
  |---|---:|---|
  | Max | 663 | SymPy → `Piecewise` → semantic drift |
  | Min | 526 | SymPy → `Piecewise` → semantic drift |
  | sin | 389 | Often combined with Min/Max |
  | sign | 333 | SymPy rewrites as `Piecewise` internally |
  | Abs | 306 | SymPy rewrites as `Piecewise` internally |
  | log | 75 | Domain issues interact with simplification |
  | sqrt | 32 | Minor contributor |

  **Top co-occurrence pairs:** Max+sin (218), Max+Min (209), Min+sin (199),
  Abs+Max (185), Max+sign (177).

  **Root cause:** The overwhelming majority of semantic rejections involve
  `Min`/`Max`/`sign`/`Abs` — operators that SymPy internally represents as
  `Piecewise`. The round-trip `tree → sympy → tree → grouping` introduces
  semantic drift because:
  1. SymPy converts `Min(a,b)` → `Piecewise((a, a<b), (b, True))` internally
  2. `sympy_to_tree()` may reconstruct this differently
  3. The string comparison of the SymPy expressions then differs, and
     `_sympy_exprs_equivalent()` either times out or returns False on
     Piecewise-heavy expressions (known issue, P12)

- **Answers to original questions:**
  1. **Which rewrite families?** Min/Max/sign/Abs account for ~80%+ of all
     rejected simplifications. DivFraction/Scale/PowRounded are NOT
     significant contributors (contrary to earlier assumption).
  2. **sympy_to_tree() or tree_node_grouping()?** Both: 60% involve both
     stages, 15% only sympy_roundtrip. The grouping step compounds drift
     from the roundtrip but is rarely the sole cause.
  3. **Conditional grouping rules?** Not the priority. The real fix is
     improving how Min/Max/sign/Abs survive the SymPy round-trip.

- **Recommended next steps:**
  1. **Short-term — Skip simplification for Piecewise-heavy trees:**
     If a tree contains `Min`/`Max`/`sign`/`Abs`, skip the SymPy round-trip
     entirely and only apply `tree_node_grouping()` (the "grouping-only"
     mode from D6 question 3). This avoids the Piecewise semantic drift.
  2. **Medium-term — Piecewise-aware equivalence check:**
     Improve `_sympy_exprs_equivalent()` to handle Piecewise expressions
     better (e.g. numerical sampling comparison as fallback).
  3. **Long-term — Direct tree-to-tree simplification:**
     Bypass SymPy entirely for structural simplifications (D6 question 1).

### D9 – RuntimeWarning suppression and data-dependent NaN quantification (→ H4)
- **Where:** `ExplainableGP.tree_to_candidate()` in `trees/_gp_engine.py`
  (line ~1238), `evaluate_tree_standalone()` in `parallel.py`
- **Context:** The H4 pre-filter analysis showed that the only trees escaping
  the 7-stage pre-filter chain are those with **data-dependent** NaN/Inf
  (e.g. `log(x)` when `x ≤ 0` in training data). These are caught by the
  `np.isfinite()` check after NumPy evaluation.
- **Quantification results (2026-04-13):**
  Benchmark: `bench_d9_evaluate_failures.py`, 10 generations, pop=200,
  MountainCar dataset, sequential mode.

  | Metric | Value |
  |---|---|
  | Total tree attempts | 2314 |
  | Evaluate failures | 124 (5.4%) |
  | Create failures | 1 (0.04%) |
  | Simplify failures | 0 |

  Error breakdown: 116× `TreeError("NaN in results")`, 4× `TreeLutError`
  (cached NaN), 3× `SympyError` (imaginary), 1× `ValueError`.
  **93.5% of all failures are data-dependent NaN** at `np.isfinite()`.

  Domain operators in failed expressions: **Div 96×, Sqrt 46×, Log 22×**.
  Gen 0 has the highest rate (13%), later generations stabilise at 2–8%.

- **Decision on question 1:** The 5.4% rate is **moderate** — not negligible
  but not alarming. A lightweight domain pre-check for `Div`/`Sqrt`/`Log`
  with known-negative training data ranges could eliminate ~70% of evaluate
  failures, but the runtime saving is small (~3–7 ms per avoided evaluation).
  **Verdict: not worth the complexity for now.** The `np.isfinite()` guard is
  fast and catches everything. Revisit only if pop sizes grow to 5000+ where
  the wasted evaluation cycles become noticeable.
- **Open question 2 (RuntimeWarning suppression):** The blanket
  `warnings.simplefilter("ignore", RuntimeWarning)` is acceptable given that
  the `isfinite()` guard catches all NaN/Inf results. Narrowing to specific
  categories (e.g. `divide`, `invalid`) would add complexity without benefit
  since no custom `eval_error_metric` bugs were observed.

### D10 – NaN-escape operator redesign (→ I1, D9)
- **Where:** `trees/_nodes.py` (node class), `_gp_engine.py` (evaluation),
  `trees/_evolution.py` (operator pool)
- **Motivation:** Formulas that are structurally good and simple but produce
  NaN/Inf on a few edge-case datapoints (e.g. `Log(x)` when some `x ≤ 0`,
  `Sqrt(x)` when `x < 0`, `Div(a, b)` when `b ≈ 0`) are currently either
  rejected or penalised. An explicit NaN-escape mechanism could rescue these
  formulas and make the NaN-handling **visible** in the tree structure
  (important for explainability).

- **Previous attempt (reverted):** An `IfNan(expr, fallback)` node was
  implemented as `IfNanDummy(sympy.Function)` + `IfNan(MathOperator)`.
  Problem: SymPy cannot simplify through an opaque `IfNanDummy` wrapper,
  so formulas wrapped in `IfNan` became dead-ends for simplification.
  Additionally, the implementation only checked `np.isinf` — imaginary
  results and actual `NaN` were not covered.

- **Current workaround:** `tree_to_candidate()` in `_gp_engine.py` uses
  NaN-tolerant scoring: trees with ≤50% non-finite results get a penalty
  value per bad datapoint instead of outright rejection. This is simple and
  effective, but the NaN-handling is invisible in the tree — bad for
  explainability and not optimisable.

- **Open design questions:**

  1. **Explicit node vs. implicit evaluation-time escaping:**
     - **Option A — Dedicated `IfNan` Node (tree-visible):**
       `IfNan(expr, fallback)` as a proper `MathOperator` subclass.
       *Pro:* NaN-handling is visible, evolvable, and explainable.
       *Con:* SymPy interaction is hard (see question 4).
     - **Option B — Evaluation-time auto-escape (current approach):**
       Replace non-finite results with penalty values during NumPy eval.
       *Pro:* Zero impact on SymPy, no new node type, simple.
       *Con:* Invisible, not optimisable, coarse penalty.
     - **Option C — Hybrid: Node for structure, transparent to SymPy:**
       `IfNan` node exists in the tree but `get_sympy_expr()` returns
       only `expr` (unwraps itself). SymPy sees and simplifies the inner
       expression; the NaN-guard re-wraps after round-trip.
       *Pro:* SymPy simplification works. *Con:* Guard can be lost during
       simplification round-trips; needs re-insertion logic.

  2. **Optimisable fallback value (via child node):**
     If implemented as a node, the fallback child could be any subtree
     (`Number`, `Symbol`, or a full expression). This means evolution can
     **optimise what happens in the NaN case** — a unique capability that
     pure evaluation-time escaping cannot provide. Example:
     `IfNan(Log(x), Scale(-0.5, x))` → uses `log(x)` where valid,
     falls back to `-0.5 * x` where `x ≤ 0`.
     *Risk:* Over-complicates if the fallback subtree grows large. Could
     mitigate by constraining fallback to terminals or depth-1 subtrees.

  3. **Coverage: NaN, Inf, and imaginary values:**
     The previous implementation only escaped `np.isinf`. A proper
     implementation must handle:
     - `NaN` (e.g. `0/0`, `sqrt(-x)` in NumPy returns NaN)
     - `±Inf` (e.g. `1/0` in float arithmetic)
     - **Imaginary results:** `asin(2)` returns `NaN` in NumPy but
       `π/2 - i·acosh(2)` in SymPy. Options:
       - **(a)** Use `np.real(result)` or `np.abs(result)` to extract the
         real part / magnitude — only works element-wise, not for subtrees.
       - **(b)** Treat imaginary as NaN (current approach via `isfinite`).
       - **(c)** Dedicated `Re(expr)` operator that extracts the real part.
         Could be useful beyond NaN-escaping but adds complexity.

  4. **SymPy transparency requirement:**
     A NaN-escape operator **must not block SymPy simplification**. If a
     formula intrinsically produces NaN (e.g. `Log(-1)` with no data
     dependency), it should be discarded — NaN-escaping is for
     **data-dependent** edge cases, not structural impossibilities.
     Approaches:
     - Option C above (unwrap for SymPy, re-wrap after).
     - Teach `tree_simplification()` to strip `IfNan` wrappers before
       the SymPy round-trip and re-insert them after `sympy_to_tree()`.
     - Mark `IfNan` subtrees as `is_fix` so simplification skips them
       (coarse, loses optimisation potential).

  5. **Node hierarchy placement:**
     If implemented, `IfNan` **must** be a proper `Node` subclass:
     - `IfNan(MathOperator)` with `xtype = ((float, float), float)`
     - `childs[0]` = expression to evaluate
     - `childs[1]` = fallback value/expression
     - `showme = "IfNan"`, `sy_str = "IfNan({0}, {1})"`
     - Custom `eval_predict_numpy_now` with lazy fallback evaluation
       (only evaluate `childs[1]` where `childs[0]` produced NaN/Inf).
     - For SymPy: either use a custom `sympy.Function` subclass (previous
       approach, problematic) or unwrap transparently (Option C).

  6. **When to apply NaN-escaping:**
     - **Per-element:** `IfNan` wraps individual operator subtrees that are
       known to produce NaN on some inputs. Evolution can place `IfNan`
       wherever needed. This is the composable, fine-grained approach.
     - **Per-tree (root-only):** `IfNan` is only allowed as the root node,
       wrapping the entire formula. Simpler but less expressive.
     - **Automatic wrapping:** After tree creation, automatically wrap
       domain-sensitive operators (`Log`, `Sqrt`, `Div`) in `IfNan` if
       the training data contains values outside their domain.
       *Risk:* Bloats trees, may interfere with crossover/mutation.

- **Recommendation (preliminary):**
  The simplest safe path is **Option B (current evaluation-time escaping)**
  enhanced with better diagnostics (log which datapoints triggered the
  penalty, track NaN-rate as a candidate attribute). This avoids all SymPy
  complications.

  If the optimisable fallback (question 2) proves valuable in practice,
  **Option C (SymPy-transparent node)** is the way forward. Implement in
  stages:
  1. First: `IfNan` node with `get_sympy_expr()` returning only `childs[0]`
     (unwrap). Test that simplification round-trips work.
  2. Then: Re-insertion logic in `tree_simplification()` to restore `IfNan`
     wrappers after SymPy round-trip.
  3. Finally: Evolution support (operator pool, mutation strategies).

- **Decision needed:** Is the optimisable fallback value worth the
  complexity? Or is the current penalty-based approach sufficient?

---

## Low Priority

### L1 – Backend-specific complexity measures
- **Where:** `tree_complexity/`
- **Ideas:** Numba/LLVM IR complexity, ASM instruction count, parallel
  critical-path complexity, branch-sensitive complexity for `Ifte`/`Piecewise`.

### L2 – Gradient tracking placeholder
- **Where:** `evaluation_context.py`
- **Idea:** JAX/PyTorch integration for gradient-based optimization.

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
- **Impact:** Low.  At `pop=50` and `crossover_rate=0.2`, only ~10 crossovers
  per generation.  The 19ms increase is ~190ms total — dwarfed by evaluation.

---

## Ideas Backlog

> Ideas migrated from the old README and other sources. Not yet prioritised
> or scoped — promote to L*/M*/H* when ready to act on them.

### I1 – NaN-escape operator ⏸️ (reverted → D10)
- **History:** An `IfNan(expr, fallback)` operator was implemented (2026-04-13)
  as `IfNanDummy(sympy.Function)` + `IfNan(MathOperator)` node class.
  **Reverted** because the SymPy function approach broke simplification —
  SymPy could not simplify through `IfNanDummy`, making trees un-reducible
  at the NaN-guard boundary.
- **Current status:** Implementation deleted. The evaluation pipeline now uses
  **NaN-tolerant scoring** in `tree_to_candidate()` (`_gp_engine.py`):
  trees with ≤50% non-finite results get a penalty value instead of rejection.
  This keeps simple-but-fragile formulas (e.g. `Log(x)` with some `x ≤ 0`)
  alive with degraded fitness.
- **Motivation preserved:** Formulas that are structurally good and simple but
  produce NaN/Inf on edge-case datapoints should remain viable GP candidates
  instead of being discarded outright.
- **Full redesign discussion:** → D10 in Design Discussions.
- **Related:** D9 (5.4% evaluate failures dominated by Div/Sqrt/Log NaN).

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

### I8 – Special constants (π, e) as terminals
- **Idea:** Allow `sympy.pi`, `sympy.E` etc. as terminal values in the
  operator pool / terminal set.

### I9 – Adaptive tournament size
- **Idea:** Adjust `tournament_size` based on population fitness distribution
  (skewness). High skew → smaller tournaments (more exploration). Low skew →
  larger tournaments (more exploitation).
- **Related:** I3 (population mining).

### I10 – GP/NN co-evolution (EM-style) 🚧 (pipeline done, paper experiments pending)
- **History:** Concept documented (2026-04-xx) as iterative EM loop.
  **Initial implementation delivered (2026-05-15)** in `benchmarks/nn_gp/`.
- **Pipeline modules (all in `benchmarks/nn_gp/`):**
  - `data_utils.py` — MinMax normalisation, GP feature matrix, residual computation
  - `nn_models.py` — PyTorch MLP, `find_minimal_nn` grid search
  - `em_loop.py` — full EM loop runner, `GPConfig`, `NNConfig`
  - `experiment_tracker.py` — crash-safe JSON persistence
  - `paper_figures.py` — auto-generates standard figures (PDF+SVG+PNG)
  - `paper_blueprint.py` — fills `docs/nn_gp_paper_template.md` with measured values
  - `run_mc.py` — MountainCar entry point; `--fast --baseline-only` for dev runs
- **Run:** `python benchmarks/nn_gp/run_mc.py --fast` (dev) or without `--fast` (full)
- **Open tasks (sorted by priority):**
  1. **Run full 3-iteration EM loop** on MountainCar and capture baseline metrics
     (NN-param reduction, residual decay, GP convergence per iteration).
     *Partial — 2026-05-17:* A `--fast --iterations 2 --gp-pop 30 --gp-gen 10`
     run completed end-to-end (~2.5 min, results in
     `.results/nn_gp/20260517-231031/`). Observation: with these tiny settings
     the NN actually **grows** from baseline 129 → 801 params; the GP candidates
     do not yet provide enough signal. A proper run (default `--gp-pop 50
     --gp-gen 20`, full 3 iterations, no `--fast`) is still pending and is the
     real I10.1 deliverable.
  2. ✅ **Wire GP config into the tracker** (2026-05-16) — `ExperimentMeta`
     now carries `gp_config` / `nn_config` dicts; `run_mc.py` populates them
     and `paper_blueprint.py` reads `pop_max_size`, `gen_end`, `epochs` from
     there. Backward-compatible JSON loading (unknown fields tolerated).
  3. ✅ **Smoke test** (2026-05-16) — `plagih/test/test_nn_gp_pipeline.py`
     runs `run_mc.run(baseline_only=True, fast=True)` end-to-end (49 s),
     asserts blueprint exists and contains **zero** unresolved `{{…}}`
     placeholders. Marked `@pytest.mark.performance` (opt-in via `--run-perf`).
  4. ✅ **Categorical-target handling documented** (2026-05-16) — Added an
     explicit *Note* block to `docs/nn_gp_paper_template.md` explaining the
     MSE-on-[0,1] trade-off vs. one-hot + cross-entropy. Implementation of
     the cross-entropy variant remains open.
  5. **Adaptive GP hyperparameters per iteration** (e.g. smaller `nodes_max` for
     residuals which should be simpler). Open question — keep fixed for now to
     ensure reproducibility, revisit after first full runs.
  6. **Second benchmark** (CartPole, SR) to validate generalisation of the
     pipeline once MountainCar produces clean results.
  7. ✅ **Per-iteration GP feature retention** (2026-05-18) — Added
     `NNConfig.feature_retention` flag with two modes: `"replace"` (default,
     current behaviour — only the latest iteration's Pareto candidates feed
     the NN) and `"accumulate"` (keep all past Pareto candidates as NN
     features). Exposed in `run_mc.py` via `--feature-retention`; recorded in
     `ExperimentMeta.nn_config` so the blueprint can report which mode was
     used. The two modes can now be A/B-compared once the first full
     baseline run finishes.
  8. **Cross-entropy variant for categorical targets** (split off from old
     point 4): one-hot encode the target, swap loss to CE, adapt the residual
     definition (probabilities vs. floats). Decide whether GP candidates emit
     logits per class or per-class scalars. Track as a separate experiment.
  9. ✅ **Promote `_render_tree_on_axes` to public API** (2026-05-18) —
     Renamed to `render_tree_on_axes` in
     `plagih/visualization/tree_renderer.py`, exported via
     `plagih.visualization.__init__.__getattr__`, and updated all callers
     (`demo_helpers.py` x3, `benchmarks/nn_gp/paper_figures.py`, the two
     internal `tree_renderer` uses). A `_render_tree_on_axes` alias remains
     for backward compatibility. Also fixed a latent bug in
     `plagih/visualization/__init__.py` whose lazy importer pointed at the
     stale top-level `visualization` package path.
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

### I15 – GUI ideas backlog
- **Status:** First desktop GUI shipped in `plagih/gui/` (PySide6).
  Documented in `docs/GUI.md`.
- **Open follow-ups:**
  1. **Click-to-render** any Pareto entry in the "Best candidate" tab,
     not just the most recently added.
  2. **Diff view** between consecutive Pareto entries (which subtree changed?).
  3. **Merged-population-tree tab** rendered via
     `plagih.population_merge.build_one_evaluation_tree`.
  4. **Subtree drill-down** with targeted-optimization scores from
     `plagih/targeted_optimization.py` (per-node intermediate values,
     Oracle Bound, Ifte / Piecewise component scores).
  5. **Finer-grained pause** *inside* a generation — requires cooperative
     check-points in `parallel.run_generation_*`.
  6. **Optional FastAPI / WebSocket adapter** so the same
     `RunController` can drive a browser view alongside the desktop UI.
  7. **Live log streaming from `plagih.logging_utils.log()`** — currently
     only events explicitly emitted by the controller reach the GUI log
     panel; standard `log()` calls only show in the terminal.

## Recent completions (changelog)

> Promote items here once delivered. Older entries can be trimmed after a
> few months — full history lives in git.

- ✅ **NN+GP first end-to-end run + bugfixes (2026-05-17)** —
  Completed the first full `run_mc.py --fast --iterations 2` run on Windows.
  Three real bugs were discovered and fixed along the way:
  (1) `experiment_tracker.finalize()` / `paper_blueprint.generate_blueprint` /
  `paper_figures.generate_all_figures` crashed on Windows with
  `UnicodeEncodeError` because their progress prints used `→` / `…`;
  replaced with ASCII (`->`, `...`).
  (2) `paper_figures._plot_gp_trees_for_iter` imported `tree_renderer` from
  the stale top-level `visualization.*` path (the module lives in
  `plagih.visualization.*` since the package restructure) and called
  `render_tree(ax=...)` with a kwarg that never existed; now uses
  `plagih.visualization.tree_renderer._render_tree_on_axes`
  (see I10.9 for promoting that to a public API).
  (3) Paper template wording said *"a reduction of X%"* even when the NN
  parameter count actually **grew** between baseline and final iteration —
  rewrote both the abstract and §4.3 to use neutral *"change … by X%"*
  with the `{{param_reduction_direction}}` word.
- ✅ **NN+GP pipeline polish (2026-05-16)** — Resolved I10.2/3/4:
  `ExperimentMeta.gp_config` / `nn_config` carry the frozen hyperparameter
  snapshot so `{{gp_pop_size}}`, `{{gp_gen_end}}`, `{{nn_epochs}}` are filled;
  added smoke test `plagih/test/test_nn_gp_pipeline.py` (49 s,
  `@pytest.mark.performance`) that asserts zero unresolved `{{…}}` placeholders
  in the generated blueprint; documented the MSE-on-[0,1] trade-off for the
  3-class MountainCar target directly in the paper template.
- ✅ **README restored (2026-05-16)** — Previous edit had silently failed and
  only the `placeholder` stub was committed; rewrote with quick-start,
  module overview, NN+GP section, and documentation index.

- ✅ **Repo cleanup (2026-05-16)** — Removed leaked GitHub token from README,
  deleted obsolete migration scripts (`migrate_trees_package.py`,
  `split_trees.py`, `_scan_trees.py`), consolidated 3 redundant setup scripts
  into a single `setup.sh`, removed `SETUP_GUIDE.md`, archived old root-level
  `paper.md` as `docs/legacy_paper_draft.md`, rewrote README without legacy
  TODO dump, added `/logs/` to `.gitignore`.
- ✅ **NN+GP pipeline scaffolding (2026-05-15)** — Initial implementation of
  EM-style co-evolution loop in `benchmarks/nn_gp/` with auto-generated paper
  blueprint. See I10 above for current open tasks.
- ✅ **Output directory unification (2026-05-15)** — Consolidated `.testruns/`
  and `results/` into a single `.results/` dotfolder.
- ✅ **P25 — Gate tree-timing CSVs behind `enable_analysis`** —
  `_analyze_generation_tree_timings()` now only writes CSV when analysis is on.
- ✅ **P12 extension + thread-based timeouts** — `_contains_piecewise_like`
  guard extended to `BaseMinMax`. `SYMPY_SIMPLIFICATION_TIMEOUT_S=5s`,
  `SYMPY_EQUIVALENCE_TIMEOUT_S=2s`.
- ✅ **P24 — Simplification performance** — Validation reduced from 3–4× SymPy
  work to 1×; `get_sympy_expr()` uses LUT cache.
- ✅ **D7 — Grouping-only simplification for Piecewise-heavy trees** — Skips
  SymPy round-trip for trees containing `Min`/`Max`/`Abs`/`Sign`, eliminating
  the entire class of semantic rejections (100% of analysed cases).

## Open next priorities

- 🔜 **D5 Phase 3 (node-level optimisation):** General node-level optimization
  with invertible operators — extend beyond Ifte/Piecewise.
- 🔜 **D8 demo notebook hardening:** Keep notebook/examples in sync with recent
  strategy additions (`mutation_terminal`, `targeted_ifte`) and add one visual
  regression smoke pass.
- 🔜 **I10 next experiments:** Run full EM loop on MountainCar (see I10 open
  tasks above) and produce the first end-to-end paper blueprint with real
  measured values.
