# Implementation Plan

> **Only open work lives here.** Completed items are **deleted** (history is
> in git; durable findings go to `PITFALLS.md` / `ARCHITECTURE.md` /
> `TARGETED_OPTIMIZATION.md`). Keep every entry short: what, where, why.
> The plan must shrink when work is done — see "Maintaining this file" below.

---

## Next up (priorised)

1. **I5 benchmark study** — ≥30 seeds / bigger populations to decide whether
   races beat single-population (current result: within noise,
   `bench_i5_races.py`).
2. **I10.1** — full 3-iteration EM loop on MountainCar
   (`python benchmarks/nn_gp/run_mc.py`, no `--fast`), fill paper blueprint
   with real values.
3. **D8** — sync `docs/demo.ipynb` with new strategies
   (`mutation_terminal`, `targeted_ifte`, `targeted_gap`, `chain_mutation`).

---

## Open tasks

### M2 – xtype system extension
Node-class constraints (e.g. `(Number, BaseOperator)`) beyond `float`/`bool`.
Needs changes in `operatorpool_to_picks` / `choose_operator_class`.
Workaround today: special-case logic in `evolve_create_random()`.

### M3 – More merge strategies
"Expand-random-tree" and "Clustered-tree" (documented in
`population_merge.py` docstring). Depends on TED infra.

### M4 – Grouping rules as class attributes
Move rules out of the central `tree_node_grouping()` — natural migration
path into the D11 rewrite engine (`trees/_rewrite.py`).

### I5 – Races, open ends
Parallel race execution (currently sequential); adaptive epoch length;
larger benchmark (→ Next up 1).

### I10 – GP/NN co-evolution, open ends
Pipeline in `benchmarks/nn_gp/` is done. Open: full run (→ Next up 3),
adaptive GP hyperparameters per iteration, second benchmark (CartPole/SR),
cross-entropy variant for categorical targets.

### D10/I1 – NaN-escape operator (decision pending)
Current: NaN-tolerant scoring in `tree_to_candidate()` (≤50% non-finite →
penalty). Question: is an explicit, evolvable `IfNan(expr, fallback)` node
worth it? Previous `sympy.Function` attempt broke simplification (reverted).
If yes: SymPy-transparent variant (unwrap in `get_sympy_expr()`, re-wrap
after round-trip). Decide only when penalty approach proves insufficient.

### L1 – Backend-specific complexity measures
Numba/LLVM/ASM complexity, branch-sensitive complexity for `Ifte`.

### L2 – Gradient tracking
JAX/PyTorch integration in `evaluation_context.py` (placeholder exists).

### L4 – Background analysis process
Run visualization/backup IO outside the evolution loop (P9).

### L5 – Crossover time grows over generations
O(n) deepcopy/traversal on growing trees. Impact low (~190 ms/gen at
pop=50). Mitigations if needed: size-aware selection, crossover rejection
at `nodes_max`.

---

## Ideas backlog (unscoped)

- **I2 Partnering:** find candidates with complementary per-row residuals
  for Piecewise composition / targeted crossover.
- **I3 Population mining:** detect pathologies (too large/similar/stagnant)
  and adapt strategy rates automatically.
- **I4 GPU evaluation:** batched 2D eval (trees × rows); merged DAG as graph.
- **I6 `nsimplify`:** clean up evolved constants (`3.333x → (10/3)x`).
- **I8 Constants π, e** as terminals.
- **I9 Adaptive tournament size** from fitness-distribution skew.
- **I11 Merged-tree viz:** structure-only variant, colour by evaluation.
- **I12 Tree styles:** raw / factorised / simplified / "better mutable".
- **I14 Showcase markdown:** visual gallery of renders/plots.
- **I15 GUI follow-ups:** click-to-render any Pareto entry; diff view;
  merged-tree tab; subtree drill-down with targeted-opt scores; pause
  inside generation; WebSocket adapter; live `log()` streaming.

---

## Design discussions (open)

### D1 – `canonicalize_children()` sort key (→ P14)
`represent_str()` is recursive per child. Benchmark vs. size-first key?
Re-sorting between mutation steps needed?

### D2 – Bytecode complexity extensions (→ P16)
Parallel critical-path complexity; branch-sensitive `Ifte` weighting;
Numba/LLVM backends worth the dependency?

### D3 – `:fix` suffix display
Better visual marker for fixed terminals than string suffix?

### D4 – Rational `Number` terminals
Allow `1/3`-style literals? Pro: compact. Contra: search space, P17.

### D5 – Targeted optimization, open ends (→ `docs/TARGETED_OPTIMIZATION.md`)
Phases 1–4 done. Open: Phase 5 (GP↔NN, → I10); approximate inverses for
`Square`/`Abs` in gap propagation; normalise `gap_mean` by subtree size?

### D6 – Idempotent simplification (→ P19)
Remaining: should grouping produce a SymPy-round-trip-stable form, or do we
drop the back-conversion entirely (→ D11.3)? Accept SymPy constant folding
(`sin(1) → 0.841`) or suppress?

---

## Decided (one line each — details in git / PITFALLS / docs)

- **H4:** Evaluation hot-path is already optimal (LUTs, early rejection,
  vectorised NumPy). Revisit only at pop ≥ 5000 (→ I4).
- **H3:** ~147 MB/worker is irreducible numpy+sympy+pandas import cost.
- **D7:** Piecewise-heavy trees (Min/Max/Abs/Sign) skip the SymPy round-trip
  (grouping-only mode) — eliminated 100% of semantic rejections.
- **D9:** No domain pre-checks for Div/Sqrt/Log; 5.4% evaluate-failure rate
  is acceptable, `np.isfinite()` guard suffices.
- **D11:** No tighter SymPy coupling, no subclassing. Own rewrite engine
  in `trees/_rewrite.py` (constant folding, neutral elements, fixpoint,
  `is_fix`-safe); grow it, shrink round-trips. Refactor hook: split
  `_nodes.py` into `_grouping.py` / `_sympy_bridge.py`.
- **I5 benchmark:** Races ≈ baseline at small budget (within seed noise).
  Keep opt-in; don't claim improvement without a larger study.
- **P26 pruning strategy:** Random stays default. Benchmark
  (`bench_p26_pruning.py`, 60 trees): deepest-first loses head-to-head
  34:23, is 12× slower, and prunes less. Root cause: each prune step
  inserts a *random* terminal — many small deep steps inject more noise
  than 1–2 big random cuts. Real lever is the replacement *value*, not
  the cut position (→ I16).
- **I16 semantic pruning:** `evolve_prune_tree(df_train=...)` replaces
  pruned branches with their mean output (bool: majority). Benchmark
  60 trees: semantic wins 39 | deepest 11 | random 4; median RMSE
  1.5 vs 18.4. Wired into all engine prune call sites.

---

## Changelog (max 5 entries, one line each — older history in git)

- 2026-09-01 **I16**: semantic pruning (mean-output replacement) +
  benchmark win 39/60; df_train wired into all prune call sites.
- 2026-09-01 **P26 follow-up**: `prune_strategy="deepest"` option added +
  benchmarked — random stays default (see Decided); spawned I16.
- 2026-09-01 **D11**: rewrite engine `_rewrite.py`; `revoke_useless_nodes()`
  now respects `is_fix`.
- 2026-09-01 **I5**: races core loop + diversity gate + benchmark; fixed
  P26 (`evolve_prune_tree` returned subtree) and P27 (races shared seed).
- 2026-09-01 **D5 Phase 4**: trunk analysis, origin templates,
  `chain_mutation` strategy — D5 Phases 1–4 complete.

---

## Maintaining this file

1. **Done ⇒ delete.** Move durable findings to `PITFALLS.md` (bugs/gotchas),
   `ARCHITECTURE.md` (structure), or the feature doc — then remove the item.
   No ✅-graveyards.
2. **Changelog is capped at 5 one-line entries.** Adding a sixth deletes the
   oldest. Full history lives in git.
3. **Decisions get one line** in "Decided". Long rationale goes into the
   commit message or the relevant doc.
4. **Measurements/diagnosis tables do not belong here** — put them in the
   benchmark script output, `PITFALLS.md`, or a dedicated doc.
5. Target size: **under ~200 lines**. If it grows past that, prune before
   adding.

