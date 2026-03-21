# Known Pitfalls

> Things that have caused bugs or confusion. Read before editing core code.

---

## P1 – Pickle strips `parent_node` / `root_node`

**Where:** `Node.__getstate__` / `Node.__setstate__` in `trees.py`

`parent_node` and `root_node` are circular back-references that would cause
pickle to serialize the entire tree multiple times. They are **excluded**
from the pickle state.

**Rule:** After any deserialization, call `tree.repair_all()`.

This applies to:
- `pickle.load()` / `pickle.loads()` (backup files)
- `ProcessPoolExecutor` IPC (data sent to/from workers)
- `copy.deepcopy()` (also uses `__getstate__`!)

**Already handled in:**
- `_update_worker_state()` in `parallel.py` – repairs incoming genepool
- `run_generation_parallel()` – repairs candidates received from workers
- `backup_load()` in `trees.py` – repairs after loading backup

**How to break it:** Add a new deserialization path without `repair_all()`.

---

## P2 – `deepcopy` also uses `__getstate__`

**Where:** `selection_tournament()` calls `copy.deepcopy(winner.get_evotree())`

Python's `deepcopy` uses `__getstate__`/`__setstate__` when they exist.
The returned tree has `parent_node = None` and `root_node = None` on all
nodes.

**Why it works anyway:** Mutation strategies (`set_new_node`, etc.) set
`parent_node = None` in their default path. `evaluate_tree_standalone`
calls `repair_depth()`. Crossover and mutation don't rely on back-refs.

**How to break it:** Write code that reads `node.parent_node` on a
deepcopy'd tree and assumes it points somewhere.

---

## P3 – `print_pop()` calls `get_sympy_expr()` on every candidate

**Where:** `end_generation()` in `trees.py`

`print_pop()` calls `candidate.full_string()` which calls
`get_sympy_expr()` for each tree. SymPy expression computation takes
~40ms per tree. With 1000 candidates, that's **~40 seconds per generation**
of pure printing overhead.

**Mitigation:** `print_pop()` is guarded by `"gggg" in PRINT_DUMMY`.
With the default `PRINT_DUMMY = "wwaaggiiffpp"`, `"gggg"` is **not**
a substring, so `print_pop` is skipped.

**How to break it:** Set `PRINT_DUMMY` to a string containing `"gggg"`
(e.g., `"wwaaggggiiiffpp"`) and run with large populations.

**Extra pitfall:** `printpl("gggg", f"...")` is **not** lazy. The f-string is
evaluated before `printpl()` checks `PRINT_DUMMY`. For expensive values like
`tree.str_as_expr()` / `get_sympy_expr()`, add a local guard first:

```python
if "gggg" in PRINT_DUMMY:
    printpl("gggg", f"...{tree.str_as_expr()}...")
```

---

## P4 – Windows parallel: everything must be picklable

**Where:** `ProcessPoolExecutor` in `parallel.py`

On Windows, `ProcessPoolExecutor` uses `spawn` (not `fork`). Every object
sent to workers is serialized via `pickle`. This means:

- **Strategies must be top-level functions** (not lambdas, not closures,
  not nested functions).
- **Error metrics must be top-level functions** – that's why `_error_rmse`,
  `_error_mse`, `_error_mae` exist in `trees.py`.
- **`_ClipAutocast`** is a picklable class, not a lambda.

**How to break it:** Pass a lambda as `eval_error_metric` to
`ExplainableGP.create()` and set `parallel=True`.

---

## P5 – Verbosity system (`cfg.verbosity` / `.env`)

**Where:** `plagih/config.py`, `plagih/util.py`

Verbosity is controlled by `cfg.verbosity` (loaded from `PLAGIH_VERBOSITY`
in `.env` or environment variables).  The check uses **substring membership**:

```python
from plagih.config import cfg
# cfg.verbosity = "wwaaggiiffpp"  (default)

# "gg" in "wwaaggiiffpp"   → True  (generation summaries printed)
# "gggg" in "wwaaggiiffpp" → False (per-candidate detail skipped)
# "pp" in "wwaaggiiffpp"   → True  (performance info printed)
```

The `printpl(msg_type, message)` function checks `if msg_type not in
cfg.verbosity: return`.

For backwards compatibility, `util.PRINT_DUMMY` is initialised from
`cfg.verbosity` at import time.  **New code should use `cfg.verbosity`
directly**, as runtime changes to `cfg.verbosity` are picked up
immediately by `printpl`/`printez`.

Legacy writes like `util.PRINT_DUMMY = "ww"` still work (module-level
variable), but they do **not** update `cfg.verbosity`.  Prefer:
```python
from plagih.config import cfg
cfg.verbosity = "ww"
```

**Levels** (from most to least common):
- `"g"` – major generation events
- `"gg"` – generation summaries
- `"ggg"` – strategy-level detail
- `"gggg"` – per-candidate detail (expensive!)

**To enable full verbosity:** Set `PLAGIH_VERBOSITY=wwwwaaaggggiiiifffpp`
in `.env` or environment.

---

## P6 – `repair_depth()` vs `repair_all()`

**Where:** `Node` in `trees.py`

| Method | Repairs | Use when |
|---|---|---|
| `repair_depth(depth)` | Only `depth` field, recursively | After structural changes (mutation, crossover) |
| `repair_all(parent, root, depth)` | `parent_node`, `root_node`, `depth` | After deserialization or full tree rebuilds |

`repair_depth()` is cheaper but does **not** fix back-references.
After pickle/deepcopy, you need `repair_all()`.

---

## P7 – Tree structural modifications

**Where:** `set_new_node()`, `replace_with()`, `set_childs()` in `trees.py`

When replacing nodes in a tree, the `set_new_node()` method:
- Copies `__class__` and `__dict__` from the new node
- Runs `revoke_useless_nodes()` to simplify (e.g., `Mul(a)` → `a`)
- Sets `parent_node = None` by default (`repair=False`)

**Rule:** After any tree modification, call at least `repair_depth()`.
If you need parent/root refs, call `repair_all()`.

---

## P8 – `gen_create_initial()` is always sequential

**Where:** `ExplainableGP.gen_create_initial()` in `trees.py`

The initial population is always created sequentially, even if
`parallel=True`. This is intentional – the initial population is small
and the parallel pool may not yet be initialized.

---

## P9 – Benchmarking with analysis enabled distorts timing

**Where:** `analyze_generation()` in `trees.py`

By default, every generation triggers visualization (merged tree rendering,
Pareto-front plots, monitoring plots, parsimony histograms) and backups.
This IO/rendering overhead **dominates** runtime for benchmarks and makes
timing results unreliable.

**Rule:** Set `enable_analysis=False` when creating GP instances for
benchmarking or performance measurement:

```python
gp = ExplainableGP.create(..., enable_analysis=False)
```

Lightweight metric recording (`GPMonitor.record_generation()`) always runs
regardless of this flag — only expensive IO operations are skipped.

**Future idea:** Run analysis in a separate background process so the main
evolution loop is never blocked by IO/rendering.

---

## P10 – `get_sympy_expr()` in monitoring was a hidden bottleneck

**Where:** `GPMonitor._compute_population_metrics()` in `monitoring.py`

The unique-expression counter originally called `get_sympy_expr()` on
**every candidate** in the population to compute diversity. For pop=1000,
this added **~3-5 seconds per generation** — more than the actual
evolution compute time.

**Fix (applied):** Replaced `str(c.tree.get_sympy_expr())` with
`str(c.tree)` which uses the fast tree string representation.

**Rule:** Never call `get_sympy_expr()` in hot paths. Use `str(tree)`,
`tree.get_lut_id()`, or `tree.represent_str()` for fast identification.

---

## P11 – Parallel worker count: physical cores, not logical

**Where:** `cpu_count_physical()` in `util.py`, `ExplainableGP.__init__`

`os.cpu_count()` returns **logical cores** (e.g. 16 on an 8-core
hyperthreaded CPU). For CPU-bound GP work, using more workers than
physical cores hurts performance because hyperthreads share execution
units and caches.

**Rule:** Use `cpu_count_physical()` from `util.py`. When `parallel=True`,
the framework now auto-detects physical cores. For explicit control,
pass the number directly, e.g. `parallel=4` or `parallel=8`.

**Current benchmark state:** With the current pre-selection + shared-memory +
chunked-batching implementation on an 8-core Windows machine, `8` workers are
best for both `pop=1000` and `pop=10000`. Do **not** hard-code the older
assumption that 4 workers are always optimal; the sweet spot depends on the
current batching strategy and population size.

---

## P12 – Relational operators on `Ifte` / `Piecewise` subtrees can hang SymPy

**Where:** `Node.get_sympy_expr()` / `sympy.Piecewise` / relational operators like `Lt`, `Le`

Nested constructions like:

```python
Lt(Ifte(...), 0)
Le(Piecewise(...), x)
```

can trigger pathological recursion or very long hangs inside SymPy on Windows
(observed in `bench_diagnose_full.py` during large parallel runs).

**Rule:** Treat relational-on-piecewise as unsupported for SymPy conversion and
fail fast with `SympyError` instead of letting SymPy recurse indefinitely.

**Already handled in:**
- `Node.get_sympy_expr()` in `trees.py` now raises `SympyError` early for this pattern.
- `run_generation_parallel()` in `parallel.py` now uses smaller runtime batches
  plus timeout/debug output so pathological tasks no longer appear as silent hangs.

**How to break it:** Remove the guard in `get_sympy_expr()` or reintroduce
large one-batch-per-worker execution without progress diagnostics.

---

## P13 – Scale is invisible to `sympy_to_tree()`

**Where:** `Scale` class in `trees.py`, `d_sym2node` mapping

`Scale` uses `sympy.Mul` as its `symfun`, and is NOT registered in `d_sym2node`.
This means `sympy_to_tree()` will always reconstruct `Scale(c, expr)` as
`Mul(c, expr)`. The Scale form is only recovered by `tree_node_grouping()`.

**Rule:** Any pipeline that round-trips through SymPy (e.g. `tree_simplification`)
must call `tree_node_grouping()` afterwards to restore Scale nodes.

**Already handled in:**
- `tree_simplification()` calls `tree_node_grouping()` then `canonicalize_children()`.

**How to break it:** Add a SymPy round-trip path that skips `tree_node_grouping()`.

---

## P14 – `canonicalize_children()` must run after tree is fully built

**Where:** `Node.canonicalize_children()` in `trees.py`

The canonical child ordering uses `represent_str()` as sort key, which requires
all children to be fully constructed. Calling it during tree *construction*
(e.g. inside `set_childs()`) would produce incorrect or crashing results.

**Rule:** Only call `canonicalize_children()` as a post-processing step:
in `tree_simplification()`, `tree_to_candidate()`, or similar finalization points.

**How to break it:** Move the sort into `set_childs()` or `__init__()`.

**Mutation invalidation:** When a mutation modifies a child node deep in the
tree, the canonical order of *all ancestor commutative nodes* may become stale.
Currently this is handled by calling `canonicalize_children()` in
`tree_to_candidate()` (after all mutations are complete). If canonicalization
were ever moved earlier in the pipeline (e.g. inside mutation operators),
every mutation would need to propagate re-sorting upwards.

**Open design trade-offs:** → see `IMPLEMENTATION_PLAN.md` § D1.

---

## P15 – Tree Edit Distance mode selection

**Where:** `plagih/tree_complexity/tree_edit_distance.py`, `eval_parsimony()` in `trees.py`

The intrinsic Zhang-Shasha TED implementation supports three label-comparison
modes via `TedConfig.mode`:

- `"structural"` – compares only node types (class). This is what the old
  `get_apted_notation()`-based distance computed, and what `eval_parsimony`
  uses internally for `"tree_edit_distance"` complexity.
- `"full"` – also compares terminal values (`Number(3) ≠ Number(5)`).
- `"structural_plus_leaf_diff"` – structural TED + leaf-diff count.

**Rule:** Choose the correct mode for the use case.  Comparing populations
for diversity → `"full"`.  Parsimony / complexity distance from origin →
`"structural"` (default in `eval_parsimony`).

**Deprecated:** `apted_distance()` and `get_apted_notation()` still exist
but emit `DeprecationWarning`.  The external `apted` package is no longer a
dependency.  Use `compute_ted(node1, node2, config)` instead.

---

## P16 – Python bytecode complexity is version-dependent

**Where:** `plagih/tree_complexity/python_bytecode_complexity.py`

The `tree_python_bytecode_count` and `tree_python_bytecode_weighted_count`
measures are based on **CPython bytecode**, not on machine assembly. This
means the exact count can change across:

- Python versions
- compiler optimisations
- bytecode format changes (`BINARY_OP`, caches, adaptive interpreter, ...)

**Rule:** Treat this as a **heuristic proof-of-concept score**, not a
hardware-stable runtime metric.

**Use cases:**
- ranking trees by rough implementation complexity
- comparing variants within the same Python/runtime environment

**Do not assume:**
- stable values across Python versions
- direct equivalence to CPU runtime
- equivalence to FLOPs or real machine instructions

**Open design tasks:** → see `IMPLEMENTATION_PLAN.md` § D2.

---

## P17 – Rounding exponents in SymPy is surprisingly hard

**Where:** `RoundDummy` in `trees/_nodes.py`, `PowRounded`, `Round`

**Context:** In GP evolution we often want `base ** round(exp)` to
constrain exponents to integers (avoiding complex numbers from fractional
powers of negative bases). Getting SymPy to cooperate with rounding
turned out to be a multi-attempt odyssey. All approaches and their
failure modes are documented below.

**SymPy upstream issue:** <https://github.com/sympy/sympy/issues/27326>
(opened 2024-11-28).

**SymPy deprecation note:** "Core operators no longer accept non-Expr
args" – see
<https://docs.sympy.org/latest/explanation/active-deprecations.html#non-expr-args-deprecated>.

### Approaches tried (and why they fail)

1. **`N(value, precision)`** – `sympy.sympify('Pow(a, N(1.234, 1))')` →
   works for numeric literals but crashes on symbols:
   `AttributeError: 'function' object has no attribute 'evalf'`.

2. **`Integer()`** – `sympy.sympify('Integer(1.234)')` truncates to `1`
   for literals, but `Integer(Symbol('a'))` raises
   `TypeError: int() argument must be a string, a bytes-like object or a
   real number, not 'Symbol'`.

3. **Modulo (`%`)** – `a - (a % 1)` truncates for floats, but
   `sympy.sympify('Pow(2, (a-(a % 1)))')` with a symbolic `a` raises
   `TypeError: unsupported operand type(s) for %: 'Symbol' and 'One'`.

4. **Built-in `round()`** – `sympy.sympify('round(1.234)')` works for
   literals, but `sympy.sympify('Pow(2, round(a))')` raises
   `TypeError: Cannot round symbolic expression`.

5. **Custom `sympy.Function` subclass (`RoundDummy`)** ✅ – the solution
   that works. A custom SymPy function whose `eval()`:
   - Returns `None` (= stays unevaluated) when the argument is symbolic.
   - Returns `sympy.Integer(round(a))` when the argument is numeric.

   This allows SymPy to carry `RoundDummy(a)` through symbolic
   manipulation and fold it to an integer when `evalf(subs=…)` is called
   or when a numeric value is substituted.

### Current implementation (`RoundDummy`)

```python
class RoundDummy(sympy.Function):
    @classmethod
    def eval(cls, a):
        if a.is_symbol:
            return None          # keep symbolic
        elif a.is_number:
            return sympy.Integer(round(a.evalf()))
```

The `__call__` override handles NumPy arrays for `lambdify`/direct
evaluation. `PowRounded` uses `RoundDummy` in its `symfun` and
`sy_str`.

### Rules

- **Always use `RoundDummy`** (or the `PowRounded` / `Round` operators)
  for rounding inside SymPy expressions. Do not use `round()`,
  `Integer()`, `N()`, or `%` directly.
- **`a.is_symbol` must be checked before `a.is_number`** inside `eval()`.
  Checking `.is_number` first can crash on certain symbol subclasses.
- Exceptions from imaginary results (e.g. `asin(tan(1))`) are caught and
  re-raised as `SympyImaginaryNumber`.

---

## P18 – `sympy.exp()` on large arguments causes `MemoryError`

**Where:** `get_sympy_expr()` → `_sym(*_cs)` in `trees/_nodes.py`

**Problem:** When GP evolves deeply nested exponential expressions
(e.g. `exp(exp(exp(large_number)))`), SymPy internally calls `mpmath`
to evaluate the result numerically. `mpmath` tries to allocate memory
proportional to the magnitude, which can be billions of bits, causing
an instant `MemoryError` that crashes the entire process.

**Concrete example from a test run:**
```
Exp(some_large_crossover_result)
→ sympy.exp(huge_float)
→ mpf_exp in mpmath
→ man << offset  # MemoryError
```

**Fix:** A `MemoryError` catch was added in `get_sympy_expr()` at the
`_r = _sym(*_cs)` call site. It re-raises as `SympyError`, which the
evaluation pipeline already handles gracefully (tree is rejected, logged,
and evolution continues).

**Rule:** Never assume that SymPy's `symfun` calls are memory-safe. Any
operator whose `symfun` involves numeric evaluation (`exp`, `Pow`, `log`
with extreme arguments) can trigger this.

---

## P19 – `tree_simplification` can grow trees (WHATHAPPENED)

**Where:** `tree_simplification()` → `sympy_to_tree()` →
`tree_node_grouping()` in `trees/_nodes.py`

**Problem:** The simplification pipeline is **not idempotent**: SymPy's
canonical form sometimes differs from the tree's grouped form, causing a
cycle where each round-trip changes the representation without shrinking
the tree.

**Observed patterns:**
1. `Div(a, b)` → SymPy: `a * b**(-1)` → `Mul(a, Pow(b, -1))` →
   grouping: `Mul(a, DivFraction(b))` — representation changes but
   semantics are identical, node count may grow.
2. `Scale(c, expr)` → SymPy doesn't know `Scale`, so `sympy_to_tree`
   rebuilds it as `Mul(c, expr)`, then `tree_node_grouping` converts
   back to `Scale(c, expr)` — but intermediate forms have different
   structure and can have more nodes.
3. `sin(1)**2` → SymPy evaluates to `~0.708` → a `Number(0.708)` node
   replaces the `Square(Sin(1))` subtree — the expression changes
   semantically ("Diff in sympy expression" warning).

**Current mitigation:**
1. **Size guard** — if the simplified tree is larger than the original,
   the original is returned unchanged.
2. **Min-size tracking** — the grouping loop tracks the smallest tree
   seen across all iterations and uses it as the result.
3. **Oscillation detection** — if a string representation repeats across
   iterations (A → B → A), the loop exits early instead of running to
   exhaustion.
4. **Structured logging** — growth and semantic diff are logged via
   `log("w", …)` instead of debug prints.

The former `CuriosityError` debug bomb at iteration 6 has been removed.

**Impact:** Trees will never grow during simplification (the size guard
ensures this). The semantic diff (case 3) is still possible but is
logged as a warning.

**Tracked as:** D6 in `IMPLEMENTATION_PLAN.md` — "Idempotent
simplification pipeline".

---

## P20 – `revoke_useless_nodes()` must never crash on neutral cleanup

**Where:** `Node.revoke_useless_nodes()` in `trees/_nodes.py`

Structural edits (especially crossover via `set_new_node()`) can transiently
create neutral forms such as `Add(a, 0)`, `Add(0, 0)`, `Mul(a, 1)`, or
`Mul(1, 1)`. These are normal cleanup cases, not "impossible" states.

**Fix (applied):**
- neutral `0`/`1` children are removed without raising `CuriosityError`
- all-neutral `Add` / `Mul` collapse to `Number(0)` / `Number(1)`
- child iteration uses a copy so removals do not skip siblings
- `Mul(..., 1)` inside `tree_node_grouping()` now delegates to normal cleanup
  instead of crashing
- `Mul` grouping removes exactly the matched child by object identity, so
  duplicate equal factors (e.g. two `Number(-1)` nodes) are not dropped twice
- malformed representation/export paths now raise structured `TreeError`,
  `ValueError`, or `SympyError` with context instead of bare `CuriosityError`

**Rule:** Do not use `CuriosityError` for expected simplification states in
cleanup code. If a case can arise during crossover/mutation, handle it
structurally and log only if further diagnosis is needed.

---

## P21 – `Node` equality must stay identity-based

**Where:** `Node` dataclass in `trees/_nodes.py`

`Node` instances carry circular back-references via `parent_node` and
`root_node` after `repair_all()`. Dataclass-generated structural `__eq__`
therefore causes two problems:

1. comparing two repaired but structurally equal trees can recurse until
   `RecursionError`
2. list membership/filtering can accidentally treat distinct but equal-valued
   nodes as the same child

**Fix (applied):** `Node` uses identity-based equality (`@dataclass(eq=False)`),
and transforms that remove a matched child do so by object identity.

**Rule:** Never use `==` for structural tree comparison. Use
`represent_str()`, `get_lut_id()`, or SymPy-level equivalence instead.

