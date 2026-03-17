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


