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

## P5 – `PRINT_DUMMY` verbosity system

**Where:** `util.py`

`PRINT_DUMMY` is a string. Verbosity is checked via **substring membership**:

```python
PRINT_DUMMY = "wwaaggiiffpp"

# "gg" in "wwaaggiiffpp"   → True  (generation summaries printed)
# "gggg" in "wwaaggiiffpp" → False (per-candidate detail skipped)
# "pp" in "wwaaggiiffpp"   → True  (performance info printed)
```

The `printpl(msg_type, message)` function checks `if msg_type not in
PRINT_DUMMY: return`.

**Levels** (from most to least common):
- `"g"` – major generation events
- `"gg"` – generation summaries
- `"ggg"` – strategy-level detail
- `"gggg"` – per-candidate detail (expensive!)

**To enable full verbosity:** Set `PRINT_DUMMY = "wwwwaaaggggiiiifffpp"`.

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
pass the number directly: `parallel=4`.

**Sweet spot:** Benchmarks show 4 workers is optimal for pop=1000
with typical operators. More workers increase IPC overhead faster
than they reduce compute time.


