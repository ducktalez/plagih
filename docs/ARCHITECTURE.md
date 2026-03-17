# plagih – Architecture & Framework Reference

> This document explains the full workflow for new users and serves as
> emergency reference for AI assistants. For pitfalls see `PITFALLS.md`.

---

## 1. What is plagih?

plagih is a **genetic programming (GP)** framework that evolves symbolic
expression trees to approximate a target function. Unlike black-box ML models,
the output is a human-readable formula – hence *explainable*.

### Core idea

```
Input:  DataFrame with feature columns + target column
Output: Pareto front of symbolic expressions (trade-off complexity vs. error)
```

Each individual in the population is a **tree of `Node` objects** representing
a mathematical/logical expression (e.g., `Add(Mul(x, 3), Sin(y))`).

---

## 2. Module overview

```
plagih/
├── config.py             # PlagihConfig singleton – loads .env, central defaults
├── trees.py              # Everything: Node hierarchy, Evolution, ExplainableGP,
│                         #   Candidate, tree operations, simplification
├── parallel.py           # Strategy system, ProcessPoolExecutor workers,
│                         #   TaskSpec/TaskResult, batching, PerformanceTracker
├── paretofront.py        # pareto_from_pop() – dominance filter
├── monitoring.py         # GPMonitor – metrics, callbacks, DataFrame export
├── evaluation_context.py # Optional unified evaluation (sympy/numpy/lambda)
├── population_merge.py   # DAG merge for batch evaluation
├── util.py               # printpl, file I/O, constants, BColors (reads cfg)
└── test/                 # pytest test suite

visualization/
├── tree_renderer.py      # Matplotlib-based tree + merged-graph rendering
└── visualize_trees.py    # Paretofront grid visualization
```

---

## 3. Class hierarchy

### 3.1 Node (tree structure)

<!-- AUTOGEN:NODE_HIERARCHY_FULL:START -->
```
Node (ABC)
    ├── NodeWithChilds
    │   ├── NodeDummy
    │   │   ├── PleaseUsePartnerOp
    │   │   └── ExprCondPair_Dummy
    │   └── BaseOperator
    │       ├── MathOperator
    │       │   ├── Trigonometry
    │       │   │   ├── Cos  (+ NoSymCapitalized)
    │       │   │   ├── Sin  (+ NoSymCapitalized)
    │       │   │   ├── Tan  (+ NoSymCapitalized)
    │       │   │   ├── Acos  (+ NoSymCapitalized)
    │       │   │   ├── Asin  (+ NoSymCapitalized)
    │       │   │   ├── Atan  (+ NoSymCapitalized)
    │       │   │   ├── Tanh  (+ NoSymCapitalized)
    │       │   │   ├── Sinh  (+ NoSymCapitalized)
    │       │   │   └── Cosh  (+ NoSymCapitalized)
    │       │   ├── BaseMinMax
    │       │   │   ├── Min  (+ ChainableOp)
    │       │   │   ├── Max  (+ ChainableOp)
    │       │   │   └── Clip  (+ CustomOperator)
    │       │   ├── Add  (+ ChainableOp)
    │       │   ├── Mul  (+ ChainableOp)
    │       │   ├── DivFraction
    │       │   ├── NthRoot
    │       │   ├── Pow
    │       │   ├── Abs
    │       │   ├── Sign  (+ NoSymCapitalized)
    │       │   ├── Log  (+ NoSymCapitalized)
    │       │   ├── Square
    │       │   ├── Exp
    │       │   ├── Exp2
    │       │   ├── Sub
    │       │   ├── Round
    │       │   ├── PowRounded
    │       │   ├── Div
    │       │   ├── Sqrt
    │       │   ├── Usub
    │       │   └── Scale
    │       ├── LogicOperator
    │       │   ├── Not
    │       │   ├── And  (+ ChainableOp)
    │       │   ├── Or  (+ ChainableOp)
    │       │   ├── Xor  (+ NoSymCapitalized, ChainableOp)
    │       │   └── ITE
    │       ├── RelationalOperator
    │       │   ├── Eq
    │       │   ├── Ne
    │       │   ├── Lt
    │       │   ├── Le
    │       │   ├── Gt
    │       │   └── Ge
    │       ├── Ifte
    │       └── Piecewise  (+ ChainableOp)
    └── Terminal
        ├── Boolean
        ├── Number
        └── Symbol

Mixins (secondary bases, not part of Node tree):
  - ChainableOp
  - CustomOperator
  - NoSymCapitalized
  - PleaseUsePartnerOp
```
<!-- AUTOGEN:NODE_HIERARCHY_FULL:END -->

**Key fields on every `Node`:**
- `childs: List[Node]` – child nodes (operators) or `[value]` (terminals)
- `parent_node: Optional[Node]` – back-reference (excluded from pickle!)
- `root_node: Optional[Node]` – back-reference (excluded from pickle!)
- `depth: Optional[int]` – depth in the tree
- `symfun` / `np_fun` – symbolic and numpy evaluation functions
- `showme` – display name for the operator

### 3.2 Candidate

```python
class Candidate:
    tree: Node        # The expression tree
    fitness: float    # Error score (lower = better)
    parsimony: int    # Complexity metric (node count)
    tag: deque[str]   # Evolution history (max 10)
```

### 3.3 Evolution

Holds the **operator pool**, **symbol list**, and **constraints** (depth_max,
nodes_max). Provides tree creation/mutation methods:

- `evolve_new_tree_depth()` – random tree
- `evolve_mutate_branch_depth()` – subtree mutation
- `evolve_mutate_point()` – single-node mutation
- `evolve_crossover()` – subtree crossover
- `evolve_create_random()` – random subtree with constraints

### 3.4 ExplainableGP (main entry point)

The top-level class. Created via `ExplainableGP.create(...)` factory method.

**Key attributes:**
- `pop_genepool: List[Candidate]` – current generation
- `pop_next: List[Candidate]` – next generation (being built)
- `paretofront: List[Candidate]` – best solutions per complexity
- `lut_tree_infos` / `lut_symex_fitness` – lookup tables (caches)
- `monitor: GPMonitor` – metrics tracker
- `_pool` – persistent `ProcessPoolExecutor` (lazily created)

---

## 4. GP lifecycle

### 4.1 Initialization

```python
gp = ExplainableGP.create(
    symbols=['x', 'y'],
    df_train=df,
    rootdir='./run_001',
    preset='math_simple',     # or 'math_full', 'with_logic'
    pop_max_size=200,
    gen_end=50,
    parallel=4,               # workers (True=auto, False=sequential)
)
```

### 4.2 Initial population

```python
gp.gen_create_initial()
```
Creates `pop_max_size` random trees, evaluates them, builds initial genepool.
Always runs **sequentially** (even if parallel is configured).

### 4.3 Generation loop

```python
strategies = [
    Strategy("reproduction", rate=0.1),
    Strategy("mutation", rate=0.3),
    Strategy("crossover", rate=0.3, crossover=True),
    Strategy("random_new", rate=0.2),
    Strategy("simplicate", rate=0.1, simplicate=True),
]

for gen in range(gp.gen_end):
    gp.run_generation(strategies)
```

`run_generation` orchestrates:

1. **Build tasks** – one `TaskSpec` per individual to create
2. **Execute** – sequential or parallel (via `run_generation_parallel`)
3. **Collect** – `Candidate` objects added to `pop_next`
4. **Finalize** – `end_generation()`:
   - Update Pareto front
   - Replace `pop_genepool` with `pop_next`
   - Record metrics via `GPMonitor`

### 4.4 Parallel execution flow

```
Main process                          Worker processes
─────────────                         ─────────────────
build TaskSpecs (1000 tasks)
split into n_workers batches
                                      _init_worker() [once at pool creation]
_update_worker_state(pop, pareto)  →  receive & repair_all()
submit batches                     →  _worker_run_batch()
                                        for each task:
                                          strategy_fn() → tree
                                          tree_simplification() [optional]
                                          evaluate_tree_standalone() → Candidate
                                      return List[TaskResult]
collect results                    ←  
repair_all() on received candidates
pop_next_append() for each
```

### 4.5 Cleanup

```python
gp.close()  # Shuts down ProcessPoolExecutor
```

---

## 5. Evaluation pipeline

Each tree is evaluated in `evaluate_tree_standalone()`:

1. **Input repair**: `repair_depth()`, `force_input_node()`, `evolve_prune_tree()`
2. **SymPy expression**: `tree.get_sympy_expr()` → symbolic form
3. **LUT check**: Has this expression been seen before? → skip evaluation
4. **NumPy prediction**: `eval_predict_sympyBatch()` via `sympy.lambdify`
5. **Error metric**: `eval_error_metric(prediction, target)` → fitness
6. **Complexity**: `complexity_metric(tree)` → parsimony
7. **Result**: `Candidate(tree, fitness, parsimony, tag)`

### LUT system (Lookup Tables)

Two caches prevent redundant computation:
- **`lut_tree_infos`**: tree-structure hash → sympy expression
- **`lut_symex_fitness`**: sympy expression string → fitness value

---

## 6. Pareto front

The Pareto front tracks **non-dominated** solutions with respect to
(parsimony, fitness). A candidate A **dominates** B iff:
- `A.parsimony ≤ B.parsimony` AND `A.fitness ≤ B.fitness`
- with at least one strict inequality.

Updated every generation via `run_update_paretofront(pop_next)`.
The front is the main output of a GP run.

---

## 7. Serialization / Backup

### Pickle optimization

`Node.__getstate__` excludes `parent_node` and `root_node` from pickle.
These circular back-references would cause pickle to serialize the entire
tree multiple times, inflating size ~5-10×.

**After any deserialization, call `tree.repair_all()` to restore back-refs.**

This affects:
- `ProcessPoolExecutor` IPC (worker ↔ main)
- `backup_save()` / `backup_load()`
- `copy.deepcopy()` (used by `selection_tournament`)

### Backup API

```python
gp.backup_save()                    # → rootdir/backup/backup.pkl
gp.backup_load()                    # ← restores gen_id, pop, pareto, monitor
```

---

## 8. Monitoring

`GPMonitor` records per-generation metrics:

```python
monitor.record_generation(gen_id, population, gen_time, pareto_updated, lut_size)
monitor.to_dataframe()      # → pandas DataFrame
monitor.export_json(path)   # → JSON file
```

Custom callbacks:
```python
monitor.on_generation(lambda metrics: print(f"Gen {metrics.gen_id}"))
monitor.on_improvement(lambda metrics: save_best(metrics))
```

---

## 9. Visualization

### Single tree
```python
from visualization.tree_renderer import render_tree
render_tree(tree, filename="my_tree", output_dir="./output")
```

### Merged population graph
```python
from visualization.tree_renderer import render_merged_tree
render_merged_tree(graph, filename="merged", display_mode="label")
```

### Pareto front grid
```python
from visualization.visualize_trees import visualize_paretofront
visualize_paretofront(gp.paretofront, output_dir=gp.rootdir)
```

---

## 10. Configuration system

### Overview

All framework-wide settings are centralised in `plagih/config.py` via the
`PlagihConfig` singleton (`cfg`).  Settings are resolved in this order:

```
.env file  ←  environment variables  ←  code-level overrides
            (lowest priority)            (highest priority)
```

1. On first import of `plagih.config`, `python-dotenv` loads the project-root
   `.env` file (if present).
2. Environment variables **override** `.env` values (standard dotenv behaviour).
3. Code-level parameters (e.g. `ExplainableGP.create(parallel=4)`) override
   everything for that particular instance.

### `.env` key reference

Copy `.env.example` → `.env` and adjust.  All keys use the `PLAGIH_` prefix.

| Key | Type | Default | Description |
|---|---|---|---|
| `PLAGIH_VERBOSITY` | str | `wwaaggiiffpp` | Substring-membership string for `printpl`/`printez` |
| `PLAGIH_DEBUG` | bool | `false` | Debug-level checks (e.g. sympy comparison) |
| `PLAGIH_SIMPLIFICATION` | bool | `false` | SymPy simplification during evolution |
| `PLAGIH_VISUALIZATION` | bool | `false` | Plots/renderings during evolution |
| `PLAGIH_MERGED_TREE` | bool | `false` | Build merged population tree per generation |
| `PLAGIH_ORIGIN_TREE` | bool | `false` | Track origin-tree metadata on candidates |
| `PLAGIH_LUT_ENABLED` | bool | `false` | Expression LUT for duplicate avoidance ⚠️ |
| `PLAGIH_PARALLEL` | int | `0` | Worker count (0=sequential) |
| `PLAGIH_FLOAT_PRECISION` | int | `3` | Decimal places for terminal formatting |
| `PLAGIH_PLOTS_INTERVAL` | int | `1` | Plot every N generations |
| `PLAGIH_BACKUP_INTERVAL` | int | `10` | Backup every N generations |
| `PLAGIH_TREE_MIN_PARSIMONY` | int | `3` | Minimum complexity for kept trees |

> ⚠️ **LUT warning:** Disabling LUT (`PLAGIH_LUT_ENABLED=false`) means every
> expression is re-evaluated even if identical ones were already seen.  For any
> non-trivial run this is **significantly** slower.  The default is `false` to
> keep the minimal profile transparent, but most users should set it to `true`.

### Minimal vs. recommended profile

The **default** profile is intentionally minimal — no simplification, no
visualisation, no parallelisation, no LUT.  This is the safest starting
point for new users and reproduces the simplest possible behaviour.

For real runs, copy `.env.example` and enable:
```
PLAGIH_LUT_ENABLED=true
PLAGIH_VISUALIZATION=true
PLAGIH_PARALLEL=4
```

### Usage in code

```python
from plagih.config import cfg

# Read
if cfg.lut_enabled:
    ...

# Runtime override
cfg.verbosity = "ww"  # suppress most output
```

Legacy module-level globals (`PRINT_DUMMY`, `DEBUG_DUMMY`, etc.) in `util.py`
are initialised from `cfg` and remain available for backwards compatibility.

---

## 11. ExplainableGP.create() parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `symbols` | list | *required* | Input variable names |
| `df_train` | DataFrame | *required* | Training data with target column |
| `rootdir` | str/Path | *required* | Output directory |
| `preset` | str | `'math_full'` | Operator preset |
| `pop_max_size` | int | `100` | Population size |
| `gen_end` | int | `50` | Number of generations |
| `depth_max` | int | `7` | Max tree depth |
| `nodes_max` | int | `40` | Max nodes per tree |
| `parallel` | bool/int/None | `None` → `.env` | Worker count |
| `enable_analysis` | bool/None | `None` → `.env` | Plots, backups, visualizations |
| `error_metric` | str | `'rmse'` | `'rmse'`, `'mse'`, `'mae'`, or callable |
| `clip_range` | tuple | `None` | `(min, max)` to clip predictions |
| `allow_chain` | bool | `False` | Allow chained operators (Add with 3+ children) |
| `target_column` | str | `'action'` | Target column in df_train |

### Built-in strategies

| Name | Description | `crossover` | `simplicate` |
|---|---|---|---|
| `reproduction` | Clone from tournament selection | | |
| `mutation` | Subtree mutation (branch) | | |
| `mutation_point` | Single-node mutation | | |
| `mutation_branch_nodes` | Branch mutation by node count | | |
| `mutation_filter` | Mutation on specific subtree types | | |
| `random_new` | Brand new random tree | | |
| `crossover` | Subtree crossover of two trees | ✓ | |
| `simplicate` | SymPy simplification of a tree | | ✓ |
| `pareto_revive` | Clone from Pareto front | | |

---

## 12. How to add a new operator

Checklist for adding a new `Node` subclass to the framework:

### 12.1 Define the class

1. **Choose a base class**: `MathOperator` (float→float), `LogicOperator`
   (bool→bool), `RelationalOperator` (float→bool), `Trigonometry`, `BaseMinMax`,
   or `BaseOperator` directly.
2. **Set required class attributes**:
   | Attribute | Example | Description |
   |---|---|---|
   | `xtype` | `((float, float), float)` | `(input_types, output_type)` |
   | `symfun` | `staticmethod(lambda *a: sympy.Add(*a))` | SymPy constructor |
   | `np_fun` | `staticmethod(np.add)` | NumPy evaluation function |
   | `showme` | `"Add"` | Display name |
   | `sy_str` | `"({0} + {1})"` | String template with `{0}`, `{1}` placeholders |
   | `repr_str` | `"Add{},[{}, {}]"` | Repr template |
3. **Optional attributes**:
   - `latex_fmt` / `latex_inline` — for LaTeX rendering
   - `is_commutative = True` — if child order doesn't matter
   - `formulae_str` — alternative string representation

### 12.2 Visualization

The `_viz_color`, `_viz_border`, `_viz_text`, `_viz_shape` attributes are
**inherited** from the base class hierarchy. No changes needed in renderers.
Override on your class only if you want a distinct colour.

### 12.3 Registration

- **If a SymPy equivalent exists**: add to `d_sym2node` (and `sym2node` if needed)
  so that `sympy_to_tree()` can reconstruct your operator from SymPy expressions.
- **If no SymPy equivalent**: don't register (like `Scale`, `Clip`). The operator
  is created only by grouping or direct creation.

### 12.4 Operator pool

Add to `Evolution.operator_presets` or pass via user `operators` dict.
Example: `{MyOp: 0.5}` means weight 0.5 relative to other operators.

### 12.5 Grouping / simplification

If your operator can be derived from simpler patterns, add a grouping rule in
`Node.tree_node_grouping()`. Example: `Mul(Number, expr)` → `Scale(Number, expr)`.

> **TODO**: Grouping rules should eventually be declared as class attributes or
> decorators on the node class itself, rather than centrally in `tree_node_grouping`.

### 12.6 Special creation logic

If your operator has input constraints beyond the `xtype` system (e.g. `Scale`
requires `childs[0]` to be a `Number`), add special-case logic in
`Evolution.evolve_create_random()`.

### 12.7 Validation

- `check_operator_pool()` validates type closure — ensure your operator doesn't
  break the float/bool ecosystem.
- Tests in `test_visualization.py` automatically check `_viz_*` attributes.
- Tests in `test_all_node_classes.py` check `symfun`/`np_fun` consistency.

---

## 13. Commutative child ordering

Commutative operators (`Add`, `Mul`, `Min`, `Max`, `And`, `Or`, `Xor`) have
`is_commutative = True`. The method `Node.canonicalize_children()` sorts
their children by string representation for a deterministic canonical form.

This is a **post-processing** step — called in:
- `tree_simplification()` (after `tree_node_grouping`)
- `tree_to_candidate()` (before LUT key generation)

**Not** called in `set_childs()` because children may be incomplete during
tree construction.

Benefits:
- Identical expressions get identical LUT keys → better cache hit rate
- Deterministic `__str__` output for debugging and comparison

### Open design discussion (TODO)

The current implementation should be revisited once benchmark data is available:

1. **Performance**: Recursive traversal + `represent_str()` sort key on every
   `tree_to_candidate()` call. May become measurable overhead on large trees.
   Alternative: sort by subtree size (`len(child)`) — cheaper and stable.

2. **Sort key**: `represent_str()` is a simple lexicographic comparison.
   SymPy's `default_sort_key` produces more canonical forms (considers type
   priority, algebraic complexity) but is significantly more expensive.
   A middle-ground: sort by subtree size first, then string as tiebreaker.

3. **Mutation invalidation**: When a mutation changes a node deep in the tree,
   the canonical order of all ancestor commutative nodes becomes stale. Currently
   handled by re-running `canonicalize_children()` in `tree_to_candidate()`
   after all mutations are complete. If canonicalization is ever moved earlier
   (e.g. between mutation steps), mutations must propagate re-sorting upwards.

See also `PITFALLS.md` P10.

---

## 14. Scale operator

`Scale(factor, expression)` is a semantically specialised `Mul` where:
- `childs[0]` is always a `Number` terminal (the scaling factor)
- `childs[1]` is a float-producing expression (operator or Symbol, not Number)

Created via:
- **Grouping**: `Mul(3.0, sin(x))` → `Scale(3.0, sin(x))` in `tree_node_grouping`
- **Direct creation**: `evolve_create_random` generates `Scale(constant, subtree)`

Purpose:
- Explicit scaling factor for better readability
- Ideal target for `evolve_mutate_filter_gauss` (tune the constant)
- Additional to `Mul` in the operator pool (not a replacement)

**SymPy mapping**: `Scale.symfun = sympy.Mul` — no dedicated SymPy type exists.
Scale is not registered in `d_sym2node`; it is produced only by grouping or
direct creation, never by `sympy_to_tree()`.

---

## 15. Future: xtype system extension

> **TODO**: Discuss whether the `xtype` system should support node-class
> constraints (e.g. `xtype = ((Number, BaseOperator), float)`) in addition
> to the current `float`/`bool` primitives.
>
> **Pro**: Universal, type-safe, enables operators like `Scale` to declare
> their constraints declaratively. Would simplify `evolve_create_random`.
>
> **Contra**: Significant changes to `operatorpool_to_picks`,
> `choose_operator_class`, `evolve_create_random`, and `choose_operator_match`.
> The current system is simple and readable.
>
> Current workaround: Operators with special constraints (e.g. `Scale`)
> handle them via `__init__` validation + special-case logic in
> `evolve_create_random`. This works but doesn't scale to many such operators.
