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
├── trees.py              # Everything: Node hierarchy, Evolution, ExplainableGP,
│                         #   Candidate, tree operations, simplification
├── parallel.py           # Strategy system, ProcessPoolExecutor workers,
│                         #   TaskSpec/TaskResult, batching, PerformanceTracker
├── paretofront.py        # pareto_from_pop() – dominance filter
├── monitoring.py         # GPMonitor – metrics, callbacks, DataFrame export
├── evaluation_context.py # Optional unified evaluation (sympy/numpy/lambda)
├── population_merge.py   # DAG merge for batch evaluation
├── util.py               # printpl, PRINT_DUMMY, file I/O, constants, BColors
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
    │       │   └── Usub
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

## 10. Configuration reference

### ExplainableGP.create() parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `symbols` | list | *required* | Input variable names |
| `df_train` | DataFrame | *required* | Training data with target column |
| `rootdir` | str/Path | *required* | Output directory |
| `preset` | str | `'math_simple'` | Operator preset |
| `pop_max_size` | int | `100` | Population size |
| `gen_end` | int | `50` | Number of generations |
| `depth_max` | int | `7` | Max tree depth |
| `nodes_max` | int | `40` | Max nodes per tree |
| `parallel` | bool/int | `False` | Worker count |
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


