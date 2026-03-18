# plagih – Architecture Reference

> Framework overview for developers and AI assistants.
> Pitfalls: `PITFALLS.md`. Open tasks: `IMPLEMENTATION_PLAN.md`.

---

## 1. What is plagih?

A **genetic programming (GP)** framework that evolves symbolic expression
trees to approximate target functions. Output is a human-readable formula –
hence *explainable AI*.

```
Input:  DataFrame with feature columns + target column
Output: Pareto front of symbolic expressions (trade-off complexity vs. error)
```

---

## 2. Module overview

```
plagih/
├── config.py             # PlagihConfig singleton – loads .env, central defaults
├── trees.py              # Node hierarchy, Evolution, ExplainableGP, Candidate,
│                         #   tree operations, simplification
├── parallel.py           # Strategy system, ProcessPoolExecutor workers, batching
├── paretofront.py        # pareto_from_pop() – dominance filter
├── monitoring.py         # GPMonitor – metrics, callbacks, DataFrame export
├── evaluation_context.py # Optional unified evaluation (sympy/numpy/lambda)
├── population_merge.py   # DAG merge for batch evaluation
├── tree_complexity/      # Isolated complexity algorithms (unary + pairwise)
├── exceptions.py         # All framework exception classes
├── logging_utils.py      # Unified logging: setup_logging(), log(), log_info(), …
├── util.py               # I/O helpers, BColors, constants, plot presets
│                         #   (re-exports exceptions.py + logging_utils.py)
└── test/                 # pytest test suite

visualization/
├── tree_renderer.py      # Matplotlib-based tree + merged-graph rendering
└── latex_renderer.py     # LaTeX/TikZ export
```

---

## 3. Node hierarchy

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
- `parent_node` / `root_node` – back-references (excluded from pickle, see P1)
- `depth: Optional[int]` – depth in tree
- `symfun` / `np_fun` – symbolic and numpy evaluation functions
- `showme` – display name

---

## 4. GP lifecycle

```
ExplainableGP.create(symbols, df_train, rootdir, ...)
  └─ gen_create_initial()        # random pop, always sequential (P8)
  └─ for gen in range(gen_end):
       run_generation(strategies) # parallel or sequential
       end_generation()           # Pareto update, swap pop, monitor
  └─ close()                     # shutdown pool
```

### Evaluation pipeline (per tree)

1. `repair_depth()` → `force_input_node()` → `evolve_prune_tree()`
2. `get_sympy_expr()` → symbolic form
3. LUT check (skip if seen)
4. `eval_predict_sympyBatch()` via `sympy.lambdify` → prediction
5. `eval_error_metric(pred, target)` → fitness
6. `complexity_metric(tree)` → parsimony
7. → `Candidate(tree, fitness, parsimony, tag)`

---

## 5. Pareto front

Non-dominated set w.r.t. (parsimony, fitness). Candidate A **dominates** B iff
A ≤ B in both dimensions and strictly < in at least one. Updated every generation.

---

## 6. Configuration system

Settings in `PlagihConfig` singleton (`cfg`), loaded from `.env` via `python-dotenv`.

```
.env file  →  environment variables  →  code-level overrides
(lowest)                                 (highest priority)
```

### Key `.env` keys

| Key | Default | Description |
|---|---|---|
| `PLAGIH_VERBOSITY` | `wwaaggiiffpp` | Substring-membership string for `printpl` |
| `PLAGIH_LUT_ENABLED` | `true` | Expression LUT for duplicate-fitness avoidance |
| `PLAGIH_PARALLEL` | `0` | Worker count (0=sequential) |
| `PLAGIH_SIMPLIFICATION` | `false` | SymPy simplification during evolution |
| `PLAGIH_VISUALIZATION` | `false` | Plots/renderings during evolution |
| `PLAGIH_DEBUG` | `false` | Debug-level checks |
| `PLAGIH_FLOAT_PRECISION` | `3` | Decimal places for terminal formatting |
| `PLAGIH_PLOTS_INTERVAL` | `1` | Plot every N generations |
| `PLAGIH_BACKUP_INTERVAL` | `10` | Backup every N generations |
| `PLAGIH_TREE_MIN_PARSIMONY` | `3` | Minimum complexity for kept trees |

---

## 7. Adding a new operator

Checklist for adding a `Node` subclass:

1. **Choose base class**: `MathOperator`, `LogicOperator`, `RelationalOperator`,
   `Trigonometry`, `BaseMinMax`, or `BaseOperator`
2. **Set class attributes**: `xtype`, `symfun`, `np_fun`, `showme`, `sy_str`, `repr_str`
3. **Optional**: `latex_fmt`, `latex_inline`, `is_commutative`
4. **Visualization**: `_viz_*` attributes are inherited — no renderer changes needed
5. **SymPy registration**: If SymPy equivalent exists → add to `d_sym2node`
6. **Operator pool**: Add to `Evolution.operator_presets` or user `operators` dict
7. **Grouping**: Add rule in `tree_node_grouping()` if derivable from simpler patterns
8. **Special creation**: Add constraints in `evolve_create_random()` if needed
9. **Validate**: `test_all_node_classes.py`, `test_visualization.py`

---

## 8. Scale operator

`Scale(factor, expr)` – specialised `Mul` where `childs[0]` is always `Number`.
Created by grouping (`Mul(3, sin(x))` → `Scale(3, sin(x))`) or direct creation.
Uses `sympy.Mul` as `symfun`, not registered in `d_sym2node`. Only recovered
by `tree_node_grouping()` after SymPy round-trips (P13).
