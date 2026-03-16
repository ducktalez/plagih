# Copilot Instructions – plagih GP Framework

> This file is injected into every AI request.  Keep it **short** and
> limited to knowledge that **cannot** be discovered from source code.
> Auto-generated sections have zero maintenance cost.
> Full details: `docs/ARCHITECTURE.md`, pitfalls: `docs/PITFALLS.md`.

## Project overview

plagih is a **genetic programming (GP)** framework for **explainable AI**.
It evolves symbolic expression trees (math/logic) via selection, mutation,
crossover, and simplification, evaluated against training data.

## Module map

<!-- AUTOGEN:MODULE_MAP:START -->
| Module | Responsibility |
|---|---|
| `plagih/trees.py` | plagih_tree contain a new implementation of trees that we use in genetic programming to display a program. (67C/22F) |
| `plagih/parallel.py` | Parallel execution engine for plagih GP. (5C/33F) |
| `plagih/paretofront.py` | Pareto-front dominance filter for GP candidates. (0C/6F) |
| `plagih/monitoring.py` | GP Monitoring Module (2C/0F) |
| `plagih/evaluation_context.py` | Unified Evaluation Context System for Plagih GP Trees. (3C/5F) |
| `plagih/population_merge.py` | Population Merge Module for plagih GP Framework (3C/8F) |
| `plagih/util.py` | *(no docstring)* (9C/28F) |
| `visualization/tree_renderer.py` | Unified Tree Visualization Module for plagih GP Framework (9C/10F) |
| `visualization/visualize_trees.py` | Tree Visualization for plagih GP Framework (0C/7F) |
<!-- AUTOGEN:MODULE_MAP:END -->

## Node hierarchy (auto-generated from `trees.py`)

<!-- AUTOGEN:NODE_HIERARCHY_COMPACT:START -->
```
Node (ABC)
    ├── NodeWithChilds
    │   ├── NodeDummy
    │   │   └── PleaseUsePartnerOp, ExprCondPair_Dummy
    │   └── BaseOperator
    │       ├── OperatorArity
    │       │   ├── LogicOperator
    │       │   │   └── Not, And (+C), Or (+C), Xor (+NC), ITE
    │       │   ├── RelationalOperator
    │       │   │   └── Eq, Ne, Lt, Le, Gt, Ge
    │       │   └── Ifte
    │       ├── MathOperator
    │       │   ├── Trigonometry
    │       │   │   └── Cos (+N), Sin (+N), Tan (+N), Acos (+N), Asin (+N), Atan (+N), Tanh (+N), Sinh (+N), Cosh (+N)
    │       │   ├── BaseMinMax
    │       │   │   └── Min (+C), Max (+C), Clip (+C)
    │       │   └── Add (+C), Mul (+C), DivFraction, NthRoot, Pow, Abs, Sign (+N), Log (+N), Square, Exp, Exp2, Sub, Round, PowRounded, Div, Sqrt, Usub
    │       └── Piecewise (+C)
    └── Terminal
        └── Boolean, Number, Symbol

Mixins: ChainableOp, CustomOperator, NoSymCapitalized, PleaseUsePartnerOp
```
<!-- AUTOGEN:NODE_HIERARCHY_COMPACT:END -->

## GP lifecycle (order matters)

```
ExplainableGP.create(symbols, df_train, rootdir, ...)
  └─ __init__  →  Evolution(symbol_list, operators, ...)
       └─ gen_create_initial()            # random pop, always sequential
       └─ for each generation:
            run_generation(strategies)     # parallel or sequential
              ├─ build TaskSpecs from Strategy list
              ├─ workers: create tree → simplify → evaluate → Candidate
              └─ main: pop_next_append(candidate)
            end_generation()
              ├─ run_update_paretofront(pop_next)
              ├─ pop_genepool = pop_next
              └─ analyze_generation()      # GPMonitor
       └─ close()                          # shutdown pool
```

## Critical pitfalls

These are **non-obvious** — an AI reading the code alone will miss them.
See `docs/PITFALLS.md` for the full list with examples.

1. **Pickle & back-references**: `parent_node`/`root_node` are excluded
   from pickle (`Node.__getstate__`). Call `tree.repair_all()` after
   **any** deserialization — including `deepcopy`, worker IPC, `backup_load`.
2. **Windows parallel**: `ProcessPoolExecutor` pickles everything.
   Strategies and error metrics must be **top-level functions** (no lambdas).
3. **`PRINT_DUMMY`** (`util.py`): Verbosity via substring membership.
   `"gg" in "wwaaggiiffpp"` → True, `"gggg"` → False.
   Always use `printpl("gg", ...)` instead of `print()`.
4. **Benchmarking**: `enable_analysis=False` disables plots/backups/rendering
   during evolution. Without it, IO overhead distorts timing.
5. **`get_sympy_expr()` is slow**: Never call it in hot paths (loops over
   population). Use `str(tree)` or `tree.get_lut_id()` for fast identification.
6. **Physical cores**: `os.cpu_count()` returns logical threads (16 on 8-core
   HT CPU). Use `cpu_count_physical()` from `util.py` for worker counts.

## Maintaining these docs

- **Module map & Node hierarchy** are auto-updated by a pre-commit hook
  (`scripts/update_copilot_instructions.py`). No manual work needed.
- **When you discover a new pitfall**, add it to `docs/PITFALLS.md` and
  add a one-liner to the list above. Same for new architectural patterns
  → update `docs/ARCHITECTURE.md`.
- **Coding conventions** (linting, type hints, test config) live in
  `pyproject.toml` — do **not** duplicate them here.
