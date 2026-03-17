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
| `plagih/trees.py` | plagih_tree contain a new implementation of trees that we use in genetic programming to display a program. (67C/24F) |
| `plagih/parallel.py` | Parallel execution engine for plagih GP. (5C/33F) |
| `plagih/paretofront.py` | Pareto-front dominance filter for GP candidates. (0C/6F) |
| `plagih/monitoring.py` | GP Monitoring Module (2C/0F) |
| `plagih/evaluation_context.py` | Unified Evaluation Context System for Plagih GP Trees. (3C/5F) |
| `plagih/population_merge.py` | Population Merge Module for plagih GP Framework (3C/8F) |
| `plagih/util.py` | *(no docstring)* (9C/31F) |
| `visualization/tree_renderer.py` | Unified Tree Visualization Module for plagih GP Framework (9C/13F) |
| `visualization/visualize_trees.py` | *(file not found)* |
<!-- AUTOGEN:MODULE_MAP:END -->

## Node hierarchy (auto-generated from `trees.py`)

<!-- AUTOGEN:NODE_HIERARCHY_COMPACT:START -->
```
Node (ABC)
    ├── NodeWithChilds
    │   ├── NodeDummy
    │   │   └── PleaseUsePartnerOp, ExprCondPair_Dummy
    │   └── BaseOperator
    │       ├── MathOperator
    │       │   ├── Trigonometry
    │       │   │   └── Cos (+N), Sin (+N), Tan (+N), Acos (+N), Asin (+N), Atan (+N), Tanh (+N), Sinh (+N), Cosh (+N)
    │       │   ├── BaseMinMax
    │       │   │   └── Min (+C), Max (+C), Clip (+C)
    │       │   └── Add (+C), Mul (+C), DivFraction, NthRoot, Pow, Abs, Sign (+N), Log (+N), Square, Exp, Exp2, Sub, Round, PowRounded, Div, Sqrt, Usub, Scale
    │       ├── LogicOperator
    │       │   └── Not, And (+C), Or (+C), Xor (+NC), ITE
    │       ├── RelationalOperator
    │       │   └── Eq, Ne, Lt, Le, Gt, Ge
    │       └── Ifte, Piecewise (+C)
    └── Terminal
        └── Boolean, Number, Symbol

Mixins: ChainableOp, CustomOperator, NoSymCapitalized, PleaseUsePartnerOp
```
<!-- AUTOGEN:NODE_HIERARCHY_COMPACT:END -->

**Node rendering attributes** (on base classes, no `isinstance` in renderers):
- `_viz_color`, `_viz_border`, `_viz_text`, `_viz_shape` — set on `MathOperator`, `LogicOperator`, `Number`, `Symbol`, `Boolean`
- `latex_fmt` — format string for special LaTeX (e.g. `Pow`, `Abs`, `Sqrt`, `Min`, `Max`)
- `latex_inline` — infix separator for LaTeX (e.g. `Add → " + "`, `Mul → r" \cdot "`)
- When adding a new node type: set these on the class or its base — **no** renderer code changes needed.

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
3. **`.env` / `PlagihConfig`** (`config.py`): All framework defaults are loaded
   from `.env` via `PlagihConfig` singleton (`cfg`).  Verbosity still uses
   substring membership (`"gg" in cfg.verbosity`).
   Always use `printpl("gg", ...)` instead of `print()`.
   Legacy globals (`PRINT_DUMMY`, `DEBUG_DUMMY`) exist in `util.py` for
   backwards compat but read from `cfg` at import time.
4. **Benchmarking**: `enable_analysis=False` disables plots/backups/rendering
   during evolution. Without it, IO overhead distorts timing.
5. **`get_sympy_expr()` is slow**: Never call it in hot paths (loops over
   population). Use `str(tree)` or `tree.get_lut_id()` for fast identification.
6. **Physical cores**: `os.cpu_count()` returns logical threads (16 on 8-core
   HT CPU). Use `cpu_count_physical()` from `util.py` for worker counts.
7. **Scale ↔ SymPy round-trip**: `Scale` maps to `sympy.Mul` and is not in
   `d_sym2node`. After any SymPy round-trip, `tree_node_grouping()` must
   run to restore Scale nodes.
8. **`canonicalize_children()` timing**: Must run **after** tree is fully
   built (post-processing only). Never call in `set_childs()` or `__init__`.
   Mutations invalidate ancestor ordering — re-run in `tree_to_candidate()`.
   Sort key, performance, and invalidation trade-offs are open for discussion
   (see `PITFALLS.md` P10).
9. **Tree edit distance modes**: `compute_ted()` supports `"structural"`,
   `"full"`, and `"structural_plus_leaf_diff"` modes via `TedConfig`.
   `eval_parsimony` uses `"structural"`. For diversity, use `"full"`.
   The external `apted` package is **no longer a dependency**.
   `apted_distance()` and `get_apted_notation()` are deprecated.

## Working behaviour

- **Proactive code review**: When working on a task, report any **bugs**,
  **code smells**, or **questionable patterns** discovered along the way —
  even if unrelated to the current task. Include a brief improvement
  suggestion for each finding.
- **Don't silently fix ambiguous findings**: Only fix a discovered bug
  directly if it is **unambiguously wrong** (e.g. missing import, off-by-one,
  typo). If the intent is unclear, the code looks like an open investigation,
  or a `print()`/comment suggests ongoing work — **ask first** or add a
  `# TODO` instead of removing/rewriting it. Debug prints with markers like
  `WHATHAPPENED`, `sfeh`, `# discuss` are investigation aids, not dead code.
- **Raise concerns**: If an approach seems risky, fragile, or
  architecturally problematic, voice the concern explicitly before or
  alongside the implementation.

## Maintaining these docs

- **Module map & Node hierarchy** are auto-updated by a pre-commit hook
  (`scripts/update_copilot_instructions.py`). No manual work needed.
- **When you discover a new pitfall**, add it to `docs/PITFALLS.md` and
  add a one-liner to the list above. Same for new architectural patterns
  → update `docs/ARCHITECTURE.md`.
- **Coding conventions** (linting, type hints, test config) live in
  `pyproject.toml` — do **not** duplicate them here.
