# Copilot Instructions – plagih GP Framework

> This file is injected into every AI request.  Keep it **short** and
> limited to knowledge that **cannot** be discovered from source code.
> Auto-generated sections have zero maintenance cost.
> Full details: `docs/ARCHITECTURE.md`, pitfalls: `docs/PITFALLS.md`,
> open tasks: `docs/IMPLEMENTATION_PLAN.md`.

## Project overview

plagih is a **genetic programming (GP)** framework for **explainable AI**.
It evolves symbolic expression trees (math/logic) via selection, mutation,
crossover, and simplification, evaluated against training data.

## Module map

<!-- AUTOGEN:MODULE_MAP:START -->
| Module | Responsibility |
|---|---|
| `plagih/trees/` | plagih.trees — Node hierarchy, evolution, and GP engine. |
| `plagih/trees/_nodes.py` | plagih_tree contain a new implementation of trees that we use in genetic programming to display a program. (62C/11F) |
| `plagih/trees/_evolution.py` | Evolution module: Candidate, NodeSelect, Evolution, and population helpers. (3C/8F) |
| `plagih/trees/_gp_engine.py` | GP Engine module: ExplainableGP and picklable helper callables. (2C/5F) |
| `plagih/parallel.py` | Parallel execution engine for plagih GP. (5C/33F) |
| `plagih/paretofront.py` | Pareto-front dominance filter for GP candidates. (0C/6F) |
| `plagih/monitoring.py` | GP Monitoring Module (2C/0F) |
| `plagih/evaluation_context.py` | Unified Evaluation Context System for Plagih GP Trees. (3C/5F) |
| `plagih/population_merge.py` | Population Merge Module for plagih GP Framework (3C/8F) |
| `plagih/util.py` | *(no docstring)* (1C/14F) |
| `visualization/tree_renderer.py` | *(file not found)* |
| `visualization/visualize_trees.py` | *(file not found)* |
<!-- AUTOGEN:MODULE_MAP:END -->

## Node hierarchy (auto-generated from `trees/_nodes.py`)

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

## Area-specific instructions

Detailed rules live in `.github/instructions/*.instructions.md` (with `applyTo`
frontmatter). They are automatically injected when editing matching files:

| File | Scope |
|---|---|
| `trees.instructions.md` | `plagih/trees/**` — pitfalls, new-operator pointer |
| `parallel.instructions.md` | `plagih/parallel.py` — pickle, IPC, batching |
| `monitoring.instructions.md` | `plagih/monitoring.py` — metrics, callbacks, DataFrame mapping |
| `evaluation-context.instructions.md` | `plagih/evaluation_context.py` — modes, LUT caching |
| `population-merge.instructions.md` | `plagih/population_merge.py` — DAG merge |
| `paretofront.instructions.md` | `plagih/paretofront.py` — dominance filter |
| `config.instructions.md` | `plagih/config.py` — PlagihConfig, verbosity |
| `logging-utils.instructions.md` | `plagih/logging_utils.py` — `log()`, verbosity gating |
| `plagih-gp.instructions.md` | `plagih_gp.py` — entry point, demos |
| `visualization.instructions.md` | `visualization/**` — rendering attributes |
| `tests.instructions.md` | `plagih/test/**` — pytest config, fixtures |
| `tree-complexity.instructions.md` | `plagih/tree_complexity/**` — TED, bytecode |
| `benchmarks.instructions.md` | `benchmarks/**` — environments, samples format |

## Key docs in `docs/`

| Document | Purpose |
|---|---|
| `ARCHITECTURE.md` | Module overview, lifecycle, node hierarchy, config reference |
| `PITFALLS.md` | Known bugs and gotchas (P1–P17). Read before editing core code. |
| `IMPLEMENTATION_PLAN.md` | Central TODO list + **Design Discussions** — add open tasks and architectural questions here, not in source code |
| `EVALUATION.md` | EvaluationContext API and examples |
| `LOGGING.md` | Hybrid logging system (`setup_logging`, `printpl` → `log_*`) |
| `BENCHMARKS.md` | Benchmark environments and data formats |
| `PARALLEL_BENCHMARK_DIAGNOSIS.md` | Performance diagnosis report (reference data) |
| `TARGETED_OPTIMIZATION.md` | Targeted per-tree optimization, pseudo-backpropagation, SoftOptimum |

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
- **Open tasks**: Add new TODOs to `docs/IMPLEMENTATION_PLAN.md` instead of
  writing `# TODO` comments in source code.
- **Design discussions**: Open architectural questions, trade-off decisions,
  and "should we?" debates go into the **Design Discussions** section of
  `docs/IMPLEMENTATION_PLAN.md` (items D1, D2, …). Do **not** embed them
  inline in `PITFALLS.md` or source code — add a cross-reference instead.
- **Raise concerns**: If an approach seems risky, fragile, or
  architecturally problematic, voice the concern explicitly before or
  alongside the implementation.

## Maintaining these docs

- **Module map & Node hierarchy** are auto-updated by a pre-commit hook
  (`scripts/update_copilot_instructions.py`). No manual work needed.
- **When you discover a new pitfall**, add it to `docs/PITFALLS.md`.
- **When you identify an open task**, add it to `docs/IMPLEMENTATION_PLAN.md`.
- **Coding conventions** (linting, type hints, test config) live in
  `pyproject.toml` — do **not** duplicate them here.
