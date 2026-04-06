# plagih — Explainable Genetic Programming

> **PL**ausible **G**enetic **I**mprovements to **H**euristics

plagih is a genetic programming (GP) framework for **explainable AI**.
It evolves symbolic expression trees (math/logic) via selection, mutation,
crossover, and simplification, evaluated against training data.
The algorithm was filed as a patent in Germany in 2023.

## Key Features

- **Tree-based GP** with typed operators (math, logic, relational, conditional)
- **SymPy-powered** symbolic simplification and unification
- **Familiarity metric** — tree edit distance to a reference program
- **Pseudo-backpropagation** for Ifte/Piecewise-focused optimization
- **Parallel evaluation** with shared memory and chunked batching
- **Pareto-front analysis** — fitness vs. complexity trade-off
- **Population merge tree** — unified view of the entire population
- **Look-Up Tables (LUT)** — deduplication of structurally and symbolically
  identical trees for major speedup

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Minimal Example

```bash
python plagih_gp.py
```

This runs a short demo (~30 s) that evolves symbolic expressions for a
toy regression problem. See `plagih_gp.py` → `demo_minimal()` for the
documented source.

For a guided interactive walkthrough, open `docs/demo.ipynb`.

## Operators

| Group | Examples |
|---|---|
| **Math** | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Abs`, `Sign`, `Square`, `Sqrt`, `Log`, `Exp`, `Scale`, `NthRoot`, `Round` |
| **Trigonometry** | `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Tanh` |
| **Min/Max** | `Min`, `Max`, `Clip` |
| **Logic** | `And`, `Or`, `Not`, `Xor` |
| **Relational** | `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` |
| **Conditional** | `Ifte` (if-then-else), `Piecewise` |

Operators are declared in `plagih/trees/_nodes.py` and configured per run via
an operator pool (see `docs/ARCHITECTURE.md`).

## Configuration (`.env`)

All framework defaults are managed via a `.env` file in the project root.
Copy `.env.example` to `.env` and adjust values as needed.

### Minimal Default Profile

The default configuration is intentionally **minimal** — every feature is off:

| Feature | Default | `.env` Key |
|---|---|---|
| SymPy Simplification | off | `PLAGIH_SIMPLIFICATION` |
| Visualisation during runs | off | `PLAGIH_VISUALIZATION` |
| Merged population tree | off | `PLAGIH_MERGED_TREE` |
| Origin tree tracking | off | `PLAGIH_ORIGIN_TREE` |
| Look-Up Tables (LUT) | **on** | `PLAGIH_LUT_ENABLED` |
| Parallelisation | 0 (sequential) | `PLAGIH_PARALLEL` |

### Recommended Profile

```dotenv
PLAGIH_LUT_ENABLED=true
PLAGIH_VISUALIZATION=true
PLAGIH_PARALLEL=4
PLAGIH_SIMPLIFICATION=true
```

### Override Hierarchy

```
.env file  →  environment variables  →  code-level parameters
(lowest)                                  (highest priority)
```

Code-level overrides (e.g. `ExplainableGP.create(parallel=8)`) always win.
See `docs/ARCHITECTURE.md` § Configuration System for full details.

## System Requirements & Benchmarks

The values below are **measured guidelines**, not guarantees.

**Test setup (2026-03-16):** Windows, 8 physical cores / 16 threads, current
parallelisation with pre-selection, shared memory for `df_train`, and chunked
batching. Details: `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md`.

### Measured Reference Values

| Population | Workers | Steady-State / Gen | Init Time | Peak RAM |
|---:|---:|---:|---:|---:|
| 1 000 | 0 | ~4.0 s | ~4.3 s | ~161 MB |
| 1 000 | 4 | ~1.8 s | ~4.7 s | ~750 MB |
| 1 000 | 8 | ~1.3 s | ~4.5 s | ~1.29 GB |
| 10 000 | 0 | ~41.6 s | ~44.4 s | ~290 MB |
| 10 000 | 4 | ~17.1 s | ~44.7 s | ~941 MB |
| 10 000 | 8 | ~11.9 s | ~45.1 s | ~1.49 GB |

> **Note:** Init times above predate the unified Generation-0 task runner.
> Re-benchmark after significant changes to batching, worker init, or SymPy
> handling. See `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md` for the full report.

### Key Observations

- The large RAM jump comes primarily from **worker processes**, not the
  population itself.
- On the benchmark system **8 workers** was the fastest configuration.

### Parameter Influence Matrix

| Parameter | Time Impact | RAM Impact | Parallelisation | Notes |
|---|---|---|---|---|
| `pop_size` | very high | medium–high | positive | Larger populations amortise parallel overhead better |
| `gen_end` | linear | low–medium | neutral | Total runtime grows linearly |
| `parallel` (workers) | low–very positive | very high | direct | More workers speed up the run but increase RAM |
| Batch size / chunking | high | low | very high | Sweet spot ~32–128 tasks per batch |
| Tree complexity (operators, depth) | high | medium | negative | More SymPy/NumPy work per tree |
| `Ifte`/`Piecewise` fraction | high | medium | potentially negative | Can stress SymPy; pathological cases are caught |
| `nodes_max`, `depth_max` | high | medium | mixed | Allow larger trees; increase search space and cost |
| `enable_analysis=True` | high | low–medium | negative | Extra IO/rendering; avoid in benchmarks |

### Practical Presets

| Goal | Recommendation |
|---|---|
| Low RAM, robust | `parallel=4` as a cautious start |
| Maximum throughput (8 physical cores) | `parallel=8` |
| Small populations / quick tests | `parallel=0` or `parallel=2` first |
| Large populations (≥10 000) | Parallelisation pays off significantly |

## Documentation

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module overview, lifecycle, node hierarchy, config reference |
| [`docs/PITFALLS.md`](docs/PITFALLS.md) | Known bugs and gotchas (P1–P20) |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Open tasks, design discussions, ideas backlog |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | EvaluationContext API and examples |
| [`docs/LOGGING.md`](docs/LOGGING.md) | Hybrid logging system (`setup_logging`, `log()`) |
| [`docs/TARGETED_OPTIMIZATION.md`](docs/TARGETED_OPTIMIZATION.md) | Pseudo-backpropagation, node-level optimisation |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark environments and data formats |
| [`docs/demo.ipynb`](docs/demo.ipynb) | Interactive feature demo (7 sections) |

## Known Framework Pitfalls

These are the main "watch out" areas when working with or extending plagih:

- **`Usub`** — unary negation; arguably not a real operator, but deeply embedded
- **`Round`** — requires `RoundDummy` for SymPy compatibility
- **`Min`/`Max`** — SymPy interaction issues with assumptions
- **`ExprCondPair`** in `Piecewise` — complex SymPy bridge behaviour
- **`DivFraction`/`Scale`** — grouping rewrites can expand trees (→ D6, D7)

See `docs/PITFALLS.md` for the full list (P1–P20).

## Compared to DEAP

- DEAP supports non-programming optimisation (array manipulation); plagih is
  **GP-only** (symbolic expression trees).
- Complexity is **node-count based**, not depth-based only.
- Trees are **recursive Node objects**, not nested lists.
- Built-in **SymPy bridge** for symbolic simplification and unification.
- Built-in **familiarity metric** (tree edit distance to reference programs).

## Input Data Format

### `behaviour_samples.csv`

Training data with observations and target actions. Header format:

| cartPos:float | cartVel:float | action0:float |
|---|---|---|
| 0.1 | 0.2 | 0.3 |

- Column names before `:` are the symbol names used in trees
- Types (`float`, `bool`) follow after `:`
- One target column (e.g. `action0`)

### `operators.csv`

One operator per line. Adding an operator multiple times increases its
selection probability. See the **Operators** table above for available choices.

## License

MIT — see [`LICENSE.txt`](LICENSE.txt).
