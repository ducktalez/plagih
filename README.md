# plagih — Explainable Genetic Programming Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**plagih** (PLAusible Genetic Improvements to Heuristics) is a genetic programming framework for **explainable AI**. It evolves symbolic expression trees (math / logic / conditional) via selection, mutation, crossover, and simplification, evaluated against training data.

> **Why this project?**  Standard neural networks are black boxes — plagih evolves human-readable formulas (e.g. `sin(x) * (a + b) > 0.5`) that can be directly inspected, debugged, and trusted. It has been applied to control-policy discovery (CartPole, MountainCar), symbolic regression, and a hybrid **NN+GP co-evolution** pipeline where a GP formula feeds feature signals into a downstream PyTorch model.

Key features:
- Tree-based GP with strongly-typed nodes (`float`, `bool`)
- Parallel evaluation via multiprocessing (Windows + Linux)
- SymPy-backed simplification and canonicalisation
- Pareto-front-based selection (fitness x parsimony)
- Built-in monitoring, visualisation, and benchmarks
- Optional **NN+GP co-evolution pipeline** (see `benchmarks/nn_gp/`)
---
## Running in PyCharm

Ready-made **Run Configurations** are committed in `.idea/runConfigurations/` and
appear automatically in PyCharm's run menu (top-right drop-down):

| Configuration | What it does | Approx. time |
|---|---|---|
| **NN+GP – Baseline only** | Trains baseline NN only (`--fast --baseline-only`), sanity check + blueprint | ~2 min |
| **NN+GP – Dev run, fast** | 3 EM iterations, small pop/gen (`--fast`), produces real blueprint | ~10 min |
| **NN+GP – Full run, 3 iterations** | Full-scale run: pop=50, gen=20, 400 epochs per NN | ~1 h |
| **NN+GP – Regenerate blueprint** | Re-renders `PAPER_BLUEPRINT.md` from an existing `experiment.json` | seconds |
| **Plagih GP – Fresh Run** | Runs `plagih_gp.py fresh` — a clean GP demo run | minutes |
| **Plagih GUI** | Opens the desktop monitoring GUI (`python plagih_gp.py gui`) — configure, start, pause and inspect runs interactively | instant |

> **Output** lands in `.results/nn_gp/<timestamp>/` (gitignored).
> Open `PAPER_BLUEPRINT.md` in that folder once the run finishes.

---
## Quick Start
```bash
# 1. Clone and enter the repo
git clone <repo-url> plagih
cd plagih
# 2. Create virtual environment & install
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
# 3. Run the minimal demo
python plagih_gp.py
```
> **LaTeX rendering (optional, Linux):** `sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super`
---
## Project structure
```
plagih/                  # Core framework
benchmarks/              # Reference problems (cp, mc, ib, sr, nn_gp)
docs/                    # Architecture, pitfalls, evaluation, etc.
scripts/                 # Maintenance utilities
.results/                # Local run artefacts (gitignored)
```
---
## Documentation
| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module overview, lifecycle, config reference |
| [`docs/PITFALLS.md`](docs/PITFALLS.md) | Known bugs and gotchas (P1-P25) - **read before editing core code** |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Open tasks, design discussions, ideas backlog |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | EvaluationContext API |
| [`docs/LOGGING.md`](docs/LOGGING.md) | Hybrid logging system |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark environments and data formats |
| [`docs/TARGETED_OPTIMIZATION.md`](docs/TARGETED_OPTIMIZATION.md) | Per-tree pseudo-backpropagation |
---
## Defining your own benchmark
Create `benchmarks/<NAME>/gp_files/samples.csv` with typed headers:
```csv
cartPos:float,cartVel:float,action:float
0.10,0.20,0.30
```
- Feature columns: any `float` / `bool` typed inputs
- Target column: e.g. `action:float`
- Add a `demo_<NAME>()` function in `plagih_gp.py` following existing patterns
- Document it in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
---
## NN+GP co-evolution pipeline
Located in `benchmarks/nn_gp/`. EM-style loop: GP evolves symbolic candidates -> their outputs become PyTorch NN input features -> smaller NN learns the residual logic -> next GP iteration targets the residual. Auto-generates a paper blueprint from a single run.
```bash
python benchmarks/nn_gp/run_mc.py --fast --baseline-only   # ~2 min sanity check
python benchmarks/nn_gp/run_mc.py --fast                   # dev run
python benchmarks/nn_gp/run_mc.py                          # full run
```
Output: `.results/nn_gp/<timestamp>/PAPER_BLUEPRINT.md` with embedded figures.
---
## Development
```bash
pip install -e ".[dev]"
pre-commit install
pytest                            # full test suite
pytest -m "not performance"       # skip slow perf tests
ruff check . && ruff format .     # lint + format
```
Coding conventions live in `pyproject.toml`. Pre-commit hooks auto-update the copilot module map and node hierarchy in `.github/copilot-instructions.md`.
---
## License
MIT - see [`LICENSE.txt`](LICENSE.txt).
