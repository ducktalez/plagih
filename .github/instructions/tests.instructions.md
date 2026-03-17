---
applyTo: "plagih/test/**"
---

# Tests – Copilot Instructions

- Config in `pyproject.toml`. Root `conftest.py` adds project root to `sys.path`.
- `plagih/test/conftest.py` provides shared fixtures via lazy imports.
- `bench_*.py` in `plagih/test/benchmarks/` are **not** collected by pytest — run directly.
- After deserialization or `deepcopy` in tests, call `repair_all()` (P1).
- New operators → run `test_all_node_classes.py` + `test_visualization.py`.
