---
applyTo: "plagih/test/**"
---

# Tests – Copilot Instructions

- Config in `pyproject.toml`. Root `conftest.py` adds project root to `sys.path`.
- `plagih/test/conftest.py` provides shared fixtures via lazy imports and
  sets `QT_QPA_PLATFORM=offscreen` so Qt-based GUI tests work headless.
- `bench_*.py` in `plagih/test/benchmarks/` are **not** collected by pytest — run directly.
- After deserialization or `deepcopy` in tests, call `repair_all()` (P1).
- New operators → run `test_all_node_classes.py` + `test_visualization.py`.
- GUI tests live in `test_gui_core.py` (Qt-free) and `test_gui_desktop.py`
  (uses PySide6 via `pytest.importorskip`; auto-skipped if PySide6 missing).
  Always wrap transient widgets in the `keep_alive` fixture to avoid
  Matplotlib's "Internal C++ object already deleted" race.
- When touching `plagih/config.py`, `plagih/monitoring.py`,
  `plagih/trees/_gp_engine.py`, or anything under `plagih/gui/**`,
  also run `pytest plagih/test/test_gui_*.py` to catch GUI breakage.
