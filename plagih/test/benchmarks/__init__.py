"""
Standalone performance benchmarks for plagih GP framework.

These are NOT pytest tests. They are standalone scripts meant to be run
directly for performance profiling and diagnostics:

    python plagih/test/benchmarks/bench_performance.py
    python plagih/test/benchmarks/bench_parallel.py
    python plagih/test/benchmarks/bench_pickle.py
    python plagih/test/benchmarks/bench_run.py
    python plagih/test/benchmarks/bench_diagnose_parallel.py

They live near the test suite for convenience, but are excluded from
pytest collection via conftest.py collect_ignore and naming convention.
"""
