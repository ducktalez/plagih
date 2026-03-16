"""
Performance benchmark for parallel vs. sequential execution.

Compares sequential and parallel GP runs with different worker counts.

Run directly:
    python plagih/test/benchmarks/bench_parallel.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from plagih.parallel import Strategy
from plagih.trees import (
    Abs,
    Add,
    And,
    Cos,
    Div,
    ExplainableGP,
    Ifte,
    Le,
    Lt,
    Max,
    Min,
    Mul,
    Not,
    Or,
    Sin,
    Square,
    Sub,
)


def _create_bench_gp(pop_size, parallel):
    """Factory that creates a GP instance for benchmarking. Returns (gp, temp_dir)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_bench_"))

    data_path = _root / "benchmarks" / "mc" / "gp_files" / "samples200.csv"
    if data_path.exists():
        df = pd.read_csv(data_path).astype("float32")
    else:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame(
            {
                "cartPos": np.random.uniform(-1.5, 1.5, n).astype("float32"),
                "cartVel": np.random.uniform(-2.0, 2.0, n).astype("float32"),
                "action": np.random.choice([0.0, 1.0, 2.0], n).astype("float32"),
            }
        )

    gp = ExplainableGP.create(
        symbols=["cartPos", "cartVel"],
        df_train=df,
        rootdir=temp_dir,
        operators={
            Add: 2,
            Mul: 2,
            Div: 1,
            Sub: 1,
            Abs: 1,
            Square: 1,
            Sin: 0.5,
            Cos: 0.5,
            Min: 1,
            Max: 1,
            Lt: 1,
            Le: 1,
            And: 1,
            Or: 1,
            Not: 1,
            Ifte: 1,
        },
        depth_max=5,
        nodes_max=25,
        pop_max_size=pop_size,
        gen_end=5,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=parallel,
        verbose=False,
        enable_analysis=False,
    )
    return gp, temp_dir


STRATEGIES = [
    Strategy("reproduction", rate=0.2, tournament_n=3),
    Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3),
    Strategy("random_new", rate=0.2, depths=[2, 3, 4]),
    Strategy("crossover", rate=0.2, crossover=True, tournament_n=3),
]


def _run_and_report(label, pop_size, parallel):
    """Run a benchmark configuration, report results, and clean up."""
    gp, temp_dir = _create_bench_gp(pop_size, parallel)
    try:
        t0 = time.perf_counter()
        gp.gen_create_initial()
        gp.run_generation(STRATEGIES)
        elapsed = time.perf_counter() - t0
        pf_size = len(gp.paretofront)
        print(f"  {label:<30s}  {elapsed:6.2f}s  pareto={pf_size}")
        assert pf_size > 0, f"{label}: empty Pareto front"
    finally:
        gp.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def bench_sequential_basic():
    """Sequential mode works and produces results."""
    _run_and_report("sequential (pop=20)", 1000, False)


def bench_parallel_2_workers():
    """Parallel mode with 2 workers works."""
    _run_and_report("parallel 2w (pop=20)", 1000, 2)


def bench_parallel_4_workers():
    """Parallel mode with 4 workers works."""
    if os.cpu_count() is not None and os.cpu_count() < 4:
        print("  Skipping 4-worker benchmark: need at least 4 CPUs")
        return
    _run_and_report("parallel 4w (pop=20)", 1000, 4)


if __name__ == "__main__":
    print("=" * 60)
    print("plagih GP — Parallel vs. Sequential Benchmark")
    print("=" * 60)
    bench_sequential_basic()
    bench_parallel_2_workers()
    bench_parallel_4_workers()
