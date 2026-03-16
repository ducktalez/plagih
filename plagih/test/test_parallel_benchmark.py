"""
Performance benchmark tests for parallel vs. sequential execution.

These tests are marked with @pytest.mark.benchmark and are excluded
from normal test runs. Run them explicitly with:
    pytest plagih/test/test_parallel_benchmark.py -m benchmark -v
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def bench_gp_factory():
    """Factory fixture that creates GP instances for benchmarking."""
    created_dirs = []

    def _create(pop_size, parallel):
        temp_dir = Path(tempfile.mkdtemp(prefix="plagih_bench_"))
        created_dirs.append(temp_dir)

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

        return ExplainableGP.create(
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
        )

    yield _create

    for d in created_dirs:
        shutil.rmtree(d, ignore_errors=True)


STRATEGIES = [
    Strategy("reproduction", rate=0.2, tournament_n=3),
    Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3),
    Strategy("random_new", rate=0.2, depths=[2, 3, 4]),
    Strategy("crossover", rate=0.2, crossover=True, tournament_n=3),
]


@pytest.mark.benchmark
class TestParallelBenchmark:
    """Benchmark tests — excluded from normal runs via -m 'not benchmark'."""

    def test_sequential_basic(self, bench_gp_factory):
        """Sequential mode works and produces results."""
        gp = bench_gp_factory(pop_size=20, parallel=False)
        gp.gen_create_initial()
        gp.run_generation(STRATEGIES)
        assert len(gp.paretofront) > 0

    def test_parallel_2_workers(self, bench_gp_factory):
        """Parallel mode with 2 workers works."""
        gp = bench_gp_factory(pop_size=20, parallel=2)
        gp.gen_create_initial()
        gp.run_generation(STRATEGIES)
        assert len(gp.paretofront) > 0

    @pytest.mark.skipif(
        os.cpu_count() is not None and os.cpu_count() < 4,
        reason="Need at least 4 CPUs",
    )
    def test_parallel_4_workers(self, bench_gp_factory):
        """Parallel mode with 4 workers works."""
        gp = bench_gp_factory(pop_size=20, parallel=4)
        gp.gen_create_initial()
        gp.run_generation(STRATEGIES)
        assert len(gp.paretofront) > 0
