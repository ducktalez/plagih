"""
Diagnostic benchmark for parallel execution overhead.

Measures pool creation, pickle sizes, initializer cost, and compares
sequential vs. parallel generation times. Useful for profiling and
optimizing the parallel pipeline.

Run directly:
    python plagih/test/benchmarks/bench_diagnose_parallel.py
"""

import os
import pickle
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from plagih.parallel import (
    BUILTIN_STRATEGIES,
    Strategy,
    _init_worker,
    _worker_run_batch,
    build_task_list,
    run_generation_sequential,
)
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

STRATEGIES = [
    Strategy("reproduction", rate=0.15, tournament_n=3),
    Strategy("mutation", rate=0.35, depth_goal=3, p_term=0.3),
    Strategy("mutation_point", rate=0.10, tournament_n=3),
    Strategy("random_new", rate=0.20, depths=[2, 3, 4], p_term=0.1),
    Strategy("crossover", rate=0.20, crossover=True, tournament_n=3),
]


def _create_test_gp(temp_dir):
    """Create a GP instance for diagnostic benchmarks."""
    data_path = _root / "benchmarks" / "mc" / "gp_files" / "samples200.csv"
    if data_path.exists():
        df = pd.read_csv(data_path).astype("float32")
    else:
        np.random.seed(42)
        n = 2000
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
        pop_max_size=100,
        gen_end=5,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=False,
        verbose=False,
        enable_analysis=False,
    )


def _noop_task(x):
    """Trivial task to measure pool overhead."""
    return x * 2


def _create_diag_gp():
    """Create a GP instance with initial population for diagnostics. Returns (gp, temp_dir)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_diag_"))
    gp = _create_test_gp(temp_dir)
    gp.gen_create_initial()
    return gp, temp_dir


def bench_bare_pool_creation():
    """Measure bare ProcessPoolExecutor creation + shutdown."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4):
        pass
    elapsed = time.perf_counter() - t0
    print(f"\nBare pool create+shutdown (4w): {elapsed * 1000:.0f}ms")


def bench_pool_with_noop_tasks():
    """Measure pool + trivial tasks."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_noop_task, i) for i in range(4)]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0
    print(f"\nPool + 4 noop tasks: {elapsed * 1000:.0f}ms")
    assert results == [0, 2, 4, 6]


def bench_pool_with_initializer(gp):
    """Measure pool creation with GP initializer (pickles evolve, df_train, etc.)."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=_init_worker,
        initargs=(
            gp.evolve,
            gp.df_train,
            gp.pop_genepool,
            gp.paretofront,
            gp.eval_autocast,
            gp.eval_error_metric,
            gp.allow_chain,
            gp.target_column,
            gp.evolve.nodes_max,
            gp.evolve.complexity_metric,
            {**BUILTIN_STRATEGIES},
        ),
    ):
        pass
    elapsed = time.perf_counter() - t0
    print(f"\nPool + initializer (no tasks): {elapsed * 1000:.0f}ms")


def bench_initargs_pickle_size(gp):
    """Measure pickle size of each component sent to workers."""
    initargs = (
        gp.evolve,
        gp.df_train,
        gp.pop_genepool,
        gp.paretofront,
        gp.eval_autocast,
        gp.eval_error_metric,
        gp.allow_chain,
        gp.target_column,
        gp.evolve.nodes_max,
        gp.evolve.complexity_metric,
        {**BUILTIN_STRATEGIES},
    )
    total = len(pickle.dumps(initargs))
    print(f"\nInitargs total pickle size: {total:,} bytes ({total / 1024:.0f} KB)")

    labels = [
        "evolve",
        "df_train",
        "pop_genepool",
        "paretofront",
        "eval_autocast",
        "eval_error_metric",
        "allow_chain",
        "target_column",
        "nodes_max",
        "complexity_metric",
        "strategy_registry",
    ]
    for label, obj in zip(labels, initargs):
        sz = len(pickle.dumps(obj))
        print(f"  - {label:<22s}: {sz:>8,} bytes")


def bench_full_parallel_generation(gp):
    """Measure full parallel generation with timing breakdown."""
    n_workers = min(4, os.cpu_count() or 4)
    tasks = build_task_list(STRATEGIES, 100)

    batches = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        batches[i % n_workers].append(task)
    batches = [b for b in batches if b]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            gp.evolve,
            gp.df_train,
            gp.pop_genepool,
            gp.paretofront,
            gp.eval_autocast,
            gp.eval_error_metric,
            gp.allow_chain,
            gp.target_column,
            gp.evolve.nodes_max,
            gp.evolve.complexity_metric,
            {**BUILTIN_STRATEGIES},
        ),
    ) as pool:
        t_pool_ready = time.perf_counter() - t0
        futures = {pool.submit(_worker_run_batch, batch): batch for batch in batches}
        t_submit = time.perf_counter() - t0
        all_results = []
        for f in as_completed(futures):
            all_results.extend(f.result())
        t_done = time.perf_counter() - t0
    t_total = time.perf_counter() - t0

    n_ok = sum(1 for r in all_results if r.error is None)
    n_err = sum(1 for r in all_results if r.error is not None)

    print(f"\nFull parallel generation (100 pop, {n_workers} workers):")
    print(f"  Pool creation + init:  {t_pool_ready * 1000:.0f}ms")
    print(f"  Submit batches:        {(t_submit - t_pool_ready) * 1000:.0f}ms")
    print(f"  Wait for results:      {(t_done - t_submit) * 1000:.0f}ms")
    print(f"  Pool shutdown:         {(t_total - t_done) * 1000:.0f}ms")
    print(f"  TOTAL:                 {t_total * 1000:.0f}ms")
    print(f"  Successful: {n_ok}, Failed: {n_err}")
    assert n_ok > 0


def bench_sequential_vs_parallel_comparison(gp):
    """Compare sequential and parallel generation times."""
    n_workers = min(4, os.cpu_count() or 4)

    # Sequential
    tasks_seq = build_task_list(STRATEGIES, 100)
    lut_t, lut_s = {}, {}
    t0 = time.perf_counter()
    candidates, _tracker = run_generation_sequential(
        tasks=tasks_seq,
        evolve=gp.evolve,
        df_train=gp.df_train,
        pop_genepool=gp.pop_genepool,
        paretofront=gp.paretofront,
        eval_autocast=gp.eval_autocast,
        eval_error_metric=gp.eval_error_metric,
        allow_chain=gp.allow_chain,
        target_column=gp.target_column,
        nodes_max=gp.evolve.nodes_max,
        complexity_metric=gp.evolve.complexity_metric,
        strategy_registry={**BUILTIN_STRATEGIES},
        lut_tree=lut_t,
        lut_symex=lut_s,
    )
    t_seq = time.perf_counter() - t0

    # Parallel
    tasks_par = build_task_list(STRATEGIES, 100)
    batches = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks_par):
        batches[i % n_workers].append(task)
    batches = [b for b in batches if b]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            gp.evolve,
            gp.df_train,
            gp.pop_genepool,
            gp.paretofront,
            gp.eval_autocast,
            gp.eval_error_metric,
            gp.allow_chain,
            gp.target_column,
            gp.evolve.nodes_max,
            gp.evolve.complexity_metric,
            {**BUILTIN_STRATEGIES},
        ),
    ) as pool:
        futures = {pool.submit(_worker_run_batch, batch): batch for batch in batches}
        all_results = []
        for f in as_completed(futures):
            all_results.extend(f.result())
    t_par = time.perf_counter() - t0

    overhead_ms = (t_par - t_seq) * 1000
    ratio = t_par / t_seq if t_seq > 0 else float("inf")

    print(f"\nSequential:  {t_seq * 1000:.0f}ms ({len(candidates)} candidates)")
    print(f"Parallel:    {t_par * 1000:.0f}ms ({n_workers} workers)")
    print(f"Overhead:    {overhead_ms:.0f}ms ({(ratio - 1) * 100:+.0f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("plagih GP — Parallel Overhead Diagnostics")
    print("=" * 60)

    bench_bare_pool_creation()
    bench_pool_with_noop_tasks()

    # Create shared GP instance for diagnostics that need it
    _gp, _temp_dir = _create_diag_gp()
    try:
        bench_pool_with_initializer(_gp)
        bench_initargs_pickle_size(_gp)
        bench_full_parallel_generation(_gp)
        bench_sequential_vs_parallel_comparison(_gp)
    finally:
        _gp.close()
        shutil.rmtree(_temp_dir, ignore_errors=True)
