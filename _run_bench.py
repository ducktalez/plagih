"""
Benchmark: Sequential vs. Parallel GP execution.
Run directly:  python _run_bench.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Setup path
_root = Path(__file__).resolve().parent
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


def create_gp(pop_size, parallel, temp_dir):
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
        pop_max_size=pop_size,
        gen_end=10,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=parallel,
        verbose=False,
    )


STRATEGIES = [
    Strategy("reproduction", rate=0.15, tournament_n=3),
    Strategy("mutation", rate=0.35, depth_goal=3, p_term=0.3),
    Strategy("mutation_point", rate=0.10, tournament_n=3),
    Strategy("random_new", rate=0.20, depths=[2, 3, 4], p_term=0.1),
    Strategy("crossover", rate=0.20, crossover=True, tournament_n=3),
]

N_GENERATIONS = 3


def run_one(label, pop_size, parallel):
    temp_dir = Path(tempfile.mkdtemp(prefix=f"plagih_bench_{label}_"))
    gp = None
    try:
        gp = create_gp(pop_size, parallel, temp_dir)
        t0 = time.perf_counter()
        gp.gen_create_initial()
        t_init = time.perf_counter() - t0

        gen_times = []
        for _ in range(N_GENERATIONS):
            tg = time.perf_counter()
            gp.run_generation(STRATEGIES)
            gen_times.append(time.perf_counter() - tg)

        t_total = time.perf_counter() - t0
        return {
            "label": label,
            "pop_size": pop_size,
            "parallel": parallel,
            "t_init": t_init,
            "gen_times": gen_times,
            "t_total": t_total,
            "t_avg": np.mean(gen_times),
        }
    finally:
        if gp is not None:
            gp.close()  # explicitly shutdown persistent worker pool
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    cpu_count = os.cpu_count() or 4
    print(f"CPUs: {cpu_count} | Python: {sys.version.split()[0]}")
    print(f"Generations: {N_GENERATIONS}\n")

    configs = [
        ("seq_1000", 10000, False),
        ("par_2w_1000", 1000, 2),
        ("par_4w_1000", 1000, 4),
        ("par_8w_1000", 1000, 8),
    ]

    results = []
    for label, pop, par in configs:
        mode = f"parallel({par}w)" if par else "sequential"
        print(f"  Running {label:<15s} pop={pop:>3d}  {mode:<16s} ...", end="", flush=True)
        try:
            r = run_one(label, pop, par)
            results.append(r)
            print(f"  {r['t_total']:6.1f}s total, {r['t_avg']:6.2f}s avg/gen")
        except Exception as e:
            print(f"  FAILED: {e}")

    # === Summary ===
    print("\n" + "=" * 75)
    print(f"{'Label':<16s} {'Pop':>4s} {'Mode':<14s} {'Init':>6s} {'Avg/Gen':>8s} {'Total':>7s} {'Speedup':>8s}")
    print("-" * 75)

    # Group by pop_size for speedup calculation
    seq_by_pop = {}
    for r in results:
        if r["parallel"] is False:
            seq_by_pop[r["pop_size"]] = r["t_avg"]

    for r in results:
        mode = f"par({r['parallel']}w)" if r["parallel"] else "seq"
        baseline = seq_by_pop.get(r["pop_size"], r["t_avg"])
        speedup = baseline / r["t_avg"] if r["t_avg"] > 0 else 0
        eff = f"({speedup / r['parallel'] * 100:.0f}%eff)" if r["parallel"] else ""
        print(
            f"{r['label']:<16s} {r['pop_size']:>4d} {mode:<14s} "
            f"{r['t_init']:>5.2f}s {r['t_avg']:>7.2f}s "
            f"{r['t_total']:>6.1f}s {speedup:>6.2f}x {eff}"
        )

    print("=" * 75)
