"""Diagnose: Where exactly is the parallel overhead?"""

import pickle
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from plagih.parallel import BUILTIN_STRATEGIES, Strategy, _init_worker, _worker_run_batch, build_task_list
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


def create_gp(pop_size, temp_dir):
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
    )


def noop_task(x):
    """Trivial task to measure pool overhead."""
    return x * 2


STRATEGIES = [
    Strategy("reproduction", rate=0.15, tournament_n=3),
    Strategy("mutation", rate=0.35, depth_goal=3, p_term=0.3),
    Strategy("mutation_point", rate=0.10, tournament_n=3),
    Strategy("random_new", rate=0.20, depths=[2, 3, 4], p_term=0.1),
    Strategy("crossover", rate=0.20, crossover=True, tournament_n=3),
]


if __name__ == "__main__":
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_diag_"))
    try:
        gp = create_gp(100, temp_dir)
        gp.gen_create_initial()

        print("=== PARALLEL OVERHEAD DIAGNOSIS ===\n")

        # 1. Measure bare pool creation + shutdown (no tasks)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as pool:
            pass
        t_pool_bare = time.perf_counter() - t0
        print(f"1. Bare pool create+shutdown (4w):   {t_pool_bare * 1000:.0f}ms")

        # 2. Measure pool + trivial task
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(noop_task, i) for i in range(4)]
            results = [f.result() for f in futures]
        t_pool_noop = time.perf_counter() - t0
        print(f"2. Pool + 4 noop tasks:              {t_pool_noop * 1000:.0f}ms")

        # 3. Measure pool creation WITH initializer (this pickles evolve, df_train, etc.)
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
        ) as pool:
            pass
        t_pool_init = time.perf_counter() - t0
        print(f"3. Pool + initializer (no tasks):    {t_pool_init * 1000:.0f}ms")

        # 4. Measure pickle size of initargs
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
        data = pickle.dumps(initargs)
        print(f"4. Initargs pickle size:             {len(data):,} bytes ({len(data) / 1024:.0f} KB)")

        # Measure individual component sizes
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
            print(f"   - {label:<22s}: {sz:>8,} bytes")

        # 5. Pool + initializer + actual batch tasks
        tasks = build_task_list(STRATEGIES, 100)
        n_workers = 4
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
            from concurrent.futures import as_completed

            all_results = []
            for f in as_completed(futures):
                all_results.extend(f.result())
            t_done = time.perf_counter() - t0
        t_total = time.perf_counter() - t0

        print("\n5. Full parallel generation (100 pop, 4 workers):")
        print(f"   Pool creation + init:  {t_pool_ready * 1000:.0f}ms")
        print(f"   Submit batches:        {(t_submit - t_pool_ready) * 1000:.0f}ms")
        print(f"   Wait for results:      {(t_done - t_submit) * 1000:.0f}ms")
        print(f"   Pool shutdown:         {(t_total - t_done) * 1000:.0f}ms")
        print(f"   TOTAL:                 {t_total * 1000:.0f}ms")
        print(f"   Successful tasks:      {sum(1 for r in all_results if r.error is None)}")
        print(f"   Failed tasks:          {sum(1 for r in all_results if r.error is not None)}")

        # 6. Sequential comparison
        from plagih.parallel import run_generation_sequential

        tasks2 = build_task_list(STRATEGIES, 100)
        lut_t, lut_s = {}, {}
        t0 = time.perf_counter()
        candidates, tracker = run_generation_sequential(
            tasks=tasks2,
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
        print(f"\n6. Sequential generation:             {t_seq * 1000:.0f}ms")
        print(f"   Successful:            {len(candidates)}")

        print("\n=== SUMMARY ===")
        print(f"Sequential:   {t_seq * 1000:.0f}ms")
        print(
            f"Parallel:     {t_total * 1000:.0f}ms (pool={t_pool_ready * 1000:.0f}ms + work={t_done - t_submit:.0f}s + shutdown={t_total - t_done:.0f}s)"
        )
        print(f"Overhead:     {(t_total - t_seq) * 1000:.0f}ms ({(t_total / t_seq - 1) * 100:.0f}% slower)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
