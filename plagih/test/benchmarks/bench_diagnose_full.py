"""
Full diagnostic benchmark: Sequential vs. Parallel GP performance.

Consolidates all benchmarks into a single executable that measures every
component of the parallel pipeline to identify bottlenecks:

  1. Pickle costs          - size and serialization time of GP state
  2. IPC overhead           - legacy vs pre-selection comparison
  3. Per-task compute time  - how much actual work each task does
  4. Result pickle cost     - worker -> main result transfer
  5. Real pool comparison   - legacy _update_worker_state vs pre-selection
  6. End-to-end comparison  - sequential vs. parallel with timing breakdown
  7. Overhead budget        - theoretical speedup analysis

Run directly:
    python plagih/test/benchmarks/bench_diagnose_full.py

Expects: enable_analysis=False (no plots/backups during benchmarks).
"""

import argparse
import atexit
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bench_output.txt"

import numpy as np
import pandas as pd

from plagih.parallel import (
    BUILTIN_STRATEGIES,
    Strategy,
    _init_worker,
    _update_worker_state,
    _worker_run_batch,
    build_task_list,
    pre_select_for_tasks,
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
    _warmup_noop,
)
from plagih.util import cpu_count_physical

# --- Configuration -----------------------------------------------------------

POP_SIZE = 1000
N_GENERATIONS = 5
COMPARE_POP_SIZES = (1000, 10_000)
BATCH_TASK_SIZES = (1, 32, 128, 0)  # 0 = auto-balanced (~tasks / worker)

STRATEGIES = [
    Strategy("reproduction", rate=0.15, tournament_n=3),
    Strategy("mutation", rate=0.35, depth_goal=3, p_term=0.3),
    Strategy("mutation_point", rate=0.10, tournament_n=3),
    Strategy("random_new", rate=0.20, depths=[2, 3, 4], p_term=0.1),
    Strategy("crossover", rate=0.20, crossover=True, tournament_n=3),
]


# --- Helpers -----------------------------------------------------------------


def _load_df():
    data_path = _root / "benchmarks" / "mc" / "gp_files" / "samples200.csv"
    if data_path.exists():
        return pd.read_csv(data_path).astype("float32")
    np.random.seed(42)
    n = 2000
    return pd.DataFrame(
        {
            "cartPos": np.random.uniform(-1.5, 1.5, n).astype("float32"),
            "cartVel": np.random.uniform(-2.0, 2.0, n).astype("float32"),
            "action": np.random.choice([0.0, 1.0, 2.0], n).astype("float32"),
        }
    )


def _create_gp(pop_size=POP_SIZE, parallel: bool | int = False):
    """Create GP instance, returns (gp, temp_dir)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_diag_"))
    gp = ExplainableGP.create(
        symbols=["cartPos", "cartVel"],
        df_train=_load_df(),
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
        enable_analysis=False,
    )
    return gp, temp_dir


def _fmt_bytes(n):
    if n < 1024:
        return f"{n} B"
    elif n < 1024**2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024**2:.1f} MB"


def _fmt_ms(seconds):
    return f"{seconds * 1000:.1f}ms"


def _section(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def _unique_ints(values):
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _auto_batch_size(n_tasks, n_workers):
    return max(1, math.ceil(n_tasks / max(1, n_workers)))


def _normalize_batch_sizes(batch_sizes, n_tasks, n_workers):
    auto_size = _auto_batch_size(n_tasks, n_workers)
    normalized = []
    seen = set()
    for raw in batch_sizes:
        size = auto_size if raw <= 0 else min(raw, n_tasks)
        if size not in seen:
            seen.add(size)
            label = f"auto({size})" if raw <= 0 else str(size)
            normalized.append((size, label))
    if auto_size not in seen:
        normalized.append((auto_size, f"auto({auto_size})"))
    return normalized


def _build_mixed_batches(tasks, target_batch_size):
    if not tasks:
        return []
    batch_count = max(1, math.ceil(len(tasks) / max(1, target_batch_size)))
    batches = [[] for _ in range(batch_count)]
    for i, task in enumerate(tasks):
        batches[i % batch_count].append(task)
    return [batch for batch in batches if batch]


def _build_preselected_tasks(gp, pop_size, seed):
    random.seed(seed)
    np.random.seed(seed % (2**31))
    tasks = build_task_list(STRATEGIES, pop_size, seed=seed)
    needs_full_pop = pre_select_for_tasks(tasks, gp.pop_genepool, gp.paretofront)
    if needs_full_pop:
        raise RuntimeError("Benchmark erwartet nur builtin strategies mit erfolgreicher pre-selection.")
    return tasks


_bench_df_train = None
_bench_df_shm = None


def _init_df_pickle_worker(df_train):
    global _bench_df_train, _bench_df_shm
    _bench_df_train = df_train
    _bench_df_shm = None


def _init_df_shm_worker(shm_name, shape, dtype_str, columns):
    global _bench_df_train, _bench_df_shm
    _bench_df_shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(_bench_df_shm.close)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_bench_df_shm.buf)
    _bench_df_train = pd.DataFrame(arr, columns=list(columns), copy=False)


def _probe_df_train_worker():
    arr = _bench_df_train.to_numpy(copy=False)
    checksum = float(arr.sum(dtype=np.float64) + arr[0, 0] + arr[-1, -1])
    return checksum, tuple(_bench_df_train.shape)


def _create_df_shared_memory(df_train):
    arr = np.ascontiguousarray(df_train.to_numpy(copy=True))
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    metadata = (shm.name, arr.shape, arr.dtype.str, tuple(df_train.columns))
    return shm, metadata, arr.nbytes


def _measure_pool_startup(initializer, initargs, n_workers, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers, initializer=initializer, initargs=initargs) as pool:
            futures = [pool.submit(_probe_df_train_worker) for _ in range(n_workers)]
            for future in futures:
                future.result()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# --- 1. Pickle Size Analysis ------------------------------------------------


def diag_pickle_sizes(gp, pop_size=POP_SIZE):
    """Measure pickle size of each component sent via IPC."""
    _section("1. PICKLE SIZE ANALYSIS")

    components = {
        "evolve": gp.evolve,
        "df_train": gp.df_train,
        "pop_genepool": gp.pop_genepool,
        "paretofront": gp.paretofront,
        "eval_autocast": gp.eval_autocast,
        "eval_error_metric": gp.eval_error_metric,
        "strategy_registry": {**BUILTIN_STRATEGIES},
    }

    total = 0
    for name, obj in components.items():
        sz = len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        total += sz
        extra = ""
        if name == "pop_genepool":
            n = len(obj)
            per_cand = sz / n if n else 0
            extra = f"  ({n} candidates, {_fmt_bytes(int(per_cand))}/candidate)"
        elif name == "df_train":
            extra = f"  ({len(obj)} rows x {len(obj.columns)} cols)"
        print(f"  {name:<22s}: {_fmt_bytes(sz):>10s}{extra}")

    print(f"  {'TOTAL':<22s}: {_fmt_bytes(total):>10s}")

    # Legacy payload
    update_payload_sz = len(pickle.dumps((gp.pop_genepool, gp.paretofront), protocol=pickle.HIGHEST_PROTOCOL))
    print(f"\n  Legacy _update_worker_state payload: {_fmt_bytes(update_payload_sz)}")

    # Pre-selected batch payload
    tasks = build_task_list(STRATEGIES, pop_size)
    pre_select_for_tasks(tasks, gp.pop_genepool, gp.paretofront)
    n_workers = min(4, cpu_count_physical())
    batches = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        batches[i % n_workers].append(task)
    total_batch_sz = sum(len(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL)) for b in batches if b)
    print(f"  Pre-selected batches ({n_workers}w): {_fmt_bytes(total_batch_sz)} total")
    ratio = (update_payload_sz * n_workers) / total_batch_sz if total_batch_sz > 0 else 0
    print(f"  Savings vs legacy x{n_workers}w: {ratio:.1f}x less IPC data")

    return update_payload_sz


# --- 2. df_train Transport: Pickle vs Shared Memory --------------------------


def diag_df_train_transport(df_train, n_workers=None):
    """Compare df_train transfer via pickle vs shared memory."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    _section(f"2. DF_TRAIN TRANSPORT (pickle vs shared memory, {n_workers} workers)")

    N = 25
    payload = pickle.dumps(df_train, protocol=pickle.HIGHEST_PROTOCOL)

    t0 = time.perf_counter()
    for _ in range(N):
        payload = pickle.dumps(df_train, protocol=pickle.HIGHEST_PROTOCOL)
    t_pickle_dumps = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        restored = pickle.loads(payload)
        _ = restored.iloc[0, 0]
    t_pickle_loads = (time.perf_counter() - t0) / N

    setup_times = []
    shm_nbytes = 0
    shm_metadata = None
    for _ in range(5):
        t0 = time.perf_counter()
        shm, shm_metadata, shm_nbytes = _create_df_shared_memory(df_train)
        setup_times.append(time.perf_counter() - t0)
        shm.close()
        shm.unlink()
    t_shm_setup = float(np.mean(setup_times))

    shm, shm_metadata, shm_nbytes = _create_df_shared_memory(df_train)
    try:
        metadata_payload = pickle.dumps(shm_metadata, protocol=pickle.HIGHEST_PROTOCOL)
        t_pool_pickle = _measure_pool_startup(_init_df_pickle_worker, (df_train,), n_workers)
        t_pool_shm = _measure_pool_startup(_init_df_shm_worker, shm_metadata, n_workers)
    finally:
        shm.close()
        shm.unlink()

    print(f"  DataFrame pickle payload:   {_fmt_bytes(len(payload))}")
    print(f"  Shared-memory raw buffer:   {_fmt_bytes(shm_nbytes)}")
    print(f"  Shared-memory metadata:     {_fmt_bytes(len(metadata_payload))}")
    print()
    print("  Local serialization cost:")
    print(f"    pickle.dumps(df_train):   {_fmt_ms(t_pickle_dumps)}")
    print(f"    pickle.loads(df_train):   {_fmt_ms(t_pickle_loads)}")
    print(f"    shm setup+copy (parent):  {_fmt_ms(t_shm_setup)}")
    print()
    print("  Real pool startup (includes worker init + first probe):")
    print(f"    Pickled DataFrame init:   {_fmt_ms(t_pool_pickle)}")
    print(f"    Shared-memory attach:     {_fmt_ms(t_pool_shm)}")
    print(f"    Shared-memory total*:     {_fmt_ms(t_pool_shm + t_shm_setup)}")
    if t_pool_shm > 0:
        print(f"    Startup speedup:          {t_pool_pickle / t_pool_shm:.2f}x")
    print("    * includes one-time parent-side shm allocation+copy")

    return {
        "pickle_payload": len(payload),
        "shm_buffer": shm_nbytes,
        "shm_metadata": len(metadata_payload),
        "pickle_init": t_pool_pickle,
        "shm_init": t_pool_shm,
        "shm_total": t_pool_shm + t_shm_setup,
    }


# --- 3. Legacy vs Pre-Selection IPC Cost ------------------------------------


def diag_ipc_comparison(gp, pop_size=POP_SIZE):
    """Compare legacy _update_worker_state vs pre-selection IPC cost."""
    _section("3. LEGACY vs PRE-SELECTION IPC COST")

    n_workers = min(4, cpu_count_physical())
    N = 10

    # Legacy: pickle full pop_genepool to each worker
    payload = (gp.pop_genepool, gp.paretofront)
    data = b""
    t0 = time.perf_counter()
    for _ in range(N):
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    t_dumps_legacy = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        pop = pickle.loads(data)
        for c in pop[0]:
            c.tree.repair_all()
    t_loads_legacy = (time.perf_counter() - t0) / N

    legacy_per_worker = t_dumps_legacy + t_loads_legacy
    legacy_total = legacy_per_worker * n_workers

    print("  LEGACY _update_worker_state (per worker):")
    print(f"    dumps: {_fmt_ms(t_dumps_legacy)}, loads+repair: {_fmt_ms(t_loads_legacy)}")
    print(f"    x {n_workers} workers: {_fmt_ms(legacy_total)}")

    # New: pre-select in main process
    t0 = time.perf_counter()
    for _ in range(N):
        tasks_copy = build_task_list(STRATEGIES, pop_size)
        pre_select_for_tasks(tasks_copy, gp.pop_genepool, gp.paretofront)
    t_preselect = (time.perf_counter() - t0) / N

    # Batch pickle cost
    tasks = build_task_list(STRATEGIES, pop_size)
    pre_select_for_tasks(tasks, gp.pop_genepool, gp.paretofront)
    batches = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        batches[i % n_workers].append(task)
    batches = [b for b in batches if b]

    t0 = time.perf_counter()
    for _ in range(N):
        for batch in batches:
            pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
    t_dumps_batches = (time.perf_counter() - t0) / N

    new_total = t_preselect + t_dumps_batches

    print("\n  PRE-SELECTION (new):")
    print(f"    pre_select_for_tasks: {_fmt_ms(t_preselect)}")
    print(f"    batch dumps ({n_workers} batches): {_fmt_ms(t_dumps_batches)}")
    print(f"    TOTAL: {_fmt_ms(new_total)}")

    print("\n  COMPARISON:")
    print(f"    Legacy:        {_fmt_ms(legacy_total)}")
    print(f"    Pre-selection: {_fmt_ms(new_total)}")
    if new_total > 0:
        print(f"    Speedup:       {legacy_total / new_total:.1f}x")

    return legacy_total, new_total


# --- 4. Per-Task Compute Time -----------------------------------------------


def diag_per_task_compute(gp, pop_size=POP_SIZE):
    """Measure average compute time per task (sequential, no IPC)."""
    _section("4. PER-TASK COMPUTE TIME (sequential, no overhead)")

    tasks = build_task_list(STRATEGIES, pop_size)
    lut_tree, lut_symex = {}, {}

    t0 = time.perf_counter()
    candidates, tracker = run_generation_sequential(
        tasks=tasks,
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
        lut_tree=lut_tree,
        lut_symex=lut_symex,
    )
    t_total = time.perf_counter() - t0

    n_tasks = len(tasks)
    n_ok = len(candidates)
    t_per_task = t_total / n_tasks if n_tasks else 0

    print(f"  Total tasks:     {n_tasks}")
    print(f"  Successful:      {n_ok}")
    print(f"  Total time:      {_fmt_ms(t_total)}")
    print(f"  Avg per task:    {_fmt_ms(t_per_task)}")
    print(f"  LUT cache hits:  tree={len(lut_tree)}, symex={len(lut_symex)}")

    summary = tracker.summary()
    print("\n  Per-strategy breakdown:")
    for tag in set(t.tag for t in tasks):
        avg = summary.get(f"strategy_{tag}_avg_time", 0)
        ok = summary.get(f"strategy_{tag}_success", 0)
        fail = summary.get(f"strategy_{tag}_fails", 0)
        print(f"    {tag:<20s}: avg={_fmt_ms(avg):>8s}  ok={ok:>4d}  fail={fail:>4d}")

    return t_total, t_per_task


# --- 5. Batch Size Sweep -----------------------------------------------------


def diag_batch_size_sweep(gp, pop_size=POP_SIZE, n_workers=None, batch_sizes=BATCH_TASK_SIZES, print_header=True):
    """Compare actual worker wall-time for different task batch sizes."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())

    n_tasks = len(build_task_list(STRATEGIES, pop_size))
    configs = _normalize_batch_sizes(batch_sizes, n_tasks, n_workers)

    if print_header:
        _section(f"5. BATCH SIZE SWEEP (pop={pop_size}, {n_workers} workers)")

    sweep_results = []
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            gp.evolve,
            gp.df_train,
            [],
            [],
            gp.eval_autocast,
            gp.eval_error_metric,
            gp.allow_chain,
            gp.target_column,
            gp.evolve.nodes_max,
            gp.evolve.complexity_metric,
            {**BUILTIN_STRATEGIES},
        ),
    )
    try:
        for future in [pool.submit(_warmup_noop) for _ in range(n_workers)]:
            future.result()

        print(f"  Total tasks: {n_tasks}")
        print(f"  {'Batch':<12s} {'#batches':>9s} {'Payload':>10s} {'Avg time':>10s} {'Speedup':>9s}")
        print(f"  {'-' * 58}")

        for cfg_idx, (batch_size, label) in enumerate(configs):
            times = []
            payload_sizes = []
            batch_counts = []

            for rep in range(2):
                tasks = _build_preselected_tasks(gp, pop_size, seed=pop_size * 10 + cfg_idx * 100 + rep)
                batches = _build_mixed_batches(tasks, batch_size)
                payload_sizes.append(
                    sum(len(pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)) for batch in batches)
                )
                batch_counts.append(len(batches))

                t0 = time.perf_counter()
                futures = {pool.submit(_worker_run_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    future.result()
                times.append(time.perf_counter() - t0)

            sweep_results.append(
                {
                    "batch_size": batch_size,
                    "label": label,
                    "avg_time": float(np.mean(times)),
                    "avg_payload": int(np.mean(payload_sizes)),
                    "avg_batches": float(np.mean(batch_counts)),
                }
            )

        baseline = sweep_results[0]["avg_time"] if sweep_results else 0
        for row in sweep_results:
            speedup = baseline / row["avg_time"] if row["avg_time"] > 0 else 0
            print(
                f"  {row['label']:<12s} {round(row['avg_batches']):>9d} "
                f"{_fmt_bytes(row['avg_payload']):>10s} {_fmt_ms(row['avg_time']):>10s} {speedup:>8.2f}x"
            )
    finally:
        pool.shutdown(wait=True)

    return sweep_results


def diag_population_batch_comparison(pop_sizes=COMPARE_POP_SIZES, batch_sizes=BATCH_TASK_SIZES, n_workers=None):
    """Compare batch-size effects side-by-side for pop=1k vs 10k."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    pop_label = " vs ".join(f"{pop:,}" for pop in pop_sizes)
    _section(f"6. BATCH + POPULATION COMPARISON ({pop_label})")

    all_results = {}
    for pop_size in pop_sizes:
        gp, temp_dir = _create_gp(pop_size)
        try:
            gp.gen_create_initial()
            print(f"\n  pop={pop_size}")
            all_results[pop_size] = diag_batch_size_sweep(
                gp,
                pop_size=pop_size,
                n_workers=n_workers,
                batch_sizes=batch_sizes,
                print_header=False,
            )
        finally:
            gp.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n  SUMMARY (smallest batch vs auto-balanced):")
    print(f"  {'Pop':>8s} {'Batch=1':>10s} {'Auto':>10s} {'Speedup':>9s} {'Payload Δ':>12s}")
    print(f"  {'-' * 57}")
    for pop_size in pop_sizes:
        rows = all_results[pop_size]
        baseline = min(rows, key=lambda row: row["batch_size"])
        auto_row = max(rows, key=lambda row: row["batch_size"])
        speedup = baseline["avg_time"] / auto_row["avg_time"] if auto_row["avg_time"] > 0 else 0
        payload_ratio = baseline["avg_payload"] / auto_row["avg_payload"] if auto_row["avg_payload"] > 0 else 0
        print(
            f"  {pop_size:>8d} {_fmt_ms(baseline['avg_time']):>10s} {_fmt_ms(auto_row['avg_time']):>10s} "
            f"{speedup:>8.2f}x {payload_ratio:>11.2f}x"
        )

    return all_results


# --- 7. Result Pickle Cost --------------------------------------------------


def diag_result_pickle_cost(gp, pop_size=POP_SIZE, batch_sizes=BATCH_TASK_SIZES, n_workers=None):
    """Measure cost of pickling TaskResults back from workers."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    _section("7. RESULT PICKLE COST (worker -> main)")

    tasks = build_task_list(STRATEGIES, 100)
    lut_tree, lut_symex = {}, {}

    from plagih.parallel import run_task_sequential

    results = []
    for task in tasks[:50]:
        r = run_task_sequential(
            task=task,
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
            lut_tree=lut_tree,
            lut_symex=lut_symex,
        )
        results.append(r)

    ok_results = [r for r in results if r.error is None]
    if not ok_results:
        print("  No successful results to measure.")
        return 0

    approx_results_per_worker = max(1, pop_size // max(1, n_workers))
    result_batch_sizes = _normalize_batch_sizes(batch_sizes, approx_results_per_worker, n_workers)

    print(f"  {'Batch':<12s} {'Payload':>10s} {'dumps':>10s}")
    print(f"  {'-' * 36}")
    measurements = []
    for batch_size, label in result_batch_sizes:
        batch = ok_results * (batch_size // len(ok_results) + 1)
        batch = batch[:batch_size]
        sz_batch = len(pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL))
        t0 = time.perf_counter()
        for _ in range(10):
            pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
        t_batch = (time.perf_counter() - t0) / 10
        measurements.append((label, sz_batch, t_batch))
        print(f"  {label:<12s} {_fmt_bytes(sz_batch):>10s} {_fmt_ms(t_batch):>10s}")

    return measurements


# --- 8. Real Pool Comparison ------------------------------------------------


def diag_real_pool_comparison(gp, pop_size=POP_SIZE, n_workers=None):
    """Compare real pool: legacy _update_worker_state vs pre-selection batch."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    _section(f"8. REAL POOL COMPARISON ({n_workers} workers)")

    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            gp.evolve,
            gp.df_train,
            [],
            [],
            gp.eval_autocast,
            gp.eval_error_metric,
            gp.allow_chain,
            gp.target_column,
            gp.evolve.nodes_max,
            gp.evolve.complexity_metric,
            {**BUILTIN_STRATEGIES},
        ),
    )
    for f in [pool.submit(_warmup_noop) for _ in range(n_workers)]:
        f.result()

    N = 3

    # Legacy: _update_worker_state only (no work)
    times_legacy = []
    for _ in range(N):
        t0 = time.perf_counter()
        futs = [pool.submit(_update_worker_state, gp.pop_genepool, gp.paretofront) for _ in range(n_workers)]
        for f in futs:
            f.result()
        times_legacy.append(time.perf_counter() - t0)

    # Pre-selection: pre-select + submit batch with work
    times_presel = []
    for _ in range(N):
        tasks = build_task_list(STRATEGIES, pop_size)
        t_ps = time.perf_counter()
        pre_select_for_tasks(tasks, gp.pop_genepool, gp.paretofront)
        batches = [[] for _ in range(n_workers)]
        for i, task in enumerate(tasks):
            batches[i % n_workers].append(task)
        batches = [b for b in batches if b]

        t0 = time.perf_counter()
        futs = {pool.submit(_worker_run_batch, batch): batch for batch in batches}
        for f in as_completed(futs):
            f.result()
        times_presel.append(time.perf_counter() - t_ps)  # includes pre-select time

    pool.shutdown(wait=True)

    avg_legacy = np.mean(times_legacy)
    avg_presel = np.mean(times_presel)

    print(f"  Legacy _update_worker_state ONLY:    avg={_fmt_ms(avg_legacy)}")
    print("    (overhead before any work starts)")
    print(f"  Pre-selection + full batch work:     avg={_fmt_ms(avg_presel)}")
    print("    (pre-select + submit + compute + return)")

    return avg_legacy, avg_presel


# --- 9. End-to-End: Sequential vs Parallel ----------------------------------


def diag_sequential_vs_parallel(pop_size=POP_SIZE, n_gens=N_GENERATIONS, print_section=True):
    """Run full sequential and parallel comparisons with timing breakdown."""
    if print_section:
        _section(f"9. END-TO-END COMPARISON (pop={pop_size}, {n_gens} gens)")
    else:
        print(f"\n  pop={pop_size} | gens={n_gens}")

    phys = cpu_count_physical()
    logical = os.cpu_count() or phys
    print(f"  Physical cores: {phys}, Logical (threads): {logical}")

    worker_counts = [0]  # 0 = sequential
    if phys >= 2:
        worker_counts.append(2)
    if phys >= 4:
        worker_counts.append(4)
    if phys >= 8:
        worker_counts.append(phys)

    results = []

    for nw in worker_counts:
        label = "sequential" if nw == 0 else f"parallel({nw}w)"
        gp, temp_dir = _create_gp(pop_size, parallel=nw if nw > 0 else False)
        try:
            t0 = time.perf_counter()
            gp.gen_create_initial()
            t_init = time.perf_counter() - t0

            gen_times = []
            for g in range(n_gens):
                tg = time.perf_counter()
                gp.run_generation(STRATEGIES)
                gen_times.append(time.perf_counter() - tg)

            t_total = time.perf_counter() - t0
            t_avg = np.mean(gen_times)

            results.append(
                {
                    "label": label,
                    "workers": nw,
                    "t_init": t_init,
                    "gen_times": gen_times,
                    "t_avg": t_avg,
                    "t_total": t_total,
                }
            )
            print(
                f"  {label:<16s}: init={_fmt_ms(t_init):>8s}  "
                f"avg/gen={_fmt_ms(t_avg):>8s}  total={t_total:.1f}s  "
                f"gens={[f'{t:.2f}' for t in gen_times]}"
            )
        finally:
            gp.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary: overall average
    seq_avg = next((r["t_avg"] for r in results if r["workers"] == 0), None)
    if seq_avg:
        print("\n  OVERALL (including pool creation in gen 1):")
        print(f"  {'Config':<16s} {'Avg/Gen':>9s} {'Speedup':>9s} {'Efficiency':>12s}")
        print(f"  {'-' * 50}")
        for r in results:
            speedup = seq_avg / r["t_avg"] if r["t_avg"] > 0 else 0
            if r["workers"] > 0:
                eff = f"{speedup / r['workers'] * 100:.0f}%"
            else:
                eff = "-"
            print(f"  {r['label']:<16s} {_fmt_ms(r['t_avg']):>9s} {speedup:>8.2f}x {eff:>12s}")

    # Summary: steady-state (gen 2+ only, amortized pool creation)
    seq_steady = next(
        (np.mean(r["gen_times"][1:]) for r in results if r["workers"] == 0 and len(r["gen_times"]) > 1),
        None,
    )
    if seq_steady and n_gens > 1:
        print("\n  STEADY-STATE (gen 2+, pool creation amortized):")
        print(f"  {'Config':<16s} {'Avg/Gen':>9s} {'Speedup':>9s} {'Efficiency':>12s}")
        print(f"  {'-' * 50}")
        for r in results:
            gt = r["gen_times"]
            steady = np.mean(gt[1:]) if len(gt) > 1 else gt[0]
            speedup = seq_steady / steady if steady > 0 else 0
            if r["workers"] > 0:
                eff = f"{speedup / r['workers'] * 100:.0f}%"
            else:
                eff = "-"
            print(f"  {r['label']:<16s} {_fmt_ms(steady):>9s} {speedup:>8.2f}x {eff:>12s}")

    return results


# --- 10. Overhead Budget ----------------------------------------------------


def diag_overhead_budget(t_ipc_legacy, t_ipc_presel, t_seq_total, pop_size=POP_SIZE, n_workers=None):
    """Calculate theoretical speedup and overhead budget."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    _section(f"10. OVERHEAD BUDGET (pop={pop_size}, {n_workers} workers)")

    t_compute = t_seq_total
    t_compute_parallel = t_compute / n_workers

    print(f"  Sequential total compute:       {_fmt_ms(t_compute)}")
    print(f"  Ideal parallel ({n_workers}w) compute:  {_fmt_ms(t_compute_parallel)}")
    print()
    print("  LEGACY (before fix):")
    est_legacy = t_compute_parallel + t_ipc_legacy
    print(f"    Overhead:  {_fmt_ms(t_ipc_legacy)}")
    print(f"    Total:     {_fmt_ms(est_legacy)}")
    if est_legacy > 0:
        print(f"    Speedup:   {t_compute / est_legacy:.2f}x")
    print()
    print("  PRE-SELECTION (after fix):")
    # t_ipc_presel already includes compute, so estimate overhead alone
    overhead_new = max(0, t_ipc_presel - t_compute_parallel)
    est_new = t_compute_parallel + overhead_new
    print(f"    Overhead:  {_fmt_ms(overhead_new)}")
    print(f"    Total:     {_fmt_ms(est_new)}")
    if est_new > 0:
        print(f"    Speedup:   {t_compute / est_new:.2f}x")


def diag_population_end_to_end_comparison(pop_sizes=COMPARE_POP_SIZES, n_gens=N_GENERATIONS):
    """Run end-to-end comparison for multiple population sizes and summarize."""
    pop_label = " vs ".join(f"{pop:,}" for pop in pop_sizes)
    _section(f"11. END-TO-END POPULATION COMPARISON ({pop_label})")

    summaries = []
    for pop_size in pop_sizes:
        results = diag_sequential_vs_parallel(pop_size=pop_size, n_gens=n_gens, print_section=False)
        seq_steady = next(
            (float(np.mean(r["gen_times"][1:])) for r in results if r["workers"] == 0 and len(r["gen_times"]) > 1),
            None,
        )
        seq_metric = seq_steady
        if seq_metric is None:
            seq_metric = next((float(r["t_avg"]) for r in results if r["workers"] == 0), None)
        parallel_rows = [r for r in results if r["workers"] > 0]
        best_parallel = min(
            parallel_rows,
            key=lambda row: float(np.mean(row["gen_times"][1:])) if len(row["gen_times"]) > 1 else row["t_avg"],
        )
        best_parallel_steady = (
            float(np.mean(best_parallel["gen_times"][1:]))
            if len(best_parallel["gen_times"]) > 1
            else best_parallel["t_avg"]
        )
        speedup = seq_metric / best_parallel_steady if seq_metric and best_parallel_steady > 0 else 0
        summaries.append(
            {
                "pop": pop_size,
                "seq_metric": seq_metric,
                "best_label": best_parallel["label"],
                "best_steady": best_parallel_steady,
                "speedup": speedup,
            }
        )

    summary_label = "steady-state, gen 2+" if n_gens > 1 else "overall (single generation)"
    print(f"\n  SUMMARY ({summary_label}):")
    print(f"  {'Pop':>8s} {'Seq':>10s} {'Best parallel':>16s} {'Parallel':>10s} {'Speedup':>9s}")
    print(f"  {'-' * 61}")
    for row in summaries:
        seq_text = _fmt_ms(row["seq_metric"]) if row["seq_metric"] is not None else "n/a"
        print(
            f"  {row['pop']:>8d} {seq_text:>10s} {row['best_label']:>16s} "
            f"{_fmt_ms(row['best_steady']):>10s} {row['speedup']:>8.2f}x"
        )

    return summaries


# --- Output Tee --------------------------------------------------------------


class _TeeWriter:
    """Write to both a file and the original stdout simultaneously.

    Intended to be used as a context manager::

        with _TeeWriter(path) as tee:
            sys.stdout = tee
            ...
    """

    def __init__(self, filepath: Path):
        self._filepath = filepath
        self._file = None
        self._stdout = sys.stdout

    def __enter__(self):
        self._file = open(self._filepath, "w", encoding="utf-8")
        return self

    def __exit__(self, *_):
        self.close()

    def write(self, msg):
        self._stdout.write(msg)
        if self._file:
            self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# --- Main --------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="plagih GP - Full Parallel Diagnostics")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Path to write benchmark results (default: {_DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=POP_SIZE,
        help=f"Population size (default: {POP_SIZE})",
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=N_GENERATIONS,
        help=f"Number of generations for end-to-end test (default: {N_GENERATIONS})",
    )
    parser.add_argument(
        "--compare-pops",
        nargs="+",
        type=int,
        default=list(COMPARE_POP_SIZES),
        help="Population sizes for direct side-by-side comparison (default: 1000 10000)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(BATCH_TASK_SIZES),
        help="Task batch sizes to compare; use 0 for auto-balanced batches (default: 1 32 128 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pop_size = args.pop
    n_generations = args.gens
    compare_pops = tuple(_unique_ints(args.compare_pops))
    batch_sizes = tuple(_unique_ints(args.batch_sizes))

    with _TeeWriter(args.output) as tee:
        sys.stdout = tee

        try:
            phys = cpu_count_physical()
            logical = os.cpu_count() or phys
            print("=" * 60)
            print("  plagih GP - Full Parallel Diagnostics")
            print(f"  Physical cores: {phys} | Logical (threads): {logical}")
            print(f"  pop={pop_size} | gens={n_generations}")
            print(f"  compare_pops={compare_pops} | batch_sizes={batch_sizes}")
            print("=" * 60)

            # Create a GP instance with initial population for component measurements
            gp, temp_dir = _create_gp(pop_size)
            gp.gen_create_initial()

            n_workers = min(4, phys)

            try:
                diag_pickle_sizes(gp, pop_size=pop_size)
                diag_df_train_transport(gp.df_train, n_workers=n_workers)
                t_ipc_legacy, t_ipc_new = diag_ipc_comparison(gp, pop_size=pop_size)
                t_seq_total, t_per_task = diag_per_task_compute(gp, pop_size=pop_size)
                diag_population_batch_comparison(pop_sizes=compare_pops, batch_sizes=batch_sizes, n_workers=n_workers)
                diag_result_pickle_cost(gp, pop_size=pop_size, batch_sizes=batch_sizes, n_workers=n_workers)
                t_pool_legacy, t_pool_presel = diag_real_pool_comparison(gp, pop_size=pop_size, n_workers=n_workers)
                diag_overhead_budget(t_pool_legacy, t_pool_presel, t_seq_total, pop_size=pop_size, n_workers=n_workers)
            finally:
                gp.close()
                shutil.rmtree(temp_dir, ignore_errors=True)

            # End-to-end population comparison (creates fresh GP instances)
            diag_population_end_to_end_comparison(pop_sizes=compare_pops, n_gens=n_generations)

            print(f"\n{'=' * 60}")
            print("  Diagnostics complete.")
            print(f"  Results saved to: {args.output}")
            print(f"{'=' * 60}")
            print("\n === DONE. ===")
        finally:
            sys.stdout = tee._stdout
