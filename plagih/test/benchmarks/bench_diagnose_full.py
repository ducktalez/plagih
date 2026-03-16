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
import os
import pickle
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
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


def _create_gp(pop_size=POP_SIZE, parallel=False):
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


# --- 1. Pickle Size Analysis ------------------------------------------------


def diag_pickle_sizes(gp):
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
    tasks = build_task_list(STRATEGIES, POP_SIZE)
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


# --- 2. Legacy vs Pre-Selection IPC Cost ------------------------------------


def diag_ipc_comparison(gp):
    """Compare legacy _update_worker_state vs pre-selection IPC cost."""
    _section("2. LEGACY vs PRE-SELECTION IPC COST")

    n_workers = min(4, cpu_count_physical())
    N = 10

    # Legacy: pickle full pop_genepool to each worker
    payload = (gp.pop_genepool, gp.paretofront)
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
        tasks_copy = build_task_list(STRATEGIES, POP_SIZE)
        pre_select_for_tasks(tasks_copy, gp.pop_genepool, gp.paretofront)
    t_preselect = (time.perf_counter() - t0) / N

    # Batch pickle cost
    tasks = build_task_list(STRATEGIES, POP_SIZE)
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


# --- 3. Per-Task Compute Time -----------------------------------------------


def diag_per_task_compute(gp):
    """Measure average compute time per task (sequential, no IPC)."""
    _section("3. PER-TASK COMPUTE TIME (sequential, no overhead)")

    tasks = build_task_list(STRATEGIES, POP_SIZE)
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


# --- 4. Result Pickle Cost --------------------------------------------------


def diag_result_pickle_cost(gp):
    """Measure cost of pickling TaskResults back from workers."""
    _section("4. RESULT PICKLE COST (worker -> main)")

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

    batch_250 = ok_results * (250 // len(ok_results) + 1)
    batch_250 = batch_250[:250]
    sz_batch = len(pickle.dumps(batch_250, protocol=pickle.HIGHEST_PROTOCOL))
    t0 = time.perf_counter()
    for _ in range(10):
        pickle.dumps(batch_250, protocol=pickle.HIGHEST_PROTOCOL)
    t_batch = (time.perf_counter() - t0) / 10

    print(f"  Batch of 250 results: {_fmt_bytes(sz_batch)}, dumps={_fmt_ms(t_batch)}")
    for nw in [2, 4, 8]:
        batch_size = POP_SIZE // nw
        est = t_batch * (batch_size / 250)
        print(f"    {nw}w x ~{batch_size} results: ~{_fmt_ms(est * nw)}")

    return t_batch


# --- 5. Real Pool Comparison ------------------------------------------------


def diag_real_pool_comparison(gp, n_workers=None):
    """Compare real pool: legacy _update_worker_state vs pre-selection batch."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    _section(f"5. REAL POOL COMPARISON ({n_workers} workers)")

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
        tasks = build_task_list(STRATEGIES, POP_SIZE)
        t_ps = time.perf_counter()
        pre_select_for_tasks(tasks, gp.pop_genepool, gp.paretofront)
        batches = [[] for _ in range(n_workers)]
        for i, task in enumerate(tasks):
            batches[i % n_workers].append(task)
        batches = [b for b in batches if b]

        t0 = time.perf_counter()
        from concurrent.futures import as_completed

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


# --- 6. End-to-End: Sequential vs Parallel ----------------------------------


def diag_sequential_vs_parallel(pop_size=POP_SIZE, n_gens=N_GENERATIONS):
    """Run full sequential and parallel comparisons with timing breakdown."""
    _section(f"6. END-TO-END COMPARISON (pop={pop_size}, {n_gens} gens)")

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
        gp, temp_dir = _create_gp(pop_size, parallel=nw if nw > 0 else 0)
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


# --- 7. Overhead Budget -----------------------------------------------------


def diag_overhead_budget(t_ipc_legacy, t_ipc_presel, t_seq_total, n_workers=None):
    """Calculate theoretical speedup and overhead budget."""
    if n_workers is None:
        n_workers = min(4, cpu_count_physical())
    _section(f"7. OVERHEAD BUDGET (pop={POP_SIZE}, {n_workers} workers)")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pop_size = args.pop
    n_generations = args.gens

    with _TeeWriter(args.output) as tee:
        sys.stdout = tee

        try:
            phys = cpu_count_physical()
            logical = os.cpu_count() or phys
            print("=" * 60)
            print("  plagih GP - Full Parallel Diagnostics")
            print(f"  Physical cores: {phys} | Logical (threads): {logical}")
            print(f"  pop={pop_size} | gens={n_generations}")
            print("=" * 60)

            # Create a GP instance with initial population for component measurements
            gp, temp_dir = _create_gp(pop_size)
            gp.gen_create_initial()

            n_workers = min(4, phys)

            try:
                diag_pickle_sizes(gp)
                t_ipc_legacy, t_ipc_new = diag_ipc_comparison(gp)
                t_seq_total, t_per_task = diag_per_task_compute(gp)
                diag_result_pickle_cost(gp)
                t_pool_legacy, t_pool_presel = diag_real_pool_comparison(gp, n_workers)
                diag_overhead_budget(t_pool_legacy, t_pool_presel, t_seq_total, n_workers)
            finally:
                gp.close()
                shutil.rmtree(temp_dir, ignore_errors=True)

            # End-to-end comparison (creates fresh GP instances)
            diag_sequential_vs_parallel(pop_size=pop_size, n_gens=n_generations)

            print(f"\n{'=' * 60}")
            print("  Diagnostics complete.")
            print(f"  Results saved to: {args.output}")
            print(f"{'=' * 60}")
        finally:
            sys.stdout = tee._stdout
