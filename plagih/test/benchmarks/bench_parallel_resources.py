"""
Resource profiler for plagih parallel end-to-end runs.

Measures CPU utilization and RAM usage of the current process tree
(main process + worker processes) while running the current GP parallelization.

Focus:
- population scaling
- worker scaling
- sequential vs parallel comparison
- effect of initialization vs per-generation work

Run directly:
    python plagih/test/benchmarks/bench_parallel_resources.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import psutil

from plagih.test.benchmarks.bench_diagnose_full import STRATEGIES, _create_gp, _fmt_ms
from plagih.util import cpu_count_physical

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bench_resources_output.txt"
_DEFAULT_POPS = (1000, 10_000)
_DEFAULT_GENS = 1
_SAMPLE_INTERVAL_S = 0.25


@dataclass
class ResourceSample:
    timestamp: float
    phase: str
    system_cpu_percent: float
    process_tree_cpu_percent: float
    rss_main: int
    rss_children: int
    rss_total: int
    worker_count: int


class ResourceSampler:
    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self._root_process = psutil.Process(os.getpid())
        self._phase = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[ResourceSample] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self._prime_process_cpu_counters()
        self._thread = threading.Thread(target=self._run, name="plagih-resource-sampler", daemon=True)
        self._thread.start()

    def mark(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
        self.samples.append(self._collect_sample())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))

    def summarize_phase(self, phase: str) -> dict[str, float | int | str] | None:
        phase_samples = [sample for sample in self.samples if sample.phase == phase]
        if not phase_samples:
            return None

        timestamps = [sample.timestamp for sample in phase_samples]
        proc_cpu = [sample.process_tree_cpu_percent for sample in phase_samples]
        sys_cpu = [sample.system_cpu_percent for sample in phase_samples]
        rss_total = [sample.rss_total for sample in phase_samples]
        rss_children = [sample.rss_children for sample in phase_samples]
        rss_main = [sample.rss_main for sample in phase_samples]
        worker_counts = [sample.worker_count for sample in phase_samples]

        return {
            "phase": phase,
            "duration_s": max(timestamps) - min(timestamps) if len(timestamps) > 1 else self.interval_s,
            "proc_cpu_avg": sum(proc_cpu) / len(proc_cpu),
            "proc_cpu_peak": max(proc_cpu),
            "sys_cpu_avg": sum(sys_cpu) / len(sys_cpu),
            "sys_cpu_peak": max(sys_cpu),
            "rss_total_avg": int(sum(rss_total) / len(rss_total)),
            "rss_total_peak": max(rss_total),
            "rss_children_peak": max(rss_children),
            "rss_main_peak": max(rss_main),
            "worker_count_peak": max(worker_counts),
        }

    def _current_phase(self) -> str:
        with self._lock:
            return self._phase

    def _prime_process_cpu_counters(self) -> None:
        for proc in self._process_tree():
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def _process_tree(self) -> list[psutil.Process]:
        try:
            procs = [self._root_process]
            procs.extend(self._root_process.children(recursive=True))
            return procs
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _collect_sample(self) -> ResourceSample:
        phase = self._current_phase()
        procs = self._process_tree()
        system_cpu = psutil.cpu_percent(interval=None)

        process_tree_cpu = 0.0
        rss_main = 0
        rss_children = 0
        worker_count = max(0, len(procs) - 1)

        for idx, proc in enumerate(procs):
            try:
                process_tree_cpu += proc.cpu_percent(interval=None)
                rss = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            if idx == 0:
                rss_main = rss
            else:
                rss_children += rss

        return ResourceSample(
            timestamp=time.perf_counter(),
            phase=phase,
            system_cpu_percent=system_cpu,
            process_tree_cpu_percent=process_tree_cpu,
            rss_main=rss_main,
            rss_children=rss_children,
            rss_total=rss_main + rss_children,
            worker_count=worker_count,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._collect_sample())
            self._stop.wait(self.interval_s)


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _resolve_worker_counts(requested: list[int] | None) -> list[int]:
    if requested:
        return requested

    phys = cpu_count_physical()
    counts = [0]
    if phys >= 2:
        counts.append(2)
    if phys >= 4:
        counts.append(4)
    if phys >= 8:
        counts.append(phys)
    return counts


def _print_phase_summary(summary: dict[str, float | int | str]) -> None:
    print(
        f"    {summary['phase']:<12s} wall={_fmt_ms(float(summary['duration_s'])):>9s} "
        f"proc_cpu(avg/peak)={float(summary['proc_cpu_avg']):>6.1f}%/{float(summary['proc_cpu_peak']):>6.1f}% "
        f"sys_cpu(avg/peak)={float(summary['sys_cpu_avg']):>5.1f}%/{float(summary['sys_cpu_peak']):>5.1f}%"
    )
    print(
        f"{'':16s}rss_total(avg/peak)={_fmt_bytes(int(summary['rss_total_avg'])):>10s}/"
        f"{_fmt_bytes(int(summary['rss_total_peak'])):>10s} "
        f"rss_children_peak={_fmt_bytes(int(summary['rss_children_peak'])):>10s} "
        f"workers_peak={int(summary['worker_count_peak'])}"
    )


def _run_profile(pop_size: int, workers: int, n_gens: int) -> dict:
    label = "sequential" if workers == 0 else f"parallel({workers}w)"
    print(f"\n  -> start {label} | pop={pop_size} | gens={n_gens}")

    gp, temp_dir = _create_gp(pop_size, parallel=workers if workers > 0 else False)
    sampler = ResourceSampler()
    phase_order: list[str] = []
    t_start = time.perf_counter()

    def mark(phase: str) -> None:
        phase_order.append(phase)
        sampler.mark(phase)

    try:
        sampler.start()

        mark("init")
        t0 = time.perf_counter()
        gp.gen_create_initial()
        t_init = time.perf_counter() - t0
        print(f"     init done: {_fmt_ms(t_init)} | genepool={len(gp.pop_genepool)}")

        gen_times = []
        for gen_idx in range(n_gens):
            phase = f"gen_{gen_idx + 1}"
            mark(phase)
            tg = time.perf_counter()
            gp.run_generation(STRATEGIES)
            dt = time.perf_counter() - tg
            gen_times.append(dt)
            print(
                f"     generation {gen_idx + 1}/{n_gens}: {_fmt_ms(dt)} | "
                f"genepool={len(gp.pop_genepool)} | pareto={len(gp.paretofront)}"
            )

        sampler.stop()
        total_wall = time.perf_counter() - t_start
        steady = sum(gen_times[1:]) / len(gen_times[1:]) if len(gen_times) > 1 else (gen_times[0] if gen_times else 0.0)
        avg_gen = sum(gen_times) / len(gen_times) if gen_times else 0.0

        phase_summaries = [
            sampler.summarize_phase(phase)
            for phase in dict.fromkeys(phase_order)
            if sampler.summarize_phase(phase) is not None
        ]

        relevant = [summary for summary in phase_summaries if summary is not None]
        peak_rss_total = max(int(summary["rss_total_peak"]) for summary in relevant) if relevant else 0
        peak_rss_children = max(int(summary["rss_children_peak"]) for summary in relevant) if relevant else 0
        peak_proc_cpu = max(float(summary["proc_cpu_peak"]) for summary in relevant) if relevant else 0.0
        peak_workers = max(int(summary["worker_count_peak"]) for summary in relevant) if relevant else 0

        print(f"     finish {label}: total={total_wall:.1f}s | avg/gen={_fmt_ms(avg_gen)} | steady={_fmt_ms(steady)}")
        for summary in relevant:
            _print_phase_summary(summary)

        return {
            "pop": pop_size,
            "label": label,
            "workers": workers,
            "t_init": t_init,
            "t_avg": avg_gen,
            "t_steady": steady,
            "t_total": total_wall,
            "peak_rss_total": peak_rss_total,
            "peak_rss_children": peak_rss_children,
            "peak_proc_cpu": peak_proc_cpu,
            "peak_workers": peak_workers,
            "phases": relevant,
        }
    finally:
        sampler.stop()
        gp.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_summary(output_path: Path, results: list[dict]) -> None:
    lines = []
    lines.append("plagih resource profiling\n")
    lines.append("========================\n")
    lines.append("\n")
    lines.append("Summary by configuration:\n")
    lines.append(
        f"{'Pop':>8s} {'Config':>14s} {'Avg/Gen':>10s} {'Steady':>10s} {'Peak RSS':>12s} {'Child RSS':>12s} {'Peak CPU':>10s} {'Workers':>8s}\n"
    )
    lines.append(f"{'-' * 92}\n")
    for row in results:
        lines.append(
            f"{row['pop']:>8d} {row['label']:>14s} {_fmt_ms(row['t_avg']):>10s} {_fmt_ms(row['t_steady']):>10s} "
            f"{_fmt_bytes(row['peak_rss_total']):>12s} {_fmt_bytes(row['peak_rss_children']):>12s} "
            f"{row['peak_proc_cpu']:>9.1f}% {row['peak_workers']:>8d}\n"
        )
    lines.append("\n")
    lines.append("Phase details:\n")
    for row in results:
        lines.append(f"\n[{row['label']} | pop={row['pop']}]\n")
        for phase in row["phases"]:
            lines.append(
                f"  {phase['phase']:<12s} wall={_fmt_ms(float(phase['duration_s'])):>9s} "
                f"proc_cpu(avg/peak)={float(phase['proc_cpu_avg']):>6.1f}%/{float(phase['proc_cpu_peak']):>6.1f}% "
                f"sys_cpu(avg/peak)={float(phase['sys_cpu_avg']):>5.1f}%/{float(phase['sys_cpu_peak']):>5.1f}% "
                f"rss_peak={_fmt_bytes(int(phase['rss_total_peak'])):>10s} "
                f"child_rss_peak={_fmt_bytes(int(phase['rss_children_peak'])):>10s} "
                f"workers_peak={int(phase['worker_count_peak'])}\n"
            )
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CPU/RAM for plagih end-to-end parallel runs.")
    parser.add_argument("--pops", nargs="+", type=int, default=list(_DEFAULT_POPS), help="Population sizes to test")
    parser.add_argument("--gens", type=int, default=_DEFAULT_GENS, help="Generations per config")
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=None,
        help="Worker counts to test. Use 0 for sequential. Default: auto [0,2,4,physical]",
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Where to write the profiling summary")
    args = parser.parse_args()

    from plagih.config import cfg

    cfg.verbosity = "ww"

    pop_sizes = tuple(args.pops)
    worker_counts = _resolve_worker_counts(args.workers)

    _section("PARALLEL RESOURCE PROFILING")
    print(f"  populations={pop_sizes} | gens={args.gens} | workers={worker_counts}")
    print(f"  output={args.output}")

    results: list[dict] = []
    for pop_size in pop_sizes:
        _section(f"POPULATION {pop_size}")
        for workers in worker_counts:
            results.append(_run_profile(pop_size=pop_size, workers=workers, n_gens=args.gens))

    _write_summary(args.output, results)
    _section("DONE")
    print(f"  profiling summary written to: {args.output}")


if __name__ == "__main__":
    main()
