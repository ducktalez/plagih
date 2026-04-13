"""H3 Diagnosis: Worker RAM overhead profiling.

Measures per-worker RSS and identifies the import/module groups responsible
for the ~1.1–1.2 GB child RSS at 8 workers (see IMPLEMENTATION_PLAN H3).

Two measurements:
  1. Import-cost profiling: measure RSS after each major import group
  2. Worker-pool RSS: launch actual workers and measure their RSS

Run directly:
    python plagih/test/benchmarks/bench_h3_worker_ram.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _get_rss_mb() -> float:
    """Current process RSS in MB (Linux only)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    # Fallback via resource module
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # kB -> MB


def profile_import_costs() -> list[dict]:
    """Measure RSS growth after each major import group."""
    results = []
    rss_before = _get_rss_mb()
    results.append({"stage": "baseline (python interpreter)", "rss_mb": rss_before, "delta_mb": 0.0})

    # Stage 1: numpy + pandas
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401

    rss_now = _get_rss_mb()
    results.append({"stage": "numpy + pandas", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
    rss_before = rss_now

    # Stage 2: sympy
    import sympy  # noqa: F401

    rss_now = _get_rss_mb()
    results.append({"stage": "sympy", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
    rss_before = rss_now

    # Stage 3: plagih.trees (node hierarchy, evolution, gp_engine)
    from plagih.trees import Evolution, ExplainableGP, Node  # noqa: F401

    rss_now = _get_rss_mb()
    results.append({"stage": "plagih.trees (full)", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
    rss_before = rss_now

    # Stage 4: plagih.parallel
    from plagih.parallel import (  # noqa: F401
        Strategy,
        evaluate_tree_standalone,
        run_task_sequential,
    )

    rss_now = _get_rss_mb()
    results.append({"stage": "plagih.parallel", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
    rss_before = rss_now

    # Stage 5: matplotlib (often imported transitively)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        rss_now = _get_rss_mb()
        results.append({"stage": "matplotlib", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
        rss_before = rss_now
    except ImportError:
        results.append({"stage": "matplotlib", "rss_mb": rss_before, "delta_mb": 0.0, "note": "not installed"})

    # Stage 6: scipy (sometimes imported by evaluation)
    try:
        import scipy  # noqa: F401

        rss_now = _get_rss_mb()
        results.append({"stage": "scipy", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
        rss_before = rss_now
    except ImportError:
        results.append({"stage": "scipy", "rss_mb": rss_before, "delta_mb": 0.0, "note": "not installed"})

    # Stage 7: tensorflow (if present — biggest offender for RAM)
    try:
        import tensorflow  # noqa: F401

        rss_now = _get_rss_mb()
        results.append({"stage": "tensorflow", "rss_mb": rss_now, "delta_mb": rss_now - rss_before})
    except ImportError:
        results.append({"stage": "tensorflow", "rss_mb": rss_before, "delta_mb": 0.0, "note": "not installed"})

    return results


def _worker_report_rss(_task_id: int) -> dict:
    """Task function run inside a worker process to report its RSS."""
    import os

    pid = os.getpid()
    rss_mb = 0.0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        import resource

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # Gather loaded module counts by package
    import sys as _sys

    package_counts: dict[str, int] = {}
    for mod_name in _sys.modules:
        top = mod_name.split(".")[0]
        package_counts[top] = package_counts.get(top, 0) + 1

    top_packages = sorted(package_counts.items(), key=lambda x: -x[1])[:15]
    return {"pid": pid, "rss_mb": rss_mb, "top_packages": dict(top_packages), "total_modules": len(_sys.modules)}


def profile_worker_pool_rss(n_workers: int = 4) -> list[dict]:
    """Launch actual plagih worker pool and measure per-worker RSS."""
    import shutil
    import tempfile

    import plagih_gp
    from plagih.test.benchmarks._tree_creation_harness import (
        load_mountaincar_train_split,
        set_benchmark_seed,
    )
    from plagih.trees import ExplainableGP

    set_benchmark_seed(0)
    df_train, _ = load_mountaincar_train_split(seed=0)
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_h3_bench_"))

    gp = ExplainableGP.create(
        symbols=["cartVel", "cartPos"],
        df_train=df_train,
        rootdir=temp_dir,
        operators=plagih_gp._build_active_test_operator_dict(),
        depth_max=7,
        nodes_max=35,
        pop_max_size=50,
        gen_end=2,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=n_workers,
        enable_analysis=False,
        verbose=False,
    )

    try:
        pool = gp._get_or_create_pool()
        if pool is None:
            return [{"error": "Pool not created (sequential mode?)"}]

        # Submit RSS report tasks to each worker
        futures = [pool.submit(_worker_report_rss, i) for i in range(n_workers)]
        results = [f.result(timeout=30) for f in futures]
        return results
    finally:
        gp.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    print("=" * 72)
    print("H3 Diagnosis: Worker RAM Overhead Profiling")
    print("=" * 72)

    # Part 1: Import cost profiling (in current process)
    print("\n--- Part 1: Import RSS Growth (this process) ---\n")
    import_costs = profile_import_costs()
    print(f"{'Stage':<35s}  {'RSS (MB)':>10s}  {'Delta (MB)':>10s}")
    print("-" * 60)
    for entry in import_costs:
        note = f"  ({entry['note']})" if "note" in entry else ""
        print(f"{entry['stage']:<35s}  {entry['rss_mb']:>10.1f}  {entry['delta_mb']:>+10.1f}{note}")

    total_import_cost = import_costs[-1]["rss_mb"] - import_costs[0]["rss_mb"] if len(import_costs) > 1 else 0
    print(f"\nTotal import overhead: {total_import_cost:.1f} MB")

    # Part 2: Actual worker pool RSS
    N_WORKERS = 4
    print(f"\n--- Part 2: Worker Pool RSS ({N_WORKERS} workers) ---\n")
    worker_results = profile_worker_pool_rss(n_workers=N_WORKERS)
    total_worker_rss = 0.0
    for wr in worker_results:
        if "error" in wr:
            print(f"  Error: {wr['error']}")
            continue
        print(f"  Worker PID {wr['pid']}: RSS = {wr['rss_mb']:.1f} MB, modules = {wr['total_modules']}")
        print(f"    Top packages: {', '.join(f'{k}({v})' for k, v in list(wr['top_packages'].items())[:8])}")
        total_worker_rss += wr["rss_mb"]

    if total_worker_rss > 0:
        avg_rss = total_worker_rss / len([w for w in worker_results if "error" not in w])
        print(f"\n  Average worker RSS: {avg_rss:.1f} MB")
        print(f"  Total worker RSS:   {total_worker_rss:.1f} MB ({N_WORKERS} workers)")
        print(f"  Main process RSS:   {_get_rss_mb():.1f} MB")

    # Part 3: Lazy-import candidates
    print("\n--- Part 3: Lazy-Import Candidates ---\n")
    candidates = []
    for entry in import_costs:
        if entry["delta_mb"] > 10.0 and entry["stage"] not in ("baseline (python interpreter)",):
            candidates.append(entry)
    if candidates:
        print("Modules with >10 MB import cost (candidates for lazy import in workers):")
        for c in sorted(candidates, key=lambda x: -x["delta_mb"]):
            print(f"  {c['stage']}: +{c['delta_mb']:.1f} MB")
    else:
        print("No single import group adds >10 MB. Worker overhead is distributed.")

    # Save results
    output = {"import_costs": import_costs, "worker_pool_rss": worker_results}
    output_path = Path(__file__).with_name("bench_h3_output.json")
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
