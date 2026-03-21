"""Shared harness for tree-creation performance diagnosis.

Focus: identify whether time is spent in raw creation, simplification, or
fitness evaluation while producing trees/candidates.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import plagih_gp
from plagih.trees import ExplainableGP


def set_benchmark_seed(seed: int = 123) -> None:
    """Set deterministic RNG state for repeatable local benchmarks."""
    random.seed(seed)
    np.random.seed(seed)


def load_mountaincar_train_split(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MountainCar dataset used by the active usability test."""
    df = pd.read_csv(Path(plagih_gp.__file__).parent.absolute() / "benchmarks/mc/gp_files/samples200.csv").astype(
        "float32"
    )
    return plagih_gp.train_test_split(df, test_size=0.2, random_state=seed)


def make_benchmark_gp(pop_max_size: int = 120, gen_end: int = 2, seed: int = 0):
    """Create a temporary GP instance configured like the active usability test."""
    set_benchmark_seed(seed)
    df_train, df_control = load_mountaincar_train_split(seed=seed)
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_tree_creation_bench_"))
    gp = ExplainableGP.create(
        symbols=["cartVel", "cartPos"],
        df_train=df_train,
        rootdir=temp_dir,
        operators=plagih_gp._build_active_test_operator_dict(),
        depth_max=7,
        nodes_max=35,
        pop_max_size=pop_max_size,
        gen_end=gen_end,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=0,
        enable_analysis=False,
        verbose=False,
    )
    return gp, temp_dir, df_control


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_timing_records(records: List[Dict[str, Any]], scenario: str) -> Dict[str, Any]:
    """Aggregate per-tree timing records into a compact diagnostic summary."""
    ok_records = [record for record in records if record.get("status") == "ok"]
    error_records = [record for record in records if record.get("status") == "error"]
    totals_ok = [float(record.get("total_ms", 0.0)) for record in ok_records]

    dominant_phase_counts = {"create": 0, "simplify": 0, "evaluate": 0, "other": 0}
    for record in ok_records:
        phase_values = {
            "create": float(record.get("create_ms_shared", 0.0)),
            "simplify": float(record.get("simplify_ms", 0.0)),
            "evaluate": float(record.get("evaluate_ms", 0.0)),
        }
        dominant = max(phase_values, key=phase_values.get)
        dominant_phase_counts[dominant] += 1

    failed_stage_counts = {"create": 0, "simplify": 0, "evaluate": 0, "other": 0}
    for record in error_records:
        stage = record.get("failed_stage")
        if stage in failed_stage_counts:
            failed_stage_counts[stage] += 1
        else:
            failed_stage_counts["other"] += 1

    slowest = sorted(ok_records, key=lambda record: float(record.get("total_ms", 0.0)), reverse=True)[:5]

    summary = {
        "scenario": scenario,
        "records": len(records),
        "ok_records": len(ok_records),
        "error_records": len(error_records),
        "mean_total_ms": float(np.mean(totals_ok)) if totals_ok else 0.0,
        "p50_total_ms": _percentile(totals_ok, 50),
        "p95_total_ms": _percentile(totals_ok, 95),
        "max_total_ms": max(totals_ok) if totals_ok else 0.0,
        "max_create_ms": max((float(record.get("create_ms_shared", 0.0)) for record in ok_records), default=0.0),
        "max_simplify_ms": max((float(record.get("simplify_ms", 0.0)) for record in ok_records), default=0.0),
        "max_evaluate_ms": max((float(record.get("evaluate_ms", 0.0)) for record in ok_records), default=0.0),
        "dominant_phase_counts": dominant_phase_counts,
        "failed_stage_counts": failed_stage_counts,
        "top_slowest": [
            {
                "tag": record.get("tag"),
                "total_ms": float(record.get("total_ms", 0.0)),
                "fitness": record.get("fitness"),
                "parsimony": record.get("parsimony"),
                "expr_short": record.get("expr_short"),
            }
            for record in slowest
        ],
    }
    return summary


def bench_raw_random_creation(iterations: int = 200, depth_max_local: int = 5, seed: int = 123) -> Dict[str, Any]:
    """Benchmark raw random-tree creation without evaluation."""
    gp, temp_dir, _ = make_benchmark_gp(pop_max_size=20, gen_end=1, seed=seed)
    try:
        records = []
        for task_index in range(iterations):
            start = time.perf_counter()
            tree = gp.evolve.evolve_create_random(xt_out=float, depth_max_local=depth_max_local, depth=0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            records.append(
                {
                    "tag": "raw_random_create",
                    "task_index": task_index,
                    "tree_index": 0,
                    "status": "ok",
                    "failed_stage": None,
                    "create_ms_shared": elapsed_ms,
                    "simplify_ms": 0.0,
                    "evaluate_ms": 0.0,
                    "total_ms": elapsed_ms,
                    "fitness": None,
                    "parsimony": len(tree),
                    "expr_short": str(tree),
                }
            )
        return summarize_timing_records(records, scenario="raw_random_creation")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bench_depth_goal_creation(iterations: int = 100, depth_goal: int = 6, seed: int = 123) -> Dict[str, Any]:
    """Benchmark targeted depth-goal creation without evaluation."""
    gp, temp_dir, _ = make_benchmark_gp(pop_max_size=20, gen_end=1, seed=seed)
    try:
        records = []
        for task_index in range(iterations):
            start = time.perf_counter()
            tree = gp.evolve.evolve_new_tree_depth(xt_out=float, depth_goal=depth_goal, p_term=0.1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            records.append(
                {
                    "tag": "depth_goal_create",
                    "task_index": task_index,
                    "tree_index": 0,
                    "status": "ok",
                    "failed_stage": None,
                    "create_ms_shared": elapsed_ms,
                    "simplify_ms": 0.0,
                    "evaluate_ms": 0.0,
                    "total_ms": elapsed_ms,
                    "fitness": None,
                    "parsimony": len(tree),
                    "expr_short": str(tree),
                }
            )
        return summarize_timing_records(records, scenario="depth_goal_creation")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bench_initial_population(pop_max_size: int = 120, seed: int = 123) -> Dict[str, Any]:
    """Benchmark initial population creation including evaluation."""
    gp, temp_dir, _ = make_benchmark_gp(pop_max_size=pop_max_size, gen_end=1, seed=seed)
    try:
        set_benchmark_seed(seed)
        gp.gen_create_initial()
        return summarize_timing_records(list(gp._latest_generation_tree_timings), scenario="initial_population")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bench_active_generation(pop_max_size: int = 120, seed: int = 123) -> Dict[str, Any]:
    """Benchmark one full active-test generation after initial population."""
    gp, temp_dir, _ = make_benchmark_gp(pop_max_size=pop_max_size, gen_end=2, seed=seed)
    try:
        set_benchmark_seed(seed)
        gp.gen_create_initial()
        set_benchmark_seed(seed + 1)
        gp.run_generation(plagih_gp._build_active_test_strategies(), parallel=False, seed=seed)
        return summarize_timing_records(list(gp._latest_generation_tree_timings), scenario="active_generation")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_tree_creation_benchmarks(pop_max_size: int = 120, seed: int = 123) -> Dict[str, Dict[str, Any]]:
    """Run the full tree-creation benchmark suite and return all summaries."""
    return {
        "raw_random_creation": bench_raw_random_creation(iterations=200, depth_max_local=5, seed=seed),
        "depth_goal_creation": bench_depth_goal_creation(iterations=100, depth_goal=6, seed=seed),
        "initial_population": bench_initial_population(pop_max_size=pop_max_size, seed=seed),
        "active_generation": bench_active_generation(pop_max_size=pop_max_size, seed=seed),
    }


def save_benchmark_summary(summary: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Save benchmark summary as JSON for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
