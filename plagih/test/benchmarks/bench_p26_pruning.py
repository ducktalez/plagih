"""P26 follow-up: random vs deepest-first pruning — which preserves semantics?

Generates oversized random trees, prunes each with both strategies (same
tree copy), and measures:

- **semantic distance**: RMSE between original and pruned predictions on
  MountainCar training data (lower = pruning changed less)
- **kept fraction**: rows where pruned output stays within 1% of original
- **size / steps / runtime**

Run directly:
    python plagih/test/benchmarks/bench_p26_pruning.py
    python plagih/test/benchmarks/bench_p26_pruning.py --trees 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from plagih.test.benchmarks._tree_creation_harness import (
    load_mountaincar_train_split,
    set_benchmark_seed,
)

NODES_MAX = 12  # prune target
CREATE_DEPTH = 7  # oversized source trees


def _make_evolution(nodes_max: int):
    import plagih_gp
    from plagih.trees._evolution import Evolution

    return Evolution(
        symbol_list=["cartVel", "cartPos"],
        operators=plagih_gp._build_active_test_operator_dict(),
        depth_max=CREATE_DEPTH + 2,
        nodes_max=nodes_max,
    )


def _safe_eval(tree, df) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            out = np.asarray(tree.eval_predict_numpy_now(df), dtype=np.float64)
        except Exception:
            return np.full(len(df), np.nan)
    if out.shape != (len(df),):
        return np.full(len(df), np.nan)
    return out


def run_benchmark(n_trees: int, seed: int = 42) -> Dict[str, Any]:
    from plagih.trees._nodes import fast_tree_copy

    set_benchmark_seed(seed)
    df, _ = load_mountaincar_train_split(seed=0)
    ev = _make_evolution(NODES_MAX)

    per_strategy: Dict[str, List[Dict[str, float]]] = {"random": [], "deepest": []}
    skipped = 0

    attempts = 0
    produced = 0
    while produced < n_trees and attempts < n_trees * 10:
        attempts += 1
        # Oversized tree: created without node budget, then prune to NODES_MAX
        tree = ev.evolve_create_random(float, CREATE_DEPTH, num_rest=-1, depth=0, p_term=0.05)
        tree.repair_all()
        if len(tree) <= NODES_MAX + 4:
            continue  # not oversized enough to be interesting

        base = _safe_eval(tree, df)
        base_finite = np.isfinite(base)
        if base_finite.sum() < len(df) * 0.8:
            skipped += 1
            continue  # original already broken — no useful comparison

        produced += 1
        for strat in ("random", "deepest"):
            copy_tree = fast_tree_copy(tree)
            copy_tree.repair_all()
            t0 = time.perf_counter()
            pruned = ev.evolve_prune_tree(copy_tree, strategy=strat)
            elapsed = time.perf_counter() - t0

            out = _safe_eval(pruned, df)
            mask = base_finite & np.isfinite(out)
            if mask.sum() == 0:
                sem_rmse = float("inf")
                kept = 0.0
            else:
                diff = out[mask] - base[mask]
                sem_rmse = float(np.sqrt(np.mean(diff**2)))
                denom = np.maximum(np.abs(base[mask]), 1e-9)
                kept = float(np.mean(np.abs(diff) / denom < 0.01))

            per_strategy[strat].append(
                {
                    "sem_rmse": sem_rmse,
                    "kept_frac": kept,
                    "size_after": len(pruned),
                    "seconds": elapsed,
                }
            )

    def _agg(records: List[Dict[str, float]]) -> Dict[str, float]:
        finite_rmse = [r["sem_rmse"] for r in records if np.isfinite(r["sem_rmse"])]
        return {
            "n": len(records),
            "rmse_median": statistics.median(finite_rmse) if finite_rmse else float("inf"),
            "rmse_mean": statistics.fmean(finite_rmse) if finite_rmse else float("inf"),
            "kept_frac_mean": statistics.fmean(r["kept_frac"] for r in records),
            "size_after_mean": statistics.fmean(r["size_after"] for r in records),
            "ms_mean": statistics.fmean(r["seconds"] for r in records) * 1000,
            "n_inf": sum(1 for r in records if not np.isfinite(r["sem_rmse"])),
        }

    # Head-to-head: per tree, which strategy has lower semantic RMSE?
    wins = {"random": 0, "deepest": 0, "tie": 0}
    for r_rec, d_rec in zip(per_strategy["random"], per_strategy["deepest"]):
        if abs(r_rec["sem_rmse"] - d_rec["sem_rmse"]) < 1e-12:
            wins["tie"] += 1
        elif d_rec["sem_rmse"] < r_rec["sem_rmse"]:
            wins["deepest"] += 1
        else:
            wins["random"] += 1

    return {
        "n_trees": produced,
        "skipped_broken": skipped,
        "random": _agg(per_strategy["random"]),
        "deepest": _agg(per_strategy["deepest"]),
        "wins": wins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P26 pruning strategy benchmark")
    parser.add_argument("--trees", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 64)
    print("P26 BENCHMARK — random vs deepest-first pruning")
    print("=" * 64)
    print(f"trees={args.trees}  nodes_max={NODES_MAX}  create_depth={CREATE_DEPTH}\n")

    result = run_benchmark(args.trees, seed=args.seed)

    print(f"trees compared: {result['n_trees']} (skipped broken: {result['skipped_broken']})\n")
    print(f"{'strategy':10s} {'rmse_med':>10s} {'rmse_mean':>10s} {'kept%':>7s} {'size':>6s} {'ms':>6s} {'inf':>4s}")
    for strat in ("random", "deepest"):
        a = result[strat]
        print(
            f"{strat:10s} {a['rmse_median']:10.4f} {a['rmse_mean']:10.4f} "
            f"{a['kept_frac_mean'] * 100:6.1f}% {a['size_after_mean']:6.1f} {a['ms_mean']:6.2f} {a['n_inf']:4d}"
        )

    w = result["wins"]
    n = result["n_trees"]
    print(f"\nhead-to-head (lower semantic RMSE): deepest {w['deepest']} | random {w['random']} | tie {w['tie']}")

    print(f"\n{'=' * 64}")
    print("VERDICT")
    print("=" * 64)
    if w["deepest"] > n * 0.6:
        print("deepest-first preserves semantics clearly better -> make it default.")
    elif w["deepest"] > w["random"]:
        print("deepest-first mildly better -> keep as opt-in, consider default later.")
    else:
        print("random pruning is not worse -> keep random as default (simpler).")

    out = Path(__file__).with_name("bench_p26_output.json")
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
