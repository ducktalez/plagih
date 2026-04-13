"""D9 Quantification: How many trees fail at the evaluate stage?

Runs a multi-generation GP loop and breaks down failure rates per stage,
including error_type and error_message patterns for evaluate-stage failures.

This answers IMPLEMENTATION_PLAN D9 question 1:
  "How many trees actually fail at the `np.isfinite()` check?"

Run directly:
    python plagih/test/benchmarks/bench_d9_evaluate_failures.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from plagih.test.benchmarks._tree_creation_harness import (
    load_mountaincar_train_split,
    set_benchmark_seed,
)


def _collect_timings_multi_gen(
    n_gens: int = 10,
    pop_max_size: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run n_gens generations and collect all per-tree timing records."""
    import shutil
    import tempfile

    import plagih_gp
    from plagih.trees import ExplainableGP

    set_benchmark_seed(seed)
    df_train, _ = load_mountaincar_train_split(seed=seed)
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_d9_bench_"))

    gp = ExplainableGP.create(
        symbols=["cartVel", "cartPos"],
        df_train=df_train,
        rootdir=temp_dir,
        operators=plagih_gp._build_active_test_operator_dict(),
        depth_max=7,
        nodes_max=35,
        pop_max_size=pop_max_size,
        gen_end=n_gens + 1,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=0,
        enable_analysis=False,
        verbose=False,
    )

    all_records: List[Dict[str, Any]] = []

    try:
        # Generation 0: initial population
        gp.gen_create_initial()
        gen0_records = list(gp._latest_generation_tree_timings)
        for r in gen0_records:
            r["gen_id"] = 0
        all_records.extend(gen0_records)

        strategies = plagih_gp._build_active_test_strategies()

        # Generations 1..n_gens
        for gen in range(1, n_gens + 1):
            set_benchmark_seed(seed + gen)
            gp.run_generation(strategies, parallel=False, seed=seed + gen)
            gen_records = list(gp._latest_generation_tree_timings)
            for r in gen_records:
                r["gen_id"] = gen
            all_records.extend(gen_records)

    finally:
        gp.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    return all_records


def _analyze_evaluate_failures(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detailed breakdown of evaluate-stage failures."""
    total = len(records)
    ok = [r for r in records if r.get("status") == "ok"]
    errors = [r for r in records if r.get("status") == "error"]

    stage_counts = Counter(r.get("failed_stage") for r in errors)

    # Focus on evaluate-stage failures
    eval_errors = [r for r in errors if r.get("failed_stage") == "evaluate"]
    eval_error_types = Counter(r.get("error_type", "?") for r in eval_errors)
    eval_error_messages = Counter(r.get("error_message", "?") for r in eval_errors)

    # Per-generation breakdown
    gen_ids = sorted(set(r.get("gen_id", -1) for r in records))
    per_gen = {}
    for gen_id in gen_ids:
        gen_recs = [r for r in records if r.get("gen_id") == gen_id]
        gen_ok = sum(1 for r in gen_recs if r.get("status") == "ok")
        gen_err = sum(1 for r in gen_recs if r.get("status") == "error")
        gen_eval_err = sum(1 for r in gen_recs if r.get("status") == "error" and r.get("failed_stage") == "evaluate")
        per_gen[gen_id] = {
            "total": len(gen_recs),
            "ok": gen_ok,
            "errors": gen_err,
            "evaluate_errors": gen_eval_err,
            "evaluate_error_rate": gen_eval_err / len(gen_recs) if gen_recs else 0.0,
        }

    # Expression patterns in evaluate failures (look for Log, Sqrt, Div, etc.)
    domain_pattern_counts = Counter()
    for r in eval_errors:
        expr = str(r.get("expr_short", ""))
        for pattern in ["Log", "Sqrt", "Div", "Pow", "Tan", "Acos", "Asin"]:
            if pattern in expr:
                domain_pattern_counts[pattern] += 1

    return {
        "total_records": total,
        "ok_records": len(ok),
        "error_records": len(errors),
        "overall_error_rate": len(errors) / total if total else 0.0,
        "stage_counts": dict(stage_counts),
        "evaluate_errors_total": len(eval_errors),
        "evaluate_error_rate": len(eval_errors) / total if total else 0.0,
        "evaluate_error_types": dict(eval_error_types),
        "evaluate_error_messages_top10": dict(eval_error_messages.most_common(10)),
        "domain_patterns_in_eval_failures": dict(domain_pattern_counts),
        "per_generation": per_gen,
    }


def main() -> int:
    print("=" * 72)
    print("D9 Quantification: Evaluate-Stage Failure Analysis")
    print("=" * 72)

    N_GENS = 10
    POP_SIZE = 200

    print(f"\nRunning {N_GENS} generations with pop={POP_SIZE}...")
    records = _collect_timings_multi_gen(n_gens=N_GENS, pop_max_size=POP_SIZE, seed=42)
    analysis = _analyze_evaluate_failures(records)

    print(f"\n{'─' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'─' * 60}")
    print(f"Total tree attempts:    {analysis['total_records']}")
    print(f"Successful (ok):        {analysis['ok_records']}")
    print(f"Failed (error):         {analysis['error_records']} ({analysis['overall_error_rate']:.1%})")
    print()
    print("Failures by stage:")
    for stage, count in sorted(analysis["stage_counts"].items(), key=lambda x: -x[1]):
        pct = count / analysis["total_records"] * 100 if analysis["total_records"] else 0
        print(f"  {stage:12s}: {count:5d} ({pct:.1f}%)")

    print(f"\n{'─' * 60}")
    print("EVALUATE-STAGE FAILURES (D9 focus)")
    print(f"{'─' * 60}")
    print(f"Evaluate failures:      {analysis['evaluate_errors_total']}")
    print(f"Evaluate failure rate:  {analysis['evaluate_error_rate']:.2%}")
    print()
    print("Error types:")
    for etype, count in sorted(analysis["evaluate_error_types"].items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    print()
    print("Error messages (top 10):")
    for msg, count in analysis["evaluate_error_messages_top10"].items():
        print(f"  [{count:3d}x] {msg[:100]}")
    print()
    print("Domain-sensitive operators in failed expressions:")
    for pattern, count in sorted(analysis["domain_patterns_in_eval_failures"].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    print(f"\n{'─' * 60}")
    print("PER-GENERATION BREAKDOWN")
    print(f"{'─' * 60}")
    print(f"{'Gen':>4s}  {'Total':>6s}  {'OK':>5s}  {'Err':>5s}  {'EvalErr':>7s}  {'EvalRate':>8s}")
    for gen_id, g in sorted(analysis["per_generation"].items()):
        print(
            f"{gen_id:4d}  {g['total']:6d}  {g['ok']:5d}  {g['errors']:5d}  "
            f"{g['evaluate_errors']:7d}  {g['evaluate_error_rate']:8.1%}"
        )

    # D9 recommendation
    eval_rate = analysis["evaluate_error_rate"]
    print(f"\n{'=' * 60}")
    print("D9 RECOMMENDATION")
    print(f"{'=' * 60}")
    if eval_rate < 0.02:
        print(f"Evaluate failure rate is LOW ({eval_rate:.2%}).")
        print("→ Domain pre-checks (Log/Sqrt/Div) are NOT worth implementing.")
        print("  The existing np.isfinite() guard is sufficient.")
    elif eval_rate < 0.10:
        print(f"Evaluate failure rate is MODERATE ({eval_rate:.2%}).")
        print("→ Domain pre-checks MIGHT be worthwhile for top failing operators.")
        print("  Check 'domain_patterns_in_eval_failures' above for candidates.")
    else:
        print(f"Evaluate failure rate is HIGH ({eval_rate:.2%}).")
        print("→ Domain pre-checks are RECOMMENDED for the top failing operators.")
        print("  This would avoid wasted evaluation cycles.")

    # Save full results
    output_path = Path(__file__).with_name("bench_d9_output.json")
    output_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
