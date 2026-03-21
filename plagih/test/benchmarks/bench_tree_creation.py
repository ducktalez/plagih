"""Focused benchmark for tree-creation bottlenecks.

Run directly:
    python plagih/test/benchmarks/bench_tree_creation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path (benchmarks/ → test/ → plagih/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from plagih.test.benchmarks._tree_creation_harness import run_tree_creation_benchmarks, save_benchmark_summary


def _print_summary_block(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"records={summary['records']} | ok={summary['ok_records']} | errors={summary['error_records']} | "
        f"mean={summary['mean_total_ms']:.2f}ms | p95={summary['p95_total_ms']:.2f}ms | max={summary['max_total_ms']:.2f}ms"
    )
    print(
        f"max_create={summary['max_create_ms']:.2f}ms | "
        f"max_simplify={summary['max_simplify_ms']:.2f}ms | "
        f"max_evaluate={summary['max_evaluate_ms']:.2f}ms"
    )
    print(f"dominant_phase_counts={summary['dominant_phase_counts']}")
    print(f"failed_stage_counts={summary['failed_stage_counts']}")
    if summary["top_slowest"]:
        slowest = summary["top_slowest"][0]
        print(
            f"slowest: tag={slowest['tag']} | total={slowest['total_ms']:.2f}ms | "
            f"parsim={slowest['parsimony']} | expr={slowest['expr_short']}"
        )


def main() -> int:
    print("=" * 72)
    print("Tree Creation Benchmark")
    print("=" * 72)

    summary = run_tree_creation_benchmarks(pop_max_size=120, seed=123)

    for name, block in summary.items():
        _print_summary_block(name, block)

    output_path = Path(__file__).with_name("bench_tree_creation_output.json")
    save_benchmark_summary(summary, output_path)
    print(f"\nSaved summary to: {output_path}")
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
