"""
Benchmark: canonicalize_children() sort-key strategies.

Compares three sort-key approaches for canonicalize_children():
  (A) represent_str()         — current implementation (string sort key)
  (B) len(child) primary, represent_str() tiebreaker
  (C) len(child) only

Measures wall-clock time on random trees of increasing size.
Run directly:  python plagih/test/benchmarks/bench_canonicalize.py
"""

from __future__ import annotations

import copy
import statistics
import time
from typing import Callable, List

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from plagih.trees import (
    Abs,
    Add,
    And,
    Cos,
    Evolution,
    Le,
    Lt,
    Max,
    Min,
    Mul,
    Node,
    Not,
    Or,
    Sign,
    Sin,
    Sqrt,
    Square,
    Sub,
)

# ---------------------------------------------------------------------------
# Sort-key variants
# ---------------------------------------------------------------------------


def _canonicalize_str(tree: Node) -> None:
    """(A) Status quo: sort by represent_str()."""
    if tree.is_term():
        return
    for cc in tree.get_childs():
        _canonicalize_str(cc)
    if getattr(tree, "is_commutative", False):
        tree.childs.sort(key=lambda c: c.represent_str(show_fixed_hint=False, cut_terms=False))


def _canonicalize_len_str(tree: Node) -> None:
    """(B) Primary: len(child), tiebreaker: represent_str()."""
    if tree.is_term():
        return
    for cc in tree.get_childs():
        _canonicalize_len_str(cc)
    if getattr(tree, "is_commutative", False):
        tree.childs.sort(key=lambda c: (len(c), c.represent_str(show_fixed_hint=False, cut_terms=False)))


def _canonicalize_len_only(tree: Node) -> None:
    """(C) Sort only by subtree size — cheapest, but no tiebreaker."""
    if tree.is_term():
        return
    for cc in tree.get_childs():
        _canonicalize_len_only(cc)
    if getattr(tree, "is_commutative", False):
        tree.childs.sort(key=lambda c: len(c))


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------


def _make_evolution() -> Evolution:
    """Create an Evolution instance with a rich operator set."""
    return Evolution(
        symbol_list=["a", "b", "c", "d"],
        operators={
            Add: 3,
            Mul: 3,
            Sub: 1,
            Abs: 1,
            Sign: 1,
            Square: 1,
            Sqrt: 0.5,
            Sin: 0.5,
            Cos: 0.5,
            Min: 1,
            Max: 1,
            Lt: 1,
            Le: 1,
            And: 1,
            Or: 1,
            Not: 1,
        },
        depth_max=8,
        nodes_max=500,
        allow_chain=False,
    )


def _generate_random_trees(evo: Evolution, count: int, depth: int) -> List[Node]:
    """Generate *count* random trees with target depth."""
    trees = []
    attempts = 0
    while len(trees) < count and attempts < count * 10:
        attempts += 1
        try:
            t = evo.evolve_create_random(xt_out=float, depth_max_local=depth)
            t.repair_all(depth=0)
            if len(t) >= 3:
                trees.append(t)
        except Exception:
            continue
    return trees


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _benchmark_variant(
    trees: List[Node],
    fn: Callable[[Node], None],
    label: str,
    warmup: int = 2,
    repeats: int = 10,
) -> dict:
    """Time a canonicalize variant on deep-copies of the given trees."""

    # Warmup
    for _ in range(warmup):
        for t in trees:
            tc = copy.deepcopy(t)
            fn(tc)

    times_ms = []
    for _ in range(repeats):
        copies = [copy.deepcopy(t) for t in trees]
        t0 = time.perf_counter()
        for tc in copies:
            fn(tc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

    return {
        "label": label,
        "trees": len(trees),
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "per_tree_us": statistics.mean(times_ms) / len(trees) * 1000.0,
    }


def _check_equivalence(trees: List[Node]) -> bool:
    """Check if (A) and (B) produce the same canonical form for all trees."""
    mismatches = 0
    for t in trees:
        t_a = copy.deepcopy(t)
        t_b = copy.deepcopy(t)
        _canonicalize_str(t_a)
        _canonicalize_len_str(t_b)
        if t_a.represent_str() != t_b.represent_str():
            mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("canonicalize_children() Sort-Key Benchmark")
    print("=" * 72)

    evo = _make_evolution()

    VARIANTS = [
        (_canonicalize_str, "(A) represent_str"),
        (_canonicalize_len_str, "(B) len+str tiebreak"),
        (_canonicalize_len_only, "(C) len only"),
    ]

    # Benchmark at different tree depths/sizes
    CONFIGS = [
        (50, 3, "depth=3, ~small"),
        (50, 5, "depth=5, ~medium"),
        (30, 7, "depth=7, ~large"),
    ]

    for n_trees, depth, desc in CONFIGS:
        print(f"\n--- {desc} ({n_trees} trees, depth_max={depth}) ---")
        trees = _generate_random_trees(evo, n_trees, depth)
        if not trees:
            print("  (no trees generated, skipping)")
            continue

        avg_size = statistics.mean(len(t) for t in trees)
        max_size = max(len(t) for t in trees)
        print(f"  Generated {len(trees)} trees, avg_size={avg_size:.1f}, max_size={max_size}")

        # Check equivalence
        mismatches = _check_equivalence(trees)
        if mismatches > 0:
            print(f"  ⚠ {mismatches}/{len(trees)} trees differ between (A) and (B)")
        else:
            print("  ✓ (A) and (B) produce identical canonical forms for all trees")

        print(f"  {'Variant':<24} {'mean_ms':>8} {'median':>8} {'stdev':>8} {'per_tree_µs':>12}")
        print(f"  {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")

        for fn, label in VARIANTS:
            result = _benchmark_variant(trees, fn, label)
            print(
                f"  {result['label']:<24} "
                f"{result['mean_ms']:>7.2f}ms "
                f"{result['median_ms']:>7.2f}ms "
                f"{result['stdev_ms']:>7.2f}ms "
                f"{result['per_tree_us']:>10.1f}µs"
            )

    print(f"\n{'=' * 72}")
    print("Done.")
    print()
    print("Interpretation:")
    print("  (A) = current implementation.")
    print("  (B) = cheaper primary key (len), string only as tiebreaker.")
    print("       If (B) is faster AND produces same results → switch.")
    print("  (C) = cheapest, but may produce different canonical forms")
    print("       (equivalent semantics, different LUT keys).")


if __name__ == "__main__":
    main()
