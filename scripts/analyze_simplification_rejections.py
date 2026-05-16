#!/usr/bin/env python
"""D7 — Frequency analysis of rejected simplification patterns.

Parses GP run logs for simplification rejection/diagnostic warnings and
produces a summary of:
  - Rejection reasons (grew tree vs changed semantics)
  - Suspected failure stages (sympy_roundtrip, tree_node_grouping, both)
  - Operator families that appear most in rejected expressions
  - Tree growth patterns (original → simplified size)
  - Most common expression patterns

Usage:
    python scripts/analyze_simplification_rejections.py [log_file ...]

    If no log files are given, scans .results/**/run.log automatically.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns for log parsing
# ---------------------------------------------------------------------------

# "Simplification rejected (grew tree (6 → 9 nodes)), keeping original: <expr>"
# "Simplification rejected (changed semantics), keeping original: <expr>"
# "Simplification rejected (grew tree (6 → 9 nodes), changed semantics), keeping original: <expr>"
RE_REJECTED = re.compile(
    r"Simplification rejected \((?P<reasons>[^)]+)\),\s*keeping original:\s*(?P<expr>.+?)(?:\s*\|\s*similar cases suppressed=(?P<suppressed>\d+))?$"
)

# "Simplification diagnostic (suspected stage: sympy_roundtrip+grouping)"
RE_DIAGNOSTIC = re.compile(r"Simplification diagnostic \(suspected stage:\s*(?P<stage>[^)]+)\)")

# Extract "grew tree (X → Y nodes)" from reasons
RE_GREW = re.compile(r"grew tree \((\d+)\s*→\s*(\d+) nodes\)")

# "tree_simplification timed out after Xs"
RE_TIMEOUT = re.compile(r"tree_simplification timed out after ([\d.]+)s")

# Operators we want to count in rejected expressions
DOMAIN_OPERATORS = [
    "Min",
    "Max",
    "Clip",  # Piecewise-like (P12)
    "Abs",
    "Sign",
    "sign",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "log",
    "Log",
    "Sqrt",
    "sqrt",
    "Div",
    "DivFraction",
    "Pow",
    "PowRounded",
    "Square",
    "Scale",
    "Ifte",
    "Piecewise",
    "Exp",
    "exp",
]


def _count_operators_in_expr(expr: str) -> Counter:
    """Count known operator names appearing in an expression string."""
    counts = Counter()
    for op in DOMAIN_OPERATORS:
        # Match as word boundary to avoid partial matches
        pattern = re.compile(rf"\b{re.escape(op)}\b")
        n = len(pattern.findall(expr))
        if n > 0:
            counts[op] += n
    return counts


def parse_log(log_path: Path) -> dict:
    """Parse a single log file and return structured rejection data."""
    rejections = []
    diagnostics = []
    timeouts = []

    text = log_path.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        # Rejection lines
        m = RE_REJECTED.search(line)
        if m:
            reasons_str = m.group("reasons")
            expr = m.group("expr").strip()
            suppressed = int(m.group("suppressed") or 0)

            grew_match = RE_GREW.search(reasons_str)
            grew = grew_match is not None
            grew_from = int(grew_match.group(1)) if grew_match else None
            grew_to = int(grew_match.group(2)) if grew_match else None
            semantic = "changed semantics" in reasons_str

            ops = _count_operators_in_expr(expr)

            rejections.append(
                {
                    "reasons_raw": reasons_str,
                    "grew": grew,
                    "grew_from": grew_from,
                    "grew_to": grew_to,
                    "semantic_changed": semantic,
                    "suppressed": suppressed,
                    "expr": expr,
                    "operators": ops,
                    "expr_len": len(expr),
                }
            )
            continue

        # Diagnostic lines
        m = RE_DIAGNOSTIC.search(line)
        if m:
            diagnostics.append({"stage": m.group("stage")})
            continue

        # Timeout lines
        m = RE_TIMEOUT.search(line)
        if m:
            timeouts.append({"timeout_s": float(m.group(1))})

    return {
        "file": str(log_path),
        "rejections": rejections,
        "diagnostics": diagnostics,
        "timeouts": timeouts,
    }


def print_analysis(results: list[dict]) -> None:
    """Print a combined frequency analysis across all parsed logs."""
    all_rejections = []
    all_diagnostics = []
    all_timeouts = []

    for r in results:
        all_rejections.extend(r["rejections"])
        all_diagnostics.extend(r["diagnostics"])
        all_timeouts.extend(r["timeouts"])

    n_rej = len(all_rejections)
    # Count including suppressed similar cases
    n_total = sum(1 + r["suppressed"] for r in all_rejections)
    n_diag = len(all_diagnostics)
    n_timeouts = len(all_timeouts)

    print("=" * 72)
    print("D7 — Simplification Rejection Frequency Analysis")
    print("=" * 72)
    print(f"\nLog files analysed: {len(results)}")
    print(f"Total rejections (logged):         {n_rej}")
    print(f"Total rejections (incl suppressed): {n_total}")
    print(f"Diagnostic messages:                {n_diag}")
    print(f"Timeouts:                           {n_timeouts}")

    if not all_rejections:
        print("\nNo rejections found.")
        return

    # --- 1. Rejection reason breakdown ---
    print("\n" + "-" * 72)
    print("1. REJECTION REASONS")
    print("-" * 72)
    reason_counter = Counter()
    for r in all_rejections:
        weight = 1 + r["suppressed"]
        if r["grew"] and r["semantic_changed"]:
            reason_counter["grew + semantic"] += weight
        elif r["grew"]:
            reason_counter["grew only"] += weight
        elif r["semantic_changed"]:
            reason_counter["semantic only"] += weight
        else:
            reason_counter["other"] += weight
    for reason, count in reason_counter.most_common():
        print(f"  {reason:30s}  {count:5d}  ({100 * count / n_total:.1f}%)")

    # --- 2. Suspected stage breakdown ---
    print("\n" + "-" * 72)
    print("2. SUSPECTED FAILURE STAGE (from diagnostics)")
    print("-" * 72)
    stage_counter = Counter(d["stage"] for d in all_diagnostics)
    for stage, count in stage_counter.most_common():
        print(f"  {stage:35s}  {count:5d}  ({100 * count / max(n_diag, 1):.1f}%)")

    # --- 3. Operators in rejected expressions ---
    print("\n" + "-" * 72)
    print("3. OPERATORS IN REJECTED EXPRESSIONS")
    print("-" * 72)
    op_counter = Counter()
    op_by_reason: dict[str, Counter] = defaultdict(Counter)
    for r in all_rejections:
        weight = 1 + r["suppressed"]
        for op, count in r["operators"].items():
            op_counter[op] += count * weight
            if r["semantic_changed"]:
                op_by_reason["semantic"][op] += count * weight
            if r["grew"]:
                op_by_reason["grew"][op] += count * weight
    print(f"\n  {'Operator':20s} {'Total':>6s}  {'Semantic':>8s}  {'Grew':>6s}")
    print(f"  {'─' * 20} {'─' * 6}  {'─' * 8}  {'─' * 6}")
    for op, total in op_counter.most_common(20):
        sem = op_by_reason["semantic"].get(op, 0)
        grew = op_by_reason["grew"].get(op, 0)
        print(f"  {op:20s} {total:6d}  {sem:8d}  {grew:6d}")

    # --- 4. Tree growth analysis ---
    print("\n" + "-" * 72)
    print("4. TREE GROWTH ANALYSIS (grew-only rejections)")
    print("-" * 72)
    grew_entries = [r for r in all_rejections if r["grew"]]
    if grew_entries:
        growths = [(r["grew_from"], r["grew_to"]) for r in grew_entries]
        deltas = [t - f for f, t in growths]
        avg_from = sum(f for f, _ in growths) / len(growths)
        avg_to = sum(t for _, t in growths) / len(growths)
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
        print(f"  Cases: {len(grew_entries)}")
        print(f"  Avg original size: {avg_from:.1f} nodes")
        print(f"  Avg simplified size: {avg_to:.1f} nodes")
        print(f"  Avg growth: +{avg_delta:.1f} nodes")
        print(f"  Max growth: +{max_delta} nodes")

        # Bucket by growth amount
        growth_buckets = Counter()
        for d in deltas:
            if d <= 2:
                growth_buckets["+1-2"] += 1
            elif d <= 5:
                growth_buckets["+3-5"] += 1
            elif d <= 10:
                growth_buckets["+6-10"] += 1
            else:
                growth_buckets["+11+"] += 1
        print("\n  Growth distribution:")
        for bucket in ["+1-2", "+3-5", "+6-10", "+11+"]:
            c = growth_buckets.get(bucket, 0)
            print(f"    {bucket:8s}  {c:4d} cases")
    else:
        print("  No grew-only rejections.")

    # --- 5. Expression complexity (length) ---
    print("\n" + "-" * 72)
    print("5. EXPRESSION COMPLEXITY (string length of rejected exprs)")
    print("-" * 72)
    lengths = sorted(r["expr_len"] for r in all_rejections)
    if lengths:
        import statistics

        print(
            f"  Min: {lengths[0]}, Median: {statistics.median(lengths):.0f}, "
            f"Mean: {statistics.mean(lengths):.0f}, Max: {lengths[-1]}"
        )
        # Bucket
        len_buckets = Counter()
        for L in lengths:
            if L < 50:
                len_buckets["<50"] += 1
            elif L < 100:
                len_buckets["50-99"] += 1
            elif L < 200:
                len_buckets["100-199"] += 1
            elif L < 500:
                len_buckets["200-499"] += 1
            else:
                len_buckets["500+"] += 1
        for bucket in ["<50", "50-99", "100-199", "200-499", "500+"]:
            c = len_buckets.get(bucket, 0)
            print(f"    {bucket:10s}  {c:4d} cases")

    # --- 6. Timeout analysis ---
    if all_timeouts:
        print("\n" + "-" * 72)
        print("6. TIMEOUTS")
        print("-" * 72)
        print(f"  Total: {n_timeouts}")
        timeout_vals = [t["timeout_s"] for t in all_timeouts]
        print(f"  Timeout values: {sorted(set(timeout_vals))}")

    # --- 7. Operator co-occurrence in semantic rejections ---
    print("\n" + "-" * 72)
    print("7. OPERATOR CO-OCCURRENCE IN SEMANTIC REJECTIONS")
    print("-" * 72)
    semantic_entries = [r for r in all_rejections if r["semantic_changed"]]
    if semantic_entries:
        cooccur = Counter()
        for r in semantic_entries:
            ops = sorted(set(r["operators"].keys()))
            if len(ops) >= 2:
                for i in range(len(ops)):
                    for j in range(i + 1, len(ops)):
                        cooccur[(ops[i], ops[j])] += 1 + r["suppressed"]
        print(f"  Semantic rejections: {len(semantic_entries)}")
        print(f"\n  {'Pair':40s} {'Count':>6s}")
        print(f"  {'─' * 40} {'─' * 6}")
        for (a, b), count in cooccur.most_common(15):
            print(f"  {a + ' + ' + b:40s} {count:6d}")
    else:
        print("  No semantic rejections found.")

    # --- 8. Top rejected expression samples ---
    print("\n" + "-" * 72)
    print("8. SAMPLE REJECTED EXPRESSIONS (first 10 semantic)")
    print("-" * 72)
    for i, r in enumerate(semantic_entries[:10]):
        expr_short = r["expr"][:120] + ("..." if len(r["expr"]) > 120 else "")
        ops_str = ", ".join(f"{op}x{c}" for op, c in r["operators"].most_common(5))
        print(f"  [{i + 1}] {expr_short}")
        print(f"       ops: {ops_str}")
        print()


def main():
    if len(sys.argv) > 1:
        log_files = [Path(p) for p in sys.argv[1:]]
    else:
        # Auto-discover
        base = Path.cwd() / ".results"
        log_files = sorted(base.glob("**/run.log"))
        if not log_files:
            print("No log files found in .results/. Pass paths as arguments.")
            sys.exit(1)

    print(f"Scanning {len(log_files)} log file(s)...\n")

    results = []
    for lf in log_files:
        r = parse_log(lf)
        n = len(r["rejections"])
        if n > 0:
            print(
                f"  {lf.relative_to(Path.cwd())}: {n} rejections, "
                f"{len(r['diagnostics'])} diagnostics, {len(r['timeouts'])} timeouts"
            )
            results.append(r)

    if not results:
        print("No simplification rejections found in any log.")
        sys.exit(0)

    print()
    print_analysis(results)


if __name__ == "__main__":
    main()
