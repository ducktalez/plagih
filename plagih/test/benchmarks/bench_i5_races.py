"""I5 validation: do multi-population races beat a single population?

Answers the open I5 question "benchmark races vs. single-population
baseline" from IMPLEMENTATION_PLAN.

Fairness
--------
The comparison is budget-matched on **total generations x population
size** (= total tree evaluations), which is the dominant cost (H4).
Three arms per seed:

  races      R races, pop P, G gens each        -> R*G*P evaluations
  base_long  1 race,  pop P, R*G gens           -> R*G*P evaluations
  base_wide  1 race,  pop R*P, G gens           -> R*G*P evaluations

`base_long` controls for "more generations", `base_wide` for "bigger
population" — races must beat *both* to be worth their complexity.

Metric: best (lowest) fitness reached.  Reported per seed plus
mean/std and pairwise win counts across seeds.

Run directly:
    python plagih/test/benchmarks/bench_i5_races.py
    python plagih/test/benchmarks/bench_i5_races.py --seeds 5 --gens 6
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from plagih.test.benchmarks._tree_creation_harness import (
    load_mountaincar_train_split,
    set_benchmark_seed,
)

# Arm labels
ARM_RACES = "races"
ARM_RACES_PLAIN = "races_plain"  # no diversity gate / no trunk reseed
ARM_BASE_LONG = "base_long"
ARM_BASE_WIDE = "base_wide"

# Fixed train split for every arm/race — only the RNG stream varies per seed.
DATA_SEED = 0


def _make_gp(rootdir: Path, pop_max_size: int, gen_end: int, data_seed: int):
    """GP instance with analysis off (P9: timing benchmarks).

    `data_seed` picks the train split — it must be identical for every arm
    and every race, otherwise fitness values are not comparable.
    """
    import plagih_gp
    from plagih.trees import ExplainableGP

    df_train, _ = load_mountaincar_train_split(seed=data_seed)
    return ExplainableGP.create(
        symbols=["cartVel", "cartPos"],
        df_train=df_train,
        rootdir=rootdir,
        operators=plagih_gp._build_active_test_operator_dict(),
        depth_max=7,
        nodes_max=35,
        pop_max_size=pop_max_size,
        gen_end=gen_end + 2,
        clip_range=(0.0, 2.0),
        error_metric="rmse",
        parallel=0,
        enable_analysis=False,
        verbose=False,
    )


def _best_fitness(candidates) -> float:
    return min((c.fitness for c in candidates), default=float("inf"))


def _run_races_arm(
    tmp_root: Path,
    n_races: int,
    pop: int,
    gens_per_race: int,
    seed: int,
    min_diversity: float,
    reseed_trunks: bool,
) -> Dict[str, Any]:
    import plagih_gp
    from plagih.population_races import run_races

    set_benchmark_seed(seed)
    races = [_make_gp(tmp_root / f"race{i}", pop, gens_per_race, DATA_SEED) for i in range(n_races)]
    t0 = time.perf_counter()
    try:
        result = run_races(
            races,
            plagih_gp._build_active_test_strategies(),
            n_epochs=gens_per_race,  # 1 generation per epoch -> frequent exchange
            gens_per_epoch=1,
            exchange_top_n=2,
            min_diversity=min_diversity,
            reseed_trunks=reseed_trunks,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        return {
            "best_fitness": _best_fitness(result.combined_pareto),
            "pareto_size": len(result.combined_pareto),
            "seconds": elapsed,
            "injected_total": sum(h.injected for h in result.history),
        }
    finally:
        for gp in races:
            gp.close()


def _run_single_arm(tmp_root: Path, name: str, pop: int, n_gens: int, seed: int) -> Dict[str, Any]:
    import plagih_gp

    set_benchmark_seed(seed)
    gp = _make_gp(tmp_root / name, pop, n_gens, DATA_SEED)
    strategies = plagih_gp._build_active_test_strategies()
    t0 = time.perf_counter()
    try:
        gp.gen_create_initial()
        for gen in range(n_gens):
            gp.run_generation(strategies, seed=seed + gen)
        return {
            "best_fitness": _best_fitness(gp.paretofront),
            "pareto_size": len(gp.paretofront),
            "seconds": time.perf_counter() - t0,
            "injected_total": 0,
        }
    finally:
        gp.close()


def run_one_seed(seed: int, n_races: int, pop: int, gens_per_race: int) -> Dict[str, Dict[str, Any]]:
    """Run all arms for one seed under a matched evaluation budget."""
    tmp_root = Path(tempfile.mkdtemp(prefix=f"plagih_i5_bench_{seed}_"))
    total_gens = n_races * gens_per_race
    try:
        return {
            ARM_RACES: _run_races_arm(
                tmp_root, n_races, pop, gens_per_race, seed, min_diversity=0.15, reseed_trunks=True
            ),
            ARM_RACES_PLAIN: _run_races_arm(
                tmp_root, n_races, pop, gens_per_race, seed, min_diversity=0.0, reseed_trunks=False
            ),
            ARM_BASE_LONG: _run_single_arm(tmp_root, "base_long", pop, total_gens, seed),
            ARM_BASE_WIDE: _run_single_arm(tmp_root, "base_wide", pop * n_races, gens_per_race, seed),
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def summarize(per_seed: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate arm results: mean/std fitness + pairwise win counts."""
    arms = [ARM_RACES, ARM_RACES_PLAIN, ARM_BASE_LONG, ARM_BASE_WIDE]
    fitness = {a: [s[a]["best_fitness"] for s in per_seed] for a in arms}
    seconds = {a: [s[a]["seconds"] for s in per_seed] for a in arms}

    def _stats(vals: List[float]) -> Dict[str, float]:
        finite = [v for v in vals if v < float("inf")]
        if not finite:
            return {"mean": float("inf"), "std": 0.0, "best": float("inf")}
        return {
            "mean": statistics.fmean(finite),
            "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
            "best": min(finite),
        }

    # Wins: how often does arm A strictly beat arm B on the same seed?
    wins: Dict[str, int] = {}
    for a in arms:
        for b in arms:
            if a == b:
                continue
            wins[f"{a}>{b}"] = sum(1 for s in per_seed if s[a]["best_fitness"] < s[b]["best_fitness"])

    return {
        "n_seeds": len(per_seed),
        "fitness": {a: _stats(fitness[a]) for a in arms},
        "seconds": {a: _stats(seconds[a]) for a in arms},
        "wins": wins,
        "per_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="I5 races vs single-population benchmark")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds (repetitions)")
    parser.add_argument("--races", type=int, default=3, help="number of races")
    parser.add_argument("--pop", type=int, default=60, help="population size per race")
    parser.add_argument("--gens", type=int, default=5, help="generations per race")
    args = parser.parse_args()

    total_gens = args.races * args.gens
    print("=" * 64)
    print("I5 BENCHMARK — races vs single population")
    print("=" * 64)
    print(f"seeds={args.seeds}  races={args.races}  pop={args.pop}  gens/race={args.gens}")
    print(f"budget per arm: {total_gens} generations x {args.pop} pop = {total_gens * args.pop} evaluations")
    print()

    per_seed: List[Dict[str, Dict[str, Any]]] = []
    for i in range(args.seeds):
        seed = 100 + i * 17
        print(f"--- seed {seed} ({i + 1}/{args.seeds}) ---", flush=True)
        res = run_one_seed(seed, args.races, args.pop, args.gens)
        for arm, r in res.items():
            print(f"  {arm:12s} fitness={r['best_fitness']:.5f}  {r['seconds']:6.1f}s  inj={r['injected_total']}")
        per_seed.append(res)

    summary = summarize(per_seed)

    print(f"\n{'=' * 64}")
    print("SUMMARY (lower fitness is better)")
    print("=" * 64)
    print(f"{'arm':14s} {'mean':>10s} {'std':>9s} {'best':>10s} {'mean_s':>8s}")
    for arm, st in summary["fitness"].items():
        sec = summary["seconds"][arm]["mean"]
        print(f"{arm:14s} {st['mean']:10.5f} {st['std']:9.5f} {st['best']:10.5f} {sec:8.1f}")

    print(f"\nHead-to-head wins (out of {summary['n_seeds']} seeds):")
    for key in [
        f"{ARM_RACES}>{ARM_BASE_LONG}",
        f"{ARM_RACES}>{ARM_BASE_WIDE}",
        f"{ARM_RACES}>{ARM_RACES_PLAIN}",
        f"{ARM_RACES_PLAIN}>{ARM_BASE_LONG}",
    ]:
        print(f"  {key:28s} {summary['wins'][key]}")

    # Verdict — win counts alone over-claim; require the effect to clear noise.
    print(f"\n{'=' * 64}")
    print("VERDICT")
    print("=" * 64)
    n = summary["n_seeds"]
    beats_long = summary["wins"][f"{ARM_RACES}>{ARM_BASE_LONG}"]
    beats_wide = summary["wins"][f"{ARM_RACES}>{ARM_BASE_WIDE}"]
    majority = beats_long > n / 2 and beats_wide > n / 2

    races_mean = summary["fitness"][ARM_RACES]["mean"]
    best_base = min(summary["fitness"][ARM_BASE_LONG]["mean"], summary["fitness"][ARM_BASE_WIDE]["mean"])
    gap = best_base - races_mean  # positive = races better
    noise = max(summary["fitness"][ARM_RACES]["std"], summary["fitness"][ARM_BASE_LONG]["std"])
    print(f"mean gap vs best baseline: {gap:+.5f}   (seed-to-seed std: {noise:.5f})")

    if majority and gap > noise:
        print("Races beat both baselines and the gap EXCEEDS seed noise.")
        print("-> Evidence that the race machinery pays off.")
    elif majority:
        print("Races win most seeds, but the gap is WITHIN seed noise.")
        print("-> Not significant at this budget. More seeds / larger runs")
        print("   are required before claiming a real improvement.")
    else:
        print("Races do NOT consistently beat the single-population baselines.")
        print("-> Complexity is not justified at this budget.")
    print(f"(seeds={n}; small samples — treat as a smoke signal, not proof.)")

    out = Path(__file__).with_name("bench_i5_output.json")
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
