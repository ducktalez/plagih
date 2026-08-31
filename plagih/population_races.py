"""
Multi-Population Races for plagih GP (I5).

Runs several ExplainableGP instances ("races") side by side and lets them
exchange their best candidates between epochs.  Optionally reseeds race
templates from shared trunk structure (D5 §3.5 / population_merge).

Concepts:
- **Race**: One independent ExplainableGP instance with its own population.
- **Epoch**: A block of generations, followed by an exchange step.
- **Exchange**: Top Pareto candidates of every race are injected as copies
  into all other races.  Optionally gated by a **diversity filter**
  (normalised structural TED) so near-duplicates are not injected.
- **Trunk reseed** (optional): After each epoch, the combined Pareto front
  is mined for shared trunks; race templates (`Evolution.origin_tree`) are
  updated so new random trees grow around proven skeletons.  The last race
  always keeps `origin_tree=None` (anti-core: free exploration).

Usage::

    from plagih.population_races import run_races

    result = run_races(
        races=[gp_a, gp_b],
        strategies=strategies,
        n_epochs=3,
        gens_per_epoch=5,
        min_diversity=0.15,  # reject near-duplicate injections
    )
    best = result.combined_pareto[0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence

from plagih.logging_utils import log

if TYPE_CHECKING:
    from plagih.trees import Candidate
    from plagih.trees._gp_engine import ExplainableGP


@dataclass
class EpochStats:
    """Per-race snapshot after one epoch."""

    epoch: int
    race_idx: int
    gen_id: int
    pop_size: int
    pareto_size: int
    best_fitness: float
    injected: int = 0
    diversity: Optional[float] = None


@dataclass
class RaceResult:
    """Outcome of a multi-population race run.

    Attributes:
        combined_pareto: Non-dominated candidates over all races (copies).
        history: One :class:`EpochStats` per race per epoch.
    """

    combined_pareto: List[Candidate] = field(default_factory=list)
    history: List[EpochStats] = field(default_factory=list)


def _best_candidates(gp: ExplainableGP, top_n: int) -> List[Candidate]:
    """Top-n Pareto candidates by fitness."""
    front = sorted(gp.paretofront, key=lambda c: c.fitness)
    return front[:top_n]


def normalized_ted(tree_a, tree_b, mode: str = "structural") -> float:
    """Structural distance in ``[0, 1]``.

    TED normalised by the summed tree size, so results are comparable
    across differently sized trees.  ``0.0`` = identical structure,
    ``1.0`` = maximally different.

    Measured reference values (small trees, ``structural`` mode)::

        identical                      0.00
        Add(a,b) vs Mul(a,b)           0.17
        Add(a,b) vs Add(a,1)           0.17
        Add(a,Mul(b,2)) vs Add(a,b)    0.25
        Add(a,Mul(b,2)) vs 1           0.67

    Useful ``min_diversity`` band is therefore roughly ``0.1 .. 0.3``.

    Args:
        tree_a: First tree.
        tree_b: Second tree.
        mode: `TedConfig` mode (``"structural"`` by default — values of
            terminals are ignored, only shape/operators count).

    Returns:
        Normalised distance.  Returns ``1.0`` if TED fails.
    """
    from plagih.tree_complexity.tree_edit_distance import TedConfig

    n_total = len(tree_a) + len(tree_b)
    if n_total == 0:
        return 0.0
    try:
        dist = tree_a.compute_ted(tree_b, TedConfig(mode=mode)).distance
    except Exception:
        return 1.0
    return min(1.0, float(dist) / n_total)


def is_diverse_enough(
    tree,
    reference_trees: Sequence,
    min_distance: float,
    mode: str = "structural",
) -> bool:
    """Check whether *tree* is structurally novel vs. *reference_trees*.

    Cheap identity pre-filter (`str(tree)`) runs first; TED is only used
    when ``min_distance > 0`` — it is O(n²·m²), so keep the reference set
    small (Pareto fronts, not whole populations).

    Args:
        tree: Candidate tree to test.
        reference_trees: Trees already present in the target population.
        min_distance: Required minimum normalised distance (0 disables
            the TED check, duplicates are still rejected).
        mode: `TedConfig` mode forwarded to :func:`normalized_ted`.

    Returns:
        True when the tree should be accepted.
    """
    tree_str = str(tree)
    ref_strs = [str(t) for t in reference_trees]
    if tree_str in ref_strs:
        return False  # exact duplicate
    if min_distance <= 0.0:
        return True
    return all(normalized_ted(tree, ref, mode=mode) >= min_distance for ref in reference_trees)


def exchange_candidates(
    races: Sequence[ExplainableGP],
    top_n: int = 2,
    min_diversity: float = 0.0,
    diversity_mode: str = "structural",
) -> List[int]:
    """Inject each race's best Pareto candidates into all other races.

    Trees are copied (`fast_tree_copy`) and re-evaluated by the receiving
    race via `tree_to_candidate` — LUTs and fitness stay race-local.
    Failing injections (size/sympy errors) are skipped silently.

    Args:
        races: Participating GP instances.
        top_n: Pareto candidates offered per donor race.
        min_diversity: Minimum normalised TED distance a donor must have
            to every tree already in the receiver's Pareto front (plus the
            donors accepted in this round).  ``0.0`` only filters exact
            duplicates.
        diversity_mode: `TedConfig` mode used for the distance.

    Returns:
        Injected-candidate count per race.
    """
    from plagih.trees._nodes import fast_tree_copy

    donors_per_race = [_best_candidates(gp, top_n) for gp in races]
    injected = [0] * len(races)

    for ii, gp in enumerate(races):
        # Reference set stays small: own Pareto front + accepted donors
        refs = [c.tree for c in gp.paretofront]
        for jj, donors in enumerate(donors_per_race):
            if ii == jj:
                continue
            for cand in donors:
                tree = fast_tree_copy(cand.tree)
                tree.repair_all()
                if not is_diverse_enough(tree, refs, min_diversity, mode=diversity_mode):
                    continue
                try:
                    new_cand = gp.tree_to_candidate(tree, raise_if_useless=False, tag="race_exchange")
                except Exception:
                    continue  # not viable in this race — skip
                gp.pop_genepool.append(new_cand)
                refs.append(new_cand.tree)
                injected[ii] += 1
        if injected[ii]:
            gp.run_update_paretofront(gp.pop_genepool)

    return injected


def reseed_templates_from_trunks(
    races: Sequence[ExplainableGP],
    min_trees: int = 2,
    min_size: int = 3,
) -> int:
    """Set race `origin_tree` templates from combined-Pareto trunks.

    Race ``i`` gets template ``i`` (best trunk first).  The **last** race is
    always reset to ``origin_tree=None`` — the anti-core race keeps
    exploring freely.  Races without a matching template are left unchanged.

    Returns:
        Number of races that received a new template.
    """
    from plagih.population_merge import suggest_origin_templates

    combined = [c for gp in races for c in gp.paretofront]
    if len(combined) < min_trees:
        return 0

    templates = suggest_origin_templates(combined, top_n=len(races), min_trees=min_trees, min_size=min_size)

    changed = 0
    for ii, gp in enumerate(races):
        if ii == len(races) - 1:
            gp.evolve.origin_tree = None  # anti-core
            continue
        if ii < len(templates):
            gp.evolve.origin_tree = templates[ii][0]
            changed += 1
    return changed


def population_diversity(gp: ExplainableGP, sample_n: int = 8, mode: str = "structural") -> float:
    """Mean pairwise normalised TED over a Pareto-front sample.

    Bounded work: at most ``sample_n`` trees → ``sample_n²/2`` TED calls.

    Returns:
        Mean distance in ``[0, 1]``; ``0.0`` for fewer than 2 trees.
    """
    trees = [c.tree for c in sorted(gp.paretofront, key=lambda c: c.fitness)[:sample_n]]
    if len(trees) < 2:
        return 0.0
    dists = [normalized_ted(trees[i], trees[j], mode=mode) for i in range(len(trees)) for j in range(i + 1, len(trees))]
    return float(sum(dists) / len(dists))


def run_races(
    races: Sequence[ExplainableGP],
    strategies,
    n_epochs: int = 3,
    gens_per_epoch: int = 5,
    exchange_top_n: int = 2,
    min_diversity: float = 0.0,
    reseed_trunks: bool = False,
    track_diversity: bool = False,
    seed: Optional[int] = None,
) -> RaceResult:
    """Run multiple GP races with periodic candidate exchange.

    Args:
        races: Pre-configured ExplainableGP instances (>= 2).  Instances
            without an initial population are initialised automatically.
        strategies: Strategy list passed to every `run_generation` call.
        n_epochs: Number of epoch blocks.
        gens_per_epoch: Generations per race per epoch.
        exchange_top_n: Pareto candidates exchanged per donor race
            (0 disables exchange).
        min_diversity: Minimum normalised TED distance for an injected
            candidate (see :func:`exchange_candidates`).
        reseed_trunks: Update race `origin_tree` templates from combined
            Pareto trunks after each exchange (last race stays free).
        track_diversity: Record per-race Pareto diversity in the history
            (costs TED calls — off by default).
        seed: Base seed.  A distinct seed is derived per race and
            generation — races must not share an RNG stream, otherwise
            they evolve identically and exchange degenerates to duplicates.

    Returns:
        :class:`RaceResult` with combined Pareto front and history.
    """
    from plagih.paretofront import pareto_from_pop
    from plagih.trees._nodes import fast_tree_copy

    if len(races) < 2:
        raise ValueError("Need at least 2 races")

    for gp_idx, gp in enumerate(races):
        if not gp.pop_genepool:
            # Distinct seeds — identical seeds would make races clones
            gp.gen_create_initial(seed=None if seed is None else seed + 1000 * gp_idx)

    result = RaceResult()

    for epoch in range(n_epochs):
        for race_idx, gp in enumerate(races):
            for gen in range(gens_per_epoch):
                gen_seed = None
                if seed is not None:
                    # Unique per race AND generation, else all races evolve alike
                    gen_seed = seed + 1000 * race_idx + 10 * epoch + gen
                gp.run_generation(strategies, seed=gen_seed)

        injected = [0] * len(races)
        if exchange_top_n > 0:
            injected = exchange_candidates(races, top_n=exchange_top_n, min_diversity=min_diversity)

        if reseed_trunks:
            n_reseeded = reseed_templates_from_trunks(races)
            if n_reseeded:
                log("ggg", f"Races epoch {epoch}: {n_reseeded} templates reseeded from trunks")

        for race_idx, gp in enumerate(races):
            best_fit = min((c.fitness for c in gp.paretofront), default=float("inf"))
            result.history.append(
                EpochStats(
                    epoch=epoch,
                    race_idx=race_idx,
                    gen_id=gp.gen_id,
                    pop_size=len(gp.pop_genepool),
                    pareto_size=len(gp.paretofront),
                    best_fitness=float(best_fit),
                    injected=injected[race_idx],
                    diversity=population_diversity(gp) if track_diversity else None,
                )
            )

    # Combined front over copies — result independent of live races
    from plagih.trees import Candidate

    all_cands = []
    for gp in races:
        for c in gp.paretofront:
            tree = fast_tree_copy(c.tree)
            tree.repair_all()
            all_cands.append(Candidate(tree, fitness=c.fitness, parsimony=c.parsimony, tag=c.get_tag()))
    result.combined_pareto = sorted(pareto_from_pop(all_cands), key=lambda c: c.fitness)

    return result
