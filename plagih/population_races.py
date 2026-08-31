"""
Multi-Population Races for plagih GP (I5).

Runs several ExplainableGP instances ("races") side by side and lets them
exchange their best candidates between epochs.  Optionally reseeds race
templates from shared trunk structure (D5 §3.5 / population_merge).

Concepts:
- **Race**: One independent ExplainableGP instance with its own population.
- **Epoch**: A block of generations, followed by an exchange step.
- **Exchange**: Top Pareto candidates of every race are injected as copies
  into all other races.
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


def exchange_candidates(races: Sequence[ExplainableGP], top_n: int = 2) -> List[int]:
    """Inject each race's best Pareto candidates into all other races.

    Trees are copied (`fast_tree_copy`) and re-evaluated by the receiving
    race via `tree_to_candidate` — LUTs and fitness stay race-local.
    Failing injections (size/sympy errors) are skipped silently.

    Returns:
        Injected-candidate count per race.
    """
    from plagih.trees._nodes import fast_tree_copy

    donors_per_race = [_best_candidates(gp, top_n) for gp in races]
    injected = [0] * len(races)

    for ii, gp in enumerate(races):
        for jj, donors in enumerate(donors_per_race):
            if ii == jj:
                continue
            for cand in donors:
                tree = fast_tree_copy(cand.tree)
                tree.repair_all()
                try:
                    new_cand = gp.tree_to_candidate(tree, raise_if_useless=False, tag="race_exchange")
                except Exception:
                    continue  # not viable in this race — skip
                gp.pop_genepool.append(new_cand)
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


def run_races(
    races: Sequence[ExplainableGP],
    strategies,
    n_epochs: int = 3,
    gens_per_epoch: int = 5,
    exchange_top_n: int = 2,
    reseed_trunks: bool = False,
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
        reseed_trunks: Update race `origin_tree` templates from combined
            Pareto trunks after each exchange (last race stays free).
        seed: Optional seed forwarded to generation runs.

    Returns:
        :class:`RaceResult` with combined Pareto front and history.
    """
    from plagih.paretofront import pareto_from_pop
    from plagih.trees._nodes import fast_tree_copy

    if len(races) < 2:
        raise ValueError("Need at least 2 races")

    for gp in races:
        if not gp.pop_genepool:
            gp.gen_create_initial(seed=seed)

    result = RaceResult()

    for epoch in range(n_epochs):
        for race_idx, gp in enumerate(races):
            for _ in range(gens_per_epoch):
                gp.run_generation(strategies, seed=seed)

        injected = [0] * len(races)
        if exchange_top_n > 0:
            injected = exchange_candidates(races, top_n=exchange_top_n)

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
