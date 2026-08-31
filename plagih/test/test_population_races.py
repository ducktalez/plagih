"""Tests for plagih.population_races (I5 multi-population races)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plagih.parallel import Strategy
from plagih.population_races import (
    EpochStats,
    RaceResult,
    exchange_candidates,
    reseed_templates_from_trunks,
    run_races,
)
from plagih.trees import Add, ExplainableGP, Lt, Mul, Sub

# =============================================================================
# Fixtures
# =============================================================================


def _make_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    a = rng.uniform(-2, 2, n)
    b = rng.uniform(-2, 2, n)
    return pd.DataFrame({"a": a, "b": b, "action": a + 0.5 * b})


def _make_gp(tmp_path: Path, name: str) -> ExplainableGP:
    return ExplainableGP.create(
        symbols=["a", "b"],
        df_train=_make_df(),
        rootdir=tmp_path / name,
        operators={Add: 1, Mul: 1, Sub: 1, Lt: 1},
        depth_max=3,
        nodes_max=12,
        pop_max_size=8,
        gen_end=100,
        error_metric="rmse",
        verbose=False,
        parallel=False,
        enable_analysis=False,
    )


@pytest.fixture
def two_races(tmp_path):
    races = [_make_gp(tmp_path, "race_a"), _make_gp(tmp_path, "race_b")]
    for gp in races:
        gp.gen_create_initial()
    return races


STRATEGIES = [
    Strategy("mutation", rate=0.5, depth_goal=2, p_term=0.3),
    Strategy("random_new", rate=0.5, depths=[2], p_term=0.2),
]


# =============================================================================
# exchange_candidates
# =============================================================================


class TestExchangeCandidates:
    def test_injects_into_other_races(self, two_races):
        pop_before = [len(gp.pop_genepool) for gp in two_races]
        injected = exchange_candidates(two_races, top_n=2)

        assert len(injected) == 2
        for gp, before, inj in zip(two_races, pop_before, injected):
            assert len(gp.pop_genepool) == before + inj

    def test_donor_trees_are_copies(self, two_races):
        exchange_candidates(two_races, top_n=1)

        ids_a = {id(c.tree) for c in two_races[0].pop_genepool}
        ids_b = {id(c.tree) for c in two_races[1].pop_genepool}
        assert not ids_a & ids_b, "Races must not share tree instances"

    def test_top_n_zero_like(self, two_races):
        injected = exchange_candidates(two_races, top_n=0)
        assert injected == [0, 0]


# =============================================================================
# reseed_templates_from_trunks
# =============================================================================


class TestReseedTemplates:
    def test_last_race_stays_free(self, two_races):
        reseed_templates_from_trunks(two_races, min_trees=2, min_size=2)
        assert two_races[-1].evolve.origin_tree is None

    def test_template_is_frozen_if_set(self, two_races):
        # Exchange first so trunks are likely shared across races
        exchange_candidates(two_races, top_n=3)
        reseed_templates_from_trunks(two_races, min_trees=2, min_size=2)

        template = two_races[0].evolve.origin_tree
        if template is not None:  # trunk existence depends on random pops
            assert template.is_fix


# =============================================================================
# run_races
# =============================================================================


class TestRunRaces:
    def test_needs_two_races(self, tmp_path):
        gp = _make_gp(tmp_path, "solo")
        with pytest.raises(ValueError):
            run_races([gp], STRATEGIES)

    def test_basic_run(self, two_races):
        result = run_races(
            two_races,
            STRATEGIES,
            n_epochs=2,
            gens_per_epoch=1,
            exchange_top_n=1,
        )

        assert isinstance(result, RaceResult)
        assert len(result.history) == 4  # 2 epochs * 2 races
        assert all(isinstance(h, EpochStats) for h in result.history)
        assert result.combined_pareto, "Combined pareto must not be empty"

    def test_combined_pareto_sorted_and_copied(self, two_races):
        result = run_races(two_races, STRATEGIES, n_epochs=1, gens_per_epoch=1, exchange_top_n=0)

        fits = [c.fitness for c in result.combined_pareto]
        assert fits == sorted(fits)

        live_ids = {id(c.tree) for gp in two_races for c in gp.paretofront}
        result_ids = {id(c.tree) for c in result.combined_pareto}
        assert not live_ids & result_ids, "Result must hold copies"

    def test_auto_initialises_population(self, tmp_path):
        races = [_make_gp(tmp_path, "x"), _make_gp(tmp_path, "y")]
        # No gen_create_initial() here — run_races must do it
        result = run_races(races, STRATEGIES, n_epochs=1, gens_per_epoch=1)
        assert result.combined_pareto

    def test_reseed_flag(self, two_races):
        result = run_races(
            two_races,
            STRATEGIES,
            n_epochs=1,
            gens_per_epoch=1,
            exchange_top_n=2,
            reseed_trunks=True,
        )
        assert result.combined_pareto
        # Anti-core race must stay template-free
        assert two_races[-1].evolve.origin_tree is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
