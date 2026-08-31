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
    is_diverse_enough,
    normalized_ted,
    population_diversity,
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

    def test_min_diversity_blocks_all(self, two_races):
        """Unreachable threshold -> nothing injected."""
        pop_before = [len(gp.pop_genepool) for gp in two_races]
        injected = exchange_candidates(two_races, top_n=3, min_diversity=1.0)

        assert injected == [0, 0]
        assert [len(gp.pop_genepool) for gp in two_races] == pop_before

    def test_no_exact_duplicates_injected(self, two_races):
        """Same race exchanged twice must not add the identical tree twice."""
        exchange_candidates(two_races, top_n=2)
        pop_after_first = [len(gp.pop_genepool) for gp in two_races]
        exchange_candidates(two_races, top_n=2)

        # Second round may still add novel trees, but never exact duplicates
        for gp in two_races:
            exchanged = [str(c.tree) for c in gp.pop_genepool if "race_exchange" in c.tag]
            assert len(exchanged) == len(set(exchanged))
        assert all(after >= before for after, before in zip([len(g.pop_genepool) for g in two_races], pop_after_first))


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
# Diversity helpers
# =============================================================================


class TestNormalizedTed:
    def test_identical_trees_zero(self):
        import sympy

        from plagih.trees import Symbol

        a = Add(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")))
        b = Add(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")))
        a.repair_all()
        b.repair_all()

        assert normalized_ted(a, b) == pytest.approx(0.0)

    def test_different_trees_positive(self):
        import sympy

        from plagih.trees import Number, Square, Symbol

        a = Add(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b")))
        b = Square(Number(1.0))
        a.repair_all()
        b.repair_all()

        assert normalized_ted(a, b) > 0.0

    def test_bounded_zero_to_one(self):
        import sympy

        from plagih.trees import Number, Symbol

        a = Add(Symbol(sympy.Symbol("a")), Mul(Symbol(sympy.Symbol("b")), Number(2.0)))
        b = Number(1.0)
        a.repair_all()
        b.repair_all()

        d = normalized_ted(a, b)
        assert 0.0 <= d <= 1.0

    def test_symmetric(self):
        import sympy

        from plagih.trees import Number, Symbol

        a = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        b = Mul(Symbol(sympy.Symbol("b")), Number(2.0))
        a.repair_all()
        b.repair_all()

        assert normalized_ted(a, b) == pytest.approx(normalized_ted(b, a))

    def test_structural_mode_ignores_values(self):
        import sympy

        from plagih.trees import Number, Symbol

        a = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        b = Add(Symbol(sympy.Symbol("a")), Number(99.0))
        a.repair_all()
        b.repair_all()

        assert normalized_ted(a, b, mode="structural") == pytest.approx(0.0)
        assert normalized_ted(a, b, mode="full") > 0.0


class TestIsDiverseEnough:
    def _trees(self):
        import sympy

        from plagih.trees import Number, Symbol

        same = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        same.repair_all()
        other = Mul(Symbol(sympy.Symbol("b")), Number(2.0))
        other.repair_all()
        return same, other

    def test_exact_duplicate_rejected(self):
        same, _ = self._trees()
        clone, _ = self._trees()
        assert is_diverse_enough(clone, [same], min_distance=0.0) is False

    def test_novel_tree_accepted(self):
        same, other = self._trees()
        assert is_diverse_enough(other, [same], min_distance=0.0) is True

    def test_empty_reference_accepts(self):
        _, other = self._trees()
        assert is_diverse_enough(other, [], min_distance=0.5) is True

    def test_high_threshold_rejects_similar(self):
        same, other = self._trees()
        # Threshold 1.0 is unreachable for any real pair
        assert is_diverse_enough(other, [same], min_distance=1.0) is False


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

    def test_min_diversity_limits_injections(self, two_races):
        """Unreachable threshold must block every injection."""
        result = run_races(
            two_races,
            STRATEGIES,
            n_epochs=1,
            gens_per_epoch=1,
            exchange_top_n=3,
            min_diversity=1.0,
        )
        assert all(h.injected == 0 for h in result.history)

    def test_track_diversity_populates_history(self, two_races):
        result = run_races(
            two_races,
            STRATEGIES,
            n_epochs=1,
            gens_per_epoch=1,
            track_diversity=True,
        )
        assert all(h.diversity is not None for h in result.history)
        assert all(0.0 <= h.diversity <= 1.0 for h in result.history)

    def test_diversity_none_by_default(self, two_races):
        result = run_races(two_races, STRATEGIES, n_epochs=1, gens_per_epoch=1)
        assert all(h.diversity is None for h in result.history)


class TestPopulationDiversity:
    def test_bounded(self, two_races):
        for gp in two_races:
            d = population_diversity(gp)
            assert 0.0 <= d <= 1.0

    def test_single_tree_is_zero(self, two_races):
        gp = two_races[0]
        gp.paretofront = gp.paretofront[:1]
        assert population_diversity(gp) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
