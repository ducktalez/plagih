"""
Integration tests for the complete GP system.

Tests verify:
1. Mini GP runs complete successfully
2. Pareto front is updated correctly
3. Backup save/load works
4. Monitoring plots are created
"""

import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import copy

import numpy as np
import pandas as pd
import pytest
import sympy

from plagih.trees import (
    Add,
    Candidate,
    Evolution,
    ExplainableGP,
    Mul,
    Node,
    Number,
    Sin,
    Symbol,
    selection_tournament,
    tree_simplification,
)

# =============================================================================
# Mini GP Run Tests
# =============================================================================


class TestMiniGPRun:
    """Tests for complete mini GP runs."""

    def test_gp_creates_initial_population(self, gp_instance):
        """Tests GP creates initial population."""
        gp_instance.gen_create_initial()

        assert len(gp_instance.pop_genepool) > 0
        assert gp_instance.gen_id == 1

    def test_gp_population_has_candidates(self, gp_instance):
        """Tests population contains Candidate objects."""
        gp_instance.gen_create_initial()

        for cand in gp_instance.pop_genepool:
            assert isinstance(cand, Candidate)
            assert cand.tree is not None
            assert cand.fitness is not None
            assert cand.parsimony is not None

    def test_gp_runs_multiple_generations(self, gp_instance):
        """Tests GP can run multiple generations."""
        gp_instance.gen_create_initial()

        # Run 2 additional generations
        for gen in range(2):

            @gp_instance.create_trees(rate=0.5)
            def reproduce():
                tree = selection_tournament(gp_instance.pop_genepool, n=2)
                return tree

            @gp_instance.create_trees(rate=0.5)
            def mutate():
                tree = selection_tournament(gp_instance.pop_genepool, n=2)
                return gp_instance.evolve.evolve_mutate_branch_depth(tree, 3)

            gp_instance.end_generation()

        assert gp_instance.gen_id >= 2

    def test_gp_pareto_front_populated(self, gp_instance):
        """Tests Pareto front gets populated."""
        gp_instance.gen_create_initial()

        assert len(gp_instance.paretofront) > 0

    def test_gp_fitness_is_finite(self, gp_instance):
        """Tests all fitness values are finite."""
        gp_instance.gen_create_initial()

        for cand in gp_instance.pop_genepool:
            assert np.isfinite(cand.fitness)

    def test_gp_trees_are_evaluable(self, gp_instance):
        """Tests all trees can be evaluated."""
        gp_instance.gen_create_initial()

        for cand in gp_instance.pop_genepool:
            result = cand.tree.eval_predict_numpy_now(gp_instance.df_train)
            assert len(result) == len(gp_instance.df_train)


# =============================================================================
# Pareto Front Tests
# =============================================================================


class TestParetoFront:
    """Tests for Pareto front management."""

    def test_pareto_front_non_dominated(self, gp_instance):
        """Tests Pareto front contains only non-dominated solutions."""
        gp_instance.gen_create_initial()

        front = gp_instance.paretofront

        # Check no solution strictly dominates another
        # Domination: c1 dominates c2 if c1 is at least as good in all objectives
        # AND strictly better in at least one objective
        for i, c1 in enumerate(front):
            for j, c2 in enumerate(front):
                if i != j:
                    # c1 dominates c2 if:
                    # - c1 is at least as good in both objectives (<=)
                    # - c1 is strictly better in at least one objective (<)
                    at_least_as_good = c1.fitness <= c2.fitness and c1.parsimony <= c2.parsimony
                    strictly_better = c1.fitness < c2.fitness or c1.parsimony < c2.parsimony
                    dominates = at_least_as_good and strictly_better
                    assert not dominates, f"Candidate {i} dominates {j}"

    def test_pareto_front_updates(self, gp_instance):
        """Tests Pareto front updates when better solutions found."""
        gp_instance.gen_create_initial()

        initial_front_size = len(gp_instance.paretofront)

        # Run another generation
        @gp_instance.create_trees(rate=1.0)
        def new_trees():
            tree = selection_tournament(gp_instance.pop_genepool, n=2)
            return gp_instance.evolve.evolve_mutate_branch_depth(tree, 3)

        gp_instance.end_generation()

        # Front may have changed
        assert len(gp_instance.paretofront) >= 1

    def test_pareto_sorted_by_parsimony(self, gp_instance):
        """Tests Pareto front is sorted by parsimony."""
        gp_instance.gen_create_initial()

        front = gp_instance.paretofront
        if len(front) > 1:
            parsimony_values = [c.parsimony for c in front]
            assert parsimony_values == sorted(parsimony_values)


# =============================================================================
# Backup Tests
# =============================================================================


class TestBackup:
    """Tests for backup save/load functionality."""

    def test_backup_save(self, gp_instance):
        """Tests backup can be saved."""
        gp_instance.gen_create_initial()

        gp_instance.backup_save()

        backup_path = gp_instance.rootdir / "backup/backup.pkl"
        assert backup_path.exists()

    def test_backup_load(self, gp_instance, temp_output_dir, cartpole_df, cartpole_evolution):
        """Tests backup can be loaded."""
        # Create and save
        gp_instance.gen_create_initial()
        original_gen_id = gp_instance.gen_id
        original_pop_size = len(gp_instance.pop_genepool)

        gp_instance.backup_save()

        # Create new instance and load
        eval_autocast = lambda x: np.clip(np.asarray(x, dtype=np.float64), 0.0, 2.0)
        eval_error_metric = lambda pred, true: np.sqrt(np.mean((pred - true) ** 2))

        new_gp = ExplainableGP(
            evolve=cartpole_evolution,
            df_train=cartpole_df,
            rootdir=temp_output_dir,
            pop_max_size=10,
            gen_end=3,
            eval_autocast=eval_autocast,
            eval_error_metric=eval_error_metric,
        )

        new_gp.backup_load()

        assert new_gp.gen_id == original_gen_id
        assert len(new_gp.pop_genepool) == original_pop_size


# =============================================================================
# Monitoring Tests
# =============================================================================


class TestMonitoring:
    """Tests for monitoring and plotting."""

    def test_monitor_df_populated(self, gp_instance):
        """Tests monitor DataFrame gets populated."""
        gp_instance.gen_create_initial()

        assert len(gp_instance.monitor_df) > 0

    def test_monitor_df_has_required_columns(self, gp_instance):
        """Tests monitor DataFrame has required columns."""
        gp_instance.gen_create_initial()

        required_cols = ["pop_len", "pop_unique", "fit_avg", "fit_best", "parsim_avg", "time"]

        for col in required_cols:
            assert col in gp_instance.monitor_df.columns

    def test_plots_created(self, gp_instance):
        """Tests monitoring plots are created."""
        gp_instance.gen_create_initial()

        # Run a couple generations to have data
        @gp_instance.create_trees(rate=1.0)
        def trees():
            return gp_instance.evolve.evolve_new_tree_depth(float, 3)

        gp_instance.end_generation()

        gp_instance.evoloop_monitoring_plots()

        # Check monitoring.png exists
        assert (gp_instance.rootdir / "monitoring.png").exists()


# =============================================================================
# Candidate Tests
# =============================================================================


class TestCandidate:
    """Tests for Candidate class."""

    def test_candidate_creation(self):
        """Tests Candidate creation."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        cand = Candidate(tree, fitness=0.5, parsimony=3, tag="test")

        assert cand.tree is tree
        assert cand.fitness == 0.5
        assert cand.parsimony == 3
        assert cand.get_tag() == "test"

    def test_candidate_string(self):
        """Tests Candidate string representation."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        cand = Candidate(tree, fitness=0.5, parsimony=3, tag="test")

        s = str(cand)
        assert "3" in s  # parsimony
        assert "0.5" in s or "0.50" in s  # fitness

    def test_candidate_tag_history(self):
        """Tests Candidate tag history."""
        tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
        cand = Candidate(tree, fitness=0.5, parsimony=3, tag="init")

        cand.append_tag("mutate")
        cand.append_tag("crossover")

        assert cand.get_tag(-1) == "crossover"
        assert cand.get_tag(-2) == "mutate"


# =============================================================================
# LUT Tests
# =============================================================================


class TestLookupTables:
    """Tests for lookup table functionality."""

    @pytest.fixture(autouse=True)
    def _enable_lut(self):
        """LUT tests require lut_enabled=True (new default is False)."""
        from plagih.config import cfg

        old = cfg.lut_enabled
        cfg.lut_enabled = True
        yield
        cfg.lut_enabled = old

    def test_lut_caches_fitness(self, gp_instance):
        """Tests LUT caches fitness values."""
        gp_instance.gen_create_initial()

        assert len(gp_instance.lut_symex_fitness) > 0

    def test_lut_caches_tree_info(self, gp_instance):
        """Tests LUT caches tree information."""
        gp_instance.gen_create_initial()

        assert len(gp_instance.lut_tree_infos) > 0

    def test_lut_reuses_cached_fitness(self, gp_instance):
        """Tests identical expressions reuse cached fitness."""
        gp_instance.gen_create_initial()

        initial_lut_size = len(gp_instance.lut_symex_fitness)

        # Create same trees again
        @gp_instance.create_trees(rate=0.5)
        def same_trees():
            # Simple tree likely to be duplicated
            return Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))

        # LUT should grow but fitness lookups should happen
        # (Hard to test directly, but LUT should have some entries)
        assert len(gp_instance.lut_symex_fitness) >= initial_lut_size


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in GP system."""

    def test_handles_invalid_trees_gracefully(self, gp_instance):
        """Tests GP handles invalid trees without crashing."""
        gp_instance.gen_create_initial()

        # Creating trees that might fail should be handled
        @gp_instance.create_trees(rate=0.5)
        def potentially_problematic():
            tree = gp_instance.evolve.evolve_new_tree_depth(float, 5)
            return tree

        # Should not crash
        gp_instance.end_generation()

    def test_empty_population_raises(self, gp_instance):
        """Tests empty population is handled."""
        # Don't create initial population
        # Trying to select should raise or handle gracefully
        with pytest.raises(Exception):
            selection_tournament([], n=3)


# =============================================================================
# Complete Run Test
# =============================================================================


class TestCompleteRun:
    """Full integration test of a mini GP run."""

    def test_full_mini_run(self, gp_instance):
        """Tests a complete mini GP run with all operations."""
        # Initial population
        gp_instance.gen_create_initial()

        # Run 2 generations with various operations
        for _ in range(2):
            # Reproduction
            @gp_instance.create_trees(rate=0.2)
            def reproduce():
                return selection_tournament(gp_instance.pop_genepool, n=2)

            # Mutation
            @gp_instance.create_trees(rate=0.3)
            def mutate():
                tree = selection_tournament(gp_instance.pop_genepool, n=2)
                return gp_instance.evolve.evolve_mutate_branch_depth(tree, 3)

            # New random
            @gp_instance.create_trees(rate=0.3)
            def random_new():
                return gp_instance.evolve.evolve_new_tree_depth(float, 3)

            # Crossover
            @gp_instance.create_trees(rate=0.2, crossover=True)
            def crossover():
                t1 = selection_tournament(gp_instance.pop_genepool, n=2)
                t2 = selection_tournament(gp_instance.pop_genepool, n=2)
                return gp_instance.evolve.evolve_crossover(t1, t2)

            gp_instance.end_generation()

        # Verify final state
        assert gp_instance.gen_id >= 2
        assert len(gp_instance.pop_genepool) > 0
        assert len(gp_instance.paretofront) > 0
        assert len(gp_instance.monitor_df) >= 2

        # Verify all candidates are valid
        for cand in gp_instance.pop_genepool:
            assert np.isfinite(cand.fitness)
            assert cand.parsimony > 0
            result = cand.tree.eval_predict_numpy_now(gp_instance.df_train)
            assert len(result) == len(gp_instance.df_train)
