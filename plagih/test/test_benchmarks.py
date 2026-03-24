"""
Tests for Benchmark Demo Functions

Tests that each benchmark demo runs correctly with minimal settings.
These are integration tests that verify the full GP pipeline works.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from plagih.trees import Abs, Add, And, Div, ExplainableGP, Ifte, Le, Lt, Mul, Or, Square, Sub, selection_tournament

# =============================================================================
# Test Data Paths
# =============================================================================

BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"


def clean_column_names(df):
    """Remove :float suffix from column names if present."""
    df.columns = [col.replace(":float", "") for col in df.columns]
    return df


# =============================================================================
# MountainCar Benchmark Tests
# =============================================================================


class TestMountainCarBenchmark:
    """Tests for the MountainCar benchmark."""

    def test_data_exists(self):
        """Test that MountainCar data files exist."""
        mc_dir = BENCHMARKS_DIR / "mc" / "gp_files"
        assert mc_dir.exists(), f"MountainCar directory not found: {mc_dir}"
        assert (mc_dir / "samples200.csv").exists(), "samples200.csv not found"

    def test_data_format(self):
        """Test that MountainCar data has correct format."""
        df = pd.read_csv(BENCHMARKS_DIR / "mc" / "gp_files" / "samples200.csv")
        df = clean_column_names(df)

        # Check columns
        assert "cartPos" in df.columns, f"Missing cartPos column. Got: {list(df.columns)}"
        assert "cartVel" in df.columns, f"Missing cartVel column. Got: {list(df.columns)}"

    @pytest.mark.performance
    def test_minimal_gp_run(self, tmp_path):
        """Test a minimal GP run with MountainCar data."""
        # Load data
        df = pd.read_csv(BENCHMARKS_DIR / "mc" / "gp_files" / "samples200.csv")
        df = clean_column_names(df)
        df = df.astype("float32")
        df_train, _ = train_test_split(df, test_size=0.2, random_state=42)

        # Operator set with both float and bool
        operators = {Add: 1, Mul: 1, Sub: 1, Lt: 1}

        # Create GP with minimal settings
        gp = ExplainableGP.create(
            symbols=["cartPos", "cartVel"],
            df_train=df_train,
            rootdir=tmp_path / "mc_test",
            operators=operators,
            depth_max=3,
            nodes_max=10,
            pop_max_size=5,
            gen_end=2,
            clip_range=(0.0, 2.0),
            error_metric="rmse",
        )

        # Create initial population
        gp.gen_create_initial()
        assert len(gp.pop_genepool) > 0, "Population should not be empty"

        # Run one generation
        @gp.create_trees(rate=1.0)
        def mutation():
            tree = selection_tournament(gp.pop_genepool, n=2)
            return gp.evolve.evolve_mutate_branch_depth(tree, depth_goal=2, p_term=0.3)

        gp.end_generation()

        # Check results
        assert len(gp.paretofront) > 0, "Pareto front should not be empty"


# =============================================================================
# CartPole Benchmark Tests
# =============================================================================


class TestCartPoleBenchmark:
    """Tests for the CartPole benchmark."""

    def test_data_exists(self):
        """Test that CartPole data files exist."""
        cp_dir = BENCHMARKS_DIR / "cp" / "gp_files"
        assert cp_dir.exists(), f"CartPole directory not found: {cp_dir}"
        assert (cp_dir / "samples.csv").exists(), "samples.csv not found"

    def test_data_format(self):
        """Test that CartPole data has correct format."""
        df = pd.read_csv(BENCHMARKS_DIR / "cp" / "gp_files" / "samples.csv")
        df = clean_column_names(df)

        # Check columns
        assert "cartPos" in df.columns, f"Missing cartPos column. Got: {list(df.columns)}"
        assert "cartVel" in df.columns, f"Missing cartVel column. Got: {list(df.columns)}"

        # Check we have enough samples
        assert len(df) > 1000, f"Expected >1000 samples, got {len(df)}"

    @pytest.mark.performance
    def test_minimal_gp_run(self, tmp_path):
        """Test a minimal GP run with CartPole data."""
        # Load data
        df = pd.read_csv(BENCHMARKS_DIR / "cp" / "gp_files" / "samples.csv")
        df = clean_column_names(df)
        df = df.astype("float32")

        # Rename for clarity
        df = df.rename(columns={"observation2": "poleAngle", "observation3": "poleVel", "action0": "action"})

        # Use subset for speed
        df_small = df.sample(n=200, random_state=42)
        df_train, _ = train_test_split(df_small, test_size=0.2, random_state=42)

        # Operator set with both float and bool
        operators = {Add: 1, Mul: 1, Lt: 1}

        # Create GP with minimal settings
        gp = ExplainableGP.create(
            symbols=["cartPos", "cartVel", "poleAngle", "poleVel"],
            df_train=df_train,
            rootdir=tmp_path / "cp_test",
            operators=operators,
            depth_max=3,
            nodes_max=8,
            pop_max_size=5,
            gen_end=2,
            clip_range=(0.0, 1.0),  # Binary classification
            error_metric="rmse",
        )

        # Create initial population
        gp.gen_create_initial()
        assert len(gp.pop_genepool) > 0, "Population should not be empty"

        # Run one generation
        @gp.create_trees(rate=1.0)
        def mutation():
            tree = selection_tournament(gp.pop_genepool, n=2)
            return gp.evolve.evolve_mutate_branch_depth(tree, depth_goal=2, p_term=0.3)

        gp.end_generation()

        # Check results
        assert len(gp.paretofront) > 0, "Pareto front should not be empty"


# =============================================================================
# Symbolic Regression Benchmark Tests
# =============================================================================


class TestSymbolicRegressionBenchmark:
    """Tests for the Symbolic Regression benchmark."""

    def test_data_exists(self):
        """Test that Symbolic Regression data files exist."""
        sr_dir = BENCHMARKS_DIR / "sr" / "gp_files"
        assert sr_dir.exists(), f"Symbolic Regression directory not found: {sr_dir}"
        assert (sr_dir / "polynomial.csv").exists(), "polynomial.csv not found"

    def test_data_format(self):
        """Test that Symbolic Regression data has correct format."""
        df = pd.read_csv(BENCHMARKS_DIR / "sr" / "gp_files" / "polynomial.csv")
        df = clean_column_names(df)

        # Check columns
        assert "x" in df.columns, f"Missing x column. Got: {list(df.columns)}"
        assert "action" in df.columns, f"Missing action column. Got: {list(df.columns)}"

        # Check data range
        assert df["x"].min() >= -2.1, "x should be >= -2"
        assert df["x"].max() <= 2.1, "x should be <= 2"

    def test_target_function(self):
        """Test that target values match f(x) = x³ + x² + x."""
        df = pd.read_csv(BENCHMARKS_DIR / "sr" / "gp_files" / "polynomial.csv")
        df = clean_column_names(df)

        # Calculate expected values
        x = df["x"].values
        expected = x**3 + x**2 + x

        # Check match (allowing for rounding)
        np.testing.assert_array_almost_equal(
            df["action"].values, expected, decimal=2, err_msg="Target values don't match f(x) = x³ + x² + x"
        )

    @pytest.mark.performance
    def test_minimal_gp_run(self, tmp_path):
        """Test a minimal GP run with Symbolic Regression data."""
        # Load data
        df = pd.read_csv(BENCHMARKS_DIR / "sr" / "gp_files" / "polynomial.csv")
        df = clean_column_names(df)
        df = df.astype("float32")
        df_train, _ = train_test_split(df, test_size=0.2, random_state=42)

        # Math-focused operators with bool
        operators = {Add: 2, Mul: 2, Sub: 1, Square: 1, Lt: 1}

        # Create GP with minimal settings
        gp = ExplainableGP.create(
            symbols=["x"],
            df_train=df_train,
            rootdir=tmp_path / "sr_test",
            operators=operators,
            depth_max=4,
            nodes_max=10,
            pop_max_size=5,
            gen_end=2,
            error_metric="mse",
        )

        # Create initial population
        gp.gen_create_initial()
        assert len(gp.pop_genepool) > 0, "Population should not be empty"

        # Run one generation
        @gp.create_trees(rate=1.0)
        def mutation():
            tree = selection_tournament(gp.pop_genepool, n=2)
            return gp.evolve.evolve_mutate_branch_depth(tree, depth_goal=2, p_term=0.2)

        gp.end_generation()

        # Check results
        assert len(gp.paretofront) > 0, "Pareto front should not be empty"


# =============================================================================
# Industrial Benchmark Tests (Data only - no GP run due to complexity)
# =============================================================================


class TestIndustrialBenchmark:
    """Tests for the Industrial Benchmark (data checks only)."""

    def test_directory_exists(self):
        """Test that Industrial Benchmark directory exists."""
        ib_dir = BENCHMARKS_DIR / "ib"
        assert ib_dir.exists(), f"Industrial Benchmark directory not found: {ib_dir}"

    def test_gp_files_exist(self):
        """Test that GP files directory exists."""
        gp_dir = BENCHMARKS_DIR / "ib" / "gp_files"
        assert gp_dir.exists(), f"GP files directory not found: {gp_dir}"


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
