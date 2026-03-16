"""
Tests for GPMonitor metrics tracking and callbacks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plagih.monitoring import GenerationMetrics, GPMonitor
from plagih.trees import Add, Candidate, Number, Symbol


def _make_candidate(fitness: float, parsimony: int) -> Candidate:
    """Create a minimal Candidate."""
    import sympy

    x = Symbol(is_fix=True)
    x.childs = [sympy.Symbol("x", real=True)]
    tree = Add(x, Number())
    tree.get_childs()[1].childs = [1.0]
    tree.repair_all()
    return Candidate(tree, fitness=fitness, parsimony=parsimony, tag="test")


def _make_population(n: int = 10) -> list:
    """Create a small population with varying fitness/parsimony."""
    return [_make_candidate(fitness=float(i) * 0.1 + 0.5, parsimony=i + 1) for i in range(n)]


# =============================================================================
# GenerationMetrics
# =============================================================================


class TestGenerationMetrics:
    def test_dict_access(self):
        m = GenerationMetrics(gen_id=0, timestamp=0.0)
        m["fit_mean"] = 1.5
        assert m["fit_mean"] == 1.5

    def test_get_with_default(self):
        m = GenerationMetrics(gen_id=0, timestamp=0.0)
        assert m.get("nonexistent", 42) == 42


# =============================================================================
# GPMonitor core
# =============================================================================


class TestGPMonitor:
    def test_empty_monitor(self):
        monitor = GPMonitor()
        assert len(monitor.generations) == 0

    def test_record_generation(self):
        monitor = GPMonitor()
        pop = _make_population()
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0)
        assert len(monitor.generations) == 1
        assert monitor.generations[0].gen_id == 0

    def test_latest_property(self):
        monitor = GPMonitor()
        pop = _make_population()
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0)
        monitor.record_generation(gen_id=1, population=pop, gen_time=0.8)
        assert monitor.latest.gen_id == 1

    def test_metrics_computed(self):
        """Standard metrics (fit_mean, parsim_mean, etc.) should be auto-computed."""
        monitor = GPMonitor(auto_compute=True)
        pop = _make_population()
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0)
        metrics = monitor.generations[0]
        # Check that standard metrics exist
        assert "fit_mean" in metrics.metrics
        assert "parsim_mean" in metrics.metrics
        assert "gen_time" in metrics.metrics

    def test_pareto_update_tracking(self):
        monitor = GPMonitor()
        pop = _make_population()
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0, pareto_updated=True)
        assert monitor.gens_since_last_pareto == 0

        monitor.record_generation(gen_id=1, population=pop, gen_time=1.0, pareto_updated=False)
        assert monitor.gens_since_last_pareto == 1

    def test_to_dataframe(self):
        monitor = GPMonitor()
        pop = _make_population()
        for i in range(3):
            monitor.record_generation(gen_id=i, population=pop, gen_time=1.0)
        df = monitor.to_dataframe()
        assert len(df) == 3
        # gen_id is the index, not a column
        assert df.index.name == "gen_id"
        assert "gen_id" not in df.columns
        assert list(df.index) == [0, 1, 2]

    def test_to_dataframe_empty(self):
        monitor = GPMonitor()
        df = monitor.to_dataframe()
        assert len(df) == 0
        assert df.index.name == "gen_id"
        assert "gen_id" not in df.columns


# =============================================================================
# Callbacks
# =============================================================================


class TestCallbacks:
    def test_on_generation_callback(self):
        monitor = GPMonitor()
        called = []
        monitor.on_generation(lambda m: called.append(m.gen_id))

        pop = _make_population()
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0)
        assert called == [0]

    def test_on_pareto_update_callback(self):
        monitor = GPMonitor()
        pareto_calls = []
        monitor.on_pareto_update(lambda m: pareto_calls.append(m.gen_id))

        pop = _make_population()
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0, pareto_updated=False)
        assert pareto_calls == []

        monitor.record_generation(gen_id=1, population=pop, gen_time=1.0, pareto_updated=True)
        assert pareto_calls == [1]

    def test_on_improvement_callback(self):
        monitor = GPMonitor()
        improvements = []
        monitor.on_improvement(lambda m, amt: improvements.append(amt))

        pop1 = [_make_candidate(2.0, 5)]
        monitor.record_generation(gen_id=0, population=pop1, gen_time=1.0)

        pop2 = [_make_candidate(1.0, 5)]  # Better fitness
        monitor.record_generation(gen_id=1, population=pop2, gen_time=1.0)

        assert len(improvements) >= 1
        assert improvements[-1] > 0  # improvement amount is positive

    def test_custom_metric(self):
        monitor = GPMonitor()
        monitor.add_custom_metric("max_parsim", lambda pop: max(c.parsimony for c in pop))

        pop = _make_population(5)
        monitor.record_generation(gen_id=0, population=pop, gen_time=1.0)
        assert monitor.generations[0].get("max_parsim") == 5
