"""
Tests for Pareto-front dominance filtering.

Tests pareto_from_pop() with various edge cases.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plagih.paretofront import pareto_from_pop
from plagih.trees import Add, Candidate, Number, Symbol


def _make_candidate(fitness: float, parsimony: int) -> Candidate:
    """Create a minimal Candidate with given fitness and parsimony."""
    x = Symbol(is_fix=True)
    import sympy

    x.childs = [sympy.Symbol("x", real=True)]
    tree = Add(x, Number())
    tree.get_childs()[1].childs = [1.0]
    return Candidate(tree, fitness=fitness, parsimony=parsimony, tag="test")


class TestParetoFromPop:
    """Tests for pareto_from_pop dominance filter."""

    def test_empty_raises(self):
        """Empty population should raise."""
        with pytest.raises(Exception):
            pareto_from_pop([])

    def test_single_candidate(self):
        """Single candidate is always on the front."""
        c = _make_candidate(1.0, 5)
        result = pareto_from_pop([c])
        assert len(result) == 1
        assert result[0] is c

    def test_dominated_removed(self):
        """A dominated candidate should not appear in the front."""
        c1 = _make_candidate(1.0, 5)  # better in both
        c2 = _make_candidate(2.0, 10)  # dominated
        result = pareto_from_pop([c1, c2])
        assert len(result) == 1
        assert result[0].get_fitness() == 1.0

    def test_non_dominated_both_kept(self):
        """Two non-dominated candidates should both appear."""
        c1 = _make_candidate(1.0, 10)  # better fitness, worse parsimony
        c2 = _make_candidate(2.0, 5)  # worse fitness, better parsimony
        result = pareto_from_pop([c1, c2])
        assert len(result) == 2

    def test_equal_candidates(self):
        """Equal candidates: only one survives (not dominated, but no strict <)."""
        c1 = _make_candidate(1.0, 5)
        c2 = _make_candidate(1.0, 5)
        result = pareto_from_pop([c1, c2])
        # Both are non-dominated w.r.t. each other (dominates requires strict <)
        assert len(result) == 2

    def test_three_candidates_chain(self):
        """c1 dominates c2, c2 dominates c3 → only c1 survives."""
        c1 = _make_candidate(1.0, 3)
        c2 = _make_candidate(2.0, 5)
        c3 = _make_candidate(3.0, 7)
        result = pareto_from_pop([c1, c2, c3])
        assert len(result) == 1

    def test_pareto_front_shape(self):
        """Classic L-shaped front: no candidate dominates another."""
        candidates = [
            _make_candidate(5.0, 1),
            _make_candidate(3.0, 3),
            _make_candidate(1.0, 5),
            _make_candidate(0.5, 10),
        ]
        result = pareto_from_pop(candidates)
        assert len(result) == 4

    def test_large_population(self):
        """Pareto front with many dominated candidates."""
        import random

        random.seed(42)
        pop = [_make_candidate(random.uniform(0, 10), random.randint(1, 20)) for _ in range(100)]
        result = pareto_from_pop(pop)
        # Front should be smaller than population
        assert len(result) < len(pop)
        # Every result should be non-dominated by every other result
        for i, a in enumerate(result):
            for j, b in enumerate(result):
                if i == j:
                    continue
                # Neither should dominate the other
                assert not (
                    a.get_parsim() <= b.get_parsim()
                    and a.get_fitness() <= b.get_fitness()
                    and (a.get_parsim() < b.get_parsim() or a.get_fitness() < b.get_fitness())
                )
