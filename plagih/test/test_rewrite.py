"""Tests for plagih.trees._rewrite (D11 local rewrite engine)."""

import numpy as np
import pandas as pd
import pytest
import sympy

from plagih.trees import Abs, Add, Boolean, Div, Max, Min, Mul, Not, Number, Sub, Symbol, Usub
from plagih.trees._rewrite import rewrite_fixpoint


def sym(name="a"):
    return Symbol(sympy.Symbol(name))


def df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


class TestConstantFolding:
    def test_add(self):
        tree = Add(Number(2.0), Number(3.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Number)
        assert float(out.get_value()) == pytest.approx(5.0)

    def test_nested_folding(self):
        tree = Mul(Add(Number(1.0), Number(2.0)), Number(4.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Number)
        assert float(out.get_value()) == pytest.approx(12.0)

    def test_abs_usub(self):
        tree = Abs(Usub(Number(4.0)))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Number)
        assert float(out.get_value()) == pytest.approx(4.0)

    def test_min_max(self):
        tree = Min(Number(3.0), Max(Number(1.0), Number(7.0)))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert float(out.get_value()) == pytest.approx(3.0)

    def test_div_by_zero_untouched(self):
        tree = Div(Number(1.0), Number(0.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Div)  # no fold, no crash

    def test_symbols_prevent_folding(self):
        tree = Add(sym(), Number(3.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Add)


class TestNeutralElements:
    def test_add_zero(self):
        tree = Add(sym(), Number(0.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Symbol)

    def test_add_chain_drops_zeros(self):
        tree = Add(sym("a"), Number(0.0), sym("b"), Number(0.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Add)
        assert len(out.get_childs()) == 2

    def test_mul_one(self):
        tree = Mul(sym(), Number(1.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Symbol)

    def test_mul_zero_absorbs(self):
        tree = Mul(sym(), Number(0.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Number)
        assert float(out.get_value()) == 0.0

    def test_sub_zero(self):
        tree = Sub(sym(), Number(0.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Symbol)

    def test_div_one(self):
        tree = Div(sym(), Number(1.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Symbol)

    def test_double_usub(self):
        tree = Usub(Usub(sym()))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Symbol)

    def test_double_not(self):
        tree = Not(Not(Boolean(True)))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Boolean)


class TestGuarantees:
    def test_idempotent(self):
        tree = Add(Mul(sym(), Number(1.0)), Number(0.0))
        tree.repair_all()
        once = rewrite_fixpoint(tree)
        first = str(once)
        twice = rewrite_fixpoint(once)
        assert str(twice) == first

    def test_never_grows(self):
        trees = [
            Add(sym(), Number(2.0)),
            Mul(Add(sym("a"), sym("b")), Number(1.0)),
            Sub(Div(sym(), Number(1.0)), Number(0.0)),
        ]
        for t in trees:
            t.repair_all()
            before = len(t)
            after = len(rewrite_fixpoint(t))
            assert after <= before

    def test_semantics_preserved(self):
        """NumPy eval before == after on every rewritten tree."""
        d = df()
        trees = [
            Add(sym("a"), Number(0.0)),
            Mul(sym("a"), Number(1.0)),
            Mul(sym("b"), Number(0.0)),
            Sub(sym("a"), Number(0.0)),
            Div(sym("b"), Number(1.0)),
            Usub(Usub(sym("a"))),
            Add(Mul(Number(2.0), Number(3.0)), sym("a")),
        ]
        for t in trees:
            t.repair_all()
            before = t.eval_predict_numpy_now(d).copy()
            out = rewrite_fixpoint(t)
            after = out.eval_predict_numpy_now(d)
            np.testing.assert_allclose(after, before, rtol=1e-12)

    def test_fix_nodes_untouched(self):
        """Frozen nodes must survive (origin_tree skeletons)."""
        inner = Add(sym(), Number(0.0))
        inner.is_fix = True
        tree = Mul(inner, Number(1.0))
        tree.repair_all()

        out = rewrite_fixpoint(tree)
        # Mul(.., 1) may resolve, but the fixed Add must stay intact
        found = [n for n in out.to_traversal("pre") if isinstance(n, Add)]
        assert found and found[0].is_fix
        assert len(found[0].get_childs()) == 2

    def test_fix_number_not_folded(self):
        n = Number(2.0)
        n.is_fix = True
        tree = Add(n, Number(3.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert isinstance(out, Add)  # fixed operand blocks folding

    def test_root_identity_preserved(self):
        tree = Add(sym(), Number(0.0))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        assert out is tree  # set_new_node swaps in place

    def test_backrefs_repaired(self):
        tree = Add(Mul(sym(), Number(1.0)), sym("b"))
        tree.repair_all()
        out = rewrite_fixpoint(tree)
        for child in out.get_childs():
            assert child.parent_node is out


class TestPipelineIntegration:
    def test_tree_simplification_uses_rewrites(self):
        from plagih.trees import tree_simplification

        tree = Add(Mul(sym(), Number(1.0)), Number(0.0))
        tree.repair_all()
        before = len(tree)

        out = tree_simplification(tree, allow_chain=False)
        assert len(out) < before

        d = df()
        np.testing.assert_allclose(
            out.eval_predict_numpy_now(d),
            d["a"].to_numpy(),
            rtol=1e-12,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
