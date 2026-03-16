"""
Tests for Node pickle optimization and repair_all() correctness.

Verifies that:
- __getstate__ excludes parent_node/root_node
- __setstate__ sets them to None
- repair_all() restores back-references after deserialization
- deepcopy also strips back-refs (uses __getstate__)
- Candidate pickle roundtrip preserves tree structure

See docs/PITFALLS.md P1, P2.
"""

import copy
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plagih.trees import Add, Candidate, Mul, Number, Sin, Symbol

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_tree():
    """Add(x, 3) – minimal tree with parent/root refs."""
    x = Symbol(is_fix=True)
    x.childs = [pytest.importorskip("sympy").Symbol("x", real=True)]
    n = Number()
    n.childs = [3.0]
    tree = Add(x, n)
    tree.repair_all()
    return tree


@pytest.fixture
def nested_tree():
    """Mul(Sin(x), Add(x, 5)) – deeper tree."""
    import sympy as sp

    x = Symbol(is_fix=True)
    x.childs = [sp.Symbol("x", real=True)]
    x2 = Symbol(is_fix=True)
    x2.childs = [sp.Symbol("x", real=True)]
    n = Number()
    n.childs = [5.0]
    inner_add = Add(x2, n)
    inner_sin = Sin(x)
    tree = Mul(inner_sin, inner_add)
    tree.repair_all()
    return tree


# =============================================================================
# __getstate__ / __setstate__
# =============================================================================


class TestGetState:
    """Tests for Node.__getstate__ pickle exclusion."""

    def test_getstate_excludes_parent(self, simple_tree):
        """parent_node must not appear in pickle state."""
        child = simple_tree.get_childs()[0]
        state = child.__getstate__()
        assert "parent_node" not in state

    def test_getstate_excludes_root(self, simple_tree):
        """root_node must not appear in pickle state."""
        child = simple_tree.get_childs()[0]
        state = child.__getstate__()
        assert "root_node" not in state

    def test_getstate_preserves_other_fields(self, simple_tree):
        """Other fields (childs, depth, is_fix) must be preserved."""
        state = simple_tree.__getstate__()
        assert "childs" in state
        assert "depth" in state

    def test_setstate_sets_none(self, simple_tree):
        """After __setstate__, parent_node and root_node should be None."""
        state = simple_tree.__getstate__()
        new_node = Add.__new__(Add)
        new_node.__setstate__(state)
        assert new_node.parent_node is None
        assert new_node.root_node is None


# =============================================================================
# Pickle roundtrip
# =============================================================================


class TestPickleRoundtrip:
    """Tests for pickle serialization/deserialization of trees."""

    def test_pickle_roundtrip_preserves_structure(self, simple_tree):
        """Tree structure (childs) survives pickle roundtrip."""
        data = pickle.dumps(simple_tree)
        restored = pickle.loads(data)
        assert len(restored.get_childs()) == len(simple_tree.get_childs())

    def test_pickle_strips_backrefs(self, simple_tree):
        """After unpickling, parent_node/root_node are None."""
        data = pickle.dumps(simple_tree)
        restored = pickle.loads(data)
        for child in restored.get_childs():
            assert child.parent_node is None
            assert child.root_node is None

    def test_repair_all_restores_backrefs(self, simple_tree):
        """repair_all() restores parent/root after pickle."""
        data = pickle.dumps(simple_tree)
        restored = pickle.loads(data)
        restored.repair_all()

        # Root's parent should be None (it IS the root)
        assert restored.parent_node is None
        # Children should point back to root
        for child in restored.get_childs():
            assert child.parent_node is restored

    def test_nested_tree_pickle_roundtrip(self, nested_tree):
        """Deeper tree: repair_all fixes all levels."""
        data = pickle.dumps(nested_tree)
        restored = pickle.loads(data)
        restored.repair_all()

        # Root
        assert restored.parent_node is None
        # Level 1
        for child in restored.get_childs():
            assert child.parent_node is restored
            # Level 2
            if child.has_childs():
                for grandchild in child.get_childs():
                    if hasattr(grandchild, "parent_node"):
                        assert grandchild.parent_node is child

    def test_pickle_size_smaller_without_backrefs(self, nested_tree):
        """Excluding back-refs should produce smaller pickle output."""
        # Pickle with __getstate__ (current)
        data_optimized = pickle.dumps(nested_tree)

        # Verify it's a reasonable size (should be < 2KB for a small tree)
        assert len(data_optimized) < 5000

    def test_candidate_pickle_roundtrip(self, simple_tree):
        """Candidate wrapping a tree survives pickle."""
        candidate = Candidate(simple_tree, fitness=0.5, parsimony=3, tag="test")
        data = pickle.dumps(candidate)
        restored = pickle.loads(data)

        assert restored.fitness == 0.5
        assert restored.parsimony == 3
        assert len(restored.tree.get_childs()) == 2

        # Repair and verify
        restored.tree.repair_all()
        for child in restored.tree.get_childs():
            assert child.parent_node is restored.tree


# =============================================================================
# deepcopy (also uses __getstate__)
# =============================================================================


class TestDeepcopy:
    """Tests that deepcopy also strips back-refs (P2)."""

    def test_deepcopy_strips_backrefs(self, simple_tree):
        """deepcopy uses __getstate__, so back-refs are stripped."""
        copied = copy.deepcopy(simple_tree)
        for child in copied.get_childs():
            assert child.parent_node is None

    def test_deepcopy_is_independent(self, simple_tree):
        """deepcopy'd tree is fully independent of original."""
        copied = copy.deepcopy(simple_tree)
        copied.repair_all()

        # Modifying copy doesn't affect original
        assert copied is not simple_tree
        assert copied.get_childs()[0] is not simple_tree.get_childs()[0]

    def test_deepcopy_repair_all_works(self, nested_tree):
        """repair_all on deepcopy'd tree sets correct back-refs."""
        copied = copy.deepcopy(nested_tree)
        copied.repair_all()

        assert copied.parent_node is None
        for child in copied.get_childs():
            assert child.parent_node is copied
